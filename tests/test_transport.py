"""Testscript voor de transport-markering (excluded_reason = "transport").

Gebruik:  python test_transport.py

Draait volledig op een tijdelijke database en een tijdelijke memory-map;
raakt de echte data niet aan. Controleert:

1. suggestie: een korte, rustige fietsrit (6 km, HR onder zone 2) krijgt een
   transport-suggestie; een echte woon-werkrit (zone 2, 37 km), een korte
   harde rit en een indoorrit niet;
2. markeren: de sessie verdwijnt uit training_activities maar blijft in
   load_activities, en het trainingslog krijgt een statusregel;
3. analyses: weektotalen tellen de rit mee (aparte categorie "transport"),
   trends/records niet;
4. brick-detectie: een gemarkeerde rit ketent niet meer met een zwemsessie,
   en een al openstaand voorstel met die rit wordt ingetrokken;
5. feedback-context: de rit doet niet mee als "vorige vergelijkbare sessie",
   maar staat wél (gemarkeerd) in het belastingsoverzicht;
6. demarkeren: alles telt weer gewoon mee.
"""

# De tests staan in tests/; zet de projectroot op sys.path zodat
# `python tests/test_<naam>.py` het tricoach-package kan importeren.
import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))


import tempfile
from datetime import datetime
from pathlib import Path

import pandas as pd

from tricoach.analysis import weekly_totals, weekly_volume
from tricoach.combos import combo_membership, detect_and_store_proposals
from tricoach.feedback_context import recent_load_block, similar_sessions_block
from tricoach.fit_parser import ParsedActivity
from tricoach.progress import personal_records
from tricoach.storage import (
    connect,
    load_activities,
    save_activity,
    training_activities,
)
from tricoach.transport import (
    mark_transport,
    suggest_transport,
    unmark_transport,
)

BOUNDS = [136, 152, 162, 171]  # zones bij LTHR 170 (Z2 = 136-151)
CONFIG = {"athlete": {"lthr": 170, "max_hr": 193,
                      "zone_pct_lthr": [0.8, 0.89, 0.95, 1.0]}}


def _act(sport: str, dag: int, uur: int, minuut: int, afstand_m: float,
         avg_hr: int, duur_s: float = 1500.0,
         sub_sport: str | None = None) -> ParsedActivity:
    # UTC-aware, net als echte FIT-starttijden (file_id.time_created).
    start = pd.Timestamp(datetime(2026, 7, dag, uur, minuut, 0), tz="UTC")
    return ParsedActivity(
        activity_key=start.isoformat(),
        sport=sport, sub_sport=sub_sport, start_time=start,
        summary={"total_timer_time": duur_s, "total_distance": afstand_m,
                 "avg_heart_rate": avg_hr, "max_heart_rate": avg_hr + 20,
                 "sport": sport},
        records=pd.DataFrame(), lengths=pd.DataFrame(),
        source_file="12345678901_ACTIVITY.fit",
    )


def main() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="tricoach_transport_"))
    conn = connect(tmp / "test.db")
    memory_dir = tmp / "memory"
    memory_dir.mkdir()

    # 1. Suggestieheuristiek.
    zwembadrit = _act("cycling", 17, 8, 3, 6000, 128, duur_s=1080)
    woonwerk = _act("cycling", 16, 7, 30, 37000, 140, duur_s=4800)
    korte_harde = _act("cycling", 15, 18, 0, 8000, 165, duur_s=1200)
    indoor = _act("cycling", 14, 19, 0, 9000, 120, duur_s=1400,
                  sub_sport="indoor_cycling")
    looprondje = _act("running", 13, 7, 0, 5000, 130, duur_s=1600)
    assert suggest_transport(zwembadrit, CONFIG), \
        "korte rustige rit hoort een transport-suggestie te krijgen"
    assert not suggest_transport(woonwerk, CONFIG), "37 km zone 2 is een training"
    assert not suggest_transport(korte_harde, CONFIG), "hard = training"
    assert not suggest_transport(indoor, CONFIG), "indoor is nooit transport"
    assert not suggest_transport(looprondje, CONFIG), "alleen fietsen"
    print("1. transport-suggestie alleen voor korte, rustige buitenritten: OK")

    # Opzet voor de rest: zwemsessie + terugrit (brick-kandidaat) + training.
    zwem = _act("swimming", 17, 7, 0, 1000, 110, duur_s=2400)
    terugrit = _act("cycling", 17, 9, 3, 6000, 130, duur_s=1080)
    for a in (zwem, terugrit, woonwerk):
        save_activity(conn, a, BOUNDS)
    (memory_dir / "trainingslog.md").write_text(
        "# Trainingslog\n"
        f"\n## 2026-07-17 Fri 09:03 — Fietsen\n\n- **Kerncijfers:** x\n"
        f"- _Sleutel `{terugrit.activity_key}` · bron `a.fit`_\n",
        encoding="utf-8")

    # Brick-voorstel ontstaat zolang de rit niet gemarkeerd is (zwem → fiets
    # binnen het gat): dat is precies de situatie vóór bevestiging.
    acts = load_activities(conn)
    detect_and_store_proposals(conn, acts, gap_min=90)
    assert terugrit.activity_key in combo_membership(conn), \
        "zonder markering hoort er een brick-voorstel te ontstaan"

    # 2. Markeren: uit de trainingssubset, wel in het volledige overzicht.
    assert mark_transport(conn, memory_dir, terugrit.activity_key)
    acts = load_activities(conn)
    trainingen = training_activities(acts)
    assert terugrit.activity_key in acts["activity_key"].tolist()
    assert terugrit.activity_key not in trainingen["activity_key"].tolist()
    log = (memory_dir / "trainingslog.md").read_text(encoding="utf-8")
    assert "gemarkeerd als transport" in log
    print("2. markering filtert de rit uit de trainingssubset + logregel: OK")

    # 3. Weektotalen tellen mee (categorie transport), records/trends niet.
    totalen = weekly_totals(acts)
    assert totalen["sessies"].sum() == 3, "transport telt mee in de weektotalen"
    assert totalen["uren_transport"].sum() > 0, \
        "transporturen horen in hun eigen kolom"
    volume = weekly_volume(acts)
    assert "transport" in volume["sport"].tolist(), \
        "transport hoort als eigen categorie in het weekvolume"
    prs = personal_records(conn, trainingen)
    langste = prs.loc[prs["Onderdeel"] == "Langste rit", "Record"].iloc[0]
    assert langste == "37.0 km", \
        f"langste rit hoort de training (37 km) te zijn, niet {langste}"
    print("3. weektotalen incl. transport (aparte categorie), records excl.: OK")

    # 4. Brick-detectie: het openstaande voorstel is bij het markeren
    # ingetrokken en komt op de trainingssubset niet terug.
    assert terugrit.activity_key not in combo_membership(conn), \
        "markeren hoort het openstaande brick-voorstel in te trekken"
    detect_and_store_proposals(conn, training_activities(load_activities(conn)),
                               gap_min=90)
    assert terugrit.activity_key not in combo_membership(conn), \
        "een transport-rit mag niet opnieuw als brick worden voorgesteld"
    print("4. brick-voorstel ingetrokken en niet opnieuw voorgesteld: OK")

    # 5. Feedback-context: niet vergelijkbaar, wel (gemarkeerd) in belasting.
    latere_rit = _act("cycling", 18, 9, 0, 20000, 140, duur_s=2700)
    vergelijkbaar = similar_sessions_block(conn, latere_rit, {"Z2": 2000})
    assert "6.0 km" not in vergelijkbaar, \
        "transport mag geen vergelijkbare eerdere sessie zijn"
    belasting = recent_load_block(conn, latere_rit)
    assert "transport/verplaatsing" in belasting, \
        "belastingsoverzicht hoort transport gemarkeerd te tonen"
    print("5. feedback-context: geen vergelijking, wel belasting (gemarkeerd): OK")

    # 6. Demarkeren: telt weer mee.
    assert unmark_transport(conn, memory_dir, terugrit.activity_key)
    trainingen = training_activities(load_activities(conn))
    assert terugrit.activity_key in trainingen["activity_key"].tolist()
    log = (memory_dir / "trainingslog.md").read_text(encoding="utf-8")
    assert "transport-markering verwijderd" in log
    print("6. demarkeren maakt er weer een training van: OK")

    print("\nAlle controles geslaagd.")


if __name__ == "__main__":
    main()
