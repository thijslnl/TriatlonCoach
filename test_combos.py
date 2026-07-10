"""Testscript voor combinatietrainingen (bricks & triatlon-trainingen).

Gebruik:  python test_combos.py

Draait volledig op een tijdelijke database; raakt de echte data niet aan.
De sessies gaan via de echte pipeline (ParsedActivity -> save_activity ->
load_activities), alleen het FIT-parsen zelf wordt overgeslagen. Controleert:

1. fietsen + hardlopen binnen de drempel op dezelfde dag -> voorgesteld als
   brick, met wisseltijd (T2) en de bakstenen-benen-analyse van de loop;
2. zwemmen + fietsen + hardlopen kort na elkaar -> triatlon-training met T1
   en T2 (en het eerdere brick-voorstel wordt door het ruimere vervangen);
3. twee sessies op dezelfde dag met uren ertussen -> NIET samengevoegd,
   en een verkeerde volgorde (loop -> fiets) ook niet;
4. bevestigen en losmaken werken, een losgemaakte groep wordt niet opnieuw
   voorgesteld, en alleen bevestigde combos tellen in de trend;
5. de race-simulatie herkent de rookie-opzet en de feedback-context bevat
   de wissel- en overgangsdata.
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from tricoach.combos import (
    combo_block,
    combo_history,
    combo_membership,
    detect_and_store_proposals,
    load_combos,
    race_similarity,
    run_transition_analysis,
    set_combo_status,
)
from tricoach.fit_parser import ParsedActivity
from tricoach.storage import connect, load_activities, load_records, save_activity

BOUNDS = [137, 152, 162, 172]
CONFIG = {"combo": {"max_gap_min": 25}, "races": [],
          "athlete": {"lthr": 170, "max_hr": 193}}


def _act(sport: str, start: str, elapsed_s: float, distance_m: float,
         records: pd.DataFrame | None = None, sub_sport: str | None = None
         ) -> ParsedActivity:
    """Eén synthetische activiteit, zoals hij uit de FIT-parser zou komen."""
    ts = pd.Timestamp(start)
    return ParsedActivity(
        activity_key=ts.isoformat(), sport=sport, sub_sport=sub_sport,
        start_time=ts,
        summary={"total_timer_time": elapsed_s, "total_elapsed_time": elapsed_s,
                 "total_distance": distance_m, "avg_heart_rate": 150},
        records=records if records is not None else pd.DataFrame(),
        lengths=pd.DataFrame(), source_file=f"test_{sport}_{start}.fit",
    )


def _run_records(start: str, km1_s: float = 360.0, rest_m: float = 4500.0,
                 rest_s: float = 1440.0) -> pd.DataFrame:
    """Seconde-data van een loop: eerste km traag (HR hoog), daarna sneller.

    Standaard: km 1 in 6:00 bij HR 165, de rest op 5:20/km bij HR 158 —
    het klassieke bakstenen-benen-profiel.
    """
    t0 = pd.Timestamp(start)
    rows = []
    for s in range(int(km1_s) + 1):
        rows.append({"timestamp": t0 + pd.Timedelta(seconds=s),
                     "distance_m": 1000.0 * s / km1_s, "heart_rate": 165,
                     "speed_ms": 1000.0 / km1_s})
    for s in range(1, int(rest_s) + 1):
        rows.append({"timestamp": t0 + pd.Timedelta(seconds=km1_s + s),
                     "distance_m": 1000.0 + rest_m * s / rest_s, "heart_rate": 158,
                     "speed_ms": rest_m / rest_s})
    return pd.DataFrame(rows)


def main() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="tricoach_combo_test_"))
    conn = connect(tmp / "test.db")

    # -- Scenario 1: brick — fiets 17:00-18:00, loop start 18:12 (T2 = 12 min).
    brick_fiets = _act("cycling", "2026-07-01 17:00:00+00:00", 3600, 25000)
    brick_loop = _act("running", "2026-07-01 18:12:00+00:00", 1800, 5500,
                      records=_run_records("2026-07-01 18:12:00+00:00"))
    # -- Scenario 3: zelfde dag, uren ertussen -> géén combo; en verkeerde
    #    volgorde (loop 08:00, fiets er direct achteraan) -> ook geen combo.
    los_fiets = _act("cycling", "2026-07-03 08:00:00+00:00", 3600, 25000)
    los_loop = _act("running", "2026-07-03 15:00:00+00:00", 1800, 5000)
    orde_loop = _act("running", "2026-07-04 08:00:00+00:00", 1800, 5000)
    orde_fiets = _act("cycling", "2026-07-04 08:40:00+00:00", 3600, 20000)

    for act in (brick_fiets, brick_loop, los_fiets, los_loop, orde_loop, orde_fiets):
        save_activity(conn, act, BOUNDS)

    acts = load_activities(conn)
    nieuw = detect_and_store_proposals(conn, acts, 25)
    combos = load_combos(conn, acts)
    assert nieuw == 1 and len(combos) == 1, f"verwacht 1 brick-voorstel, kreeg {nieuw}/{len(combos)}"
    brick = combos[0]
    assert brick["kind"] == "brick" and brick["status"] == "voorgesteld"
    assert len(brick["transitions"]) == 1
    t2 = brick["transitions"][0]
    assert t2["label"] == "T2" and abs(t2["seconds"] - 720) < 1, t2
    print(f"1. Brick voorgesteld: T2 = {t2['seconds']:.0f} s ✓ "
          "(losse en verkeerd geordende sessies niet samengevoegd ✓)")

    # Bakstenen-benen-analyse van de loop.
    analyse = run_transition_analysis(load_records(conn, brick_loop.activity_key))
    assert analyse is not None and analyse["basis"].startswith("eerste 1")
    e, r = analyse["eerste"], analyse["rest"]
    assert abs(e["tempo_s_per_km"] - 360) < 5 and abs(r["tempo_s_per_km"] - 320) < 5
    assert abs(e["gem_hr"] - 165) < 1 and abs(r["gem_hr"] - 158) < 1
    print(f"2. Bakstenen benen: {e['tempo_s_per_km']:.0f} -> "
          f"{r['tempo_s_per_km']:.0f} s/km, HR {e['gem_hr']:.0f} -> "
          f"{r['gem_hr']:.0f} ✓")

    # -- Scenario 2: triatlon-training — zwem (open water) 08:00-08:20,
    #    fiets 08:25 (T1 = 5 min), loop 09:13 (T2 = 8 min na fietseinde 09:05).
    tri_zwem = _act("swimming", "2026-07-05 08:00:00+00:00", 1200, 500,
                    sub_sport="open_water")
    tri_fiets = _act("cycling", "2026-07-05 08:25:00+00:00", 2400, 20000)
    tri_loop = _act("running", "2026-07-05 09:13:00+00:00", 1800, 5500,
                    records=_run_records("2026-07-05 09:13:00+00:00"))
    # Eerst alleen fiets+loop opslaan: dat geeft een brick-voorstel dat daarna
    # door het ruimere triatlon-voorstel vervangen moet worden zodra het
    # zwem-bestand (later geüpload) erbij komt.
    save_activity(conn, tri_fiets, BOUNDS)
    save_activity(conn, tri_loop, BOUNDS)
    detect_and_store_proposals(conn, load_activities(conn), 25)
    save_activity(conn, tri_zwem, BOUNDS)
    detect_and_store_proposals(conn, load_activities(conn), 25)

    acts = load_activities(conn)
    combos = load_combos(conn, acts)
    tri = [c for c in combos if c["kind"] == "triatlon"]
    assert len(tri) == 1 and len(combos) == 2, \
        f"verwacht brick + triatlon, kreeg {[(c['kind'], c['status']) for c in combos]}"
    tri = tri[0]
    labels = {t["label"]: t["seconds"] for t in tri["transitions"]}
    assert abs(labels["T1"] - 300) < 1 and abs(labels["T2"] - 480) < 1, labels
    # Totaal = eerste start (08:00) tot laatste einde (09:43), incl. wissels.
    assert abs(tri["totaal_s"] - 6180) < 1, tri["totaal_s"]
    print(f"3. Triatlon-training herkend (brick-voorstel vervangen door de "
          f"ruimere groep ✓): T1 = {labels['T1']:.0f} s, T2 = {labels['T2']:.0f} s ✓")

    # -- Race-simulatie: 500/20000/5500 lijkt op de rookie-opzet.
    sim = race_similarity({m["sport"]: m["distance_m"] for m in tri["members"]},
                          CONFIG)
    assert sim and "Rookie" in sim, sim
    print(f"4. Race-simulatie: {sim.split(':')[0]} ✓")

    # -- Scenario 4: bevestigen en losmaken.
    set_combo_status(conn, tri["combo_id"], "bevestigd")
    leden = combo_membership(conn)
    assert leden[tri_zwem.activity_key]["status"] == "bevestigd"
    historie = combo_history(conn, acts, load_records)
    assert len(historie) == 1 and historie.iloc[0]["t1_s"] == 300
    assert abs(historie.iloc[0]["delta_tempo_s_per_km"] - 40) < 5
    print("5. Bevestigen: combo telt mee in trend (T1/T2 + overgangsdelta) ✓")

    brick_id = [c for c in combos if c["kind"] == "brick"][0]["combo_id"]
    set_combo_status(conn, brick_id, "losgemaakt")
    nieuw = detect_and_store_proposals(conn, load_activities(conn), 25)
    leden = combo_membership(conn)
    assert nieuw == 0 and brick_fiets.activity_key not in leden, \
        "losgemaakte groep werd opnieuw voorgesteld"
    # Losmaken kan ook bij een bevestigde combo.
    set_combo_status(conn, tri["combo_id"], "losgemaakt")
    assert combo_history(conn, load_activities(conn), load_records).empty
    print("6. Losmaken: voorstel komt niet terug, ook bevestigde combo los te "
          "maken ✓")

    # -- Feedback-context: de loop die de triatlon afsloot krijgt het
    #    wissel-/overgangsblok mee (onafhankelijk van de opgeslagen combos).
    blok = combo_block(conn, tri_loop, load_activities(conn), CONFIG, load_records)
    assert blok is not None and "T2" in blok and "T1" in blok
    assert "Bakstenen-benen-analyse" in blok and "Rookie" in blok
    print("7. Feedback-context bevat T1/T2, bakstenen-benen-analyse en "
          "race-simulatie ✓")

    conn.close()
    print(f"\nAlle combo-tests geslaagd. (tijdelijke data: {tmp})")


if __name__ == "__main__":
    main()
