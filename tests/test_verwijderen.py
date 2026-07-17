"""Testscript voor het verwijderen van sessies (soft delete + memory-boekhouding).

Gebruik:  python test_verwijderen.py

Draait volledig op een tijdelijke database en een tijdelijke memory-map;
raakt de echte data niet aan. Controleert:

1. soft delete: sessie verdwijnt uit load_activities maar blijft in de database;
2. dedup: een herimport van een verwijderde activity_key blijft "verwijderd";
3. trainingslog: de entry krijgt een statusregel bij verwijderen en herstellen;
4. feedback-context: de feedback van een verwijderde sessie vervalt als
   "vorige feedback" en de adherence-check slaat hem over;
5. herstel: alles telt weer mee en de vervallen-markering is weg;
6. definitief wissen: de rijen zijn echt weg, dus de dedup vergeet de sessie.
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

from tricoach.feedback import last_proposed_adjustment
from tricoach.feedback_context import _last_md_section
from tricoach.fit_parser import ParsedActivity
from tricoach.removal import purge_session, remove_session, restore_session
from tricoach.storage import (
    activity_exists,
    connect,
    is_deleted,
    load_activities,
    load_deleted_activities,
    save_activity,
    soft_delete_activity,
)

BOUNDS = [137, 152, 162, 172]


def _act(key_suffix: str, hour: int) -> ParsedActivity:
    """Minimale fiets-activiteit voor de tests."""
    start = pd.Timestamp(datetime(2026, 7, 4, hour, 0, 0))
    return ParsedActivity(
        activity_key=start.isoformat(),
        sport="cycling", sub_sport=None, start_time=start,
        summary={"total_timer_time": 3600.0, "total_distance": 15000.0,
                 "avg_heart_rate": 91, "max_heart_rate": 123},
        records=pd.DataFrame(), lengths=pd.DataFrame(),
        source_file=f"test_{key_suffix}.fit",
    )


def main() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="tricoach_test_"))
    conn = connect(tmp / "test.db")
    memory_dir = tmp / "memory"
    memory_dir.mkdir()

    a1, a2 = _act("casual", 6), _act("training", 12)
    save_activity(conn, a1, BOUNDS)
    save_activity(conn, a2, BOUNDS)
    key = a1.activity_key

    # Memory-bestanden zoals de import/feedback ze zou achterlaten.
    (memory_dir / "trainingslog.md").write_text(
        "# Trainingslog\n"
        f"\n## 2026-07-04 Sat 06:00 — Fietsen\n\n- **Kerncijfers:** x\n"
        f"- _Sleutel `{key}` · bron `test_casual.fit`_\n"
        f"\n## 2026-07-04 Sat 12:00 — Fietsen\n\n- **Kerncijfers:** y\n"
        f"- _Sleutel `{a2.activity_key}` · bron `test_training.fit`_\n",
        encoding="utf-8")
    (memory_dir / "feedback.md").write_text(
        "# Feedback per training\n"
        f"\n## 2026-07-03 Fri — Zwemmen\n\n- **Feedback:** oud\n"
        "- **Voorgestelde aanpassing:** rustiger zwemmen\n"
        "- _Sleutel `2026-07-03T06:19:43`_\n"
        f"\n## 2026-07-04 Sat — Fietsen\n\n- **Feedback:** casual rit\n"
        "- **Voorgestelde aanpassing:** korter fietsen\n"
        f"- _Sleutel `{key}`_\n",
        encoding="utf-8")

    # 1. Soft delete: weg uit load_activities, maar nog wél in de database.
    assert remove_session(conn, memory_dir, key, reason="verkeerd bestand")
    assert key not in load_activities(conn)["activity_key"].tolist()
    assert key in load_deleted_activities(conn)["activity_key"].tolist()
    assert is_deleted(conn, key)
    print("1. soft delete filtert de sessie weg, rij blijft bestaan: OK")

    # 2. Dedup: de sleutel blijft bekend, dus een herimport wordt geen 'nieuw'.
    assert activity_exists(conn, key)
    assert not save_activity(conn, a1, BOUNDS), "herimport mag niet opnieuw opslaan"
    assert is_deleted(conn, key), "sessie moet ná herimport verwijderd blijven"
    print("2. herimport van een verwijderde sessie blijft verwijderd: OK")

    # 3. Trainingslog: statusregel bij de juiste entry, andere entry ongemoeid.
    log = (memory_dir / "trainingslog.md").read_text(encoding="utf-8")
    entry = log.split("\n## ")[1]  # de 06:00-entry
    assert "verwijderd op" in entry and "verkeerd bestand" in entry
    assert "verwijderd op" not in log.split("\n## ")[2]
    print("3. trainingslog-entry heeft een verwijderd-notitie: OK")

    # 4. Feedback-context: de vervallen sectie telt niet meer mee.
    laatste = _last_md_section(memory_dir / "feedback.md")
    assert laatste is not None and "casual rit" not in laatste, \
        "vorige feedback mag niet van de verwijderde sessie komen"
    assert last_proposed_adjustment(memory_dir) == "rustiger zwemmen"
    print("4. feedback van de verwijderde sessie vervalt als context: OK")

    # 5. Herstel: telt weer mee, markering weg.
    assert restore_session(conn, memory_dir, key)
    assert key in load_activities(conn)["activity_key"].tolist()
    assert "casual rit" in _last_md_section(memory_dir / "feedback.md")
    assert last_proposed_adjustment(memory_dir) == "korter fietsen"
    log = (memory_dir / "trainingslog.md").read_text(encoding="utf-8")
    assert "hersteld op" in log
    print("5. herstellen maakt alles ongedaan: OK")

    # 6. Definitief wissen: rij weg, dedup vergeet de sessie.
    assert soft_delete_activity(conn, key)
    assert purge_session(conn, memory_dir, key)
    assert not activity_exists(conn, key)
    assert save_activity(conn, a1, BOUNDS), "na wissen is een herimport weer 'nieuw'"
    print("6. definitief wissen maakt herimport weer mogelijk: OK")

    print("\nAlle controles geslaagd.")


if __name__ == "__main__":
    main()
