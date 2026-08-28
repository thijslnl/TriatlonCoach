"""Testscript voor tricoach.analysis: tempo-bij-gelijke-hartslag en weekvolume.

Gebruik:  python tests/test_analysis.py

Regressietest voor een concrete bug: een overwegend harde loopsessie (tempo-
of intervaltraining) kan tijdens het opwarmen kort door de zone-2-hartslag-
band komen. pace_at_hr() middelde vroeger de snelheid over al die seconden,
ongeacht of de sessie als geheel rustig was — met als gevolg een misleidende
uitschieter (8:54 min/km op een sessie die overall op 6:55 min/km liep, 87%
van de tijd in Z3). Nu telt alleen een overwegend rustige sessie
(intensity_category == "rustig") mee.
"""

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))


import tempfile
from pathlib import Path

import pandas as pd

from tricoach.analysis import pace_at_hr
from tricoach.fit_parser import ParsedActivity
from tricoach.storage import connect, load_activities, save_activity

BOUNDS = [136, 150, 156, 164]  # loop-zonegrenzen (Z2..Z5-ondergrenzen)

GESLAAGD, GEFAALD = [], []


def check(naam: str, voorwaarde: bool, toelichting: str = "") -> None:
    if voorwaarde:
        GESLAAGD.append(naam)
        print(f"  ✅ {naam}" + (f" — {toelichting}" if toelichting else ""))
    else:
        GEFAALD.append(naam)
        print(f"  ❌ {naam}" + (f" — {toelichting}" if toelichting else ""))


def _run(start: str, seconden: int, snelheid_ms: float, hr: int) -> ParsedActivity:
    """Een rustige loopsessie: constante snelheid en hartslag."""
    ts = pd.Timestamp(start)
    rows = [{"timestamp": ts + pd.Timedelta(seconds=s), "heart_rate": hr,
             "speed_ms": snelheid_ms, "distance_m": snelheid_ms * s}
            for s in range(seconden)]
    return ParsedActivity(
        activity_key=ts.isoformat(), sport="running", sub_sport=None,
        start_time=ts, summary={"total_timer_time": seconden, "avg_heart_rate": hr},
        records=pd.DataFrame(rows), lengths=pd.DataFrame(),
        source_file=f"test_run_{start}.fit",
    )


def _tempo_run_met_opwarmen(start: str) -> ParsedActivity:
    """Een overwegend harde sessie: 6 min opwarmen dóór zone 2 (2.6 m/s,
    traag) op weg naar Z3, dan 30 min stevig in Z3 (4.0 m/s)."""
    ts = pd.Timestamp(start)
    rows = []
    for s in range(360):  # 6 min opwarmen: HR 140 (zone 2), traag tempo
        rows.append({"timestamp": ts + pd.Timedelta(seconds=s), "heart_rate": 140,
                     "speed_ms": 2.6, "distance_m": 2.6 * s})
    for s in range(360, 360 + 1800):  # 30 min Z3: HR 160, veel sneller
        rows.append({"timestamp": ts + pd.Timedelta(seconds=s), "heart_rate": 160,
                     "speed_ms": 4.0, "distance_m": 2.6 * 360 + 4.0 * (s - 360)})
    return ParsedActivity(
        activity_key=ts.isoformat(), sport="running", sub_sport=None,
        start_time=ts, summary={"total_timer_time": len(rows), "avg_heart_rate": 156},
        records=pd.DataFrame(rows), lengths=pd.DataFrame(),
        source_file=f"test_tempo_{start}.fit",
    )


def test_harde_sessie_niet_in_zone2_trend() -> None:
    print("\n== Alleen rustige sessies tellen mee in 'tempo bij gelijke hartslag' ==")
    tmp = Path(tempfile.mkdtemp(prefix="tricoach_analysis_test_"))
    conn = connect(tmp / "test.db")

    rustig1 = _run("2026-08-01 07:00:00+00:00", 1800, 2.8, 140)   # 30 min, echt Z2
    rustig2 = _run("2026-08-08 07:00:00+00:00", 1800, 2.9, 141)
    tempo = _tempo_run_met_opwarmen("2026-08-15 07:00:00+00:00")  # overwegend Z3

    for act in (rustig1, rustig2, tempo):
        save_activity(conn, act, BOUNDS)

    acts = load_activities(conn)
    trend = pace_at_hr(conn, acts, "running", (136, 150))

    check("beide rustige sessies staan in de trend", len(trend) == 2,
          f"{len(trend)} sessie(s): {list(trend['start_time'])}")
    check("de tempo-sessie met opwarmen door zone 2 zit er NIET in",
          tempo.activity_key not in
          {pd.Timestamp(t).isoformat() for t in trend["start_time"]},
          "zou een misleidende uitschieter zijn (opwarmtempo != sessietempo)")

    conn.close()


def main() -> int:
    print("tricoach.analysis — tests")
    for test in (test_harde_sessie_niet_in_zone2_trend,):
        test()
    print(f"\n{'=' * 60}")
    print(f"{len(GESLAAGD)} geslaagd, {len(GEFAALD)} gefaald")
    for naam in GEFAALD:
        print(f"  ❌ {naam}")
    return 1 if GEFAALD else 0


if __name__ == "__main__":
    sys.exit(main())
