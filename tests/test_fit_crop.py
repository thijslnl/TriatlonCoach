"""Testscript voor het bijsnijden van record-berichten op het sessievenster.

Gebruik:  python tests/test_fit_crop.py

Regressietest voor een concrete brick op 13-08-2026: het geëxporteerde
.fit-bestand van de loopsessie bevatte de récord-berichten van de hele
opname (fietsen + wissel + lopen), terwijl het session-bericht netjes tot de
loop beperkt bleef. Zonder bijsnijden lekte het fietstempo de trends in
(een "tempo in zone 2" van 2:23 min/km i.p.v. de echte ~6:20 min/km) zodra de
fietshartslag toevallig in de loop-zone viel. ``parse_fit`` snijdt de
record-DataFrame nu bij op het eigen [start_time, start_time+duur]-venster
van de sessie zelf.

Draait tegen het echte gearchiveerde origineel in ``uploads/``; dat bestand
bevat GPS/hartslagdata en gaat niet mee in git (zie .gitignore). Zonder dat
bestand (bijv. een verse clone) slaat de test zichzelf over in plaats van te
falen.
"""

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))


from pathlib import Path

from tricoach.fit_parser import parse_fit

GESLAAGD, GEFAALD = [], []


def check(naam: str, voorwaarde: bool, toelichting: str = "") -> None:
    if voorwaarde:
        GESLAAGD.append(naam)
        print(f"  ✅ {naam}" + (f" — {toelichting}" if toelichting else ""))
    else:
        GEFAALD.append(naam)
        print(f"  ❌ {naam}" + (f" — {toelichting}" if toelichting else ""))


FIXTURE = Path(__file__).resolve().parent.parent / "uploads/2026/08/2026-08-13_1722_23963784502.fit"


def test_brick_loop_niet_besmet_door_fietsdata() -> None:
    print("\n== Record-venster van een brick-loop wordt bijgesneden op de eigen sessie ==")
    if not FIXTURE.exists():
        print(f"  (overgeslagen: {FIXTURE} niet aanwezig — persoonlijke data, niet in git)")
        return

    with open(FIXTURE, "rb") as f:
        act = parse_fit(f, source_name=FIXTURE.name)

    assert act is not None
    duur = act.summary.get("total_elapsed_time") or act.summary.get("total_timer_time")
    check("sessie is de loop van 13-08-2026", act.sport == "running")
    check("geen records vóór de eigen sessiestart",
          act.records.empty or act.records["timestamp"].min() >= act.start_time,
          f"eerste record {act.records['timestamp'].min() if not act.records.empty else '-'} "
          f"vs sessiestart {act.start_time}")
    if not act.records.empty:
        span = (act.records["timestamp"].max() - act.records["timestamp"].min()).total_seconds()
        check("recordvenster past bij de sessieduur (geen fietsdata meer lek)",
              span <= duur + 5, f"venster {span:.0f}s vs duur {duur:.0f}s")
        check("gemiddelde snelheid is loop-tempo, geen fietstempo",
              act.records["speed_ms"].mean() < 4.0,
              f"gem. {act.records['speed_ms'].mean():.2f} m/s")


def main() -> int:
    print("Bijsnijden van record-berichten — tests")
    test_brick_loop_niet_besmet_door_fietsdata()
    print(f"\n{'=' * 60}")
    print(f"{len(GESLAAGD)} geslaagd, {len(GEFAALD)} gefaald")
    if GEFAALD:
        print("Gefaald: " + ", ".join(GEFAALD))
    return 1 if GEFAALD else 0


if __name__ == "__main__":
    raise SystemExit(main())
