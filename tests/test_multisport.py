"""Testscript voor multisport-FIT-bestanden (triatlon/brick in één opname).

Gebruik:  python tests/test_multisport.py

Regressietest voor de bug waarbij een multisport-bestand (zwemmen →
transition → fietsen → transition → hardlopen, vijf session-berichten in één
.fit-bestand) instortte tot precies één activiteit: parse_fit() schreef alle
session-berichten in dezelfde platte dict, dus de laatste sessie (hardlopen)
overschreef alle voorgaande. De zwemsessie en beide wisselsessies verdwenen
zonder spoor — niet overgeslagen met een waarschuwing, maar nooit gebouwd.

parse_fit() geeft nu een lijst activiteiten terug (één per session-bericht),
elk met een eigen bijgesneden record-/lengtevenster. Zie ook
memory/beslissingen.md voor de volledige diagnose en fix.

Draait tegen echte, gearchiveerde originelen in ``uploads/``; die bevatten
GPS/hartslagdata en gaan niet mee in git (zie .gitignore). Zonder die
bestanden (bijv. een verse clone) slaan de tests zichzelf over in plaats van
te falen.
"""

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))


import warnings
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


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MULTISPORT_FIXTURE = (
    PROJECT_ROOT / "uploads/2026/08/2026-08-22_1050_24070215691.fit")
POOL_SWIM_FIXTURE = (
    PROJECT_ROOT / "uploads/2026/08/2026-08-24_0818_24093133247.fit")
STANDALONE_CYCLING_FIXTURE = (
    PROJECT_ROOT / "uploads/2026/08/2026-08-13_1554_23963563990.fit")


def test_multisport_alle_vijf_sessies() -> None:
    print("\n== Multisport-bestand levert alle vijf sessies op ==")
    if not MULTISPORT_FIXTURE.exists():
        print(f"  (overgeslagen: {MULTISPORT_FIXTURE} niet aanwezig — "
              "persoonlijke data, niet in git)")
        return

    with warnings.catch_warnings(record=True) as gevangen:
        warnings.simplefilter("always")
        with open(MULTISPORT_FIXTURE, "rb") as f:
            acts = parse_fit(f, source_name=MULTISPORT_FIXTURE.name)

    check("geen waarschuwingen (alle sessies bruikbaar)", not gevangen,
          "; ".join(str(w.message) for w in gevangen))
    check("vijf activiteiten uit één bestand", len(acts) == 5, f"{len(acts)} stuks")

    sporten = [a.sport for a in acts]
    check("volgorde: zwem, wissel, fiets, wissel, loop",
          sporten == ["swimming", "transition", "cycling", "transition", "running"],
          str(sporten))

    sleutels = [a.activity_key for a in acts]
    check("elke sessie heeft een unieke sleutel", len(set(sleutels)) == len(sleutels),
          str(sleutels))

    zwem = acts[0]
    check("zwemsessie heeft de juiste afstand (~950,8 m)",
          zwem.distance_m is not None and abs(zwem.distance_m - 950.83) < 1,
          f"{zwem.distance_m} m")
    check("zwemsessie heeft de juiste duur (~1559,6 s)",
          abs(zwem.duration_s - 1559.589) < 1, f"{zwem.duration_s} s")
    check("zwemsessie heeft geen enkele GPS-fix (open water, geen fix)",
          "lat" not in zwem.records.columns or zwem.records["lat"].notna().sum() == 0,
          "0 fixes verwacht, geen crash")
    check("zwemsessie heeft toch hartslagdata",
          "heart_rate" in zwem.records.columns and zwem.records["heart_rate"].notna().any())

    t1, t2 = acts[1], acts[3]
    check("T1 heeft een aannemelijke duur (~5 min)",
          250 < t1.duration_s < 400, f"{t1.duration_s:.0f} s")
    check("T2 heeft een aannemelijke duur (~1-2 min)",
          40 < t2.duration_s < 150, f"{t2.duration_s:.0f} s")

    fiets, loop = acts[2], acts[4]
    check("fietssessie uit dit bestand heeft GPS en de juiste afstand",
          "lat" in fiets.records.columns and fiets.records["lat"].notna().all()
          and abs(fiets.distance_m - 38141.74) < 1)
    check("loopsessie heeft GPS en de juiste afstand",
          "lat" in loop.records.columns and loop.records["lat"].notna().all()
          and abs(loop.distance_m - 10396.57) < 1)

    # Geen enkel record-venster mag over de grens van de volgende sessie
    # heen lekken — dat was precies de kern van de oorspronkelijke bug.
    lek = False
    for eerder, later in zip(acts, acts[1:]):
        if eerder.records.empty:
            continue
        laatste = eerder.records["timestamp"].max()
        if laatste > later.start_time:
            lek = True
    check("geen enkele sessie lekt voorbij de start van de volgende", not lek)


def test_gewone_zwembadsessie_blijft_werken() -> None:
    print("\n== Losse zwembadsessie (zonder multisport) importeert ongewijzigd ==")
    if not POOL_SWIM_FIXTURE.exists():
        print(f"  (overgeslagen: {POOL_SWIM_FIXTURE} niet aanwezig)")
        return
    with open(POOL_SWIM_FIXTURE, "rb") as f:
        acts = parse_fit(f, source_name=POOL_SWIM_FIXTURE.name)
    check("precies één activiteit uit een single-session bestand", len(acts) == 1,
          f"{len(acts)} stuks")
    check("het blijft een zwemsessie", acts and acts[0].sport == "swimming")
    check("banendata (lengths) komt nog steeds mee", acts and not acts[0].lengths.empty)


def test_losse_fietssessie_met_gps_blijft_werken() -> None:
    print("\n== Losse fietssessie met GPS importeert ongewijzigd ==")
    if not STANDALONE_CYCLING_FIXTURE.exists():
        print(f"  (overgeslagen: {STANDALONE_CYCLING_FIXTURE} niet aanwezig)")
        return
    with open(STANDALONE_CYCLING_FIXTURE, "rb") as f:
        acts = parse_fit(f, source_name=STANDALONE_CYCLING_FIXTURE.name)
    check("precies één activiteit", len(acts) == 1, f"{len(acts)} stuks")
    check("het blijft een fietssessie met GPS",
          acts and acts[0].sport == "cycling"
          and "lat" in acts[0].records.columns
          and acts[0].records["lat"].notna().any())


def main() -> int:
    print("Multisport-FIT-bestanden — tests")
    test_multisport_alle_vijf_sessies()
    test_gewone_zwembadsessie_blijft_werken()
    test_losse_fietssessie_met_gps_blijft_werken()
    print(f"\n{'=' * 60}")
    print(f"{len(GESLAAGD)} geslaagd, {len(GEFAALD)} gefaald")
    if GEFAALD:
        print("Gefaald: " + ", ".join(GEFAALD))
    return 1 if GEFAALD else 0


if __name__ == "__main__":
    raise SystemExit(main())
