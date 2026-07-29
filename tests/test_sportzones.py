"""Tests voor de sport-afhankelijke drempels, zones en de FTP-afleiding.

Draaien vanuit de projectroot::

    python tests/test_sportzones.py

Loopt de verificatiepunten van de uitbreiding langs:

1. loop-LTHR 170 → de %LTHR-grenzen, en fiets-LTHR 164 → eigen, lagere
   grenzen voor fietsen (Z2 131–145 i.p.v. 136–150);
2. FTP leeg → fietssessies op fiets-LTHR-hartslagzones, gemarkeerd als
   tussenoplossing;
3. FTP ingevuld → fietssessies op %FTP-vermogenszones (Coggan);
4. zwemmen krijgt nergens een zone-oordeel;
5. de herberekening zet elke sport op zijn eigen drempel;
6. een ramptest wordt herkend en levert een FTP-voorstel (75% van de beste
   minuut) maar géén fiets-LTHR; een gewone duurrit levert niets op;
7. een 20-minutentest levert FTP én fiets-LTHR uit één inspanning;
8. drempels die op een schatting rusten worden als voorlopig gemeld;
9. oude configs migreren en de geschiedenis splitst per sport.
"""

import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from tricoach.config import normalize_athlete
from tricoach.fit_parser import ParsedActivity
from tricoach.lthr import BIKE_FTP, RUN_LTHR, append_entry, load_history
from tricoach.ramptest import (
    RAMP_FTP_FACTOR,
    best_20min_window,
    ftp_proposal,
    is_ramp_test,
)
from tricoach.sportzones import (
    CYCLING,
    METHOD_HR,
    METHOD_NONE,
    METHOD_POWER,
    RUNNING,
    SWIMMING,
    bike_lthr,
    clear_provisional,
    hr_zone_bounds,
    mark_provisional,
    run_lthr,
    threshold_notes,
    zone2_range,
    zone_model,
)
from tricoach.storage import connect, recompute_zones, save_activity
from tricoach.zones import time_in_zones, zone_for_hr

# De athlete-config zoals hij na de herijking in config.yaml staat.
ATHLETE = {
    "max_hr": 193,
    "zone_pct_lthr": [0.80, 0.89, 0.95, 1.00],
    "thresholds": {
        "running": {"lthr": 170, "lthr_source": "Garmin-schatting (bevestigd)"},
        "cycling": {"lthr": 164, "ftp": None},
    },
}


def _records(start: datetime, hrs: list[int] | None = None,
             watts: list[float] | None = None) -> pd.DataFrame:
    """Seconde-data met één meetpunt per seconde."""
    n = len(hrs if hrs is not None else watts)
    data = {"timestamp": [start + timedelta(seconds=i) for i in range(n)]}
    if hrs is not None:
        data["heart_rate"] = hrs
    if watts is not None:
        data["power"] = watts
    return pd.DataFrame(data)


def _activity(sport: str, start: datetime, records: pd.DataFrame,
              sub_sport: str | None = None) -> ParsedActivity:
    return ParsedActivity(
        activity_key=f"{sport}-{start:%Y%m%d%H%M%S}",
        sport=sport,
        sub_sport=sub_sport,
        start_time=start,
        summary={"total_timer_time": float(len(records)),
                 "total_distance": 5000.0,
                 "avg_heart_rate": 145},
        records=records,
        lengths=pd.DataFrame(),
        source_file="test.fit",
    )


# --------------------------------------------------------------------------

def test_drempels_per_sport():
    """Lopen op 170, fietsen op 164 — dezelfde hartslag, andere zone."""
    assert run_lthr(ATHLETE) == 170
    assert bike_lthr(ATHLETE) == 164
    # De fiets-drempel hoort 5-10 bpm onder de loop-drempel te liggen.
    assert 5 <= run_lthr(ATHLETE) - bike_lthr(ATHLETE) <= 10

    loop = hr_zone_bounds(ATHLETE, RUNNING)
    fiets = hr_zone_bounds(ATHLETE, CYCLING)
    assert loop == [136, 151, 162, 170], loop
    assert fiets == [131, 146, 156, 164], fiets

    # %LTHR-indeling op de loopdrempel.
    assert zone_for_hr(135, loop) == "Z1"
    assert zone_for_hr(136, loop) == "Z2"
    assert zone_for_hr(150, loop) == "Z2"
    assert zone_for_hr(151, loop) == "Z3"
    assert zone_for_hr(161, loop) == "Z3"
    assert zone_for_hr(162, loop) == "Z4"
    assert zone_for_hr(170, loop) == "Z5"

    # Dít is de kern van de uitbreiding: HR 148 is bij lopen nog rustig (Z2)
    # maar op de fiets al de grijze zone (Z3), omdat de drempel daar lager is.
    assert zone_for_hr(148, loop) == "Z2"
    assert zone_for_hr(148, fiets) == "Z3"
    assert zone2_range(ATHLETE, RUNNING) == (136, 150)
    assert zone2_range(ATHLETE, CYCLING) == (131, 145)
    print("1. loop-LTHR 170 → Z2 136–150; fiets-LTHR 164 → Z2 131–145; "
          "HR 148 = lopen Z2 maar fietsen Z3: OK")


def test_fiets_zonder_ftp_is_tussenoplossing():
    """Zonder FTP: hartslagzones op de fiets-LTHR, expliciet als tussenoplossing."""
    model = zone_model(ATHLETE, CYCLING)
    assert model.method == METHOD_HR
    assert model.provisional is True
    assert "tussenoplossing" in model.reason
    assert model.threshold == 164
    print(f"2. FTP leeg → fiets op %LTHR {model.threshold} "
          f"({model.bounds_text()}), gemarkeerd als tussenoplossing: OK")


def test_fiets_met_ftp_is_vermogen():
    """Met FTP: Coggan-vermogenszones zijn primair; ritten zonder power vallen terug."""
    met_ftp = {**ATHLETE, "thresholds": {
        "running": {"lthr": 170}, "cycling": {"lthr": 164, "ftp": 240}}}
    model = zone_model(met_ftp, CYCLING)
    assert model.method == METHOD_POWER
    assert model.provisional is False
    assert model.threshold == 240
    # Coggan: 55/75/90/105/120% van 240 W.
    assert [round(w) for w in model.bounds] == [132, 180, 216, 252, 288]

    # Een rit zonder vermogensdata valt terug op de fiets-hartslagzones, maar
    # dat is geen "tussenoplossing" — de instelling klopt, de rit mist de data.
    zonder_power = zone_model(met_ftp, CYCLING, has_power=False)
    assert zonder_power.method == METHOD_HR
    assert zonder_power.threshold == 164
    assert zonder_power.provisional is False
    assert "geen vermogensdata" in zonder_power.reason
    print(f"3. FTP 240 W → {model.bounds_text()}; rit zonder power valt terug "
          "op de fiets-LTHR: OK")


def test_zwemmen_krijgt_geen_zones():
    """Zwemmen heeft geen drempel, geen grenzen en geen zonetijd."""
    model = zone_model(ATHLETE, SWIMMING)
    assert model.method == METHOD_NONE
    assert model.has_zones is False
    assert model.threshold is None
    assert model.bounds_text() == "—"
    assert hr_zone_bounds(ATHLETE, SWIMMING) is None
    assert zone2_range(ATHLETE, SWIMMING) is None

    # Ook met hartslagdata blijft de zonetijd leeg.
    rec = _records(datetime(2026, 7, 20, 6, 0), hrs=[150] * 600)
    tiz = time_in_zones(rec, None)
    assert sum(tiz.values()) == 0, tiz
    print("4. zwemmen: geen drempel, geen grenzen, geen zonetijd: OK")


def test_herberekening_per_sport(tmp: Path):
    """recompute_zones zet elke sessie op de drempel van háár eigen sport."""
    conn = connect(tmp / "zones.db")
    start = datetime(2026, 7, 20, 6, 0)

    # Drie sessies, alle drie 10 minuten op HR 148 — een hartslag die per sport
    # in een andere zone valt: lopen Z2 (136-150), fietsen Z3 (146-155).
    loop = _activity(RUNNING, start, _records(start, hrs=[148] * 600))
    rit = _activity(CYCLING, start + timedelta(hours=2),
                    _records(start + timedelta(hours=2), hrs=[148] * 600))
    zwem = _activity(SWIMMING, start + timedelta(hours=4),
                     _records(start + timedelta(hours=4), hrs=[148] * 600))

    # Bewust met de VERKEERDE (oude, uniforme) grenzen opgeslagen, zodat de
    # herberekening echt iets te corrigeren heeft.
    oud = [137, 152, 162, 171]
    for act in (loop, rit, zwem):
        assert save_activity(conn, act, oud)
    voor = dict(conn.execute(
        "SELECT sport, z2_s FROM activities").fetchall())
    assert voor[RUNNING] > 0 and voor[CYCLING] > 0 and voor[SWIMMING] > 0

    n = recompute_zones(conn, ATHLETE)
    assert n == 3, n
    na = {sport: rij for sport, *rij in conn.execute(
        "SELECT sport, z1_s, z2_s, z3_s, z4_s, z5_s, pct_in_zone2 "
        "FROM activities").fetchall()}

    # Hardlopen: HR 148 zit in Z2 (136-150).
    assert na[RUNNING][1] > 0 and na[RUNNING][2] == 0, na[RUNNING]
    # Fietsen: dezelfde HR 148 zit in Z3 (146-155) — eigen, lagere drempel.
    assert na[CYCLING][2] > 0 and na[CYCLING][1] == 0, na[CYCLING]
    # Zwemmen: alles leeg, geen percentage.
    assert sum(na[SWIMMING][:5]) == 0, na[SWIMMING]
    assert na[SWIMMING][5] is None, na[SWIMMING]
    conn.close()
    print("5. herberekening: HR 148 → lopen Z2, fietsen Z3, zwemmen geen zones; "
          "zwem-percentage gewist: OK")


def test_ramptest_en_ftp_voorstel():
    """Een oplopend blokkig indoorprofiel is een ramptest; een duurrit niet."""
    start = datetime(2026, 8, 1, 19, 0)
    # Ramptest: 12 trappen van 60 s, 100 W → 320 W in stappen van 20 W.
    watts = []
    for stap in range(12):
        watts += [100 + stap * 20] * 60
    rec = _records(start, watts=watts)

    assert is_ramp_test(rec, indoor=True) is True
    assert is_ramp_test(rec, indoor=False) is False, "buiten is geen ramptest"

    voorstel = ftp_proposal(rec, indoor=True)
    assert voorstel is not None
    assert voorstel.factor == RAMP_FTP_FACTOR
    assert voorstel.confidence == "hoog"
    # Beste minuut is de laatste trap (320 W); 75% daarvan is 240 W.
    assert abs(voorstel.basis_watt - 320) < 5, voorstel.basis_watt
    assert abs(voorstel.ftp_watt - 240) < 5, voorstel.ftp_watt

    # Een ramptest geeft GEEN fiets-LTHR: de hartslag is aan het eind maximaal,
    # niet drempelniveau, en zou de drempel dus flink overschatten.
    ramp_met_hr = _records(start, watts=watts)
    ramp_met_hr["heart_rate"] = [110 + i // 12 for i in range(len(watts))]
    assert ftp_proposal(ramp_met_hr, indoor=True).lthr_bpm is None

    # Een vlakke duurrit van een uur is geen test en levert geen voorstel op.
    duur = _records(start, watts=[180.0] * 3600)
    assert is_ramp_test(duur, indoor=True) is False
    assert ftp_proposal(duur, indoor=True) is None
    print(f"6. ramptest herkend → FTP-voorstel {voorstel.ftp_watt:.0f} W "
          f"({voorstel.basis_watt:.0f} W × {RAMP_FTP_FACTOR:.0%}), géén "
          "fiets-LTHR; duurrit levert geen voorstel: OK")


def test_20min_test_levert_beide_drempels():
    """Een 20-minutentest geeft FTP én fiets-LTHR uit één inspanning."""
    start = datetime(2026, 8, 8, 18, 0)
    # 10 min inrijden op 120 W / HR 130, dan 20 min voluit op 250 W / HR 165,
    # dan 10 min uitrijden. Buiten, dus geen ramptest-herkenning.
    watts = [120.0] * 600 + [250.0] * 1200 + [120.0] * 600
    hrs = [130] * 600 + [165] * 1200 + [125] * 600
    rec = _records(start, hrs=hrs, watts=watts)

    venster = best_20min_window(rec)
    assert venster is not None
    assert abs(venster["avg_watt"] - 250) < 5, venster["avg_watt"]
    assert abs(venster["avg_hr"] - 165) < 2, venster["avg_hr"]

    voorstel = ftp_proposal(rec, indoor=False)
    assert voorstel is not None
    assert voorstel.window_s == 1200
    # FTP = 95% van 250 W; fiets-LTHR = gem. HR over datzelfde blok.
    assert abs(voorstel.ftp_watt - 237.5) < 3, voorstel.ftp_watt
    assert voorstel.lthr_bpm == 165, voorstel.lthr_bpm
    assert "fiets-LTHR" in voorstel.as_text()
    print(f"7. 20-minutentest → FTP {voorstel.ftp_watt:.0f} W "
          f"(95% van {voorstel.basis_watt:.0f} W) én fiets-LTHR "
          f"{voorstel.lthr_bpm} bpm uit één inspanning: OK")


def test_voorlopige_drempels_worden_gemeld():
    """Geschatte drempels worden als voorlopig gemeld, met herzieningsdatum."""
    from datetime import date

    # Niets gemarkeerd → geen meldingen.
    assert threshold_notes(ATHLETE) == []

    athlete = {
        "max_hr": 193,
        "zone_pct_lthr": [0.8, 0.89, 0.95, 1.0],
        "thresholds": {"running": {"lthr": 170},
                       "cycling": {"lthr": 164, "ftp": 210}},
    }
    mark_provisional(athlete, CYCLING, "geschat, niet gemeten",
                     review_after=date(2026, 9, 23))
    mark_provisional(athlete, CYCLING, "handmatig ingevuld",
                     review_after=date(2026, 9, 23), kind="ftp")

    # Vóór de herzieningsdatum: melding met de datum erbij.
    notes = threshold_notes(athlete, today=date(2026, 8, 1))
    assert len(notes) == 2, notes
    assert "fiets-LTHR (164 bpm) is voorlopig" in notes[0]
    assert "FTP (210 W) is voorlopig" in notes[1]
    assert all("herzien rond 23-09-2026" in n for n in notes)

    # Ná de herzieningsdatum: dringender formulering.
    laat = threshold_notes(athlete, today=date(2026, 10, 1))
    assert all("verstreken" in n for n in laat), laat

    # De loopdrempel is niet gemarkeerd en komt dus niet voor.
    assert not any("loop-LTHR" in n for n in notes)

    # Na een echte meting valt de markering weg.
    clear_provisional(athlete, CYCLING, kind="ftp")
    assert len(threshold_notes(athlete, today=date(2026, 8, 1))) == 1
    print("8. voorlopige drempels gemeld met herzieningsdatum, dringender na "
          "het verstrijken, weg na een meting: OK")


def test_config_migratie_en_geschiedenis(tmp: Path):
    """Oude platte config migreert; de drempelgeschiedenis splitst per sport."""
    oud = {"max_hr": 193, "lthr": 170, "ftp": None,
           "zone_pct_lthr": [0.8, 0.89, 0.95, 1.0]}
    normalize_athlete(oud)
    assert "lthr" not in oud and "ftp" not in oud
    assert oud["thresholds"]["running"]["lthr"] == 170
    assert oud["thresholds"]["cycling"]["lthr"] == 162  # 170 - 8
    # Idempotent: nog eens draaien verandert niets.
    kopie = {k: v for k, v in oud.items()}
    normalize_athlete(kopie)
    assert kopie == oud

    # Geschiedenis: de oude 3-koloms vorm wordt gelezen als loop-LTHR en er
    # kan een FTP-regel naast staan zonder dat die de loopgrafiek vervuilt.
    mem = tmp / "mem"
    mem.mkdir()
    (mem / "lthr_geschiedenis.md").write_text(
        "# LTHR-geschiedenis\n\n| Datum | LTHR | Opmerking |\n|---|---|---|\n"
        "| 2026-06-12 | 171 | Startwaarde |\n", encoding="utf-8")
    append_entry(mem, 170, "Bevestigd via instellingen-tab", kind=RUN_LTHR)
    append_entry(mem, 240, "Ramptest op de Kickr", kind=BIKE_FTP)

    loop_hist = load_history(mem, 170, kind=RUN_LTHR)
    assert list(loop_hist["waarde"]) == [171, 170], list(loop_hist["waarde"])
    ftp_hist = load_history(mem, 170, kind=BIKE_FTP)
    assert list(ftp_hist["waarde"]) == [240]
    print("9. oude config migreert (idempotent); drempelgeschiedenis gesplitst "
          "per sport, oude regels gelden als loop-LTHR: OK")


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="tricoach_sportzones_") as d:
        tmp = Path(d)
        test_drempels_per_sport()
        test_fiets_zonder_ftp_is_tussenoplossing()
        test_fiets_met_ftp_is_vermogen()
        test_zwemmen_krijgt_geen_zones()
        test_herberekening_per_sport(tmp)
        test_ramptest_en_ftp_voorstel()
        test_20min_test_levert_beide_drempels()
        test_voorlopige_drempels_worden_gemeld()
        test_config_migratie_en_geschiedenis(tmp)
    print("\nAlle sportzone-tests geslaagd ✓")


if __name__ == "__main__":
    main()
