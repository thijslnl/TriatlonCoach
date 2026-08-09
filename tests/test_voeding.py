"""Tests voor de voedingsplanner (tricoach.nutrition).

Draait op een tijdelijke database met verzonnen sessies; de echte
data/training.db wordt niet aangeraakt. Starten vanuit de projectroot:

    python tests/test_voeding.py

De vier verificatiescenario's uit de opdracht staan onderaan als
``verificatie_*``; die zijn de eigenlijke acceptatietest.
"""

import sqlite3
import sys
import tempfile
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tricoach.nutrition import duration as dur_mod
from tricoach.nutrition import products as prod_mod
from tricoach.nutrition import rules, store
from tricoach.nutrition.duration import DurationEstimate, LegEstimate, LegRequest
from tricoach.nutrition.plan import (
    AidStation,
    PlanRequest,
    build_plan,
    build_segments,
    candidate_slots,
)

GESLAAGD, GEFAALD = [], []


def check(naam: str, voorwaarde: bool, toelichting: str = "") -> None:
    """Eén assertie met leesbare uitvoer (zelfde stijl als de andere tests)."""
    if voorwaarde:
        GESLAAGD.append(naam)
        print(f"  ✅ {naam}" + (f" — {toelichting}" if toelichting else ""))
    else:
        GEFAALD.append(naam)
        print(f"  ❌ {naam}" + (f" — {toelichting}" if toelichting else ""))


# ------------------------------------------------------------- testfixtures --

ATLEET = {
    "max_hr": 193,
    "thresholds": {
        "running": {"lthr": 170},
        "cycling": {"lthr": 164, "ftp": 264.0},
    },
    "zone_pct_lthr": [0.8, 0.89, 0.95, 1.0],
}

GEWICHT_KG = 104.0


def _acts(rijen: list[dict]) -> pd.DataFrame:
    """Een activities-DataFrame zoals storage.load_activities hem oplevert."""
    basis = {
        "activity_key": "", "sport": "running", "sub_sport": "generic",
        "start_time": None, "duration_s": 0.0, "active_s": None,
        "distance_m": 0.0, "avg_hr": None, "avg_speed_ms": None,
        "np_power": None, "avg_power": None, "is_indoor": 0,
        "temperature_c": None, "excluded_reason": None,
    }
    uit = []
    for i, r in enumerate(rijen):
        rij = dict(basis)
        rij.update(r)
        rij["activity_key"] = rij["activity_key"] or f"key-{i}"
        uit.append(rij)
    df = pd.DataFrame(uit)
    df["start_time"] = pd.to_datetime(df["start_time"], utc=True)
    return df


def eigen_sessies() -> pd.DataFrame:
    """Sessies die lijken op de echte historie: fietsritten rond 37 km op
    tempo-vermogen, duurlopen in zone 2, en een paar open water-zwemsessies."""
    vandaag = pd.Timestamp.now(tz="UTC").normalize()
    rijen = []
    ritten = [(37256, 4444, 226.7, 19.0), (37175, 4737, 216.8, 21.0),
              (46419, 5603, 209.3, 18.0), (37843, 4824, 208.1, 22.0),
              (37156, 4677, 200.7, 17.0), (37197, 4401, 217.6, 16.0),
              (42950, 4940, 219.2, 20.0)]
    for i, (m, s, np_w, temp) in enumerate(ritten):
        rijen.append({
            "sport": "cycling", "sub_sport": "road",
            "start_time": vandaag - pd.Timedelta(days=3 + i * 4),
            "distance_m": m, "duration_s": s, "active_s": s,
            "np_power": np_w, "avg_hr": 140, "temperature_c": temp,
        })
    lopen = [(18094, 7172, 147, 22.3), (17504, 7274, 148, 25.9),
             (12313, 4776, 143, 19.4), (10188, 3860, 146, 18.3),
             (10014, 3954, 144, 22.8), (8019, 3169, 148, 23.8),
             (9047, 3796, 146, 17.0), (7749, 3229, 144, 15.6)]
    for i, (m, s, hr, temp) in enumerate(lopen):
        rijen.append({
            "sport": "running",
            "start_time": vandaag - pd.Timedelta(days=2 + i * 5),
            "distance_m": m, "duration_s": s, "active_s": s,
            "avg_hr": hr, "temperature_c": temp,
        })
    for i, (m, s) in enumerate([(877, 1174), (1518, 2137)]):
        rijen.append({
            "sport": "swimming", "sub_sport": "open_water",
            "start_time": vandaag - pd.Timedelta(days=5 + i * 20),
            "distance_m": m, "duration_s": s, "active_s": s,
        })
    return _acts(rijen)


def tijdelijke_db() -> sqlite3.Connection:
    """Lege database in een tijdelijke map (raakt de echte data niet)."""
    tmp = Path(tempfile.mkdtemp(prefix="voedingstest-"))
    return sqlite3.connect(tmp / "test.db")


def producten(*namen: str) -> list[prod_mod.Product]:
    """De standaardproducten met deze namen, in de opgegeven volgorde."""
    per_naam = {p.name: p for p in prod_mod.DEFAULT_PRODUCTS}
    return [per_naam[n] for n in namen]


def vaste_duur(sport: str, seconden: float, afstand_m: float | None = None
               ) -> DurationEstimate:
    """Een duurschatting met één onderdeel en een vastgezette duur."""
    return DurationEstimate(legs=[LegEstimate(
        sport=sport, low_s=seconden, mid_s=seconden, high_s=seconden,
        distance_m=afstand_m, basis="test")])


def codes(plan) -> set[str]:
    return {w.code for w in plan.warnings}


# ------------------------------------------------------------------- regels --

def test_koolhydraatbanden():
    print("\n== Koolhydraatbanden per duur ==")
    kort = rules.carb_target(60 * 60)
    check("< 75 min: 0-30 g/uur", (kort.lo_g_h, kort.hi_g_h) == (0.0, 30.0))
    check("< 75 min: advies is 'niet nodig'", not kort.needed,
          f"advies {kort.advice_g_h:.0f} g/uur")

    midden = rules.carb_target(120 * 60)
    check("75-150 min: 30-60 g/uur", (midden.lo_g_h, midden.hi_g_h) == (30.0, 60.0))
    check("75-150 min: advies in de band",
          midden.lo_g_h <= midden.advice_g_h <= midden.hi_g_h,
          f"{midden.advice_g_h:.0f} g/uur")

    tussen = rules.carb_target(170 * 60)
    check("2,5-3 uur: ~60 g/uur", tussen.advice_g_h == 60.0)

    lang = rules.carb_target(4 * 3600)
    check("> 3 uur: 60-90 g/uur", (lang.lo_g_h, lang.hi_g_h) == (60.0, 90.0))
    check("> 3 uur: advies binnen de band",
          lang.lo_g_h <= lang.advice_g_h <= lang.hi_g_h, f"{lang.advice_g_h:.0f} g/uur")

    zeer_lang = rules.carb_target(7 * 3600)
    check("zeer lang: advies aan de bovenkant", zeer_lang.advice_g_h == 90.0,
          f"{zeer_lang.advice_g_h:.0f} g/uur")


def test_plafonds_en_vocht():
    print("\n== Plafonds, vocht en natrium ==")
    check("single-source plafond 60 g/uur",
          rules.absorption_cap_g_h(has_dual_source=False) == 60.0)
    check("dual-source plafond 90 g/uur",
          rules.absorption_cap_g_h(has_dual_source=True) == 90.0)
    check("getrainde darm 120 g/uur",
          rules.absorption_cap_g_h(True, trained_gut=True) == 120.0)

    check("vocht 20 °C = 600 ml/uur", rules.fluid_ml_per_hour(20.0) == 600.0)
    check("vocht koud >= 400 ml/uur", rules.fluid_ml_per_hour(2.0) == 400.0)
    check("vocht hitte <= 800 ml/uur", rules.fluid_ml_per_hour(35.0) == 800.0)

    check("natrium gematigd ~300 mg/l",
          rules.sodium_mg_per_liter(15.0, 5 * 3600) == 300.0)
    check("natrium 25 °C ~600 mg/l",
          rules.sodium_mg_per_liter(25.0, 5 * 3600) == 600.0)
    check("natrium tot 1000 mg/l alleen bij hitte én lange duur",
          rules.sodium_mg_per_liter(32.0, 5 * 3600) == 1000.0
          and rules.sodium_mg_per_liter(32.0, 2 * 3600) == 600.0)

    check("8% van 500 ml = 40 g", rules.max_carbs_from_fluid_g(500) == 40.0)
    check("cafeïneplafond 3 mg/kg",
          rules.caffeine_cap_mg(104.0) == 312.0)


def test_conservatieve_bron():
    print("\n== Onbekende koolhydraatbron telt als single-source ==")
    lidl = producten("Lidl HealthyFit Isotonic (schepje)")[0]
    check("bron staat als 'onbekend'", lidl.source == prod_mod.SOURCE_UNKNOWN)
    check("wordt conservatief als single gerekend",
          lidl.effective_source == prod_mod.SOURCE_SINGLE)
    check("selectie met alleen Lidl heeft geen dual-source",
          not prod_mod.has_dual_source([lidl]))


# ----------------------------------------------------------------- tijdlijn --

def test_timing():
    print("\n== Timing van de innamemomenten ==")
    schatting = vaste_duur("cycling", 3 * 3600, 90000)
    segmenten = build_segments(schatting, 3 * 3600)
    slots = candidate_slots(segmenten, 3 * 3600, rules.INTAKE_INTERVAL_MIN,
                            rules.FIRST_INTAKE_MIN)
    check("eerste moment niet bij de start",
          slots and rules.FIRST_INTAKE_RANGE_MIN[0] * 60 <= slots[0]
          <= rules.FIRST_INTAKE_RANGE_MIN[1] * 60,
          f"eerste op {slots[0] / 60:.0f} min")
    tussenruimte = [b - a for a, b in zip(slots, slots[1:])]
    check("interval tussen 20 en 30 minuten",
          all(rules.INTAKE_INTERVAL_RANGE_MIN[0] * 60 <= t
              <= rules.INTAKE_INTERVAL_RANGE_MIN[1] * 60 for t in tussenruimte))
    check("laatste moment niet in de slotminuten",
          slots[-1] <= 3 * 3600 - rules.LAST_INTAKE_BEFORE_END_MIN * 60)


def test_cafeine_in_tweede_helft():
    print("\n== Cafeïne alleen in de tweede helft, binnen 3 mg/kg ==")
    schatting = vaste_duur("cycling", 4 * 3600, 120000)
    verzoek = PlanRequest(
        session_type="cycling", intensity="racetempo", temp_c=20.0,
        weight_kg=GEWICHT_KG, override_duration_s=4 * 3600, target_g_h=80.0)
    plan = build_plan(verzoek, producten("SIS Beta Fuel gel",
                                         "SIS Beta Fuel gel + cafeïne"), schatting)
    caf = [e for e in plan.events if e.caffeine_mg > 0]
    check("cafeïnegels ingepland", bool(caf), f"{len(caf)} stuks")
    check("alle cafeïne na de helft",
          all(e.t_s >= plan.total_s / 2 for e in caf),
          "; ".join(e.time_label for e in caf))
    plafond = rules.caffeine_cap_mg(GEWICHT_KG)
    check("binnen het cafeïneplafond",
          plan.totals["caffeine_mg"] <= plafond,
          f"{plan.totals['caffeine_mg']:.0f} van max {plafond:.0f} mg")


def test_natriumtekort():
    print("\n== Natriumtekort wordt gemeld ==")
    schatting = vaste_duur("cycling", 4 * 3600, 120000)
    verzoek = PlanRequest(session_type="cycling", intensity="racetempo",
                          temp_c=30.0, weight_kg=GEWICHT_KG,
                          override_duration_s=4 * 3600)
    plan = build_plan(verzoek, producten("SIS Beta Fuel gel"), schatting)
    check("tekort gemeld bij hitte zonder natriumbron",
          "natrium_laag" in codes(plan),
          next((w.text[:80] for w in plan.warnings if w.code == "natrium_laag"), ""))


def test_verzorgingsposten():
    print("\n== Verzorgingsposten verlagen wat er mee moet ==")
    schatting = vaste_duur("cycling", 4 * 3600, 120000)
    basis = dict(session_type="cycling", intensity="racetempo", temp_c=25.0,
                 weight_kg=GEWICHT_KG, override_duration_s=4 * 3600)
    sel = producten("SIS Beta Fuel gel", "SIS Beta Fuel drank (sachet)")
    zonder = build_plan(PlanRequest(**basis), sel, schatting)
    met = build_plan(PlanRequest(**basis, aid_stations=[AidStation(0, 40.0),
                                                        AidStation(0, 80.0)]),
                     sel, schatting)

    def bidons(plan):
        regel = next(c for c in plan.carry if c.label == "Bidons")
        return int(regel.amount.split("×")[0])

    check("met posten hoeven er minder bidons mee",
          bidons(met) < bidons(zonder), f"{bidons(met)} i.p.v. {bidons(zonder)}")
    regel = next(c for c in met.carry if c.label == "Bidons")
    check("bijvulmomenten benoemd", "bijvullen" in regel.detail, regel.detail)


def test_duurschatting_uit_eigen_data():
    print("\n== Duurschatting komt uit de eigen sessies ==")
    conn = tijdelijke_db()
    acts = eigen_sessies()
    schatting = dur_mod.estimate_duration(
        conn, acts, ATLEET, [LegRequest("cycling", distance_m=90000)],
        "racetempo", 20.0)
    been = schatting.legs[0]
    check("schatting op basis van meerdere eigen ritten", been.n_sessions >= 3,
          been.basis)
    check("bandbreedte, geen puntschatting", been.low_s < been.mid_s < been.high_s,
          been.range_text())
    check("vermogensvenster in de onderbouwing", "vermogen" in been.basis,
          been.basis)

    warm = dur_mod.estimate_duration(
        conn, acts, ATLEET, [LegRequest("running", distance_m=21100)],
        "rustig", 30.0)
    koel = dur_mod.estimate_duration(
        conn, acts, ATLEET, [LegRequest("running", distance_m=21100)],
        "rustig", 15.0)
    check("warmte kost tijd", warm.mid_s > koel.mid_s,
          f"{warm.range_text()} bij 30 °C tegen {koel.range_text()} bij 15 °C")

    pct, uit_data = dur_mod.heat_pct_per_degree(
        dur_mod._session_speeds(conn, acts, "running"))
    check("temperatuureffect uit eigen data of expliciete terugval",
          0 < pct <= dur_mod.HEAT_FIT_MAX_PCT_PER_C,
          f"{pct:.2f} %/°C ({'eigen sessies' if uit_data else 'richtlijn'})")
    conn.close()


def test_opslag_en_terugkoppeling():
    print("\n== Plan opslaan en achteraf invullen hoe het viel ==")
    conn = tijdelijke_db()
    schatting = vaste_duur("cycling", 3 * 3600, 90000)
    plan = build_plan(
        PlanRequest(session_type="cycling", intensity="racetempo", temp_c=20.0,
                    weight_kg=GEWICHT_KG, override_duration_s=3 * 3600),
        producten("SIS Beta Fuel gel"), schatting)

    plan_id = store.save_plan(conn, plan, "Testrit 90 km", date(2026, 8, 9))
    check("plan opgeslagen", plan_id > 0)
    check("plan terug te lezen", store.load_plan_json(conn, plan_id) is not None)

    store.save_feedback(conn, plan_id, actual_carbs_g=300.0,
                        actual_duration_s=3 * 3600, gut=store.GUT_GOOD,
                        note="ging prima")
    regels = store.tolerance_summary(conn)
    check("tolerantie zichtbaar over tijd",
          any("viel goed" in r and "100 g/uur" in r for r in regels),
          " | ".join(regels))

    store.save_feedback(conn, plan_id, actual_carbs_g=360.0,
                        actual_duration_s=3 * 3600, gut=store.GUT_BAD, note="")
    regels = store.tolerance_summary(conn)
    check("herinvullen corrigeert in plaats van te dupliceren",
          len(store.tolerance_history(conn)) == 1 and
          any("klachten" in r for r in regels), " | ".join(regels))
    conn.close()


def test_productdatabase():
    print("\n== Productdatabase is bewerkbaar ==")
    conn = tijdelijke_db()
    lijst = prod_mod.load_products(conn)
    check("wordt gevuld met de eigen producten",
          len(lijst) == len(prod_mod.DEFAULT_PRODUCTS), f"{len(lijst)} producten")
    check("SIS Beta Fuel gel staat erin op 40 g dual-source",
          any(p.name == "SIS Beta Fuel gel" and p.carbs_g == 40
              and p.source == prod_mod.SOURCE_DUAL for p in lijst))

    df = prod_mod.products_dataframe(lijst)
    df.loc[df["name"] == "SIS Go Isotonic gel", "carbs_g"] = 21.0
    prod_mod.save_products(conn, prod_mod.products_from_dataframe(df))
    opnieuw = {p.name: p for p in prod_mod.load_products(conn)}
    check("aangepaste waarde blijft bewaard",
          opnieuw["SIS Go Isotonic gel"].carbs_g == 21.0)

    prod_mod.save_products(conn, [prod_mod.Product(name="Eigen reep", kind="vast",
                                                   carbs_g=25.0)])
    check("lijst uit de editor is de waarheid (verwijderen werkt)",
          [p.name for p in prod_mod.load_products(conn)] == ["Eigen reep"])
    prod_mod.reset_products(conn)
    check("terugzetten op de standaardlijst kan",
          len(prod_mod.load_products(conn)) == len(prod_mod.DEFAULT_PRODUCTS))
    conn.close()


# -------------------------------------------------------- verificatiescenario's --

def verificatie_1_negentig_km_fietsen():
    print("\n== VERIFICATIE 1: 90 km fietsen op racetempo bij 20 °C ==")
    conn = tijdelijke_db()
    acts = eigen_sessies()
    schatting = dur_mod.estimate_duration(
        conn, acts, ATLEET, [LegRequest("cycling", distance_m=90000)],
        "racetempo", 20.0)
    lo_min, hi_min = schatting.low_s / 60, schatting.high_s / 60
    check("duurschatting rond 3:05-3:25",
          170 <= lo_min <= 195 and 175 <= hi_min <= 215,
          f"{schatting.range_text()}")

    doel = rules.carb_target(schatting.mid_s)
    check("advies 60-90 g/uur", (doel.lo_g_h, doel.hi_g_h) == (60.0, 90.0),
          doel.as_text())

    plan = build_plan(
        PlanRequest(session_type="cycling", intensity="racetempo", temp_c=20.0,
                    weight_kg=GEWICHT_KG,
                    legs=[LegRequest("cycling", distance_m=90000)]),
        producten("SIS Go Isotonic gel"), schatting)
    check("waarschuwing: single-source plafond op 60 g/uur",
          "single_source_plafond" in codes(plan),
          next((w.text[:120] for w in plan.warnings
                if w.code == "single_source_plafond"), ""))
    check("plan afgetopt op 60 g/uur", plan.planned_g_h == 60.0,
          f"{plan.planned_g_h:.0f} g/uur (gevraagd {plan.requested_g_h:.0f})")
    check("tijdlijn heeft kilometermarkering",
          all(e.km is not None for e in plan.events),
          f"laatste op km {plan.events[-1].km:.1f}" if plan.events else "")
    conn.close()


def verificatie_2_uur_rustig_lopen():
    print("\n== VERIFICATIE 2: 1 uur rustig lopen ==")
    schatting = vaste_duur("running", 3600)
    plan = build_plan(
        PlanRequest(session_type="running", intensity="rustig", temp_c=18.0,
                    weight_kg=GEWICHT_KG, override_duration_s=3600),
        producten("SIS Go Isotonic gel"), schatting)
    check("advies: niet nodig", plan.planned_g_h == 0.0,
          f"{plan.planned_g_h:.0f} g/uur")
    check("melding 'meestal niet nodig'", "niet_nodig" in codes(plan),
          next((w.text[:90] for w in plan.warnings if w.code == "niet_nodig"), ""))
    check("geen gels ingepland", not plan.events, f"{len(plan.events)} momenten")


def verificatie_3_middenafstand():
    print("\n== VERIFICATIE 3: middenafstand 1,9/90/21,1 met Beta Fuel ==")
    conn = tijdelijke_db()
    acts = eigen_sessies()
    legs = [LegRequest("swimming", distance_m=1900),
            LegRequest("cycling", distance_m=90000),
            LegRequest("running", distance_m=21100)]
    schatting = dur_mod.estimate_duration(conn, acts, ATLEET, legs, "racetempo", 20.0)
    check("wissels meegeteld", len(schatting.transitions) == 2,
          ", ".join(f"{l} {s / 60:.0f} min" for l, s in schatting.transitions))

    plan = build_plan(
        PlanRequest(session_type="triathlon", intensity="racetempo", temp_c=20.0,
                    weight_kg=GEWICHT_KG, legs=legs),
        producten("SIS Beta Fuel gel", "SIS Beta Fuel drank (sachet)"), schatting)

    totaal = plan.totals["carbs_g"]
    check("totaal rond 450-550 g", 430 <= totaal <= 570, f"{totaal:.0f} g")
    check("binnen het dual-source plafond",
          plan.totals["carbs_per_feedable_hour"] <= rules.DUAL_SOURCE_CAP_G_H + 1,
          f"{plan.totals['carbs_per_feedable_hour']:.0f} g/uur over de eetbare tijd")

    wissels = [e for e in plan.events if "wissel" in e.note]
    check("T1 en T2 als expliciete eetmomenten", len(wissels) >= 2,
          ", ".join(f"{e.time_label} {e.segment}" for e in wissels))
    check("niets ingepland tijdens het zwemmen",
          not any(e.segment == "Zwemmen" for e in plan.events))
    check("melding dat zwemmen niet eetbaar is",
          "zwemmen_niet_eetbaar" in codes(plan))
    conn.close()


def verificatie_4_lidl_drank():
    print("\n== VERIFICATIE 4: alleen Lidl-drank, doel 80 g/uur ==")
    schatting = vaste_duur("cycling", 4 * 3600, 120000)
    basis = dict(session_type="cycling", intensity="racetempo", temp_c=20.0,
                 weight_kg=GEWICHT_KG, override_duration_s=4 * 3600,
                 target_g_h=80.0)
    alleen_drank = build_plan(
        PlanRequest(**basis), producten("Lidl HealthyFit Isotonic (schepje)"),
        schatting)
    check("waarschuwing over de concentratie",
          {"concentratie", "concentratie_geen_gel"} & codes(alleen_drank),
          next((w.text[:140] for w in alleen_drank.warnings
                if w.code.startswith("concentratie")), ""))
    check("drank blijft isotoon (<= 8%)",
          alleen_drank.drink.concentration_pct <= rules.ISOTONIC_PCT_HI + 0.05,
          f"{alleen_drank.drink.concentration_pct:.1f}%")
    check("meldt dat er geen gel is om naar te verschuiven",
          "concentratie_geen_gel" in codes(alleen_drank))

    met_gel = build_plan(
        PlanRequest(**basis),
        producten("Lidl HealthyFit Isotonic (schepje)", "SIS Go Isotonic gel"),
        schatting)
    check("met gel erbij: automatische verschuiving naar gels",
          "concentratie" in codes(met_gel) and bool(met_gel.events),
          f"{sum(e.carbs_g for e in met_gel.events):.0f} g uit gels naast "
          f"{met_gel.drink.carbs_g:.0f} g uit drank")
    check("Lidl telt conservatief als single-source (plafond 60 g/uur)",
          met_gel.planned_g_h == 60.0 and "single_source_plafond" in codes(met_gel),
          f"{met_gel.planned_g_h:.0f} g/uur")


def main() -> int:
    print("Voedingsplanner — tests")
    for test in (test_koolhydraatbanden, test_plafonds_en_vocht,
                 test_conservatieve_bron, test_timing,
                 test_cafeine_in_tweede_helft, test_natriumtekort,
                 test_verzorgingsposten, test_duurschatting_uit_eigen_data,
                 test_opslag_en_terugkoppeling, test_productdatabase,
                 verificatie_1_negentig_km_fietsen, verificatie_2_uur_rustig_lopen,
                 verificatie_3_middenafstand, verificatie_4_lidl_drank):
        test()

    print(f"\n{'=' * 60}")
    print(f"{len(GESLAAGD)} geslaagd, {len(GEFAALD)} gefaald")
    for naam in GEFAALD:
        print(f"  ❌ {naam}")
    return 1 if GEFAALD else 0


if __name__ == "__main__":
    sys.exit(main())
