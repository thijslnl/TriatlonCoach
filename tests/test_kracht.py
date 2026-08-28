"""Testscript voor de krachtmodule (tricoach.strength).

Gebruik:  python tests/test_kracht.py

Draait op een tijdelijke database; raakt de echte data niet aan. De
pure-rekenfuncties (Epley/volume/beste-set/deload/kwaliteitssessie) worden
rechtstreeks met opgebouwde DataFrames getest; rotatie, historie en
weekvolume gaan via de echte opslaglaag (catalog.py/store.py), zodat ook de
SQL-kant (inclusief numpy.int64-ids zoals een DataFrame ze aanlevert) wordt
gedekt.
"""

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))


import tempfile
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from tricoach.strength import catalog, rules, store
from tricoach.storage import connect

GESLAAGD, GEFAALD = [], []


def check(naam: str, voorwaarde: bool, toelichting: str = "") -> None:
    if voorwaarde:
        GESLAAGD.append(naam)
        print(f"  ✅ {naam}" + (f" — {toelichting}" if toelichting else ""))
    else:
        GEFAALD.append(naam)
        print(f"  ❌ {naam}" + (f" — {toelichting}" if toelichting else ""))


def _conn():
    tmp = Path(tempfile.mkdtemp(prefix="tricoach_kracht_test_"))
    return connect(tmp / "test.db")


ATHLETE = {"thresholds": {"running": {"lthr": 170},
                          "cycling": {"lthr": 162, "ftp": 264.0}}}


# ---------------------------------------------------------- seed & schema --

def test_seed_en_schema() -> None:
    print("\n== Seed & schema ==")
    conn = _conn()
    ex = catalog.load_exercises(conn)
    tpl = catalog.load_templates(conn)
    check("13 oefeningen geseed", len(ex) == 13, f"{len(ex)} stuks")
    check("2 sjablonen geseed", len(tpl) == 2, f"{len(tpl)} stuks")
    items_a = catalog.load_template_items(conn, int(tpl.iloc[0]["id"]))
    items_b = catalog.load_template_items(conn, int(tpl.iloc[1]["id"]))
    check("template A heeft 6 items", len(items_a) == 6, f"{len(items_a)} stuks")
    check("template B heeft 7 items", len(items_b) == 7, f"{len(items_b)} stuks")

    ids_voor = set(ex["id"])
    catalog.ensure_tables(conn)  # tweede aanroep
    ex2 = catalog.load_exercises(conn)
    check("ensure_tables() is idempotent", len(ex2) == 13 and set(ex2["id"]) == ids_voor,
          f"{len(ex2)} stuks, zelfde id's: {set(ex2['id']) == ids_voor}")
    conn.close()


# -------------------------------------------------------------- rotatie --

def test_rotatie() -> None:
    print("\n== Rotatie: A → B → A, ook na toevoegen/deactiveren ==")
    conn = _conn()
    tpl = catalog.load_templates(conn)
    a_id, b_id = int(tpl.iloc[0]["id"]), int(tpl.iloc[1]["id"])

    nxt = store.next_template(conn)
    check("zonder workouts: A voorgesteld", nxt["code"] == "A", nxt["code"])

    wid_a = store.start_workout(conn, a_id, phase=1)
    store.complete_workout(conn, wid_a, session_rpe=6)
    nxt = store.next_template(conn)
    check("na voltooide A: B voorgesteld", nxt["code"] == "B", nxt["code"])

    wid_b = store.start_workout(conn, b_id, phase=1)
    store.complete_workout(conn, wid_b, session_rpe=6)
    nxt = store.next_template(conn)
    check("na voltooide B: weer A (wrap-around)", nxt["code"] == "A", nxt["code"])

    wid_niet_voltooid = store.start_workout(conn, b_id, phase=1)
    nxt = store.next_template(conn)
    check("een NIET-voltooide workout verschuift de rotatie niet",
          nxt["code"] == "A", nxt["code"])
    store.soft_delete_workout(conn, wid_niet_voltooid)

    c_id = catalog.add_template(conn, "C", "Extra", sort_order=3)
    wid_a2 = store.start_workout(conn, a_id, phase=1)
    store.complete_workout(conn, wid_a2, session_rpe=6)
    nxt = store.next_template(conn)
    check("na een 3e sjabloon: A → B → C → A klopt (nu B)",
          nxt["code"] == "B", nxt["code"])
    wid_b2 = store.start_workout(conn, b_id, phase=1)
    store.complete_workout(conn, wid_b2, session_rpe=6)
    nxt = store.next_template(conn)
    check("... dan C", nxt["code"] == "C", nxt["code"])

    catalog.set_template_active(conn, b_id, False)
    wid_c = store.start_workout(conn, c_id, phase=1)
    store.complete_workout(conn, wid_c, session_rpe=6)
    nxt = store.next_template(conn)
    check("na B deactiveren: na C volgt direct A (B overgeslagen)",
          nxt["code"] == "A", nxt["code"])

    wid_a3 = store.start_workout(conn, a_id, phase=1)
    store.complete_workout(conn, wid_a3, session_rpe=6)
    wid_soft = store.start_workout(conn, b_id, phase=1)  # B is inactief maar bestond nog
    store.complete_workout(conn, wid_soft, session_rpe=6)
    store.soft_delete_workout(conn, wid_soft)
    nxt = store.next_template(conn)
    check("een soft-deleted workout telt niet mee in de rotatie",
          nxt["code"] == "C", nxt["code"])

    conn.close()


def test_laatst_gebruikte_template_inactief() -> None:
    print("\n== Rotatie: laatst gebruikte sjabloon inmiddels inactief ==")
    conn = _conn()
    tpl = catalog.load_templates(conn)
    a_id, b_id = int(tpl.iloc[0]["id"]), int(tpl.iloc[1]["id"])
    wid = store.start_workout(conn, b_id, phase=1)
    store.complete_workout(conn, wid, session_rpe=6)
    catalog.set_template_active(conn, b_id, False)
    nxt = store.next_template(conn)
    check("valt terug op het eerste actieve sjabloon", nxt["code"] == "A", nxt["code"])
    conn.close()


# ----------------------------------------------------------- opwarmsets --

def _sets(rijen: list[dict]) -> pd.DataFrame:
    basis = {"is_warmup": 0, "reps": None, "weight_kg": None, "seconds": None, "meters": None}
    return pd.DataFrame([{**basis, **r} for r in rijen])


def test_opwarmsets() -> None:
    print("\n== Opwarmsets tellen niet mee ==")
    # De opwarmset heeft bewust een HOGERE schijnbare e1RM (150 x 3 -> 165)
    # dan elke echte werkset, juist om te toetsen dat de filter hem negeert
    # ondanks dat hij numeriek zou "winnen" als hij meetelde.
    sets = _sets([
        {"weight_kg": 150, "reps": 3, "is_warmup": 1},
        {"weight_kg": 100, "reps": 5, "is_warmup": 0},
        {"weight_kg": 100, "reps": 5, "is_warmup": 0},
        {"weight_kg": 110, "reps": 5, "is_warmup": 0},   # de echte beste werkset
    ])
    beste_werkset_e1rm = rules.epley_1rm(110, 5)
    e1rm = rules.workout_1rm(sets)
    check("opwarmset met een hogere schijnbare e1RM wordt genegeerd",
          abs(e1rm - beste_werkset_e1rm) < 0.01,
          f"e1RM={e1rm:.1f} (opwarmset zou {rules.epley_1rm(150, 3):.1f} geven)")
    vol_met = rules.workout_volume(sets)
    vol_zonder = rules.workout_volume(_sets([r for r in sets.to_dict("records")
                                             if not r["is_warmup"]]))
    check("volume telt alleen werksets", vol_met == vol_zonder, f"{vol_met} == {vol_zonder}")
    beste = rules.workout_best_set(sets)
    check("opwarmset wordt nooit beste set", beste["weight_kg"] == 110, beste["weight_kg"])

    workouts = pd.DataFrame([{"id": 1, "started_at": "2026-08-01", "is_deload": 0}])
    serie = rules.progression_series(sets.assign(workout_id=1), workouts, "weight")
    pr = rules.personal_records(serie, "weight")
    check("PR-bepaling gebruikt alleen werksets (via progression_series)",
          abs(pr["e1rm"]["waarde"] - beste_werkset_e1rm) < 0.01,
          f"PR e1RM={pr['e1rm']['waarde']:.1f}")


# ------------------------------------------------------- progressie --

def test_progressie_per_load_type() -> None:
    print("\n== Progressie per load_type ==")
    check("Epley: 100kg x 5 reps", abs(rules.epley_1rm(100, 5) - 100 * (1 + 5 / 30)) < 1e-9)
    check("Epley: geen gewicht -> None", rules.epley_1rm(None, 5) is None)
    check("Epley: 0 reps -> None", rules.epley_1rm(100, 0) is None)

    gewicht_sets = _sets([{"weight_kg": 100, "reps": 8}, {"weight_kg": 100, "reps": 8}])
    totals = rules.workout_totals(gewicht_sets, "weight", target_reps=8)
    check("weight: volume = som(reps*kg)", totals["volume_kg"] == 1600, totals["volume_kg"])

    bw_sets = _sets([{"reps": 10}, {"reps": 8}, {"reps": 7}])
    totals_bw = rules.workout_totals(bw_sets, "bodyweight")
    check("bodyweight: totaal aantal reps", totals_bw["total_reps"] == 25, totals_bw)

    tijd_sets = _sets([{"seconds": 30}, {"seconds": 35}])
    totals_t = rules.workout_totals(tijd_sets, "time")
    check("time: totale seconden", totals_t["total_seconds"] == 65, totals_t)

    afstand_sets = _sets([{"meters": 40}, {"meters": 40}])
    totals_d = rules.workout_totals(afstand_sets, "distance")
    check("distance: totale meters", totals_d["total_meters"] == 80, totals_d)

    per_side_sets = _sets([
        {"reps": 8, "side": "left"}, {"reps": 8, "side": "right"},
        {"reps": 10, "side": "left"}, {"reps": 10, "side": "right"},
    ])
    totals_ps = rules.workout_totals(per_side_sets, "bodyweight")
    check("per_side: links en rechts tellen als losse sets, niet gemiddeld",
          totals_ps["total_reps"] == 36, totals_ps)


# --------------------------------------------------------- gewichtsuggestie --

def test_gewichtsuggestie() -> None:
    print("\n== Gewichtsuggestie ==")
    conn = _conn()
    tpl = catalog.load_templates(conn)
    a_id, b_id = int(tpl.iloc[0]["id"]), int(tpl.iloc[1]["id"])
    ex = catalog.load_exercises(conn)
    deadlift = int(ex[ex["name"] == "Deadlift"].iloc[0]["id"])

    leeg = store.last_values_for_exercise(conn, deadlift)
    check("leeg bij nooit eerder gedaan", leeg.empty)

    wid1 = store.start_workout(conn, a_id, phase=1)
    store.log_set(conn, wid1, deadlift, 1, reps=8, weight_kg=100)
    store.log_set(conn, wid1, deadlift, 2, reps=8, weight_kg=100, is_warmup=True)
    store.complete_workout(conn, wid1, session_rpe=7)

    # Een sessie waar deadlift NIET in een sjabloon zit maar wel losstaand
    # gelogd wordt (bijv. een vrije sessie): toont dat "ongeacht sjabloon" klopt.
    wid2 = store.start_workout(conn, b_id, phase=1)
    store.log_set(conn, wid2, deadlift, 1, reps=6, weight_kg=110)

    vorige = store.last_values_for_exercise(conn, deadlift, exclude_workout_id=wid2)
    check("vindt de vorige keer ook buiten hetzelfde sjabloon",
          not vorige.empty and vorige.iloc[0]["weight_kg"] == 100,
          vorige["weight_kg"].tolist() if not vorige.empty else None)
    check("sluit de lopende sessie uit (exclude_workout_id)",
          wid2 not in set(vorige["workout_id"]) if not vorige.empty else True)
    check("sluit opwarmsets uit", (vorige["is_warmup"] == 0).all() if not vorige.empty else True)
    conn.close()


# --------------------------------------------------- deactiveren/hernoemen --

def test_deactiveren_behoudt_historie() -> None:
    print("\n== Deactiveren behoudt historie ==")
    conn = _conn()
    tpl = catalog.load_templates(conn)
    a_id = int(tpl.iloc[0]["id"])
    ex = catalog.load_exercises(conn)
    deadlift = int(ex[ex["name"] == "Deadlift"].iloc[0]["id"])

    wid = store.start_workout(conn, a_id, phase=1)
    store.log_set(conn, wid, deadlift, 1, reps=8, weight_kg=100)
    store.complete_workout(conn, wid, session_rpe=7)
    aantal_voor = len(catalog.load_exercises(conn))

    catalog.set_exercise_active(conn, deadlift, False)

    check("verdwijnt uit load_exercises(only_active=True)",
          deadlift not in set(catalog.load_exercises(conn, only_active=True)["id"]))
    items_a_actief = catalog.load_template_items(conn, a_id, only_active=True)
    check("verdwijnt uit load_template_items(only_active=True)",
          deadlift not in set(items_a_actief["exercise_id"]))

    sets = store.load_sets(conn, workout_id=wid)
    check("blijft in load_sets()", len(sets) == 1 and sets.iloc[0]["exercise_id"] == deadlift)
    workouts = store.load_workouts(conn)
    serie = rules.progression_series(sets, workouts, "weight")
    check("blijft in progression_series()", len(serie) == 1)
    pr = rules.personal_records(serie, "weight")
    check("blijft in personal_records()", pr.get("e1rm") is not None)

    aantal_na = len(catalog.load_exercises(conn))
    check("rijtelling ongewijzigd (geen delete)", aantal_voor == aantal_na,
          f"{aantal_voor} -> {aantal_na}")

    catalog.set_exercise_active(conn, deadlift, True)
    check("heractiveren werkt",
          deadlift in set(catalog.load_exercises(conn, only_active=True)["id"]))
    conn.close()


def test_hernoemen_werkt_door() -> None:
    print("\n== Hernoemen werkt door in de historie ==")
    conn = _conn()
    tpl = catalog.load_templates(conn)
    a_id = int(tpl.iloc[0]["id"])
    ex = catalog.load_exercises(conn)
    deadlift = int(ex[ex["name"] == "Deadlift"].iloc[0]["id"])

    wid = store.start_workout(conn, a_id, phase=1)
    store.log_set(conn, wid, deadlift, 1, reps=8, weight_kg=100)
    store.complete_workout(conn, wid, session_rpe=7)

    catalog.update_exercise(conn, deadlift, name="Trap bar deadlift")
    sets = store.load_sets(conn, workout_id=wid)
    check("oude workout toont de NIEUWE naam (live join)",
          sets.iloc[0]["exercise_name"] == "Trap bar deadlift", sets.iloc[0]["exercise_name"])
    check("de snapshot bevat nog de oude naam (vangnet)",
          sets.iloc[0]["exercise_name_snapshot"] == "Deadlift",
          sets.iloc[0]["exercise_name_snapshot"])

    catalog.set_exercise_active(conn, deadlift, False)
    sets2 = store.load_sets(conn, workout_id=wid)
    check("blijft de nieuwe naam tonen ook na latere deactivering",
          sets2.iloc[0]["exercise_name"] == "Trap bar deadlift")
    conn.close()


# ------------------------------------------------------------------ deload --

def test_deload_signaal() -> None:
    print("\n== Deload-signaal ==")
    vandaag = date(2026, 9, 1)

    def workouts_df(n, weken_geleden_deload=None):
        rijen = []
        for i in range(n):
            gestart = vandaag - timedelta(weeks=i)
            rijen.append({"started_at": gestart.isoformat(), "completed_at": "x",
                         "is_deload": 0})
        if weken_geleden_deload is not None:
            rijen.append({
                "started_at": (vandaag - timedelta(weeks=weken_geleden_deload)).isoformat(),
                "completed_at": "x", "is_deload": 1,
            })
        return pd.DataFrame(rijen)

    check("9 workouts, geen deload -> False (te weinig workouts)",
          rules.suggest_deload(workouts_df(9), today=vandaag) is False)
    check("10 workouts, laatste deload 8 weken geleden -> True",
          rules.suggest_deload(workouts_df(10, weken_geleden_deload=8), today=vandaag) is True)
    check("10 workouts, deload 3 weken geleden -> False",
          rules.suggest_deload(workouts_df(10, weken_geleden_deload=3), today=vandaag) is False)
    check("direct na het loggen van een deload -> False",
          rules.suggest_deload(workouts_df(10, weken_geleden_deload=0), today=vandaag) is False)
    check("50 dagen na die deload -> True (venster verstreken)",
          rules.suggest_deload(workouts_df(10, weken_geleden_deload=0),
                               today=vandaag + timedelta(days=50)) is True)
    check("meldingstekst noemt 60%", "60%" in rules.DELOAD_MELDING)


def test_deload_verlaagt_geen_pr() -> None:
    print("\n== Een deload-sessie verlaagt geen PR ==")
    conn = _conn()
    tpl = catalog.load_templates(conn)
    a_id = int(tpl.iloc[0]["id"])
    ex = catalog.load_exercises(conn)
    deadlift = int(ex[ex["name"] == "Deadlift"].iloc[0]["id"])

    wid1 = store.start_workout(conn, a_id, phase=1)
    store.log_set(conn, wid1, deadlift, 1, reps=5, weight_kg=120)
    store.complete_workout(conn, wid1, session_rpe=8)

    wid2 = store.start_workout(conn, a_id, phase=1, is_deload=True)
    store.log_set(conn, wid2, deadlift, 1, reps=5, weight_kg=72)  # 60% van 120
    store.complete_workout(conn, wid2, session_rpe=4)

    sets = store.load_sets(conn, exercise_id=deadlift)
    workouts = store.load_workouts(conn)
    serie = rules.progression_series(sets, workouts, "weight")
    pr = rules.personal_records(serie, "weight")
    check("PR blijft op 120 kg na een lichtere deload",
          abs(pr["e1rm"]["waarde"] - rules.epley_1rm(120, 5)) < 0.01,
          f"PR e1RM={pr['e1rm']['waarde']:.1f}")

    wid3 = store.start_workout(conn, a_id, phase=1, is_deload=True)
    store.log_set(conn, wid3, deadlift, 1, reps=5, weight_kg=140)  # toevallig zwaarder
    store.complete_workout(conn, wid3, session_rpe=5)
    sets = store.load_sets(conn, exercise_id=deadlift)
    serie = rules.progression_series(sets, store.load_workouts(conn), "weight")
    pr2 = rules.personal_records(serie, "weight")
    check("een deload met toevallig hoger gewicht verhoogt het PR NIET",
          abs(pr2["e1rm"]["waarde"] - rules.epley_1rm(120, 5)) < 0.01,
          f"PR e1RM={pr2['e1rm']['waarde']:.1f} (deloads categorisch uitgesloten)")
    check("de deload-rij staat wél in progression_series met is_deload=True",
          bool(serie[serie["workout_id"] == wid3]["is_deload"].iloc[0]))
    conn.close()


# --------------------------------------------------------- kwaliteitssessie --

def _act(sport, start, duration_s, avg_hr=None, np_power=None, avg_power=None,
        excluded_reason=None):
    return {
        "activity_key": f"{sport}-{start}", "sport": sport,
        "start_time": pd.Timestamp(start, tz="UTC"), "duration_s": duration_s,
        "avg_hr": avg_hr, "np_power": np_power, "avg_power": avg_power,
        "excluded_reason": excluded_reason,
    }


def test_kwaliteitssessie_en_conflict() -> None:
    print("\n== Kwaliteitssessie & sessieconflict ==")
    op_dag = date(2026, 8, 20)

    interval_gisteren = _act("running", "2026-08-19 18:00:00", 2700, avg_hr=160)  # >153
    lange_rit_morgen = _act("cycling", "2026-08-21 09:00:00", 9000, avg_hr=130)   # >2u
    rustige_duurloop_vandaag = _act("running", "2026-08-20 07:00:00", 3600, avg_hr=140)
    zwem_3_dagen_terug = _act("swimming", "2026-08-17 07:00:00", 1800, avg_hr=165)
    transport = _act("cycling", "2026-08-20 12:00:00", 600, avg_hr=170,
                     excluded_reason="transport")

    check("intervalloop (HR>90%*LTHR) is een kwaliteitssessie",
          rules.is_quality_session(interval_gisteren, ATHLETE))
    check("lange rit (>2u) is een kwaliteitssessie",
          rules.is_quality_session(lange_rit_morgen, ATHLETE))
    check("rustige duurloop is GEEN kwaliteitssessie",
          not rules.is_quality_session(rustige_duurloop_vandaag, ATHLETE))

    acts = pd.DataFrame([interval_gisteren, lange_rit_morgen, rustige_duurloop_vandaag,
                        zwem_3_dagen_terug])
    treffers = rules.check_session_conflict(acts, op_dag, ATHLETE)
    check("precies 2 treffers (interval gisteren + lange rit morgen)",
          len(treffers) == 2, [t["when"] for t in treffers])
    check("de rustige duurloop van vandaag geeft geen treffer",
          not any(t["activity_key"] == rustige_duurloop_vandaag["activity_key"] for t in treffers))
    check("een sessie van 3 dagen terug valt buiten het venster",
          not any(t["activity_key"] == zwem_3_dagen_terug["activity_key"] for t in treffers))

    fietsrit_np = _act("cycling", "2026-08-20 08:00:00", 3600, np_power=240)  # >0.85*264=224.4
    check("fietsrit met NP boven 85% FTP is een kwaliteitssessie",
          rules.is_quality_session(fietsrit_np, ATHLETE))
    geen_ftp = dict(ATHLETE, thresholds={**ATHLETE["thresholds"],
                                        "cycling": {"lthr": 162, "ftp": None}})
    check("zonder FTP valt fietsen terug op HR, geen crash",
          rules.is_quality_session(dict(fietsrit_np, avg_hr=155), geen_ftp) is not None)

    # check_session_conflict zelf kent geen excluded_reason — dat filter hoort
    # bij de aanroeper (training_activities()), precies zoals transport.py's
    # eigen classificatie ook los staat van de excluded_reason-markering.
    # Een korte, hartige transportrit ZOU dus best als kwaliteitssessie
    # aanslaan als hij niet vooraf wordt weggefilterd:
    acts_ongefilterd = pd.DataFrame([transport])
    treffers_ongefilterd = rules.check_session_conflict(acts_ongefilterd, op_dag, ATHLETE)
    check("zonder vooraf filteren telt een hartige transportrit wél mee "
          "(bevestigt dat de aanroeper training_activities() moet gebruiken)",
          len(treffers_ongefilterd) == 1, len(treffers_ongefilterd))

    acts_gefilterd = acts_ongefilterd[acts_ongefilterd["excluded_reason"].isna()]
    treffers_gefilterd = rules.check_session_conflict(acts_gefilterd, op_dag, ATHLETE)
    check("na training_activities()-achtig filteren (excluded_reason weg) "
          "geeft de transportrit geen treffer meer",
          len(treffers_gefilterd) == 0, len(treffers_gefilterd))


# ------------------------------------------------------------------ weekvolume --

def test_weekvolume() -> None:
    print("\n== Weekvolume ==")
    conn = _conn()
    tpl = catalog.load_templates(conn)
    a_id = int(tpl.iloc[0]["id"])
    ex = catalog.load_exercises(conn)
    deadlift = int(ex[ex["name"] == "Deadlift"].iloc[0]["id"])
    pullups = int(ex[ex["name"] == "Pull-ups"].iloc[0]["id"])

    wid = store.start_workout(conn, a_id, phase=1)
    store.log_set(conn, wid, deadlift, 1, reps=8, weight_kg=100)
    store.log_set(conn, wid, deadlift, 2, reps=8, weight_kg=100, is_warmup=True)
    store.log_set(conn, wid, pullups, 1, reps=10)  # geen gewicht: telt niet mee
    store.complete_workout(conn, wid, session_rpe=7)

    wid2 = store.start_workout(conn, a_id, phase=1)  # niet voltooid
    store.log_set(conn, wid2, deadlift, 1, reps=8, weight_kg=999)

    vol = store.weekly_strength_volume(conn)
    check("alleen voltooide workouts, alleen werksets, alleen gewicht",
          len(vol) == 1 and vol.iloc[0]["volume_kg"] == 800,
          vol.to_dict("records"))

    from tricoach.analysis import add_week
    from tricoach.strength.rules import iso_week_label
    nu = pd.Timestamp.now()
    check("weeklabel-formaat identiek aan analysis.add_week() op dezelfde datum",
          iso_week_label(pd.Series([nu])).iloc[0]
          == add_week(pd.DataFrame({"start_time": [nu]}))["week"].iloc[0])
    conn.close()


# ------------------------------------------------------------- fase-instelling --

def test_fase_instelling() -> None:
    print("\n== Fase-instelling ==")
    check("default fase 1", rules.current_phase({}) == 1)
    check("fase 3 uit config", rules.current_phase({"training": {"phase": 3}}) == 3)
    check("geklemd op 3 bij een te hoge waarde",
          rules.current_phase({"training": {"phase": 9}}) == 3)
    check("valt terug op 1 bij een ongeldige waarde",
          rules.current_phase({"training": {"phase": "twee"}}) == 1)

    item = {"target_sets_p1": 3, "target_reps_p1": "8", "target_sets_p2": None,
           "target_reps_p2": None}
    check("phase_target valt terug op fase 1 bij een lege fase 2",
          rules.phase_target(item, 2) == (3, "8"))
    check("phase_target geeft de eigen waarde als die er is",
          rules.phase_target({**item, "target_sets_p2": 4, "target_reps_p2": "6"}, 2) == (4, "6"))

    check('parse_target_reps("max") -> None', rules.parse_target_reps("max") is None)
    check('parse_target_reps("8") -> 8', rules.parse_target_reps("8") == 8)
    check('parse_target_reps("40m") -> 40', rules.parse_target_reps("40m") == 40)
    check('parse_target_reps("30s") -> 30', rules.parse_target_reps("30s") == 30)


# ------------------------------------------------------------ workout soft-delete --

def test_workout_soft_delete() -> None:
    print("\n== Workout soft-delete ==")
    conn = _conn()
    tpl = catalog.load_templates(conn)
    a_id = int(tpl.iloc[0]["id"])
    ex = catalog.load_exercises(conn)
    deadlift = int(ex[ex["name"] == "Deadlift"].iloc[0]["id"])

    wid = store.start_workout(conn, a_id, phase=1)
    store.log_set(conn, wid, deadlift, 1, reps=8, weight_kg=100)
    store.complete_workout(conn, wid, session_rpe=7)

    ok = store.soft_delete_workout(conn, wid)
    check("soft_delete_workout geeft True", ok)
    check("verdwijnt uit load_workouts()", wid not in set(store.load_workouts(conn)["id"]))
    check("verdwijnt uit weekly_strength_volume", store.weekly_strength_volume(conn).empty)
    nxt = store.next_template(conn)
    check("verdwijnt uit de rotatie (terug bij A: geen voltooide workouts meer)",
          nxt["code"] == "A", nxt["code"])

    ok2 = store.restore_workout(conn, wid)
    check("restore_workout herstelt hem", ok2 and wid in set(store.load_workouts(conn)["id"]))
    rij = conn.execute("SELECT COUNT(*) FROM strength_workout WHERE id=?", (wid,)).fetchone()
    check("de rij heeft altijd bestaan (geen echte delete)", rij[0] == 1)
    conn.close()


def main() -> int:
    print("Krachtmodule — tests")
    for test in (
        test_seed_en_schema, test_rotatie, test_laatst_gebruikte_template_inactief,
        test_opwarmsets, test_progressie_per_load_type, test_gewichtsuggestie,
        test_deactiveren_behoudt_historie, test_hernoemen_werkt_door,
        test_deload_signaal, test_deload_verlaagt_geen_pr,
        test_kwaliteitssessie_en_conflict, test_weekvolume, test_fase_instelling,
        test_workout_soft_delete,
    ):
        test()
    print(f"\n{'=' * 60}")
    print(f"{len(GESLAAGD)} geslaagd, {len(GEFAALD)} gefaald")
    for naam in GEFAALD:
        print(f"  ❌ {naam}")
    return 1 if GEFAALD else 0


if __name__ == "__main__":
    sys.exit(main())
