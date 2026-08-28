"""Testscript voor de coremodule (tricoach.coretraining + tricoach.streak).

Gebruik:  python tests/test_core.py

De streak-/freeze-algoritmiek (tricoach.streak) wordt rechtstreeks met
set[date] getest — puur, geen database nodig. De opslaglaag
(tricoach.coretraining) wordt tegen een tijdelijke database getest, inclusief
de freeze-reconciliatie en de "korte dag"-knop.
"""

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))


import sqlite3
import tempfile
from datetime import date, timedelta
from pathlib import Path

from tricoach import coretraining, streak
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
    tmp = Path(tempfile.mkdtemp(prefix="tricoach_core_test_"))
    return connect(tmp / "test.db")


def _reeks(start: date, n: int) -> set:
    return {start + timedelta(days=i) for i in range(n)}


# ------------------------------------------------------ seed & dubbele invoer --

def test_seed_en_dubbele_invoer() -> None:
    print("\n== Seed & dubbele invoer ==")
    conn = _conn()
    ex = coretraining.load_core_exercises(conn)
    check("9 core-oefeningen geseed", len(ex) == 9, f"{len(ex)} stuks")
    check("2 categorieën", set(ex["category"]) == {"mobiliteit", "romp"})
    eid = int(ex.iloc[0]["id"])
    vandaag = date(2026, 8, 1)

    ok1 = coretraining.log_exercise(conn, eid, vandaag)
    check("eerste keer loggen lukt", ok1)
    ok2 = coretraining.log_exercise(conn, eid, vandaag)
    check("tweede keer dezelfde dag: False, geen crash", ok2 is False)
    n = conn.execute("SELECT COUNT(*) FROM core_log WHERE exercise_id=? AND log_date=?",
                     (eid, vandaag.isoformat())).fetchone()[0]
    check("er staat maar 1 rij", n == 1, n)

    faalde = False
    try:
        conn.execute(
            "INSERT INTO core_log (exercise_id, exercise_name_snapshot, log_date, logged_at) "
            "VALUES (?,?,?,?)", (eid, "x", vandaag.isoformat(), "2026-08-01T12:00:00"))
    except sqlite3.IntegrityError:
        faalde = True
    check("een rauwe dubbele INSERT gooit echt een IntegrityError (de constraint bestaat)",
          faalde)
    conn.close()


def test_tijdzone() -> None:
    print("\n== Tijdzone: log_date is de meegegeven lokale datum ==")
    conn = _conn()
    ex = coretraining.load_core_exercises(conn)
    eid = int(ex.iloc[0]["id"])
    d = date(2026, 8, 15)
    coretraining.log_exercise(conn, eid, d)
    opgeslagen = conn.execute(
        "SELECT log_date FROM core_log WHERE exercise_id=?", (eid,)).fetchone()[0]
    check("log_date == date.today()-achtige lokale datum, letterlijk overgenomen",
          opgeslagen == d.isoformat(), opgeslagen)
    conn.close()


# ---------------------------------------------------------- dagcompleet-drempel --

def test_dagcompleet_drempel() -> None:
    print("\n== Dagcompleet-drempel ==")
    counts = {date(2026, 8, 1): 1, date(2026, 8, 2): 2, date(2026, 8, 3): 6}
    check("1 oefening bij drempel 1 -> compleet",
          streak.is_dag_compleet(counts, date(2026, 8, 1), threshold=1))
    check("0 oefeningen (niet gelogde dag) -> niet compleet",
          not streak.is_dag_compleet(counts, date(2026, 8, 5), threshold=1))
    check("drempel 3: 2 oefeningen -> niet compleet",
          not streak.is_dag_compleet(counts, date(2026, 8, 2), threshold=3))
    check("drempel 3: 6 oefeningen -> wel compleet",
          streak.is_dag_compleet(counts, date(2026, 8, 3), threshold=3))
    check("6 oefeningen -> volledige dag (cosmetisch)",
          streak.is_volledige_dag(counts, date(2026, 8, 3)))
    check("5 oefeningen -> geen volledige dag",
          not streak.is_volledige_dag({date(2026, 8, 4): 5}, date(2026, 8, 4)))


# ------------------------------------------------------------------- streak --

def test_streak_basis() -> None:
    print("\n== Streak: basisgedrag en ankerpunt ==")
    vandaag = date(2026, 8, 20)
    compleet = _reeks(vandaag - timedelta(days=4), 5)  # 5 dagen t/m vandaag
    s = streak.current_streak(compleet, set(), vandaag)
    check("5 aaneengesloten dagen t/m vandaag -> streak 5", s == 5, s)

    compleet_zonder_vandaag = _reeks(vandaag - timedelta(days=4), 4)  # 4 dagen t/m gisteren
    s2 = streak.current_streak(compleet_zonder_vandaag, set(), vandaag)
    check("vandaag nog niets, 4 dagen t/m gisteren wel -> streak 4 (anker schuift naar gisteren)",
          s2 == 4, s2)

    s3 = streak.current_streak(set(), set(), vandaag)
    check("lege log -> streak 0", s3 == 0)

    compleet_oud = {vandaag - timedelta(days=10)}
    s4 = streak.current_streak(compleet_oud, set(), vandaag)
    check("vandaag én gisteren niets (alleen een oude losse dag) -> streak 0", s4 == 0, s4)


def test_freeze_redt_een_dag() -> None:
    print("\n== Freeze redt één gemiste dag ==")
    vandaag = date(2026, 8, 20)
    compleet = (_reeks(vandaag - timedelta(days=6), 3)   # -6,-5,-4
               | _reeks(vandaag - timedelta(days=2), 3))  # -2,-1,vandaag
    # dag -3 ontbreekt: precies één gat tussen twee complete reeksen
    bevroren = streak.compute_freezes(compleet, vandaag)
    dag_min3 = vandaag - timedelta(days=3)
    check("de ontbrekende dag wordt bevroren", dag_min3 in bevroren, bevroren)
    s = streak.current_streak(compleet, bevroren, vandaag)
    check("streak = 6 (de bevroren dag zelf telt niet mee in het getal)", s == 6, s)


def test_twee_opeenvolgende_dagen_breekt() -> None:
    print("\n== Twee opeenvolgende gemiste dagen breken de streak ==")
    vandaag = date(2026, 8, 20)
    compleet = (_reeks(vandaag - timedelta(days=7), 3)   # -7,-6,-5
               | _reeks(vandaag - timedelta(days=2), 3))  # -2,-1,vandaag
    # -4 en -3 ontbreken allebei: een gat van twee dagen
    bevroren = streak.compute_freezes(compleet, vandaag)
    check("geen enkele freeze toegekend bij een gat van twee dagen",
          len(bevroren) == 0, bevroren)
    s = streak.current_streak(compleet, bevroren, vandaag)
    check("de streak begint na het gat opnieuw (loopt niet door tot -7)", s == 3, s)


def test_maandcap() -> None:
    print("\n== Maximaal twee freezes per kalendermaand ==")
    vandaag = date(2026, 8, 20)
    # Complete dagen met daartussen drie losse eendagsgaten: 1, 4, 7, 10, 13.
    compleet = {vandaag - timedelta(days=d) for d in (0, 2, 4, 6, 8, 10, 12, 14, 16, 18)}
    bevroren = streak.compute_freezes(compleet, vandaag)
    check("precies de eerste twee eendagsgaten (chronologisch) worden bevroren",
          len(bevroren) == 2, sorted(bevroren))

    volgende_maand = date(2026, 9, 15)
    compleet2 = compleet | {volgende_maand - timedelta(days=1), volgende_maand + timedelta(days=1)}
    bevroren2 = streak.compute_freezes(compleet2, volgende_maand + timedelta(days=1))
    freezes_september = {d for d in bevroren2 if d.month == 9}
    check("een eendagsgat in de volgende kalendermaand krijgt weer een freeze "
          "(cap is per maand, niet rollend)",
          len(freezes_september) == 1, freezes_september)


def test_terugwerkend_invullen() -> None:
    print("\n== Terugwerkend invullen herberekent de streak correct ==")
    conn = _conn()
    ex = coretraining.load_core_exercises(conn)
    eid = int(ex.iloc[0]["id"])
    vandaag = date(2026, 8, 20)

    for d in (vandaag - timedelta(days=6), vandaag - timedelta(days=5),
             vandaag - timedelta(days=4), vandaag - timedelta(days=2),
             vandaag - timedelta(days=1), vandaag):
        coretraining.log_exercise(conn, eid, d)

    stats_voor = coretraining.streak_stats(conn, today=vandaag)
    gat = vandaag - timedelta(days=3)
    check("het gat is bevroren vóór het invullen",
          gat.isoformat() in {r[0] for r in
                              conn.execute("SELECT used_on FROM core_streak_freeze").fetchall()})
    check("streak vóór invullen is 6 (de freeze telt niet mee)",
          stats_voor["current"] == 6, stats_voor["current"])

    coretraining.log_exercise(conn, eid, gat)
    stats_na = coretraining.streak_stats(conn, today=vandaag)
    check("de freeze verdwijnt uit core_streak_freeze (afgeleide cache, herberekend)",
          gat.isoformat() not in {r[0] for r in
                                  conn.execute("SELECT used_on FROM core_streak_freeze").fetchall()})
    check("streak na invullen is 7 (één langer, freeze niet meer nodig)",
          stats_na["current"] == 7, stats_na["current"])

    coretraining.unlog_exercise(conn, eid, gat)
    stats_terug = coretraining.streak_stats(conn, today=vandaag)
    check("unlog_exercise draait dit om: het gat (en de freeze) komen terug",
          stats_terug["current"] == 6, stats_terug["current"])
    conn.close()


def test_langste_streak_en_consistentie() -> None:
    print("\n== Langste streak & consistentiepercentage ==")
    basis = date(2026, 7, 1)
    compleet = _reeks(basis, 10) | _reeks(basis + timedelta(days=15), 20)
    langste = streak.longest_streak(compleet, set())
    check("langste streak is de langste reeks (20)", langste == 20, langste)

    # Een freeze overbrugt ook in longest_streak zonder zelf te tellen.
    compleet_met_gat = _reeks(basis, 5) | {basis + timedelta(days=6)}  # gat op dag 5
    freezes = {basis + timedelta(days=5)}
    langste2 = streak.longest_streak(compleet_met_gat, freezes)
    check("een bevroren gat overbrugt longest_streak (5+1=6 dagen aaneengesloten)",
          langste2 == 6, langste2)

    vandaag = date(2026, 8, 30)
    dagen = [vandaag - timedelta(days=i) for i in range(30)]
    compleet_pct = set(dagen[:21])  # 21 van de 30
    pct = streak.consistency_pct(compleet_pct, vandaag, days=30)
    check("consistentie: 21 van 30 dagen -> 70.0%", pct == 70.0, pct)
    pct_met_freeze = streak.consistency_pct(compleet_pct | set(), vandaag, days=30)
    check("een bevroren dag (niet in complete_days) verhoogt het percentage niet",
          pct_met_freeze == pct, pct_met_freeze)


def test_mijlpalen_en_meldingen() -> None:
    print("\n== Mijlpalen & 'nooit twee keer missen' ==")
    check("7 is een mijlpaal", streak.milestone_reached(7) == 7)
    check("8 is geen mijlpaal", streak.milestone_reached(8) is None)
    check("alle mijlpalen kloppen met MILESTONES",
          all(streak.milestone_reached(m) == m for m in streak.MILESTONES))

    vandaag = date(2026, 8, 20)
    compleet = {vandaag - timedelta(days=5)}  # historie, maar niet gisteren/vandaag
    msg = streak.never_twice_message(compleet, vandaag)
    check("melding verschijnt: gisteren gemist, vandaag nog niets",
          msg == streak.NEVER_TWICE_TEKST)

    compleet_vandaag = compleet | {vandaag}
    check("melding verdwijnt zodra er vandaag iets gelogd is",
          streak.never_twice_message(compleet_vandaag, vandaag) is None)

    check("geen melding bij een compleet lege log (geen verwijt aan nieuwe gebruiker)",
          streak.never_twice_message(set(), vandaag) is None)


# --------------------------------------------------------------- korte dag --

def test_korte_dag() -> None:
    print("\n== Korte dag ==")
    conn = _conn()
    d = date(2026, 8, 10)
    gelogd = coretraining.log_korte_dag(conn, d)
    check("logt precies 3 oefeningen", len(gelogd) == 3, gelogd)
    check("de dag is compleet (drempel 1)",
          streak.is_dag_compleet(coretraining.day_counts(conn), d))

    ex = coretraining.load_core_exercises(conn)
    side_plank = int(ex[ex["name"] == "Side plank"].iloc[0]["id"])
    coretraining.set_core_exercise_active(conn, side_plank, False)
    d2 = date(2026, 8, 11)
    gelogd2 = coretraining.log_korte_dag(conn, d2)
    check("na deactiveren van Side plank: nog 2 gelogd", len(gelogd2) == 2, gelogd2)
    check("Side plank zit niet in het resultaat", "Side plank" not in gelogd2)

    dead_bug = int(ex[ex["name"] == "Dead bug"].iloc[0]["id"])
    coretraining.update_core_exercise(conn, dead_bug, name="Deadbug")
    d3 = date(2026, 8, 12)
    gelogd3 = coretraining.log_korte_dag(conn, d3)
    check("hernoemen breekt niks: 'Deadbug' wordt gewoon gevonden via is_quick_day "
          "(geen naam-matching, dat was de bekende wrijving die dit ontwerp juist vermijdt)",
          "Deadbug" in gelogd3, gelogd3)

    aantal_voor = conn.execute("SELECT COUNT(*) FROM core_log WHERE log_date=?",
                               (d.isoformat(),)).fetchone()[0]
    coretraining.log_korte_dag(conn, d)  # nogmaals dezelfde dag
    aantal_na = conn.execute("SELECT COUNT(*) FROM core_log WHERE log_date=?",
                             (d.isoformat(),)).fetchone()[0]
    check("tweemaal dezelfde dag: geen duplicaten, geen crash", aantal_voor == aantal_na)
    conn.close()


# ------------------------------------------------------ verwaarloosde oefeningen --

def test_verwaarloosd() -> None:
    print("\n== Verwaarloosde oefeningen ==")
    conn = _conn()
    ex = coretraining.load_core_exercises(conn)
    vandaag = date(2026, 8, 20)
    e1, e2, e3 = [int(x) for x in ex["id"].iloc[:3]]

    coretraining.log_exercise(conn, e1, vandaag - timedelta(days=15))  # verwaarloosd
    coretraining.log_exercise(conn, e2, vandaag - timedelta(days=3))   # niet verwaarloosd
    # e3: nooit gedaan

    dagen = coretraining.dagen_sinds_laatste(conn, today=vandaag)
    check("juiste dagenteller voor e1 (15)", dagen[e1] == 15, dagen[e1])
    check("juiste dagenteller voor e2 (3)", dagen[e2] == 3, dagen[e2])
    check("None voor een nooit gedane oefening (e3)", dagen[e3] is None)

    lijst = coretraining.verwaarloosd(conn, drempel=10, today=vandaag)
    ids_in_lijst = {r["exercise_id"] for r in lijst}
    check("e1 (15 dagen) staat in de verwaarloosd-lijst", e1 in ids_in_lijst)
    check("e3 (nooit gedaan) staat er ook in", e3 in ids_in_lijst)
    check("e2 (3 dagen) staat er niet in", e2 not in ids_in_lijst)

    coretraining.set_core_exercise_active(conn, e1, False)
    lijst2 = coretraining.verwaarloosd(conn, drempel=10, today=vandaag)
    check("een gedeactiveerde oefening verdwijnt uit de verwaarloosd-lijst",
          e1 not in {r["exercise_id"] for r in lijst2})
    conn.close()


# ----------------------------------------------------- deactiveren behoudt historie --

def test_deactiveren_behoudt_core_historie() -> None:
    print("\n== Deactiveren behoudt core-historie ==")
    conn = _conn()
    ex = coretraining.load_core_exercises(conn)
    eid = int(ex.iloc[0]["id"])
    d = date(2026, 8, 5)
    coretraining.log_exercise(conn, eid, d)

    coretraining.set_core_exercise_active(conn, eid, False)
    check("verdwijnt uit load_core_exercises(only_active=True)",
          eid not in set(coretraining.load_core_exercises(conn, only_active=True)["id"]))

    logs = coretraining.logs_for_period(conn, d, d)
    check("logs_for_period bevat de gedeactiveerde oefening nog steeds",
          eid in set(logs["exercise_id"]))
    check("... met exercise_is_active=0 (zodat de UI hem kan doorstrepen)",
          logs[logs["exercise_id"] == eid].iloc[0]["exercise_is_active"] == 0)

    check("de dag telt nog mee voor complete_days/streak",
          d in coretraining.complete_days(conn))

    coretraining.update_core_exercise(conn, eid, name="Nieuwe naam")
    logs2 = coretraining.logs_for_period(conn, d, d)
    check("hernoemen na deactiveren toont de nieuwe naam in de historie",
          logs2[logs2["exercise_id"] == eid].iloc[0]["exercise_name"] == "Nieuwe naam")
    conn.close()


# ------------------------------------------------------------- heatmap-matrix --

def test_heatmap_matrix() -> None:
    print("\n== Heatmap-matrix ==")
    start, eind = date(2026, 8, 1), date(2026, 8, 14)
    compleet = {date(2026, 8, 3), date(2026, 8, 5)}
    bevroren = {date(2026, 8, 4)}
    z, x_labels, y_labels, hover = streak.heatmap_matrix(compleet, bevroren, start, eind)
    check("7 rijen (ma..zo)", len(z) == 7 and len(y_labels) == 7)
    check("elke rij heeft evenveel kolommen als x_labels", all(len(rij) == len(x_labels) for rij in z))

    def _waarde(d: date):
        maandag = start - timedelta(days=start.weekday())
        week_idx = (d - maandag).days // 7
        dag_idx = d.weekday()
        return z[dag_idx][week_idx]

    check("complete dag krijgt waarde 1", _waarde(date(2026, 8, 3)) == 1)
    check("bevroren dag krijgt waarde 2", _waarde(date(2026, 8, 4)) == 2)
    check("gewone lege dag krijgt waarde 0", _waarde(date(2026, 8, 6)) == 0)

    maandag = start - timedelta(days=start.weekday())
    if maandag < start:
        check("dagen vóór start zijn None", z[0][0] is None)
    eind_zondag = eind + timedelta(days=6 - eind.weekday())
    if eind_zondag > eind:
        check("dagen ná eind zijn None", z[6][-1] is None)


def main() -> int:
    print("Coremodule — tests")
    for test in (
        test_seed_en_dubbele_invoer, test_tijdzone, test_dagcompleet_drempel,
        test_streak_basis, test_freeze_redt_een_dag, test_twee_opeenvolgende_dagen_breekt,
        test_maandcap, test_terugwerkend_invullen, test_langste_streak_en_consistentie,
        test_mijlpalen_en_meldingen, test_korte_dag, test_verwaarloosd,
        test_deactiveren_behoudt_core_historie, test_heatmap_matrix,
    ):
        test()
    print(f"\n{'=' * 60}")
    print(f"{len(GESLAAGD)} geslaagd, {len(GEFAALD)} gefaald")
    for naam in GEFAALD:
        print(f"  ❌ {naam}")
    return 1 if GEFAALD else 0


if __name__ == "__main__":
    sys.exit(main())
