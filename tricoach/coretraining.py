"""Dagelijkse mobiliteit/core: oefeningen loggen en de streak bijhouden.

Drie tabellen: ``core_exercise`` (de catalogus), ``core_log`` (één vinkje per
oefening per dag) en ``core_streak_freeze`` (een **afgeleide cache**, geen
gebruikersinvoer — zie :func:`_reconcile_freezes`). De streak-/freezerekenlogica
zelf staat los in :mod:`tricoach.streak` (puur, geen ``sqlite3``); deze module
is de dunne laag die dat voedt met data uit ``core_log``.

**Nooit hard verwijderd.** Zelfde regels als :mod:`tricoach.strength.catalog`:
aanmaken via een kale ``INSERT``, wijzigen via ``UPDATE ... WHERE id = ?``,
deactiveren via ``is_active``. Geen ``DELETE``, geen ``reset_*()``.

**Tijdzone.** ``log_date`` en alle datumvelden gebruiken ``date.today()`` —
de naïeve lokale serverklok (de container draait op ``TZ=Europe/Amsterdam``,
zie ``docker-compose.yml``) — niet :func:`tricoach.formatting.local_time`.
Die laatste converteert UTC-oorsprong-tijdstempels (FIT-starttijden); een
vinkje dat je nu zet heeft geen UTC-oorsprong. Dit is dezelfde conventie als
``transport.py``/``removal.py``/``body.py`` voor administratieve datums.
"""

import sqlite3
from datetime import date, datetime, timedelta

import pandas as pd

from tricoach import streak

CATEGORIES = ("mobiliteit", "romp")
MAX_BACKFILL_DAYS = 14

CORE_SCHEMA = """
CREATE TABLE IF NOT EXISTS core_exercise (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    category TEXT NOT NULL,
    dosage TEXT,
    search_term TEXT,
    sort_order INTEGER NOT NULL,
    is_active INTEGER DEFAULT 1,
    is_quick_day INTEGER DEFAULT 0,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS core_log (
    id INTEGER PRIMARY KEY,
    exercise_id INTEGER NOT NULL REFERENCES core_exercise(id),
    exercise_name_snapshot TEXT NOT NULL,
    log_date TEXT NOT NULL,
    logged_at TEXT NOT NULL,
    UNIQUE(exercise_id, log_date)
);
CREATE TABLE IF NOT EXISTS core_streak_freeze (
    id INTEGER PRIMARY KEY,
    used_on TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_core_log_date ON core_log(log_date);
CREATE UNIQUE INDEX IF NOT EXISTS idx_core_freeze_day ON core_streak_freeze(used_on);
"""

# (naam, categorie, dosering, zoekterm, is_quick_day)
SEED_EXERCISES: list[tuple] = [
    ("Heupbuiger-stretch", "mobiliteit", "60 sec per kant", "couch stretch", 1),
    ("90/90 heupdraai", "mobiliteit", "8x per kant", "90/90 hip switch", 0),
    ("Open book", "mobiliteit", "8x per kant", "open book thoracic rotation", 0),
    ("Kat-koe", "mobiliteit", "10x", "cat cow", 0),
    ("Enkelmobiliteit", "mobiliteit", "10x per kant", "knee to wall ankle mobility", 0),
    ("Dead bug", "romp", "2x10 per kant", "dead bug exercise", 1),
    ("Side plank", "romp", "2x30 sec per kant", "side plank", 1),
    ("Bird dog", "romp", "2x8 per kant", "bird dog exercise", 0),
    ("Plank", "romp", "2x45 sec", "plank exercise", 0),
]


def ensure_tables(conn: sqlite3.Connection) -> None:
    """Maak de coretabellen aan en vul ze bij eerste gebruik (idempotent)."""
    conn.executescript(CORE_SCHEMA)
    leeg = conn.execute("SELECT COUNT(*) FROM core_exercise").fetchone()[0] == 0
    if leeg:
        _seed(conn)
    conn.commit()


def _seed(conn: sqlite3.Connection) -> None:
    nu = datetime.now().isoformat(timespec="seconds")
    for i, (naam, categorie, dosage, search_term, quick) in enumerate(SEED_EXERCISES, start=1):
        conn.execute(
            "INSERT INTO core_exercise (name, category, dosage, search_term, "
            "sort_order, is_active, is_quick_day, created_at) VALUES (?,?,?,?,?,1,?,?)",
            (naam, categorie, dosage, search_term, i, quick, nu))


# --------------------------------------------------------------- oefeningen --


def load_core_exercises(conn: sqlite3.Connection, only_active: bool = False) -> pd.DataFrame:
    ensure_tables(conn)
    vraag = "SELECT * FROM core_exercise"
    if only_active:
        vraag += " WHERE is_active = 1"
    vraag += " ORDER BY sort_order"
    return pd.read_sql_query(vraag, conn)


def get_core_exercise(conn: sqlite3.Connection, exercise_id: int) -> dict | None:
    ensure_tables(conn)
    cur = conn.execute("SELECT * FROM core_exercise WHERE id = ?", (int(exercise_id),))
    row = cur.fetchone()
    if row is None:
        return None
    kolommen = [d[0] for d in cur.description]
    return dict(zip(kolommen, row))


def add_core_exercise(conn: sqlite3.Connection, name: str, category: str,
                      dosage: str = "", search_term: str = "",
                      sort_order: int | None = None, is_quick_day: bool = False) -> int:
    ensure_tables(conn)
    if category not in CATEGORIES:
        raise ValueError(f"onbekende categorie: {category!r}")
    if sort_order is None:
        rij = conn.execute(
            "SELECT COALESCE(MAX(sort_order), 0) + 1 FROM core_exercise").fetchone()
        sort_order = rij[0]
    nu = datetime.now().isoformat(timespec="seconds")
    cur = conn.execute(
        "INSERT INTO core_exercise (name, category, dosage, search_term, "
        "sort_order, is_active, is_quick_day, created_at) VALUES (?,?,?,?,?,1,?,?)",
        (name.strip(), category, dosage.strip(), search_term.strip(), sort_order,
         int(bool(is_quick_day)), nu))
    conn.commit()
    return int(cur.lastrowid)


def update_core_exercise(conn: sqlite3.Connection, exercise_id: int, **fields) -> bool:
    """Wijzig muteerbare velden, incl. ``is_quick_day``. Het ``id`` zelf
    staat nooit in ``fields``."""
    ensure_tables(conn)
    toegestaan = {"name", "category", "dosage", "search_term", "is_quick_day"}
    zetten = {k: v for k, v in fields.items() if k in toegestaan}
    if not zetten:
        return False
    if "is_quick_day" in zetten:
        zetten["is_quick_day"] = int(bool(zetten["is_quick_day"]))
    kolommen = ", ".join(f"{k} = ?" for k in zetten)
    cur = conn.execute(
        f"UPDATE core_exercise SET {kolommen} WHERE id = ?",
        (*zetten.values(), int(exercise_id)))
    conn.commit()
    return cur.rowcount > 0


def set_core_exercise_active(conn: sqlite3.Connection, exercise_id: int, active: bool) -> bool:
    ensure_tables(conn)
    cur = conn.execute(
        "UPDATE core_exercise SET is_active = ? WHERE id = ?",
        (int(bool(active)), int(exercise_id)))
    conn.commit()
    return cur.rowcount > 0


def reorder_core(conn: sqlite3.Connection, id_order: list[int]) -> None:
    """Schrijf alleen ``sort_order`` per id — nooit een delete/insert."""
    ensure_tables(conn)
    for i, exercise_id in enumerate(id_order):
        conn.execute("UPDATE core_exercise SET sort_order = ? WHERE id = ?",
                     (i, int(exercise_id)))
    conn.commit()


# -------------------------------------------------------------------- loggen --


def log_exercise(conn: sqlite3.Connection, exercise_id: int, log_date: date) -> bool:
    """Vink één oefening af op ``log_date``.

    ``INSERT OR IGNORE``: geeft ``False`` terug (geen exception) als er al
    een rij voor dit paar bestond — de nette afhandeling van de unique
    constraint. Werkt daarna de freeze-cache bij (zie
    :func:`_reconcile_freezes`).
    """
    ensure_tables(conn)
    exercise_id = int(exercise_id)
    ex = get_core_exercise(conn, exercise_id)
    naam = ex["name"] if ex else "?"
    nu = datetime.now().isoformat(timespec="seconds")
    cur = conn.execute(
        "INSERT OR IGNORE INTO core_log (exercise_id, exercise_name_snapshot, "
        "log_date, logged_at) VALUES (?,?,?,?)",
        (exercise_id, naam, log_date.isoformat(), nu))
    conn.commit()
    if cur.rowcount > 0:
        _reconcile_freezes(conn)
    return cur.rowcount > 0


def unlog_exercise(conn: sqlite3.Connection, exercise_id: int, log_date: date) -> bool:
    """Zet een vinkje weer uit — het corrigeren van een mistik, geen
    historie wissen (de unique constraint maakt een tombstone hier
    onwerkbaar)."""
    ensure_tables(conn)
    cur = conn.execute(
        "DELETE FROM core_log WHERE exercise_id = ? AND log_date = ?",
        (int(exercise_id), log_date.isoformat()))
    conn.commit()
    if cur.rowcount > 0:
        _reconcile_freezes(conn)
    return cur.rowcount > 0


def log_korte_dag(conn: sqlite3.Connection, log_date: date) -> list[str]:
    """Log alle actieve ``is_quick_day``-oefeningen op ``log_date`` in één
    keer. Geeft de namen van de gelogde oefeningen terug (leeg als er geen
    enkele quick-day-oefening actief is)."""
    ensure_tables(conn)
    rijen = conn.execute(
        "SELECT id, name FROM core_exercise WHERE is_quick_day = 1 AND is_active = 1"
    ).fetchall()
    gelogd = []
    for exercise_id, naam in rijen:
        log_exercise(conn, exercise_id, log_date)
        gelogd.append(naam)
    return gelogd


def logs_for_period(conn: sqlite3.Connection, start: date, end: date) -> pd.DataFrame:
    """``core_log`` in ``[start, end]`` met de live naam (val terug op de
    snapshot) en ``is_active`` van de oefening — bevat dus ook
    gedeactiveerde-maar-in-de-periode-gelogde oefeningen."""
    ensure_tables(conn)
    vraag = (
        "SELECT l.*, COALESCE(e.name, l.exercise_name_snapshot) AS exercise_name, "
        "COALESCE(e.is_active, 0) AS exercise_is_active, e.category "
        "FROM core_log l LEFT JOIN core_exercise e ON e.id = l.exercise_id "
        "WHERE l.log_date >= ? AND l.log_date <= ? ORDER BY l.log_date"
    )
    return pd.read_sql_query(vraag, conn, params=(start.isoformat(), end.isoformat()))


# ------------------------------------------------------------------- streak --


def day_counts(conn: sqlite3.Connection) -> dict[date, int]:
    """Aantal gelogde oefeningen per dag, over de hele geschiedenis."""
    ensure_tables(conn)
    rijen = conn.execute(
        "SELECT log_date, COUNT(*) FROM core_log GROUP BY log_date").fetchall()
    return {date.fromisoformat(d): n for d, n in rijen}


def complete_days(conn: sqlite3.Connection, threshold: int = streak.DEFAULT_THRESHOLD) -> set[date]:
    counts = day_counts(conn)
    return {d for d, n in counts.items() if n >= threshold}


def _reconcile_freezes(conn: sqlite3.Connection, today: date | None = None) -> set[date]:
    """Herbereken welke dagen door een freeze gered worden en trek
    ``core_streak_freeze`` daaraan gelijk.

    Deze tabel is een 100% afgeleide cache van ``core_log`` — geen
    gebruikersinvoer, dus herbouwen is geen schending van "nooit hard
    verwijderen". Nodig omdat "terugwerkend invullen herberekent de streak
    correct" met eager/permanente freeze-rijen onhaalbaar is: een achteraf
    ingevulde dag zou anders een freeze blijven verbruiken die niet meer
    nodig is en de maandquota onterecht opslokken.
    """
    ensure_tables(conn)
    vandaag = today or date.today()
    bevroren = streak.compute_freezes(complete_days(conn), vandaag)
    huidig = {date.fromisoformat(r[0]) for r in
             conn.execute("SELECT used_on FROM core_streak_freeze").fetchall()}
    te_wissen = huidig - bevroren
    te_toevoegen = bevroren - huidig
    for d in te_wissen:
        conn.execute("DELETE FROM core_streak_freeze WHERE used_on = ?", (d.isoformat(),))
    for d in te_toevoegen:
        conn.execute(
            "INSERT OR IGNORE INTO core_streak_freeze (used_on) VALUES (?)",
            (d.isoformat(),))
    conn.commit()
    return bevroren


def frozen_days(conn: sqlite3.Connection, today: date | None = None) -> set[date]:
    """De dagen die momenteel door een freeze gered zijn (voor de kalenderheatmap)."""
    return _reconcile_freezes(conn, today)


def streak_stats(conn: sqlite3.Connection, today: date | None = None) -> dict:
    """Eén aanroep die de complete statistiektab voedt."""
    ensure_tables(conn)
    vandaag = today or date.today()
    compleet = complete_days(conn)
    bevroren = _reconcile_freezes(conn, vandaag)
    huidige_streak = streak.current_streak(compleet, bevroren, vandaag)
    maand = (vandaag.year, vandaag.month)
    freezes_deze_maand = sum(1 for d in bevroren if (d.year, d.month) == maand)
    counts = day_counts(conn)
    return {
        "current": huidige_streak,
        "longest": streak.longest_streak(compleet, bevroren),
        "consistency_30d": streak.consistency_pct(compleet, vandaag, 30),
        "freezes_this_month": freezes_deze_maand,
        "freezes_left": max(0, streak.MAX_FREEZES_PER_MONTH - freezes_deze_maand),
        "milestone": streak.milestone_reached(huidige_streak),
        "never_twice": streak.never_twice_message(compleet, vandaag),
        "today_complete": streak.is_dag_compleet(counts, vandaag),
        "today_full": streak.is_volledige_dag(counts, vandaag),
    }


# ------------------------------------------------------ verwaarloosde oefeningen --


def dagen_sinds_laatste(conn: sqlite3.Connection, today: date | None = None) -> dict[int, int | None]:
    """Per actieve oefening het aantal dagen sinds de laatste log; ``None``
    als hij nog nooit gedaan is."""
    ensure_tables(conn)
    vandaag = today or date.today()
    actief = load_core_exercises(conn, only_active=True)
    laatste = dict(conn.execute(
        "SELECT exercise_id, MAX(log_date) FROM core_log GROUP BY exercise_id").fetchall())
    uit: dict[int, int | None] = {}
    for _, ex in actief.iterrows():
        eid = int(ex["id"])
        gelogd = laatste.get(eid)
        uit[eid] = (vandaag - date.fromisoformat(gelogd)).days if gelogd else None
    return uit


def verwaarloosd(conn: sqlite3.Connection, drempel: int = 10,
                 today: date | None = None) -> list[dict]:
    """Actieve oefeningen die ``>= drempel`` dagen niet gedaan zijn, of nooit."""
    ensure_tables(conn)
    dagen = dagen_sinds_laatste(conn, today=today)
    actief = load_core_exercises(conn, only_active=True).set_index("id")
    uit = []
    for eid, n in dagen.items():
        if n is None or n >= drempel:
            uit.append({"exercise_id": eid, "name": actief.loc[eid, "name"],
                       "dagen": n})
    return uit
