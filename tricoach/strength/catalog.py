"""De bewerkbare oefeningen- en sjabloonbibliotheek voor krachttraining.

Drie tabellen: ``strength_exercise`` (de oefeningen), ``strength_template``
(A, B, ...) en ``strength_template_item`` (welke oefening in welke volgorde
en met welke doelsets/reps per fase bij welk sjabloon hoort).

**Nooit een blanket overschrijving.** Anders dan
:mod:`tricoach.nutrition.products` (waar de rij-sleutel de productnaam is en
niets ernaar refereert) hebben deze drie tabellen een ``INTEGER PRIMARY
KEY`` die door loggegevens wordt aangehaald (``strength_set.exercise_id``,
``strength_template_item.exercise_id``). Een ``DELETE`` + volledige
herinsert vanuit een editor zou die id's hernummeren en de historie
verweesd achterlaten. Daarom hier uitsluitend: aanmaken via een kale
``INSERT`` (geen ``id`` meegeven), wijzigen via ``UPDATE ... WHERE id = ?``,
deactiveren via ``is_active``. Geen ``DELETE``, geen ``reset_*()``.
"""

import sqlite3
from datetime import datetime

import pandas as pd

LOAD_TYPES = ("weight", "bodyweight", "time", "distance")

CATALOG_SCHEMA = """
CREATE TABLE IF NOT EXISTS strength_exercise (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    load_type TEXT NOT NULL,
    per_side INTEGER DEFAULT 0,
    search_term TEXT,
    cue TEXT,
    is_active INTEGER DEFAULT 1,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS strength_template (
    id INTEGER PRIMARY KEY,
    code TEXT NOT NULL,
    name TEXT NOT NULL,
    sort_order INTEGER NOT NULL,
    is_active INTEGER DEFAULT 1
);
CREATE TABLE IF NOT EXISTS strength_template_item (
    id INTEGER PRIMARY KEY,
    template_id INTEGER NOT NULL REFERENCES strength_template(id),
    exercise_id INTEGER NOT NULL REFERENCES strength_exercise(id),
    sort_order INTEGER NOT NULL,
    target_sets_p1 INTEGER, target_reps_p1 TEXT,
    target_sets_p2 INTEGER, target_reps_p2 TEXT,
    target_sets_p3 INTEGER, target_reps_p3 TEXT,
    is_active INTEGER DEFAULT 1
);
CREATE INDEX IF NOT EXISTS idx_strength_item_template
    ON strength_template_item(template_id);
"""

# Seed-data: de startopzet uit het trainingsprogramma. Na de eerste keer
# volledig bewerkbaar via de beheertab; deze lijsten worden alleen gebruikt
# als de tabellen nog leeg zijn (zie ensure_tables()).
SEED_EXERCISES: list[dict] = [
    {"name": "Deadlift", "load_type": "weight", "per_side": 0,
     "search_term": "conventional deadlift technique",
     "cue": "Rug recht, stang dicht bij de schenen, heupen omhoog en naar voren."},
    {"name": "Bulgarian split squat", "load_type": "weight", "per_side": 1,
     "search_term": "bulgarian split squat",
     "cue": "Romp rechtop, knie zakt recht naar beneden, niet naar voren over de teen."},
    {"name": "Zercher squat", "load_type": "weight", "per_side": 0,
     "search_term": "zercher squat",
     "cue": "Ellebogen hoog houden zodat de stang niet uit de elleboogplooi rolt."},
    {"name": "Kuitheffen gestrekt been", "load_type": "weight", "per_side": 1,
     "search_term": "single leg calf raise",
     "cue": "Volledige rek onderaan, korte pauze bovenaan — niet doorveren."},
    {"name": "Kuitheffen gebogen knie (soleus)", "load_type": "weight", "per_side": 1,
     "search_term": "bent knee calf raise soleus",
     "cue": "Knie licht gebogen houden — daar zit precies het verschil met de gestrekte variant."},
    {"name": "Pull-ups", "load_type": "bodyweight", "per_side": 0,
     "search_term": "pull up form",
     "cue": "Volledig hangen onderaan, kin over de stang, geen zwaai."},
    {"name": "Pogo hops", "load_type": "bodyweight", "per_side": 0,
     "search_term": "pogo hops plyometric drill",
     "cue": "Stijve enkels, kort en snel contact met de grond, niet diep door de knieën."},
    {"name": "Box jump", "load_type": "bodyweight", "per_side": 0,
     "search_term": "box jump technique",
     "cue": "Landen zacht met beide voeten tegelijk, rustig van de box af stappen."},
    {"name": "Barbell hip thrust", "load_type": "weight", "per_side": 0,
     "search_term": "barbell hip thrust",
     "cue": "Bovenaan de billen hard aanspannen, kin naar de borst, geen overstrekking in de rug."},
    {"name": "Romanian deadlift", "load_type": "weight", "per_side": 0,
     "search_term": "romanian deadlift technique",
     "cue": "Knieën bijna gestrekt, de stang scheert langs de benen, rug recht."},
    {"name": "Suitcase carry", "load_type": "distance", "per_side": 1,
     "search_term": "suitcase carry",
     "cue": "Romp recht houden, niet naar de kant van de gewichten laten hangen."},
    {"name": "Dead bug", "load_type": "bodyweight", "per_side": 1,
     "search_term": "dead bug exercise",
     "cue": "Onderrug de hele tijd tegen de vloer gedrukt houden."},
    {"name": "Side plank", "load_type": "time", "per_side": 1,
     "search_term": "side plank",
     "cue": "Heupen omhoog in één rechte lijn, niet laten doorzakken."},
]

# (code, naam, sort_order)
SEED_TEMPLATES: list[dict] = [
    {"code": "A", "name": "Kracht", "sort_order": 1},
    {"code": "B", "name": "Explosief en romp", "sort_order": 2},
]

# Per sjabloon-code: lijst van (oefeningnaam, sort_order, p1, p2, p3) waarbij
# pN = (sets, reps-notatie).
SEED_ITEMS: dict[str, list[tuple]] = {
    "A": [
        ("Deadlift", 1, (3, "8"), (4, "5"), (4, "3")),
        ("Bulgarian split squat", 2, (3, "10"), (4, "8"), (4, "6")),
        ("Zercher squat", 3, (3, "10"), (3, "8"), (3, "6")),
        ("Kuitheffen gestrekt been", 4, (3, "12"), (3, "10"), (3, "8")),
        ("Kuitheffen gebogen knie (soleus)", 5, (3, "15"), (3, "12"), (3, "10")),
        ("Pull-ups", 6, (3, "max"), (3, "max"), (3, "max")),
    ],
    "B": [
        ("Pogo hops", 1, (3, "15"), (3, "20"), (4, "20")),
        ("Box jump", 2, (2, "5"), (3, "5"), (4, "5")),
        ("Barbell hip thrust", 3, (3, "12"), (3, "8"), (3, "6")),
        ("Romanian deadlift", 4, (3, "10"), (3, "8"), (3, "6")),
        ("Suitcase carry", 5, (3, "40m"), (3, "40m"), (3, "40m")),
        ("Dead bug", 6, (3, "8"), (3, "10"), (3, "10")),
        ("Side plank", 7, (3, "30s"), (3, "40s"), (3, "45s")),
    ],
}


def ensure_tables(conn: sqlite3.Connection) -> None:
    """Maak de catalogustabellen aan en vul ze bij eerste gebruik (idempotent)."""
    conn.executescript(CATALOG_SCHEMA)
    leeg = conn.execute("SELECT COUNT(*) FROM strength_exercise").fetchone()[0] == 0
    if leeg:
        _seed(conn)
    conn.commit()


def _seed(conn: sqlite3.Connection) -> None:
    nu = datetime.now().isoformat(timespec="seconds")
    naam_naar_id: dict[str, int] = {}
    for ex in SEED_EXERCISES:
        cur = conn.execute(
            "INSERT INTO strength_exercise (name, load_type, per_side, "
            "search_term, cue, is_active, created_at) VALUES (?,?,?,?,?,1,?)",
            (ex["name"], ex["load_type"], ex["per_side"], ex["search_term"],
             ex["cue"], nu))
        naam_naar_id[ex["name"]] = cur.lastrowid

    code_naar_id: dict[str, int] = {}
    for tpl in SEED_TEMPLATES:
        cur = conn.execute(
            "INSERT INTO strength_template (code, name, sort_order, is_active) "
            "VALUES (?,?,?,1)", (tpl["code"], tpl["name"], tpl["sort_order"]))
        code_naar_id[tpl["code"]] = cur.lastrowid

    for code, items in SEED_ITEMS.items():
        template_id = code_naar_id[code]
        for naam, sort_order, p1, p2, p3 in items:
            conn.execute(
                "INSERT INTO strength_template_item (template_id, exercise_id, "
                "sort_order, target_sets_p1, target_reps_p1, target_sets_p2, "
                "target_reps_p2, target_sets_p3, target_reps_p3, is_active) "
                "VALUES (?,?,?,?,?,?,?,?,?,1)",
                (template_id, naam_naar_id[naam], sort_order,
                 p1[0], p1[1], p2[0], p2[1], p3[0], p3[1]))


# --------------------------------------------------------------- oefeningen --


def load_exercises(conn: sqlite3.Connection, only_active: bool = False) -> pd.DataFrame:
    """Alle oefeningen, op naam gesorteerd."""
    ensure_tables(conn)
    vraag = "SELECT * FROM strength_exercise"
    if only_active:
        vraag += " WHERE is_active = 1"
    vraag += " ORDER BY name"
    return pd.read_sql_query(vraag, conn)


def get_exercise(conn: sqlite3.Connection, exercise_id: int) -> dict | None:
    ensure_tables(conn)
    cur = conn.execute(
        "SELECT * FROM strength_exercise WHERE id = ?", (int(exercise_id),))
    row = cur.fetchone()
    if row is None:
        return None
    kolommen = [d[0] for d in cur.description]
    return dict(zip(kolommen, row))


def add_exercise(conn: sqlite3.Connection, name: str, load_type: str,
                 per_side: int = 0, search_term: str = "", cue: str = "") -> int:
    """Voeg één nieuwe oefening toe. Geeft het nieuwe ``id`` terug."""
    ensure_tables(conn)
    if load_type not in LOAD_TYPES:
        raise ValueError(f"onbekend load_type: {load_type!r}")
    nu = datetime.now().isoformat(timespec="seconds")
    cur = conn.execute(
        "INSERT INTO strength_exercise (name, load_type, per_side, "
        "search_term, cue, is_active, created_at) VALUES (?,?,?,?,?,1,?)",
        (name.strip(), load_type, int(bool(per_side)), search_term.strip(),
         cue.strip(), nu))
    conn.commit()
    return int(cur.lastrowid)


def update_exercise(conn: sqlite3.Connection, exercise_id: int, **fields) -> bool:
    """Wijzig muteerbare velden van een bestaande oefening. Het ``id`` zelf
    staat nooit in ``fields`` en wordt genegeerd als het er per ongeluk in zit."""
    ensure_tables(conn)
    toegestaan = {"name", "load_type", "per_side", "search_term", "cue"}
    zetten = {k: v for k, v in fields.items() if k in toegestaan}
    if not zetten:
        return False
    kolommen = ", ".join(f"{k} = ?" for k in zetten)
    cur = conn.execute(
        f"UPDATE strength_exercise SET {kolommen} WHERE id = ?",
        (*zetten.values(), int(exercise_id)))
    conn.commit()
    return cur.rowcount > 0


def set_exercise_active(conn: sqlite3.Connection, exercise_id: int, active: bool) -> bool:
    ensure_tables(conn)
    cur = conn.execute(
        "UPDATE strength_exercise SET is_active = ? WHERE id = ?",
        (int(bool(active)), int(exercise_id)))
    conn.commit()
    return cur.rowcount > 0


# ----------------------------------------------------------------- sjablonen --


def load_templates(conn: sqlite3.Connection, only_active: bool = False) -> pd.DataFrame:
    ensure_tables(conn)
    vraag = "SELECT * FROM strength_template"
    if only_active:
        vraag += " WHERE is_active = 1"
    vraag += " ORDER BY sort_order"
    return pd.read_sql_query(vraag, conn)


def add_template(conn: sqlite3.Connection, code: str, name: str,
                 sort_order: int | None = None) -> int:
    ensure_tables(conn)
    if sort_order is None:
        rij = conn.execute(
            "SELECT COALESCE(MAX(sort_order), 0) + 1 FROM strength_template").fetchone()
        sort_order = rij[0]
    cur = conn.execute(
        "INSERT INTO strength_template (code, name, sort_order, is_active) "
        "VALUES (?,?,?,1)", (code.strip(), name.strip(), sort_order))
    conn.commit()
    return int(cur.lastrowid)


def update_template(conn: sqlite3.Connection, template_id: int, **fields) -> bool:
    ensure_tables(conn)
    toegestaan = {"code", "name", "sort_order"}
    zetten = {k: v for k, v in fields.items() if k in toegestaan}
    if not zetten:
        return False
    kolommen = ", ".join(f"{k} = ?" for k in zetten)
    cur = conn.execute(
        f"UPDATE strength_template SET {kolommen} WHERE id = ?",
        (*zetten.values(), int(template_id)))
    conn.commit()
    return cur.rowcount > 0


def set_template_active(conn: sqlite3.Connection, template_id: int, active: bool) -> bool:
    ensure_tables(conn)
    cur = conn.execute(
        "UPDATE strength_template SET is_active = ? WHERE id = ?",
        (int(bool(active)), int(template_id)))
    conn.commit()
    return cur.rowcount > 0


# ------------------------------------------------------------ sjabloon-items --


def load_template_items(conn: sqlite3.Connection, template_id: int,
                        only_active: bool = True) -> pd.DataFrame:
    """De items van één sjabloon, met de live oefeningnaam erbij.

    Filtert op ``is_active`` van zowel het item als de oefening als
    ``only_active`` — een gedeactiveerde oefening verdwijnt zo automatisch
    uit het invoerscherm zonder dat het sjabloon hoeft te worden aangepast.
    """
    ensure_tables(conn)
    vraag = (
        "SELECT i.*, e.name AS exercise_name, e.load_type, e.per_side, "
        "e.search_term, e.cue, e.is_active AS exercise_is_active "
        "FROM strength_template_item i "
        "JOIN strength_exercise e ON e.id = i.exercise_id "
        "WHERE i.template_id = ?"
    )
    if only_active:
        vraag += " AND i.is_active = 1 AND e.is_active = 1"
    vraag += " ORDER BY i.sort_order"
    return pd.read_sql_query(vraag, conn, params=(int(template_id),))


def add_template_item(conn: sqlite3.Connection, template_id: int, exercise_id: int,
                      targets: dict, sort_order: int | None = None) -> int:
    """``targets`` bevat desgewenst ``target_sets_p1``..``target_reps_p3``."""
    ensure_tables(conn)
    template_id, exercise_id = int(template_id), int(exercise_id)
    if sort_order is None:
        rij = conn.execute(
            "SELECT COALESCE(MAX(sort_order), 0) + 1 FROM strength_template_item "
            "WHERE template_id = ?", (template_id,)).fetchone()
        sort_order = rij[0]
    velden = ["target_sets_p1", "target_reps_p1", "target_sets_p2", "target_reps_p2",
              "target_sets_p3", "target_reps_p3"]
    waarden = [targets.get(v) for v in velden]
    cur = conn.execute(
        "INSERT INTO strength_template_item (template_id, exercise_id, sort_order, "
        + ", ".join(velden) + ", is_active) VALUES (?,?,?,?,?,?,?,?,?,1)",
        (template_id, exercise_id, sort_order, *waarden))
    conn.commit()
    return int(cur.lastrowid)


def update_template_item(conn: sqlite3.Connection, item_id: int, **fields) -> bool:
    ensure_tables(conn)
    toegestaan = {"sort_order", "target_sets_p1", "target_reps_p1",
                 "target_sets_p2", "target_reps_p2", "target_sets_p3", "target_reps_p3"}
    zetten = {k: v for k, v in fields.items() if k in toegestaan}
    if not zetten:
        return False
    kolommen = ", ".join(f"{k} = ?" for k in zetten)
    cur = conn.execute(
        f"UPDATE strength_template_item SET {kolommen} WHERE id = ?",
        (*zetten.values(), int(item_id)))
    conn.commit()
    return cur.rowcount > 0


def set_template_item_active(conn: sqlite3.Connection, item_id: int, active: bool) -> bool:
    ensure_tables(conn)
    cur = conn.execute(
        "UPDATE strength_template_item SET is_active = ? WHERE id = ?",
        (int(bool(active)), int(item_id)))
    conn.commit()
    return cur.rowcount > 0


def reorder(conn: sqlite3.Connection, table: str, id_order: list[int]) -> None:
    """Schrijf alleen ``sort_order`` per id — nooit een delete/insert.

    ``table`` is ``"strength_template"``, ``"strength_template_item"`` of
    ``"strength_exercise"`` (die laatste heeft geen sort_order-kolom en is
    dus niet geldig; opgenomen als expliciete lijst tegen typefouten).
    """
    if table not in ("strength_template", "strength_template_item"):
        raise ValueError(f"reorder() ondersteunt geen tabel {table!r}")
    ensure_tables(conn)
    for i, item_id in enumerate(id_order):
        conn.execute(f"UPDATE {table} SET sort_order = ? WHERE id = ?", (i, int(item_id)))
    conn.commit()
