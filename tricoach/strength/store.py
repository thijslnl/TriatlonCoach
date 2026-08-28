"""Loggen van krachtsessies en sets, en de leesqueries die erop bouwen.

Twee tabellen: ``strength_workout`` (één sessie) en ``strength_set`` (één set
binnen die sessie). ``strength_workout.deleted_at`` is een toevoeging op de
oorspronkelijke opdracht-schema — soft delete voor een per ongeluk gestarte
sessie (verkeerd getikt op "start"), zodat die niet in de rotatie of de
historie blijft hangen. Precies :func:`tricoach.storage.soft_delete_activity`
se patroon: de rij blijft bestaan en is herstelbaar; :func:`load_workouts`
is het ene filterpunt, zoals ``load_activities()`` dat is voor sessies.

**Write-through.** :func:`log_set` commit direct per set — er wordt nergens
een groeiende lijst in ``session_state`` opgebouwd die pas bij het afsluiten
naar de database gaat. Dat is hoe "een sessie kan tussentijds opgeslagen en
later afgemaakt worden" hier waargemaakt wordt: een browserrefresh of een
container-herstart halverwege een sessie kost je geen gelogde sets.
"""

import sqlite3
from datetime import datetime

import pandas as pd

from tricoach.strength import catalog
from tricoach.strength.rules import next_template_id, working_sets

WORKOUT_SCHEMA = """
CREATE TABLE IF NOT EXISTS strength_workout (
    id INTEGER PRIMARY KEY,
    template_id INTEGER REFERENCES strength_template(id),
    template_code_snapshot TEXT NOT NULL,
    phase INTEGER NOT NULL,
    started_at TEXT NOT NULL,
    completed_at TEXT,
    legs_feel INTEGER,
    session_rpe REAL,
    is_deload INTEGER DEFAULT 0,
    warmup_done INTEGER DEFAULT 0,
    notes TEXT,
    deleted_at TEXT
);
CREATE TABLE IF NOT EXISTS strength_set (
    id INTEGER PRIMARY KEY,
    workout_id INTEGER NOT NULL REFERENCES strength_workout(id),
    exercise_id INTEGER NOT NULL REFERENCES strength_exercise(id),
    exercise_name_snapshot TEXT NOT NULL,
    set_number INTEGER NOT NULL,
    reps INTEGER,
    weight_kg REAL,
    seconds INTEGER,
    meters REAL,
    side TEXT DEFAULT 'both',
    rpe REAL,
    is_warmup INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_strength_set_workout ON strength_set(workout_id);
CREATE INDEX IF NOT EXISTS idx_strength_set_exercise ON strength_set(exercise_id);
"""


def ensure_tables(conn: sqlite3.Connection) -> None:
    """Maak de workout-/settabellen aan (idempotent). Vereist ook de
    catalogustabellen (foreign-key-achtige verwijzingen, niet afgedwongen
    door SQLite maar wel inhoudelijk nodig)."""
    catalog.ensure_tables(conn)
    conn.executescript(WORKOUT_SCHEMA)
    conn.commit()


# ------------------------------------------------------------------ workouts --


def load_workouts(conn: sqlite3.Connection, include_deleted: bool = False) -> pd.DataFrame:
    """Alle workouts, nieuwste eerst — HET filterpunt voor de soft delete.

    ``COALESCE(t.code, w.template_code_snapshot)`` laat het sjablooncode
    altijd zien, ook als het sjabloon zelf later gedeactiveerd is (die join
    filtert niet op ``is_active``).
    """
    ensure_tables(conn)
    vraag = (
        "SELECT w.*, COALESCE(t.code, w.template_code_snapshot) AS template_code "
        "FROM strength_workout w LEFT JOIN strength_template t ON t.id = w.template_id"
    )
    if not include_deleted:
        vraag += " WHERE w.deleted_at IS NULL"
    vraag += " ORDER BY w.started_at DESC, w.id DESC"
    df = pd.read_sql_query(vraag, conn)
    if not df.empty:
        df["started_at"] = pd.to_datetime(df["started_at"])
    return df


def start_workout(conn: sqlite3.Connection, template_id: int | None, phase: int,
                  legs_feel: int | None = None, is_deload: bool = False) -> int:
    """Begin een nieuwe sessie. Geeft het nieuwe ``id`` terug."""
    ensure_tables(conn)
    template_id = None if template_id is None else int(template_id)
    code = "?"
    if template_id is not None:
        rij = conn.execute(
            "SELECT code FROM strength_template WHERE id = ?", (template_id,)).fetchone()
        if rij:
            code = rij[0]
    nu = datetime.now().isoformat(timespec="seconds")
    cur = conn.execute(
        "INSERT INTO strength_workout (template_id, template_code_snapshot, phase, "
        "started_at, legs_feel, is_deload) VALUES (?,?,?,?,?,?)",
        (template_id, code, int(phase), nu, legs_feel, int(bool(is_deload))))
    conn.commit()
    return int(cur.lastrowid)


def open_workout(conn: sqlite3.Connection) -> dict | None:
    """De meest recente onafgemaakte, niet-verwijderde sessie, of ``None``."""
    ensure_tables(conn)
    cur = conn.execute(
        "SELECT * FROM strength_workout WHERE completed_at IS NULL "
        "AND deleted_at IS NULL ORDER BY started_at DESC, id DESC LIMIT 1")
    row = cur.fetchone()
    if row is None:
        return None
    kolommen = [d[0] for d in cur.description]
    return dict(zip(kolommen, row))


def complete_workout(conn: sqlite3.Connection, workout_id: int,
                     session_rpe: float | None = None, notes: str = "") -> None:
    ensure_tables(conn)
    nu = datetime.now().isoformat(timespec="seconds")
    conn.execute(
        "UPDATE strength_workout SET completed_at = ?, session_rpe = ?, notes = ? "
        "WHERE id = ?", (nu, session_rpe, notes.strip(), int(workout_id)))
    conn.commit()


def update_workout(conn: sqlite3.Connection, workout_id: int, **fields) -> bool:
    ensure_tables(conn)
    toegestaan = {"legs_feel", "session_rpe", "is_deload", "warmup_done", "notes", "phase"}
    zetten = {k: v for k, v in fields.items() if k in toegestaan}
    if not zetten:
        return False
    kolommen = ", ".join(f"{k} = ?" for k in zetten)
    cur = conn.execute(
        f"UPDATE strength_workout SET {kolommen} WHERE id = ?",
        (*zetten.values(), int(workout_id)))
    conn.commit()
    return cur.rowcount > 0


def soft_delete_workout(conn: sqlite3.Connection, workout_id: int) -> bool:
    """Markeer een sessie als verwijderd; de rij en de sets blijven bestaan."""
    ensure_tables(conn)
    nu = datetime.now().isoformat(timespec="seconds")
    cur = conn.execute(
        "UPDATE strength_workout SET deleted_at = ? WHERE id = ? AND deleted_at IS NULL",
        (nu, int(workout_id)))
    conn.commit()
    return cur.rowcount > 0


def restore_workout(conn: sqlite3.Connection, workout_id: int) -> bool:
    ensure_tables(conn)
    cur = conn.execute(
        "UPDATE strength_workout SET deleted_at = NULL WHERE id = ? "
        "AND deleted_at IS NOT NULL", (int(workout_id),))
    conn.commit()
    return cur.rowcount > 0


# ---------------------------------------------------------------------- sets --


def log_set(conn: sqlite3.Connection, workout_id: int, exercise_id: int, set_number: int,
           reps: int | None = None, weight_kg: float | None = None,
           seconds: int | None = None, meters: float | None = None,
           side: str = "both", rpe: float | None = None, is_warmup: bool = False) -> int:
    """Log één set en commit direct (write-through, zie moduledocstring)."""
    ensure_tables(conn)
    workout_id, exercise_id = int(workout_id), int(exercise_id)
    ex = catalog.get_exercise(conn, exercise_id)
    naam = ex["name"] if ex else "?"
    cur = conn.execute(
        "INSERT INTO strength_set (workout_id, exercise_id, exercise_name_snapshot, "
        "set_number, reps, weight_kg, seconds, meters, side, rpe, is_warmup) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        (workout_id, exercise_id, naam, set_number, reps, weight_kg, seconds,
         meters, side, rpe, int(bool(is_warmup))))
    conn.commit()
    return int(cur.lastrowid)


def update_set(conn: sqlite3.Connection, set_id: int, **fields) -> bool:
    ensure_tables(conn)
    toegestaan = {"reps", "weight_kg", "seconds", "meters", "side", "rpe", "is_warmup"}
    zetten = {k: v for k, v in fields.items() if k in toegestaan}
    if not zetten:
        return False
    kolommen = ", ".join(f"{k} = ?" for k in zetten)
    cur = conn.execute(
        f"UPDATE strength_set SET {kolommen} WHERE id = ?",
        (*zetten.values(), int(set_id)))
    conn.commit()
    return cur.rowcount > 0


def delete_set(conn: sqlite3.Connection, set_id: int) -> bool:
    """Verwijder één set — de correctie van een verkeerde tik binnen een
    lopende sessie, geen historie wissen."""
    ensure_tables(conn)
    cur = conn.execute("DELETE FROM strength_set WHERE id = ?", (int(set_id),))
    conn.commit()
    return cur.rowcount > 0


def load_sets(conn: sqlite3.Connection, workout_id: int | None = None,
             exercise_id: int | None = None) -> pd.DataFrame:
    """Sets met de live oefeningnaam (val terug op de snapshot), van
    niet-verwijderde workouts."""
    ensure_tables(conn)
    vraag = (
        "SELECT s.*, COALESCE(e.name, s.exercise_name_snapshot) AS exercise_name "
        "FROM strength_set s "
        "JOIN strength_workout w ON w.id = s.workout_id AND w.deleted_at IS NULL "
        "LEFT JOIN strength_exercise e ON e.id = s.exercise_id"
    )
    waar, params = [], []
    if workout_id is not None:
        waar.append("s.workout_id = ?")
        params.append(int(workout_id))
    if exercise_id is not None:
        waar.append("s.exercise_id = ?")
        params.append(int(exercise_id))
    if waar:
        vraag += " WHERE " + " AND ".join(waar)
    vraag += " ORDER BY s.set_number"
    return pd.read_sql_query(vraag, conn, params=params)


# --------------------------------------------------------------------- rotatie --


def next_template(conn: sqlite3.Connection) -> dict | None:
    """Het sjabloon dat nu aan de beurt is (zie
    :func:`tricoach.strength.rules.next_template_id`)."""
    ensure_tables(conn)
    actief = catalog.load_templates(conn, only_active=True)
    if actief.empty:
        return None
    rij = conn.execute(
        "SELECT template_id FROM strength_workout "
        "WHERE completed_at IS NOT NULL AND deleted_at IS NULL "
        "ORDER BY started_at DESC, id DESC LIMIT 1").fetchone()
    laatste_id = rij[0] if rij else None
    volgende_id = next_template_id(actief.to_dict("records"), laatste_id)
    if volgende_id is None:
        return None
    return actief[actief["id"] == volgende_id].iloc[0].to_dict()


# ---------------------------------------------------------- gewichtsuggestie --


def last_values_for_exercise(conn: sqlite3.Connection, exercise_id: int,
                             exclude_workout_id: int | None = None) -> pd.DataFrame:
    """Werksets van de meest recente voltooide workout waarin deze oefening
    voorkomt (ongeacht sjabloon). Leeg = nooit eerder gedaan."""
    ensure_tables(conn)
    exercise_id = int(exercise_id)
    vraag = (
        "SELECT s.workout_id, w.started_at FROM strength_set s "
        "JOIN strength_workout w ON w.id = s.workout_id "
        "WHERE s.exercise_id = ? AND w.completed_at IS NOT NULL "
        "AND w.deleted_at IS NULL AND s.is_warmup = 0"
    )
    params: list = [exercise_id]
    if exclude_workout_id is not None:
        vraag += " AND s.workout_id != ?"
        params.append(int(exclude_workout_id))
    vraag += " ORDER BY w.started_at DESC, w.id DESC LIMIT 1"
    rij = conn.execute(vraag, params).fetchone()
    if rij is None:
        return pd.DataFrame()
    workout_id = rij[0]
    sets = load_sets(conn, workout_id=workout_id, exercise_id=exercise_id)
    return working_sets(sets).sort_values("set_number").reset_index(drop=True)


# --------------------------------------------------------------- weekvolume --


def weekly_strength_volume(conn: sqlite3.Connection) -> pd.DataFrame:
    """Wekelijks totaal krachtvolume (kg): voltooide, niet-verwijderde
    workouts x werksets met een gewicht. Kolommen: ``week``, ``volume_kg``."""
    ensure_tables(conn)
    from tricoach.strength.rules import iso_week_label

    vraag = (
        "SELECT w.started_at, s.reps, s.weight_kg FROM strength_set s "
        "JOIN strength_workout w ON w.id = s.workout_id "
        "WHERE w.completed_at IS NOT NULL AND w.deleted_at IS NULL "
        "AND s.is_warmup = 0 AND s.weight_kg IS NOT NULL"
    )
    df = pd.read_sql_query(vraag, conn)
    if df.empty:
        return pd.DataFrame(columns=["week", "volume_kg"])
    df["week"] = iso_week_label(df["started_at"])
    df["volume"] = df["reps"].fillna(0) * df["weight_kg"].fillna(0)
    out = df.groupby("week", as_index=False)["volume"].sum().rename(
        columns={"volume": "volume_kg"})
    return out.sort_values("week").reset_index(drop=True)
