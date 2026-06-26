"""SQLite-opslag voor geparste activiteiten.

Drie tabellen:

- ``activities``: één rij per sessie met de samenvatting én de vooraf
  berekende tijd-in-zones (z1_s..z5_s), zodat het dashboard snel kan tekenen.
- ``records``: de seconde-data (hartslag, snelheid, ...) per sessie.
- ``lengths``: baandata bij zwemmen (slagtype, slagen, tijd per baan).

De primaire sleutel van ``activities`` is ``activity_key`` (de starttijd uit
het FIT-bestand); een tweede import van dezelfde activiteit wordt daardoor
genegeerd in plaats van gedupliceerd.
"""

import json
import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd

from tricoach.fit_parser import ParsedActivity
from tricoach.zones import ZONE_NAMES, pct_in_zone2, time_in_zones

SCHEMA = """
CREATE TABLE IF NOT EXISTS activities (
    activity_key  TEXT PRIMARY KEY,
    sport         TEXT NOT NULL,
    sub_sport     TEXT,
    start_time    TEXT NOT NULL,
    duration_s    REAL,
    distance_m    REAL,
    avg_hr        INTEGER,
    max_hr        INTEGER,
    avg_speed_ms  REAL,
    avg_cadence   REAL,
    total_ascent  REAL,
    calories      REAL,
    pool_length   REAL,
    num_lengths   INTEGER,
    total_strokes INTEGER,
    z1_s INTEGER, z2_s INTEGER, z3_s INTEGER, z4_s INTEGER, z5_s INTEGER,
    pct_in_zone2       REAL,
    aerobic_efficiency REAL,
    user_note      TEXT,
    wind_speed     REAL,
    wind_direction REAL,
    wind_gusts     REAL,
    summary_json  TEXT,
    source_file   TEXT,
    imported_at   TEXT
);
CREATE TABLE IF NOT EXISTS records (
    activity_key TEXT NOT NULL REFERENCES activities(activity_key),
    timestamp    TEXT NOT NULL,
    heart_rate   INTEGER,
    speed_ms     REAL,
    distance_m   REAL,
    cadence      REAL,
    altitude_m   REAL
);
CREATE INDEX IF NOT EXISTS idx_records_key ON records(activity_key);
CREATE TABLE IF NOT EXISTS lengths (
    activity_key     TEXT NOT NULL REFERENCES activities(activity_key),
    timestamp        TEXT,
    total_timer_time REAL,
    total_strokes    INTEGER,
    swim_stroke      TEXT
);
CREATE INDEX IF NOT EXISTS idx_lengths_key ON lengths(activity_key);
"""


def connect(db_path: Path | str) -> sqlite3.Connection:
    """Open (en maak zo nodig) de database, inclusief schema."""
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.executescript(SCHEMA)
    _migrate(conn)
    return conn


# Kolommen die later zijn toegevoegd; bestaande databases krijgen ze via een
# ALTER TABLE (CREATE TABLE IF NOT EXISTS voegt geen kolommen toe aan een
# bestaande tabel). De waarden blijven NULL tot een import of recompute ze vult.
_ADDED_COLUMNS = {
    "pct_in_zone2": "REAL",
    "aerobic_efficiency": "REAL",
    # Vrije opmerking bij de upload en de automatisch opgehaalde winddata
    # (Open-Meteo). Blijven NULL bij oude sessies, bij zwemmen en bij sessies
    # zonder GPS of internet.
    "user_note": "TEXT",
    "wind_speed": "REAL",
    "wind_direction": "REAL",
    "wind_gusts": "REAL",
}


def _migrate(conn: sqlite3.Connection) -> None:
    """Voeg ontbrekende kolommen toe aan een al bestaande activities-tabel."""
    have = {row[1] for row in conn.execute("PRAGMA table_info(activities)")}
    for name, decl in _ADDED_COLUMNS.items():
        if name not in have:
            conn.execute(f"ALTER TABLE activities ADD COLUMN {name} {decl}")
    conn.commit()


def aerobic_efficiency(sport: str, avg_speed_ms: float | None,
                       avg_hr: float | None) -> float | None:
    """Aerobe efficiëntie = gemiddelde snelheid (m/s) per hartslag.

    Hoger = meer snelheid bij dezelfde hartslag. Alleen zinvol voor lopen en
    fietsen; zwemmen valt af omdat de pols-HR onder water onbetrouwbaar is.
    Geeft None als de snelheid of hartslag ontbreekt.
    """
    if sport not in ("running", "cycling"):
        return None
    if not avg_speed_ms or not avg_hr:
        return None
    return avg_speed_ms / avg_hr


def activity_exists(conn: sqlite3.Connection, activity_key: str) -> bool:
    """Is deze activiteit al eens geïmporteerd?"""
    row = conn.execute(
        "SELECT 1 FROM activities WHERE activity_key = ?", (activity_key,)
    ).fetchone()
    return row is not None


def save_activity(
    conn: sqlite3.Connection,
    act: ParsedActivity,
    zone_bounds: list[int],
    user_note: str | None = None,
    wind: "WindData | None" = None,
) -> bool:
    """Sla één activiteit op. Geeft False terug als hij al bestond (dedup).

    ``user_note`` is de vrije opmerking die de gebruiker bij de upload typte
    (optioneel). ``wind`` is de automatisch opgehaalde Open-Meteo-winddata
    (optioneel; ``None`` bij zwemmen, geen GPS of geen internet).
    """
    if activity_exists(conn, act.activity_key):
        return False

    s = act.summary
    tiz = (
        time_in_zones(act.records, zone_bounds)
        if not act.records.empty and "heart_rate" in act.records
        else dict.fromkeys(ZONE_NAMES, 0)
    )
    avg_speed = s.get("enhanced_avg_speed") or s.get("avg_speed")
    avg_hr = s.get("avg_heart_rate")
    # Vooraf berekend en gecachet, zodat de tabel niet bij elke render alle
    # FIT-records opnieuw hoeft te parsen.
    pct_z2 = pct_in_zone2(tiz["Z1"], tiz["Z2"], tiz["Z3"], tiz["Z4"], tiz["Z5"])
    eff = aerobic_efficiency(act.sport, avg_speed, avg_hr)

    conn.execute(
        """INSERT INTO activities (
               activity_key, sport, sub_sport, start_time, duration_s, distance_m,
               avg_hr, max_hr, avg_speed_ms, avg_cadence, total_ascent, calories,
               pool_length, num_lengths, total_strokes,
               z1_s, z2_s, z3_s, z4_s, z5_s,
               pct_in_zone2, aerobic_efficiency,
               user_note, wind_speed, wind_direction, wind_gusts,
               summary_json, source_file, imported_at
           ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            act.activity_key, act.sport, act.sub_sport, act.start_time.isoformat(),
            s.get("total_timer_time"), s.get("total_distance"),
            avg_hr, s.get("max_heart_rate"),
            avg_speed,
            s.get("avg_running_cadence") or s.get("avg_cadence"),
            s.get("total_ascent"), s.get("total_calories"),
            s.get("pool_length"),
            s.get("num_active_lengths") or s.get("num_lengths"),
            s.get("total_strokes"),
            tiz["Z1"], tiz["Z2"], tiz["Z3"], tiz["Z4"], tiz["Z5"],
            pct_z2, eff,
            ((user_note or "").strip() or None),
            wind.speed_kmh if wind else None,
            wind.direction_deg if wind else None,
            wind.gusts_kmh if wind else None,
            json.dumps(s, default=str), act.source_file,
            datetime.now().isoformat(timespec="seconds"),
        ),
    )

    if not act.records.empty:
        df = act.records.copy()
        df["activity_key"] = act.activity_key
        df["timestamp"] = df["timestamp"].astype(str)
        # Zorg dat alle schemakolommen bestaan (zwemrecords missen bijv. cadans).
        for col in ["heart_rate", "speed_ms", "distance_m", "cadence", "altitude_m"]:
            if col not in df:
                df[col] = None
        df[["activity_key", "timestamp", "heart_rate", "speed_ms",
            "distance_m", "cadence", "altitude_m"]].to_sql(
            "records", conn, if_exists="append", index=False)

    if not act.lengths.empty:
        df = act.lengths.copy()
        df["activity_key"] = act.activity_key
        df["timestamp"] = df["timestamp"].astype(str)
        df["swim_stroke"] = df["swim_stroke"].astype(str)
        df[["activity_key", "timestamp", "total_timer_time",
            "total_strokes", "swim_stroke"]].to_sql(
            "lengths", conn, if_exists="append", index=False)

    conn.commit()
    return True


def load_activities(conn: sqlite3.Connection) -> pd.DataFrame:
    """Alle sessies als DataFrame, nieuwste eerst."""
    df = pd.read_sql_query(
        "SELECT * FROM activities ORDER BY start_time DESC", conn)
    if not df.empty:
        df["start_time"] = pd.to_datetime(df["start_time"])
    return df


def load_records(conn: sqlite3.Connection, activity_key: str) -> pd.DataFrame:
    """Seconde-data van één sessie."""
    df = pd.read_sql_query(
        "SELECT * FROM records WHERE activity_key = ? ORDER BY timestamp",
        conn, params=(activity_key,))
    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def recompute_zones(conn: sqlite3.Connection, zone_bounds: list[int]) -> int:
    """Herreken de tijd-in-zones én de afgeleide maten van álle activiteiten.

    Nodig wanneer de LTHR (en dus de zones) wijzigt: de z1_s..z5_s-kolommen
    en het daaruit afgeleide ``pct_in_zone2`` zijn bij import berekend met de
    toenmalige grenzen. ``aerobic_efficiency`` hangt niet van de zones af, maar
    wordt hier meegenomen zodat één herberekening alle gecachete maten dekt
    (bijv. ook handig om een bestaande database eenmalig te vullen). De
    seconde-data staat in ``records``, dus herrekenen kan altijd. Geeft het
    aantal bijgewerkte activiteiten terug.
    """
    rows = conn.execute(
        "SELECT activity_key, sport, avg_speed_ms, avg_hr FROM activities"
    ).fetchall()
    for key, sport, avg_speed_ms, avg_hr in rows:
        records = load_records(conn, key)
        tiz = (
            time_in_zones(records, zone_bounds)
            if not records.empty and "heart_rate" in records
            else dict.fromkeys(ZONE_NAMES, 0)
        )
        pct_z2 = pct_in_zone2(tiz["Z1"], tiz["Z2"], tiz["Z3"], tiz["Z4"], tiz["Z5"])
        eff = aerobic_efficiency(sport, avg_speed_ms, avg_hr)
        conn.execute(
            "UPDATE activities SET z1_s=?, z2_s=?, z3_s=?, z4_s=?, z5_s=?, "
            "pct_in_zone2=?, aerobic_efficiency=? WHERE activity_key=?",
            (tiz["Z1"], tiz["Z2"], tiz["Z3"], tiz["Z4"], tiz["Z5"],
             pct_z2, eff, key),
        )
    conn.commit()
    return len(rows)


def load_lengths(conn: sqlite3.Connection, activity_key: str) -> pd.DataFrame:
    """Baandata van één zwemsessie."""
    return pd.read_sql_query(
        "SELECT * FROM lengths WHERE activity_key = ? ORDER BY timestamp",
        conn, params=(activity_key,))


def swim_active_seconds(conn: sqlite3.Connection) -> dict[str, float]:
    """Actieve zwemtijd (s) per activiteit: de som van de actieve banen.

    De ``lengths``-tabel bevat alleen actieve banen — rustpauzes aan de kant
    zijn bij het parsen al weggefilterd — dus de som van ``total_timer_time``
    is de zuivere zwemtijd zonder rust. Dat is een betrouwbaardere noemer voor
    het tempo per 100 m dan de totale timer-duur (``duration_s``), die de rust
    aan de kant meetelt en het tempo dus te traag laat lijken.

    Geeft een dict ``{activity_key: seconden}``; sessies zonder baandata
    (bijv. open water of een loop/fietssessie) ontbreken simpelweg, zodat de
    aanroeper kan terugvallen op de totale duur.
    """
    rows = conn.execute(
        "SELECT activity_key, SUM(total_timer_time) FROM lengths "
        "GROUP BY activity_key"
    ).fetchall()
    return {key: total for key, total in rows if total}
