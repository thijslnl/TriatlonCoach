"""Dagelijkse wellness-data uit Garmin Connect: opslag en trendhulpen.

Aparte SQLite-tabel ``wellness`` (los van de trainingen), één rij per dag met
rustpols, HRV, slaap, body battery, stress, VO2 max en training readiness.
De data komt binnen via :mod:`tricoach.garmin_sync`; dit bestand weet niets
van de Garmin-API en is daardoor los te testen.

Waarom deze data: de atleet is bewust vroeger gaan slapen (22:00 i.p.v.
23:00–24:00) voor beter herstel. Rustpols en HRV over tijd zijn de objectieve
maat of dat aanslaat — en een betere graadmeter voor de belasting dan het
gevoel van de ochtend. Dag-tot-dag-ruis is groot; overal waar deze data
getoond of meegewogen wordt geldt: de 7-daagse trend telt, niet de losse dag.
"""

import sqlite3
from datetime import date, timedelta

import pandas as pd

WELLNESS_SCHEMA = """
CREATE TABLE IF NOT EXISTS wellness (
    day                TEXT PRIMARY KEY,   -- datum (YYYY-MM-DD)
    resting_hr         INTEGER,            -- rustpols (bpm)
    hrv_last_night     REAL,               -- nachtelijk HRV-gemiddelde (ms)
    hrv_weekly_avg     REAL,               -- Garmin's 7-daags HRV-gemiddelde (ms)
    hrv_status         TEXT,               -- BALANCED / UNBALANCED / LOW / ...
    hrv_baseline_low   REAL,               -- onderkant persoonlijke baseline (ms)
    hrv_baseline_high  REAL,               -- bovenkant persoonlijke baseline (ms)
    sleep_s            REAL,               -- totale slaapduur (s)
    deep_s             REAL, light_s REAL, rem_s REAL, awake_s REAL,
    sleep_score        INTEGER,            -- Garmin-slaapscore (0-100)
    body_battery_high  INTEGER, body_battery_low INTEGER,
    stress_avg         INTEGER,            -- gemiddelde stressscore van de dag
    vo2max_run         REAL,               -- VO2 max hardlopen (Garmin-schatting)
    vo2max_bike        REAL,               -- VO2 max fietsen
    training_readiness INTEGER,            -- readiness-score (0-100)
    readiness_level    TEXT,               -- LOW / MODERATE / HIGH / ...
    synced_at          TEXT
);
"""

# Kolommen die een sync kan vullen (alles behalve de sleutel en synced_at).
VALUE_COLS = [
    "resting_hr", "hrv_last_night", "hrv_weekly_avg", "hrv_status",
    "hrv_baseline_low", "hrv_baseline_high",
    "sleep_s", "deep_s", "light_s", "rem_s", "awake_s", "sleep_score",
    "body_battery_high", "body_battery_low", "stress_avg",
    "vo2max_run", "vo2max_bike", "training_readiness", "readiness_level",
]


def ensure_table(conn: sqlite3.Connection) -> None:
    """Maak de tabel aan als hij nog niet bestaat (idempotent)."""
    conn.executescript(WELLNESS_SCHEMA)


def upsert_day(conn: sqlite3.Connection, day: date, values: dict) -> None:
    """Sla de wellness-waarden van één dag op (insert of update).

    Een her-sync overschrijft bestaande waarden alléén met echte data:
    ontbrekende velden (None) laten wat er al stond intact, zodat een
    gedeeltelijk mislukte sync (bijv. alleen de slaap-endpoint viel uit)
    nooit eerder opgehaalde waarden wist.
    """
    ensure_table(conn)
    cleaned = {k: values.get(k) for k in VALUE_COLS}
    cols = ["day", *VALUE_COLS, "synced_at"]
    updates = ", ".join(
        f"{c} = COALESCE(excluded.{c}, {c})" for c in VALUE_COLS
    )
    conn.execute(
        f"INSERT INTO wellness ({','.join(cols)}) "
        f"VALUES ({','.join('?' * len(cols))}) "
        f"ON CONFLICT(day) DO UPDATE SET {updates}, synced_at = excluded.synced_at",
        [day.isoformat(), *[cleaned[k] for k in VALUE_COLS],
         pd.Timestamp.now().isoformat(timespec="seconds")],
    )
    conn.commit()


def day_is_complete(conn: sqlite3.Connection, day: date) -> bool:
    """Heeft deze dag de kernwaarden (rustpols én HRV) al binnen?

    Gebruikt om bij een sync oude, al gevulde dagen over te slaan — recente
    dagen worden altijd opnieuw opgehaald omdat Garmin ze gedurende de dag
    nog bijwerkt.
    """
    ensure_table(conn)
    row = conn.execute(
        "SELECT 1 FROM wellness WHERE day = ? "
        "AND resting_hr IS NOT NULL AND hrv_last_night IS NOT NULL",
        (day.isoformat(),),
    ).fetchone()
    return row is not None


def load_wellness(conn: sqlite3.Connection) -> pd.DataFrame:
    """Alle wellness-dagen als DataFrame, oudste eerst (voor de trendlijnen)."""
    ensure_table(conn)
    df = pd.read_sql_query("SELECT * FROM wellness ORDER BY day", conn)
    if not df.empty:
        df["day"] = pd.to_datetime(df["day"])
    return df


def with_rolling(df: pd.DataFrame, days: int = 7) -> pd.DataFrame:
    """Voeg voortschrijdende gemiddelden toe (``<kolom>_ma``) voor de
    ruisgevoelige dagreeksen — de trend is wat telt, niet de losse dag.

    Het venster is op kalenderdagen (``{days}D`` over een datum-index), zodat
    ontbrekende dagen niet stiekem het venster oprekken.
    """
    if df.empty:
        return df
    out = df.copy().set_index("day")
    for col in ("resting_hr", "hrv_last_night", "sleep_s", "stress_avg"):
        serie = out[col].dropna()
        if serie.empty:
            continue
        out[f"{col}_ma"] = serie.rolling(f"{days}D", min_periods=2).mean()
    return out.reset_index()


def in_range(df: pd.DataFrame, start: date | None, end: date | None) -> pd.DataFrame:
    """Filter wellness-dagen op een (optioneel) datumbereik."""
    if df.empty:
        return df
    out = df
    if start is not None:
        out = out[out["day"] >= pd.Timestamp(start)]
    if end is not None:
        out = out[out["day"] <= pd.Timestamp(end)]
    return out


def _iso_week(times: pd.Series) -> pd.Series:
    """ISO-weeklabel ('2026-W24') uit een datumreeks."""
    iso = times.dt.isocalendar()
    return iso["year"].astype(str) + "-W" + iso["week"].astype(str).str.zfill(2)


def weekly_recovery(conn: sqlite3.Connection) -> pd.DataFrame:
    """Gemiddelde rustpols en HRV per ISO-week, voor de kruising met het
    weekvolume: zo wordt zichtbaar hoe het herstel meebeweegt met zware weken."""
    df = load_wellness(conn)
    if df.empty:
        return pd.DataFrame()
    df = df.copy()
    df["week"] = _iso_week(df["day"])
    return df.groupby("week", as_index=False).agg(
        rustpols=("resting_hr", "mean"),
        hrv=("hrv_last_night", "mean"),
    )


# ------------------------------------------------- herstelcontext voor feedback --

def recovery_snapshot(conn: sqlite3.Connection,
                      on_day: date | None = None) -> dict | None:
    """De herstelstand van (rond) een dag, t.o.v. het 7-daags gemiddelde.

    Voor de feedback-context: de meest recente wellness-dag op of vóór
    ``on_day`` (hooguit 2 dagen oud, anders is de data niet actueel genoeg om
    mee te wegen), plus de gemiddelden over de 7 dagen ervoor. Geeft None als
    er geen bruikbare data is — de feedback gaat dan gewoon zonder
    herstelcontext door.
    """
    ensure_table(conn)
    on_day = on_day or date.today()
    df = load_wellness(conn)
    if df.empty:
        return None
    df = df[df["day"] <= pd.Timestamp(on_day)]
    if df.empty:
        return None
    vandaag = df.iloc[-1]
    dag = vandaag["day"].date()
    if (on_day - dag).days > 2:
        return None

    venster = df[(df["day"] >= pd.Timestamp(dag - timedelta(days=7)))
                 & (df["day"] < pd.Timestamp(dag))]

    def _gem(col: str) -> float | None:
        serie = venster[col].dropna()
        return float(serie.mean()) if len(serie) >= 3 else None

    def _val(col: str):
        v = vandaag[col]
        return None if pd.isna(v) else v

    snap = {
        "dag": dag,
        "rustpols": _val("resting_hr"),
        "rustpols_7d": _gem("resting_hr"),
        "hrv": _val("hrv_last_night"),
        "hrv_7d": _gem("hrv_last_night"),
        "hrv_status": _val("hrv_status"),
        "hrv_baseline_low": _val("hrv_baseline_low"),
        "hrv_baseline_high": _val("hrv_baseline_high"),
        "slaap_s": _val("sleep_s"),
        "slaap_score": _val("sleep_score"),
        "readiness": _val("training_readiness"),
        "readiness_level": _val("readiness_level"),
    }
    if snap["rustpols"] is None and snap["hrv"] is None:
        return None
    return snap
