"""Opslag van voedingsplannen en de terugkoppeling achteraf.

Twee tabellen naast de bestaande:

- ``nutrition_plans`` — een opgeslagen plan als JSON (de invoer én het
  berekende plan), optioneel gekoppeld aan een geplande datum en later aan de
  sessie die eruit voortkwam (``activity_key``).
- ``nutrition_feedback`` — wat er daadwerkelijk in ging en hoe de maag
  reageerde. Eén rij per plan.

Waarom het plan als JSON en niet uitgesplitst over kolommen: een plan is een
momentopname van een berekening met de toen geldende producten en drempels.
Zou je het genormaliseerd opslaan, dan verandert een later gewijzigd product
met terugwerkende kracht wat er "gepland" was — en juist de vergelijking tussen
plan en praktijk is hier het punt.

De terugkoppeling is het waardevolste deel van deze module: na een paar
sessies staat er zwart op wit wat *jouw* darm aankan ("100 g/uur viel goed op
9 augustus"), en dat weegt zwaarder dan welke algemene richtlijn ook.
"""

import json
import sqlite3
from dataclasses import asdict
from datetime import date, datetime

import pandas as pd

from tricoach.nutrition.plan import NutritionPlan

# De mogelijke maagreacties, van goed naar slecht. De volgorde is betekenisvol:
# de tolerantiegeschiedenis zoekt de hóógste inname die nog "goed" viel.
GUT_GOOD, GUT_MILD, GUT_BAD = "goed", "licht ongemak", "klachten"
GUT_OPTIONS = (GUT_GOOD, GUT_MILD, GUT_BAD)
GUT_ICON = {GUT_GOOD: "✅", GUT_MILD: "🟡", GUT_BAD: "🔴"}

PLAN_SCHEMA = """
CREATE TABLE IF NOT EXISTS nutrition_plans (
    plan_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    name         TEXT,
    planned_date TEXT,
    created_at   TEXT,
    activity_key TEXT,
    summary      TEXT,
    plan_json    TEXT
);
CREATE TABLE IF NOT EXISTS nutrition_feedback (
    plan_id        INTEGER PRIMARY KEY,
    recorded_at    TEXT,
    activity_key   TEXT,
    actual_carbs_g REAL,
    actual_duration_s REAL,
    gut            TEXT,
    note           TEXT
);
"""


def ensure_tables(conn: sqlite3.Connection) -> None:
    """Maak de plan- en terugkoppelingstabellen aan (idempotent)."""
    conn.executescript(PLAN_SCHEMA)
    conn.commit()


def _plan_as_dict(plan: NutritionPlan) -> dict:
    """Het plan als JSON-serialiseerbare dict (dataclasses helemaal uitgeklapt)."""
    return json.loads(json.dumps(asdict(plan), default=str))


def save_plan(conn: sqlite3.Connection, plan: NutritionPlan, name: str,
              planned_date: date | None = None,
              activity_key: str | None = None) -> int:
    """Sla een plan op en geef het ``plan_id`` terug."""
    ensure_tables(conn)
    cur = conn.execute(
        "INSERT INTO nutrition_plans (name, planned_date, created_at, "
        "activity_key, summary, plan_json) VALUES (?,?,?,?,?,?)",
        (name.strip() or "Naamloos plan",
         planned_date.isoformat() if planned_date else None,
         datetime.now().isoformat(timespec="seconds"),
         activity_key,
         plan.summary_text(),
         json.dumps(_plan_as_dict(plan), ensure_ascii=False)),
    )
    conn.commit()
    return int(cur.lastrowid)


def load_plans(conn: sqlite3.Connection) -> pd.DataFrame:
    """Alle opgeslagen plannen met hun terugkoppeling, nieuwste eerst."""
    ensure_tables(conn)
    df = pd.read_sql_query(
        "SELECT p.plan_id, p.name, p.planned_date, p.created_at, p.activity_key, "
        "p.summary, f.recorded_at, f.actual_carbs_g, f.actual_duration_s, "
        "f.gut, f.note "
        "FROM nutrition_plans p LEFT JOIN nutrition_feedback f "
        "ON f.plan_id = p.plan_id ORDER BY p.plan_id DESC", conn)
    return df


def load_plan_json(conn: sqlite3.Connection, plan_id: int) -> dict | None:
    """Het opgeslagen plan van één ``plan_id`` als dict, of None."""
    ensure_tables(conn)
    row = conn.execute(
        "SELECT plan_json FROM nutrition_plans WHERE plan_id = ?", (plan_id,)
    ).fetchone()
    if row is None or not row[0]:
        return None
    return json.loads(row[0])


def delete_plan(conn: sqlite3.Connection, plan_id: int) -> bool:
    """Verwijder een plan en de bijbehorende terugkoppeling."""
    ensure_tables(conn)
    conn.execute("DELETE FROM nutrition_feedback WHERE plan_id = ?", (plan_id,))
    cur = conn.execute("DELETE FROM nutrition_plans WHERE plan_id = ?", (plan_id,))
    conn.commit()
    return cur.rowcount > 0


def link_activity(conn: sqlite3.Connection, plan_id: int, activity_key: str | None) -> bool:
    """Koppel een plan aan de sessie waarin het is uitgevoerd (of maak los)."""
    ensure_tables(conn)
    cur = conn.execute(
        "UPDATE nutrition_plans SET activity_key = ? WHERE plan_id = ?",
        (activity_key, plan_id))
    conn.commit()
    return cur.rowcount > 0


def save_feedback(conn: sqlite3.Connection, plan_id: int, actual_carbs_g: float | None,
                  actual_duration_s: float | None, gut: str, note: str = "",
                  activity_key: str | None = None) -> None:
    """Leg vast wat er werkelijk in ging en hoe de maag reageerde.

    Eén rij per plan: opnieuw invullen corrigeert de vorige invoer.
    """
    ensure_tables(conn)
    conn.execute(
        "INSERT OR REPLACE INTO nutrition_feedback (plan_id, recorded_at, "
        "activity_key, actual_carbs_g, actual_duration_s, gut, note) "
        "VALUES (?,?,?,?,?,?,?)",
        (plan_id, datetime.now().isoformat(timespec="seconds"), activity_key,
         actual_carbs_g, actual_duration_s,
         gut if gut in GUT_OPTIONS else GUT_GOOD, (note or "").strip()),
    )
    conn.commit()


def tolerance_history(conn: sqlite3.Connection) -> pd.DataFrame:
    """De eigen tolerantie over tijd: wat ging erin en hoe viel het?

    Per ingevulde terugkoppeling de werkelijke inname per uur, de maagreactie en
    de datum. Dit is de reeks waaruit blijkt wat jouw darm aankan — waardevoller
    dan een algemene richtlijn, omdat hij over jou gaat.
    """
    ensure_tables(conn)
    df = pd.read_sql_query(
        "SELECT p.plan_id, p.name, COALESCE(p.planned_date, date(f.recorded_at)) AS datum, "
        "f.actual_carbs_g, f.actual_duration_s, f.gut, f.note "
        "FROM nutrition_feedback f JOIN nutrition_plans p ON p.plan_id = f.plan_id "
        "ORDER BY datum", conn)
    if df.empty:
        return df
    uren = df["actual_duration_s"] / 3600
    df["g_per_uur"] = (df["actual_carbs_g"] / uren).where(uren > 0)
    df["datum"] = pd.to_datetime(df["datum"], errors="coerce")
    return df


def tolerance_summary(conn: sqlite3.Connection) -> list[str]:
    """Leesbare regels over de eigen tolerantie, bijv. '100 g/uur viel goed op ...'.

    Geeft de hoogste inname per maagreactie: de bovengrens die goed viel is het
    interessantste getal, en de laagste die klachten gaf de belangrijkste
    waarschuwing.
    """
    df = tolerance_history(conn)
    if df.empty or df["g_per_uur"].isna().all():
        return []
    uit = []
    goed = df[(df["gut"] == GUT_GOOD) & df["g_per_uur"].notna()]
    if not goed.empty:
        beste = goed.loc[goed["g_per_uur"].idxmax()]
        uit.append(f"{GUT_ICON[GUT_GOOD]} {beste['g_per_uur']:.0f} g/uur viel goed "
                   f"op {beste['datum']:%d-%m-%Y} ({beste['name']}).")
    slecht = df[(df["gut"] == GUT_BAD) & df["g_per_uur"].notna()]
    if not slecht.empty:
        laagste = slecht.loc[slecht["g_per_uur"].idxmin()]
        uit.append(f"{GUT_ICON[GUT_BAD]} {laagste['g_per_uur']:.0f} g/uur gaf klachten "
                   f"op {laagste['datum']:%d-%m-%Y} ({laagste['name']}).")
    mild = df[(df["gut"] == GUT_MILD) & df["g_per_uur"].notna()]
    if not mild.empty:
        hoogste = mild.loc[mild["g_per_uur"].idxmax()]
        uit.append(f"{GUT_ICON[GUT_MILD]} {hoogste['g_per_uur']:.0f} g/uur gaf licht "
                   f"ongemak op {hoogste['datum']:%d-%m-%Y} ({hoogste['name']}).")
    return uit
