"""Data-bewerkingen voor het dashboard: weekvolumes, zonetijden en trends.

Alle functies werken op de DataFrames uit ``storage`` en geven DataFrames
terug die direct te plotten zijn. Hier zit geen Streamlit- of plotly-code;
dat houdt de berekeningen testbaar los van de presentatie.
"""

import json
import sqlite3

import pandas as pd

from tricoach.formatting import derive_speed_ms
from tricoach.progress import trimp_per_session
from tricoach.storage import load_records
from tricoach.zones import ZONE_NAMES, intensity_category


def add_week(activities: pd.DataFrame) -> pd.DataFrame:
    """Voeg een ISO-weeklabel toe (bijv. '2026-W23') voor groeperen per week."""
    df = activities.copy()
    iso = df["start_time"].dt.isocalendar()
    df["week"] = iso["year"].astype(str) + "-W" + iso["week"].astype(str).str.zfill(2)
    return df


def weekly_volume(activities: pd.DataFrame) -> pd.DataFrame:
    """Trainingsuren per week per sport (lange vorm: week, sport, uren)."""
    df = add_week(activities)
    out = (
        df.groupby(["week", "sport"], as_index=False)["duration_s"].sum()
        .rename(columns={"duration_s": "uren"})
    )
    out["uren"] = out["uren"] / 3600
    return out.sort_values("week")


def weekly_zone_time(activities: pd.DataFrame) -> pd.DataFrame:
    """Minuten per hartslagzone per week (lange vorm: week, zone, minuten)."""
    df = add_week(activities)
    zone_cols = [f"{z.lower()}_s" for z in ZONE_NAMES]
    melted = df.melt(
        id_vars="week", value_vars=zone_cols,
        var_name="zone", value_name="seconden",
    )
    melted["zone"] = melted["zone"].str.removesuffix("_s").str.upper()
    out = melted.groupby(["week", "zone"], as_index=False)["seconden"].sum()
    out["minuten"] = out["seconden"] / 60
    return out.sort_values(["week", "zone"])


def weekly_intensity_share(activities: pd.DataFrame) -> pd.DataFrame:
    """Intensiteitsverdeling per week: aandeel rustig (Z1–Z2), grijze zone
    (Z3) en hard (Z4–Z5) als percentage van de gemeten zonetijd.

    Lange vorm (week, categorie, pct) voor een 100%-gestapelde weekbalk —
    de klassieke 80/20-check voor duursporters.
    """
    df = add_week(activities)
    per_week = df.groupby("week")[["z1_s", "z2_s", "z3_s", "z4_s", "z5_s"]].sum()
    per_week = per_week[per_week.sum(axis=1) > 0]
    if per_week.empty:
        return pd.DataFrame()
    totaal = per_week.sum(axis=1)
    breed = pd.DataFrame({
        "week": per_week.index,
        "Rustig (Z1–Z2)": (per_week["z1_s"] + per_week["z2_s"]) / totaal * 100,
        "Grijze zone (Z3)": per_week["z3_s"] / totaal * 100,
        "Hard (Z4–Z5)": (per_week["z4_s"] + per_week["z5_s"]) / totaal * 100,
    })
    lang = breed.melt(id_vars="week", var_name="categorie", value_name="pct")
    return lang.sort_values("week")


def weekly_totals(activities: pd.DataFrame) -> pd.DataFrame:
    """Weektotalen voor het overzicht: sessies, uren (totaal en per sport),
    kilometers en TRIMP per week, nieuwste week eerst.

    ``delta_uren`` is het verschil in trainingsuren met de eerstvolgende
    oudere week in de tabel.
    """
    df = add_week(activities)
    df["trimp"] = trimp_per_session(activities)["trimp"]
    per_week = df.groupby("week").agg(
        sessies=("activity_key", "count"),
        uren=("duration_s", "sum"),
        km=("distance_m", "sum"),
        trimp=("trimp", "sum"),
    )
    per_week["uren"] = per_week["uren"] / 3600
    per_week["km"] = per_week["km"] / 1000
    per_sport = (df.groupby(["week", "sport"])["duration_s"].sum() / 3600).unstack()
    for sport in ("swimming", "cycling", "running"):
        kolom = per_sport[sport] if sport in per_sport else 0.0
        per_week[f"uren_{sport}"] = kolom
    per_week = per_week.fillna(0.0).sort_index(ascending=False).reset_index()
    per_week["delta_uren"] = per_week["uren"].diff(-1)
    return per_week


def pace_at_hr(
    conn: sqlite3.Connection,
    activities: pd.DataFrame,
    sport: str,
    hr_range: tuple[int, int],
    min_seconds: int = 300,
) -> pd.DataFrame:
    """Tempo bij gelijke hartslag: per sessie de gemiddelde snelheid van alle
    meetpunten binnen ``hr_range`` (bijv. Z2).

    Dit is de belangrijkste trendmaat: wordt het tempo bij dezelfde hartslag
    sneller, dan groeit de aerobe basis. Sessies met minder dan
    ``min_seconds`` aan meetpunten in de range worden overgeslagen, anders
    vertekenen een paar losse seconden het beeld.
    """
    lo, hi = hr_range
    rows = []
    for _, act in activities[activities["sport"] == sport].iterrows():
        rec = load_records(conn, act["activity_key"])
        if rec.empty or "heart_rate" not in rec:
            continue
        in_range = rec[(rec["heart_rate"] >= lo) & (rec["heart_rate"] <= hi)]
        in_range = in_range.dropna(subset=["speed_ms"])
        in_range = in_range[in_range["speed_ms"] > 0.5]  # stilstand eruit
        if len(in_range) < min_seconds:  # records zijn ~1/s
            continue
        speed = in_range["speed_ms"].mean()
        rows.append({
            "start_time": act["start_time"],
            "speed_ms": speed,
            "tempo_min_per_km": (1000 / speed) / 60,
            "snelheid_kmh": speed * 3.6,
            "meetpunten": len(in_range),
        })
    return pd.DataFrame(rows).sort_values("start_time") if rows else pd.DataFrame()


def aerobic_efficiency_trend(
    activities: pd.DataFrame, margin_pct: float = 2.0
) -> dict[str, dict]:
    """Per sessie een trendpijl voor de aerobe efficiëntie t.o.v. een eerdere,
    vergelijkbare sessie.

    Werkt volledig op de al gecachete kolommen (``aerobic_efficiency`` en de
    z*_s-zonetijden); er worden hier geen FIT-records meer geparset.

    Vergelijkingsregel (hybride, like-for-like met terugval):
    - Kies de meest recente eerdere sessie van **dezelfde sport** met dezelfde
      intensiteitscategorie (rustig/intensief). Dat is de zuivere vergelijking.
    - Is die er niet, val dan terug op de meest recente eerdere sessie van
      dezelfde sport (``exact=False``) en markeer dat als waarschuwing, zodat
      de weergave kan tonen dat het geen gelijke-intensiteitsvergelijking is.
    - Geen eerdere sessie, of geen efficiëntie (o.a. zwemmen): geen pijl.

    Geeft een dict ``activity_key -> {symbol, delta_pct, ref, exact, note}``.
    """
    out: dict[str, dict] = {}
    df = activities.sort_values("start_time")

    for sport in ("running", "cycling"):
        history: list[dict] = []  # eerdere sessies met geldige efficiëntie
        for _, row in df[df["sport"] == sport].iterrows():
            key = row["activity_key"]
            eff = row.get("aerobic_efficiency")
            cat = intensity_category(
                row["z1_s"], row["z2_s"], row["z3_s"], row["z4_s"], row["z5_s"])

            if eff is None or pd.isna(eff):
                out[key] = {"symbol": "—", "delta_pct": None, "ref": None,
                            "exact": True, "note": "geen snelheid/hartslag"}
                continue

            same_cat = [h for h in history if h["cat"] == cat]
            if same_cat:
                prev, exact = same_cat[-1], True
            elif history:
                prev, exact = history[-1], False
            else:
                prev = None

            if prev is None:
                out[key] = {"symbol": "—", "delta_pct": None, "ref": None,
                            "exact": True, "note": "geen eerdere sessie"}
            else:
                delta = 100.0 * (eff - prev["eff"]) / prev["eff"]
                symbol = "▲" if delta > margin_pct else "▼" if delta < -margin_pct else "▬"
                note = ("vergeleken met de dichtstbijzijnde sessie (geen eerdere "
                        "sessie van gelijke intensiteit)") if not exact else ""
                out[key] = {"symbol": symbol, "delta_pct": delta,
                            "ref": prev["start_time"], "exact": exact, "note": note}

            history.append({"eff": eff, "cat": cat, "start_time": row["start_time"]})

    # Zwemmen en overige sporten: geen efficiëntiepijl (onbetrouwbare pols-HR).
    for _, row in df[~df["sport"].isin(("running", "cycling"))].iterrows():
        out[row["activity_key"]] = {
            "symbol": "—", "delta_pct": None, "ref": None, "exact": True,
            "note": "zwemmen: pols-HR onder water onbetrouwbaar",
        }
    return out


def run_power(activities: pd.DataFrame) -> pd.DataFrame:
    """Loopvermogen per sessie uit de FIT-samenvatting (summary_json).

    Garmin schat loopvermogen aan de pols; genormaliseerd vermogen waar
    beschikbaar, anders het gemiddelde. Indicatief, maar wél consistent
    tussen sessies van hetzelfde horloge.
    """
    runs = activities[activities["sport"] == "running"].sort_values("start_time")
    rows = []
    for _, act in runs.iterrows():
        try:
            summary = json.loads(act.get("summary_json") or "{}")
        except (TypeError, ValueError):
            continue
        vermogen = summary.get("normalized_power") or summary.get("avg_power")
        if vermogen:
            rows.append({"start_time": act["start_time"], "watt": float(vermogen)})
    return pd.DataFrame(rows)


def swim_length_matrix(conn: sqlite3.Connection, activities: pd.DataFrame) -> pd.DataFrame:
    """Tijd per baan per zwemsessie als matrix voor een heatmap.

    Rijen: sessies (index = startdatum), kolommen: baannummer (1..n), waarde:
    seconden per baan. Kortere sessies krijgen NaN in de staart.
    """
    swims = activities[activities["sport"] == "swimming"].sort_values("start_time")
    rows = {}
    for _, act in swims.iterrows():
        lengths = pd.read_sql_query(
            "SELECT total_timer_time FROM lengths WHERE activity_key = ? "
            "ORDER BY timestamp", conn, params=(act["activity_key"],))
        if lengths.empty:
            continue
        rows[act["start_time"]] = lengths["total_timer_time"].reset_index(drop=True)
    if not rows:
        return pd.DataFrame()
    matrix = pd.DataFrame(rows).T
    matrix.columns = matrix.columns + 1  # baannummers vanaf 1
    return matrix


def stroke_distribution(conn: sqlite3.Connection, activities: pd.DataFrame) -> pd.DataFrame:
    """Banen per slagtype per zwemsessie (lange vorm: start_time, slag, banen)."""
    swims = activities[activities["sport"] == "swimming"].sort_values("start_time")
    rows = []
    for _, act in swims.iterrows():
        lengths = pd.read_sql_query(
            "SELECT swim_stroke, COUNT(*) AS banen FROM lengths "
            "WHERE activity_key = ? GROUP BY swim_stroke",
            conn, params=(act["activity_key"],))
        for _, r in lengths.iterrows():
            rows.append({"start_time": act["start_time"],
                         "slag": r["swim_stroke"], "banen": int(r["banen"])})
    return pd.DataFrame(rows)


def swim_per_session(conn: sqlite3.Connection, activities: pd.DataFrame) -> pd.DataFrame:
    """SWOLF en tempo per zwemsessie, voor de zwemtrend-grafiek."""
    swims = activities[activities["sport"] == "swimming"].copy()
    if swims.empty:
        return pd.DataFrame()
    rows = []
    for _, act in swims.iterrows():
        lengths = pd.read_sql_query(
            "SELECT * FROM lengths WHERE activity_key = ?",
            conn, params=(act["activity_key"],))
        swolf = (
            (lengths["total_timer_time"] + lengths["total_strokes"]).mean()
            if not lengths.empty else None
        )
        # Tempo uit afstand en de zuivere zwemtijd (som van de actieve banen),
        # niet uit avg_speed_ms: dat veld ontbreekt op sommige sessies (bijv.
        # een samengevoegde zwemsessie) waardoor die anders uit de grafiek viel.
        # Geen baandata -> terugval op de totale timer-duur.
        actief_s = lengths["total_timer_time"].sum() if not lengths.empty else None
        zwemtijd = actief_s if actief_s and actief_s > 0 else act["duration_s"]
        speed = derive_speed_ms(act["distance_m"], zwemtijd)
        rows.append({
            "start_time": act["start_time"],
            "swolf": swolf,
            "tempo_s_per_100m": 100 / speed if speed else None,
            "afstand_m": act["distance_m"],
        })
    return pd.DataFrame(rows).sort_values("start_time")
