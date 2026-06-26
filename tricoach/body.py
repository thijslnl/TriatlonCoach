"""Lichaamssamenstelling: handmatige invoer, geschiedenis en trendduiding.

Aparte SQLite-tabel ``body_composition`` (los van de trainingen), één rij per
meetdatum met de waarden van de slimme weegschaal (Fitdays). De ruwe getallen
staan in SQLite; ``memory/lichaamssamenstelling.md`` houdt een leesbaar logboek
plus een korte, door gemma opgestelde duiding van de trend.

**Toon is bewust neutraal.** De tool volgt lichaamssamenstelling als data voor
sportprestatie, niet als afvalcoach: focus op trends over tijd, weinig nadruk
op BMI (slechte maat bij veel spiermassa), en géén calorie-, dieet- of
afvaldoelen. Die lijn zit ook in de systeemprompt voor de trendduiding.
"""

import base64
import json
import re
import sqlite3
from datetime import date
from pathlib import Path

import pandas as pd

from tricoach.llm.router import LLMRouter

# Veldmetadata: (kolom, label, eenheid, stap). Alleen ``measured_on`` (datum) en
# minstens één waarde zijn nodig; de rest mag leeg blijven. De volgorde stuurt
# zowel het invoerformulier als de logregel.
FIELDS = [
    ("weight_kg", "Gewicht", "kg", 0.1),
    ("bmi", "BMI", "", 0.1),
    ("fat_pct", "Lichaamsvet", "%", 0.1),
    ("fat_mass_kg", "Vetmassa", "kg", 0.1),
    ("lean_mass_kg", "Vetvrije massa", "kg", 0.1),
    ("muscle_mass_kg", "Spiermassa", "kg", 0.1),
    ("skeletal_muscle_pct", "Skeletspier", "%", 0.1),
    ("bone_mass_kg", "Botmassa", "kg", 0.1),
    ("water_pct", "Lichaamswater", "%", 0.1),
    ("visceral_fat", "Visceraal vet", "", 0.1),
    ("bmr_kcal", "BMR", "kcal", 1.0),
    ("metabolic_age", "Lichaamsleeftijd", "jaar", 1.0),
]
VALUE_COLS = [f[0] for f in FIELDS]
LABELS = {f[0]: f[1] for f in FIELDS}
UNITS = {f[0]: f[2] for f in FIELDS}

# De vier kernreeksen voor de trendgrafieken (los en gecombineerd).
TREND_FIELDS = ["weight_kg", "fat_pct", "muscle_mass_kg", "visceral_fat"]

BODY_SCHEMA = """
CREATE TABLE IF NOT EXISTS body_composition (
    measured_on          TEXT PRIMARY KEY,   -- meetdatum (YYYY-MM-DD)
    weight_kg            REAL,
    bmi                  REAL,
    fat_pct              REAL,
    fat_mass_kg          REAL,
    lean_mass_kg         REAL,
    muscle_mass_kg       REAL,
    skeletal_muscle_pct  REAL,
    bone_mass_kg         REAL,
    water_pct            REAL,
    visceral_fat         REAL,
    bmr_kcal             REAL,
    metabolic_age        REAL,
    created_at           TEXT
);
"""


def ensure_table(conn: sqlite3.Connection) -> None:
    """Maak de tabel aan als hij nog niet bestaat (idempotent)."""
    conn.executescript(BODY_SCHEMA)


def save_measurement(
    conn: sqlite3.Connection, measured_on: date, values: dict[str, float | None]
) -> None:
    """Sla één meting op (of overschrijf een bestaande van dezelfde datum).

    Alleen ingevulde velden worden bewaard; ontbrekende waarden worden NULL.
    De datum is de primaire sleutel, zodat een typefout corrigeren kan door de
    meting opnieuw in te voeren.
    """
    ensure_table(conn)
    cleaned = {k: values.get(k) for k in VALUE_COLS}
    cols = ["measured_on", *VALUE_COLS, "created_at"]
    params = [
        measured_on.isoformat(),
        *[cleaned[k] for k in VALUE_COLS],
        pd.Timestamp.now().isoformat(timespec="seconds"),
    ]
    placeholders = ",".join("?" * len(cols))
    conn.execute(
        f"INSERT OR REPLACE INTO body_composition ({','.join(cols)}) "
        f"VALUES ({placeholders})",
        params,
    )
    conn.commit()


def delete_measurement(conn: sqlite3.Connection, measured_on: date) -> None:
    """Verwijder de meting van één datum (no-op als die niet bestaat)."""
    ensure_table(conn)
    conn.execute(
        "DELETE FROM body_composition WHERE measured_on = ?",
        (measured_on.isoformat(),),
    )
    conn.commit()


def load_measurements(conn: sqlite3.Connection) -> pd.DataFrame:
    """Alle metingen als DataFrame, oudste eerst (voor de trendlijnen)."""
    ensure_table(conn)
    df = pd.read_sql_query(
        "SELECT * FROM body_composition ORDER BY measured_on", conn)
    if not df.empty:
        df["measured_on"] = pd.to_datetime(df["measured_on"])
    return df


def in_range(df: pd.DataFrame, start: date | None, end: date | None) -> pd.DataFrame:
    """Filter metingen op een (optioneel) datumbereik."""
    if df.empty:
        return df
    out = df
    if start is not None:
        out = out[out["measured_on"] >= pd.Timestamp(start)]
    if end is not None:
        out = out[out["measured_on"] <= pd.Timestamp(end)]
    return out


def normalized_trends(df: pd.DataFrame, fields: list[str]) -> pd.DataFrame:
    """Reeksen geïndexeerd op de eerste meting (=100), voor één gecombineerde
    grafiek met verschillende eenheden op dezelfde schaal (relatieve verandering)."""
    if df.empty:
        return pd.DataFrame()
    rows = []
    for col in fields:
        serie = df[["measured_on", col]].dropna()
        if serie.empty:
            continue
        basis = serie[col].iloc[0]
        if not basis:
            continue
        for _, r in serie.iterrows():
            rows.append({
                "measured_on": r["measured_on"],
                "reeks": LABELS.get(col, col),
                "index": r[col] / basis * 100,
            })
    return pd.DataFrame(rows)


def _iso_week(times: pd.Series) -> pd.Series:
    """ISO-weeklabel ('2026-W24') uit een datumreeks, tijdzone-veilig."""
    naive = times.dt.tz_localize(None) if times.dt.tz is not None else times
    iso = naive.dt.isocalendar()
    return iso["year"].astype(str) + "-W" + iso["week"].astype(str).str.zfill(2)


def weight_vs_cycling(conn: sqlite3.Connection, acts: pd.DataFrame) -> pd.DataFrame:
    """Gewichtstrend naast wekelijkse fietsafstand (richting power-to-weight).

    Vermogen wordt niet structureel opgeslagen, dus we gebruiken fietsafstand
    per week als prestatieproxy. Geeft per ISO-week de gemiddelde gewichtsmeting
    en de getrapte kilometers terug.
    """
    metingen = load_measurements(conn)
    if metingen.empty or acts.empty:
        return pd.DataFrame()

    m = metingen.dropna(subset=["weight_kg"]).copy()
    m["week"] = _iso_week(m["measured_on"])
    gewicht = m.groupby("week", as_index=False)["weight_kg"].mean()

    rides = acts[acts["sport"] == "cycling"].copy()
    if rides.empty:
        return gewicht.assign(fiets_km=0.0)
    rides["week"] = _iso_week(rides["start_time"])
    km = (rides.groupby("week", as_index=False)["distance_m"].sum()
          .assign(fiets_km=lambda d: d["distance_m"] / 1000)
          .drop(columns="distance_m"))
    return gewicht.merge(km, on="week", how="left").fillna({"fiets_km": 0.0})


# ----------------------------------------------------- logboek + trendduiding --

HEADER = """# Lichaamssamenstelling

Leesbaar logboek van de weegschaalmetingen (Fitdays), nieuwste onderaan, plus
een korte door gemma opgestelde duiding van de trend. De ruwe getallen staan in
SQLite (tabel `body_composition`). Neutrale data voor sportprestatie — geen
afval- of dieetdoelen.
"""

BODY_SYSTEM = (
    "Je bent een nuchtere sportcoach die lichaamssamenstelling volgt als "
    "neutrale data voor sportprestatie, niet als afvalcoach. Je krijgt een reeks "
    "weegschaalmetingen over tijd. Schrijf in het Nederlands 2 tot 4 zinnen over "
    "de TREND (de richting van vetpercentage en spiermassa over de weken), niet "
    "over een momentopname. Belangrijk: geef GEEN calorie-, dieet- of afvaldoelen "
    "en GEEN streefgetallen voor gewicht. BMI is een slechte maat voor gespierde "
    "mensen (deze atleet heeft ~80 kg spiermassa); leg er weinig nadruk op en de "
    "meeste op vetpercentage en behoud van spiermassa. Blijf feitelijk en "
    "neutraal-bemoedigend. Geen aanhalingstekens."
)


def _measurement_line(measured_on: date, values: dict[str, float | None]) -> str:
    """Compacte, leesbare regel met de ingevulde waarden van één meting."""
    parts = []
    for col, label, eenheid, _ in FIELDS:
        val = values.get(col)
        if val is None or (isinstance(val, float) and pd.isna(val)):
            continue
        parts.append(f"{label} {val:g}{eenheid}")
    return ", ".join(parts) if parts else "(geen waarden)"


def log_measurement(memory_dir: Path, measured_on: date, values: dict) -> None:
    """Schrijf de ruwe meting als leesbare regel naar lichaamssamenstelling.md."""
    path = memory_dir / "lichaamssamenstelling.md"
    if not path.exists():
        path.write_text(HEADER, encoding="utf-8")
    entry = f"\n## {measured_on:%Y-%m-%d} — meting\n\n- {_measurement_line(measured_on, values)}\n"
    with open(path, "a", encoding="utf-8") as f:
        f.write(entry)


def _trend_table(df: pd.DataFrame, n: int = 12) -> str:
    """De laatste n metingen als compacte teksttabel voor de gemma-prompt."""
    cols = ["measured_on", *TREND_FIELDS]
    recent = df[cols].tail(n)
    regels = []
    for _, r in recent.iterrows():
        stukjes = [r["measured_on"].strftime("%Y-%m-%d")]
        for col in TREND_FIELDS:
            if pd.notna(r[col]):
                stukjes.append(f"{LABELS[col]} {r[col]:g}{UNITS[col]}")
        regels.append("- " + ", ".join(stukjes))
    return "\n".join(regels)


def summarize_trend(router: LLMRouter, conn: sqlite3.Connection, memory_dir: Path) -> str | None:
    """Laat gemma een korte, neutrale trendduiding schrijven en leg die vast.

    Pas zinvol vanaf twee metingen; bij één meting is er nog geen trend en geven
    we None terug (de meting zelf is dan al gelogd).
    """
    df = load_measurements(conn)
    if len(df) < 2:
        return None

    prompt = (
        "Metingen (oudste eerst):\n"
        f"{_trend_table(df)}\n\n"
        "Geef een korte, neutrale duiding van de trend."
    )
    try:
        duiding = router.ask("body_trend", prompt, system=BODY_SYSTEM)
    except Exception:
        return None  # Ollama onbereikbaar: de meting is al opgeslagen en gelogd.

    path = memory_dir / "lichaamssamenstelling.md"
    if not path.exists():
        path.write_text(HEADER, encoding="utf-8")
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"\n  - _Trendduiding ({date.today():%Y-%m-%d}):_ {duiding}\n")
    return duiding


# --------------------------------------------------- screenshot-extractie (opt) --

EXTRACT_SYSTEM = (
    "Je leest een screenshot van een weegschaal-app (Fitdays) uit. Geef de "
    "waarden terug als platte JSON met deze sleutels (laat een sleutel weg als "
    "de waarde niet zichtbaar is): "
    + ", ".join(VALUE_COLS)
    + ". Gebruik punten als decimaalteken, geen eenheden, geen extra tekst."
)


def extract_from_screenshot(router: LLMRouter, image_bytes: bytes) -> dict[str, float]:
    """Lees weegschaalwaarden uit een screenshot via het multimodale gemma-model.

    Optionele hulp bij de handmatige invoer: het resultaat vult het formulier
    voor, de gebruiker controleert het. Geeft een dict met herkende velden;
    bij twijfel of mislukking een lege/gedeeltelijke dict.
    """
    b64 = base64.b64encode(image_bytes).decode("ascii")
    raw = router.ask_with_images(
        "body_trend",  # zelfde lokale model; routing body_trend -> ollama
        "Lees de zichtbare lichaamssamenstellingswaarden uit deze screenshot.",
        images=[b64],
        system=EXTRACT_SYSTEM,
    )
    return _parse_extracted(raw)


def _parse_extracted(raw: str) -> dict[str, float]:
    """Haal het JSON-blok uit het modelantwoord en behoud alleen bekende velden."""
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        return {}
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return {}
    out = {}
    for col in VALUE_COLS:
        if col in data and data[col] is not None:
            try:
                out[col] = float(data[col])
            except (TypeError, ValueError):
                continue
    return out
