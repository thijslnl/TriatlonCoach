"""Voortgangsanalyses: belasting, efficiëntie, racevoorspelling en records.

Vier blokken, allemaal gevoed door de SQLite-data:

- **Belasting (TRIMP/CTL/ATL/ACWR)**: elke sessie krijgt een belastingsscore
  uit de tijd per hartslagzone. Daaruit volgen een fitheidslijn (CTL, traag
  voortschrijdend gemiddelde), een vermoeidheidslijn (ATL, snel) en hun
  verhouding (ACWR) als signaal voor verantwoorde opbouw.
- **Efficiëntie**: efficiency factor (meters per minuut per hartslag) en
  decoupling (hoeveel zakt de efficiëntie in de tweede helft van een sessie).
- **Racevoorspelling**: huidige niveaus vertaald naar de standaardafstanden
  (1,5 km zwemmen / 40 km fietsen / 10 km lopen).
- **Records & zwemprogressie**: persoonlijke records en de ontwikkeling van
  het crawl-aandeel en de SWOLF per slagtype.
"""

import math
import sqlite3

import numpy as np
import pandas as pd

from tricoach.storage import load_records

# Gewicht per hartslagzone voor de TRIMP-score (minuten × gewicht).
TRIMP_WEIGHTS = {"z1_s": 1, "z2_s": 2, "z3_s": 3, "z4_s": 4, "z5_s": 5}

CTL_TAU = 42  # dagen; fitheid reageert traag
ATL_TAU = 7   # dagen; vermoeidheid reageert snel

# Riegel-exponent: hoe looptijden schalen met de afstand.
RIEGEL = 1.06


# ------------------------------------------------------------- belasting --

def trimp_per_session(acts: pd.DataFrame) -> pd.DataFrame:
    """TRIMP-score per sessie (kolommen: start_time, sport, trimp)."""
    df = acts.copy()
    df["trimp"] = sum(df[col].fillna(0) / 60 * w for col, w in TRIMP_WEIGHTS.items())
    return df[["start_time", "sport", "trimp"]]


def load_curves(acts: pd.DataFrame) -> pd.DataFrame:
    """Dagelijkse fitheid (CTL) en vermoeidheid (ATL) als exponentieel
    voortschrijdende gemiddelden van de dagelijkse TRIMP.

    Geeft een DataFrame per dag: datum, trimp, ctl, atl, acwr.
    """
    per_sessie = trimp_per_session(acts)
    per_dag = (
        per_sessie.assign(datum=per_sessie["start_time"].dt.date)
        .groupby("datum")["trimp"].sum()
    )
    dagen = pd.date_range(min(per_dag.index), pd.Timestamp.today().date(), freq="D")
    trimp = pd.Series(0.0, index=dagen)
    for d, v in per_dag.items():
        trimp[pd.Timestamp(d)] = v

    k_ctl = 1 - math.exp(-1 / CTL_TAU)
    k_atl = 1 - math.exp(-1 / ATL_TAU)
    ctl, atl = [], []
    c = a = 0.0
    for v in trimp:
        c += (v - c) * k_ctl
        a += (v - a) * k_atl
        ctl.append(c)
        atl.append(a)

    out = pd.DataFrame({"datum": dagen, "trimp": trimp.values, "ctl": ctl, "atl": atl})
    out["acwr"] = np.where(out["ctl"] > 1, out["atl"] / out["ctl"], np.nan)
    return out


def acwr_status(acwr: float | None) -> tuple[str, str]:
    """Stoplicht bij de acute:chronische verhouding: (label, kleur-emoji)."""
    if acwr is None or np.isnan(acwr):
        return "Nog te weinig data", "⚪"
    if acwr < 0.8:
        return "Ruimte om op te bouwen", "🟦"
    if acwr <= 1.3:
        return "Gezonde opbouw", "🟢"
    if acwr <= 1.5:
        return "Oppassen: snelle stijging", "🟠"
    return "Risico op overbelasting", "🔴"


# ----------------------------------------------------------- efficiëntie --

def efficiency_factor(acts: pd.DataFrame) -> pd.DataFrame:
    """Efficiency factor per sessie: meters per minuut, per hartslag.

    Hogere EF bij vergelijkbare sessies = grotere aerobe basis. Alleen
    zinvol voor lopen en fietsen (zwem-hartslag is minder betrouwbaar).
    """
    df = acts[acts["sport"].isin(["running", "cycling"])].dropna(
        subset=["avg_speed_ms", "avg_hr"]).copy()
    df["ef"] = (df["avg_speed_ms"] * 60) / df["avg_hr"]
    return df[["start_time", "sport", "ef", "avg_hr", "duration_s"]].sort_values("start_time")


def decoupling(conn: sqlite3.Connection, acts: pd.DataFrame,
               min_duration_s: int = 1800) -> pd.DataFrame:
    """HR-decoupling per sessie: efficiëntieverlies in de tweede helft.

    We splitsen de sessie op het tijdsmidden en vergelijken snelheid/HR van
    beide helften: positieve waarden betekenen dat de hartslag relatief
    oploopt (cardiac drift). Onder de ~5% bij een duurtraining duidt op een
    goede aerobe basis. Alleen voor loop-/fietssessies vanaf 30 minuten.
    """
    rows = []
    kandidaten = acts[
        acts["sport"].isin(["running", "cycling"])
        & (acts["duration_s"] >= min_duration_s)
    ]
    for _, act in kandidaten.iterrows():
        rec = load_records(conn, act["activity_key"])
        rec = rec.dropna(subset=["heart_rate", "speed_ms"])
        rec = rec[rec["speed_ms"] > 0.5]  # stilstand eruit
        if len(rec) < 600:
            continue
        midden = rec["timestamp"].iloc[0] + (rec["timestamp"].iloc[-1] - rec["timestamp"].iloc[0]) / 2
        h1 = rec[rec["timestamp"] <= midden]
        h2 = rec[rec["timestamp"] > midden]
        ef1 = h1["speed_ms"].mean() / h1["heart_rate"].mean()
        ef2 = h2["speed_ms"].mean() / h2["heart_rate"].mean()
        rows.append({
            "start_time": act["start_time"],
            "sport": act["sport"],
            "decoupling_pct": (ef1 - ef2) / ef1 * 100,
            "duur_s": act["duration_s"],
        })
    return pd.DataFrame(rows).sort_values("start_time") if rows else pd.DataFrame()


# ------------------------------------------------------ racevoorspelling --

def race_prediction(acts: pd.DataFrame) -> dict:
    """Ruwe voorspelling van de standaard/olympische racetijden (1,5 / 40 / 10 km).

    Geeft per onderdeel seconden (of None bij te weinig data) plus een
    totaal inclusief een wisselbuffer. Bewust simpel gehouden: lopen via
    Riegel-schaling vanaf de beste recente loop, fietsen en zwemmen via
    het recente sessiegemiddelde.
    """
    pred: dict[str, float | None] = {"zwem_1500": None, "fiets_40k": None, "loop_10k": None}

    runs = acts[(acts["sport"] == "running") & (acts["distance_m"] >= 3000)].dropna(
        subset=["avg_speed_ms"])
    if not runs.empty:
        beste = runs.loc[runs["avg_speed_ms"].idxmax()]
        t1, d1 = beste["duration_s"], beste["distance_m"]
        pred["loop_10k"] = t1 * (10000 / d1) ** RIEGEL

    rides = acts[(acts["sport"] == "cycling") & (acts["distance_m"] >= 15000)].dropna(
        subset=["avg_speed_ms"])
    if not rides.empty:
        snelste = rides["avg_speed_ms"].max()
        pred["fiets_40k"] = 40000 / snelste

    swims = acts[acts["sport"] == "swimming"].dropna(subset=["avg_speed_ms"])
    if not swims.empty:
        # Meest recente sessie: het zwemniveau verandert nu het snelst.
        recent = swims.sort_values("start_time").iloc[-1]
        pred["zwem_1500"] = 1500 / recent["avg_speed_ms"]

    pred["wissels"] = 5 * 60  # ruwe buffer voor T1 + T2 (iets ruimer op olympische afstand)
    delen = [pred["zwem_1500"], pred["fiets_40k"], pred["loop_10k"]]
    pred["totaal"] = sum(delen) + pred["wissels"] if all(d is not None for d in delen) else None
    return pred


def readiness(acts: pd.DataFrame) -> list[tuple[str, str]]:
    """Gereedheid per discipline als (emoji, tekst), op basis van racevolume."""
    out = []
    swims = acts[acts["sport"] == "swimming"]
    langste_zwem = swims["distance_m"].max() if not swims.empty else 0
    if langste_zwem >= 1500:
        out.append(("✅", f"Zwemmen: langste sessie {langste_zwem:.0f} m — racevolume gehaald."))
    elif langste_zwem > 0:
        out.append(("⚠️", f"Zwemmen: langste sessie {langste_zwem:.0f} m van de 1500 m — "
                          "bouw rustig uit, de crawlcursus gaat hierbij helpen."))
    else:
        out.append(("⚠️", "Zwemmen: nog geen sessies."))

    rides = acts[acts["sport"] == "cycling"]
    langste_rit = rides["distance_m"].max() if not rides.empty else 0
    out.append(("✅" if langste_rit >= 40000 else "⚠️",
                f"Fietsen: langste rit {langste_rit / 1000:.0f} km "
                f"({'ruim boven' if langste_rit >= 40000 else 'onder'} de 40 km van de race)."))

    runs = acts[acts["sport"] == "running"]
    langste_loop = runs["distance_m"].max() if not runs.empty else 0
    out.append(("✅" if langste_loop >= 10000 else "⚠️",
                f"Lopen: langste loop {langste_loop / 1000:.1f} km "
                f"({'ruim boven' if langste_loop >= 10000 else 'onder'} de 10 km van de race)."))
    return out


# ------------------------------------------------- records & zwemprogressie --

def _fastest_window(records: pd.DataFrame, target_m: float) -> float | None:
    """Snelste tijd (s) waarin binnen één sessie ``target_m`` meter is afgelegd."""
    df = records.dropna(subset=["distance_m"]).sort_values("timestamp")
    if df.empty or df["distance_m"].iloc[-1] - df["distance_m"].iloc[0] < target_m:
        return None
    # Via total_seconds(), niet via astype(int64): de interne resolutie van
    # timestamps verschilt per pandas-versie (ns of µs).
    tijd = (df["timestamp"] - df["timestamp"].iloc[0]).dt.total_seconds().to_numpy()
    afstand = df["distance_m"].to_numpy()

    beste = None
    j = 0
    for i in range(len(afstand)):
        while j < len(afstand) and afstand[j] - afstand[i] < target_m:
            j += 1
        if j == len(afstand):
            break
        duur = tijd[j] - tijd[i]
        if beste is None or duur < beste:
            beste = duur
    return beste


def personal_records(conn: sqlite3.Connection, acts: pd.DataFrame) -> pd.DataFrame:
    """Persoonlijke records over alle geïmporteerde sessies."""
    records = []

    def voeg_toe(onderdeel, waarde, start_time):
        records.append({"Onderdeel": onderdeel, "Record": waarde,
                        "Datum": f"{start_time:%d-%m-%Y}"})

    runs = acts[acts["sport"] == "running"]
    if not runs.empty:
        langste = runs.loc[runs["distance_m"].idxmax()]
        voeg_toe("Langste loop", f"{langste['distance_m'] / 1000:.2f} km", langste["start_time"])
        beste_5k, beste_5k_act = None, None
        for _, act in runs.iterrows():
            t = _fastest_window(load_records(conn, act["activity_key"]), 5000)
            if t is not None and (beste_5k is None or t < beste_5k):
                beste_5k, beste_5k_act = t, act
        if beste_5k is not None:
            m, s = divmod(int(beste_5k), 60)
            voeg_toe("Snelste 5 km (binnen een loop)", f"{m}:{s:02d}",
                     beste_5k_act["start_time"])

    rides = acts[acts["sport"] == "cycling"]
    if not rides.empty:
        langste = rides.loc[rides["distance_m"].idxmax()]
        voeg_toe("Langste rit", f"{langste['distance_m'] / 1000:.1f} km", langste["start_time"])
        serieus = rides[rides["distance_m"] >= 15000]
        if not serieus.empty:
            snelste = serieus.loc[serieus["avg_speed_ms"].idxmax()]
            voeg_toe("Snelste rit (≥15 km)", f"{snelste['avg_speed_ms'] * 3.6:.1f} km/h",
                     snelste["start_time"])

    swims = acts[acts["sport"] == "swimming"]
    if not swims.empty:
        langste = swims.loc[swims["distance_m"].idxmax()]
        voeg_toe("Langste zwemsessie", f"{langste['distance_m']:.0f} m", langste["start_time"])
        met_tempo = swims.dropna(subset=["avg_speed_ms"])
        if not met_tempo.empty:
            snelste = met_tempo.loc[met_tempo["avg_speed_ms"].idxmax()]
            sec = 100 / snelste["avg_speed_ms"]
            voeg_toe("Snelste 100 m (sessiegemiddelde)",
                     f"{int(sec // 60)}:{int(sec % 60):02d} /100m", snelste["start_time"])

    return pd.DataFrame(records)


def progress_summary_text(conn: sqlite3.Connection, acts: pd.DataFrame) -> str:
    """Alle voortgangsstatistieken als platte tekst, voor in LLM-prompts."""
    delen = []

    curves = load_curves(acts)
    laatste = curves.iloc[-1]
    acwr = f"{laatste['acwr']:.2f}" if not np.isnan(laatste["acwr"]) else "n.v.t."
    delen.append(
        f"Belasting vandaag: fitheid (CTL) {laatste['ctl']:.1f}, "
        f"vermoeidheid (ATL) {laatste['atl']:.1f}, ACWR {acwr}."
    )

    ef = efficiency_factor(acts)
    if not ef.empty:
        regels = [f"  - {r['start_time']:%d-%m} {r['sport']}: EF {r['ef']:.2f} "
                  f"(gem. HR {r['avg_hr']:.0f})" for _, r in ef.iterrows()]
        delen.append("Efficiency factor per sessie (m/min per hartslag):\n" + "\n".join(regels))

    dec = decoupling(conn, acts)
    if not dec.empty:
        regels = [f"  - {r['start_time']:%d-%m} {r['sport']}: {r['decoupling_pct']:+.1f}%"
                  for _, r in dec.iterrows()]
        delen.append("HR-decoupling per sessie (richtwaarde: < 5% bij duurtraining):\n"
                     + "\n".join(regels))

    prs = personal_records(conn, acts)
    if not prs.empty:
        regels = [f"  - {r['Onderdeel']}: {r['Record']} ({r['Datum']})" for _, r in prs.iterrows()]
        delen.append("Persoonlijke records:\n" + "\n".join(regels))

    aandeel, swolf = swim_progression(conn, acts)
    if not aandeel.empty:
        regels = [f"  - {r['start_time']:%d-%m}: {r['crawl_pct']:.0f}% borstcrawl"
                  for _, r in aandeel.iterrows()]
        delen.append("Crawl-aandeel per zwemsessie:\n" + "\n".join(regels))
    if not swolf.empty:
        regels = [f"  - {r['start_time']:%d-%m} {r['slag']}: SWOLF {r['swolf']:.0f}"
                  for _, r in swolf.iterrows()]
        delen.append("SWOLF per slagtype:\n" + "\n".join(regels))

    return "\n\n".join(delen)


def swim_progression(conn: sqlite3.Connection, acts: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Zwemontwikkeling: (crawl-aandeel per sessie, SWOLF per slagtype per sessie)."""
    swims = acts[acts["sport"] == "swimming"].sort_values("start_time")
    aandeel_rows, swolf_rows = [], []
    for _, act in swims.iterrows():
        lengths = pd.read_sql_query(
            "SELECT swim_stroke, total_timer_time, total_strokes FROM lengths "
            "WHERE activity_key = ?", conn, params=(act["activity_key"],))
        if lengths.empty:
            continue
        aandeel_rows.append({
            "start_time": act["start_time"],
            "crawl_pct": (lengths["swim_stroke"] == "freestyle").mean() * 100,
        })
        per_slag = lengths.assign(
            swolf=lengths["total_timer_time"] + lengths["total_strokes"]
        ).groupby("swim_stroke")["swolf"].mean()
        for slag, swolf in per_slag.items():
            swolf_rows.append({"start_time": act["start_time"], "slag": slag, "swolf": swolf})
    return pd.DataFrame(aandeel_rows), pd.DataFrame(swolf_rows)
