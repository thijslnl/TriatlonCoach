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

from tricoach.formatting import derive_speed_ms
from tricoach.storage import load_lengths, load_records


def swim_speed_ms(conn: sqlite3.Connection, act: pd.Series) -> float | None:
    """Gemiddelde zwemsnelheid (m/s) uit afstand en de zuivere zwemtijd.

    De zwemtijd is de som van de actieve banen (``lengths.total_timer_time``),
    dus zonder rust aan de kant; ontbreekt baandata, dan geldt de totale
    timer-duur. Leunt bewust niet op ``avg_speed_ms``, dat op sommige sessies
    (zoals een samengevoegde zwemsessie) ontbreekt. None bij onbruikbare data.
    """
    lengths = pd.read_sql_query(
        "SELECT total_timer_time FROM lengths WHERE activity_key = ?",
        conn, params=(act["activity_key"],))
    actief = lengths["total_timer_time"].sum() if not lengths.empty else None
    zwemtijd = actief if actief and actief > 0 else act["duration_s"]
    return derive_speed_ms(act["distance_m"], zwemtijd)

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

    Geeft een DataFrame per dag: datum, trimp, ctl, atl, tsb, acwr. TSB
    ("vorm") is fitheid minus vermoeidheid: positief = fris, sterk negatief =
    diep in de vermoeidheid.
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
    out["tsb"] = out["ctl"] - out["atl"]
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

# Standaard/olympische afstand als terugval wanneer een race geen
# gestructureerde afstanden heeft (of er nog geen race is ingesteld).
STANDARD_RACE = {"swim_m": 1500.0, "bike_m": 40000.0, "run_m": 10000.0}


def race_distances(race: dict | None) -> dict:
    """De zwem-/fiets-/loopafstand (m) van een race uit config.yaml.

    Leest de gestructureerde velden ``swim_m``/``bike_m``/``run_m`` en valt
    per ontbrekend veld terug op de standaardafstand.
    """
    race = race or {}
    return {k: float(race.get(k) or v) for k, v in STANDARD_RACE.items()}


def race_prediction(conn: sqlite3.Connection, acts: pd.DataFrame,
                    race: dict | None = None) -> dict:
    """Ruwe voorspelling van de racetijden voor de ingestelde afstanden.

    Geeft per onderdeel seconden (of None bij te weinig data) plus een
    totaal inclusief een wisselbuffer. Bewust simpel gehouden: lopen via
    Riegel-schaling vanaf de beste recente loop, fietsen en zwemmen via
    het recente sessiegemiddelde. De zwemsnelheid komt uit afstand en zwemtijd
    (zie :func:`swim_speed_ms`), niet uit het soms ontbrekende avg_speed_ms.
    """
    afst = race_distances(race)
    pred: dict = {"zwem": None, "fiets": None, "loop": None, "afstanden": afst}

    runs = acts[(acts["sport"] == "running") & (acts["distance_m"] >= 3000)].dropna(
        subset=["avg_speed_ms"])
    if not runs.empty:
        beste = runs.loc[runs["avg_speed_ms"].idxmax()]
        t1, d1 = beste["duration_s"], beste["distance_m"]
        pred["loop"] = t1 * (afst["run_m"] / d1) ** RIEGEL

    rides = acts[(acts["sport"] == "cycling") & (acts["distance_m"] >= 15000)].dropna(
        subset=["avg_speed_ms"])
    if not rides.empty:
        snelste = rides["avg_speed_ms"].max()
        pred["fiets"] = afst["bike_m"] / snelste

    swims = acts[acts["sport"] == "swimming"]
    if not swims.empty:
        # Meest recente sessie: het zwemniveau verandert nu het snelst.
        recent = swims.sort_values("start_time").iloc[-1]
        speed = swim_speed_ms(conn, recent)
        if speed:
            pred["zwem"] = afst["swim_m"] / speed

    pred["wissels"] = 5 * 60  # ruwe buffer voor T1 + T2
    delen = [pred["zwem"], pred["fiets"], pred["loop"]]
    pred["totaal"] = sum(delen) + pred["wissels"] if all(d is not None for d in delen) else None
    return pred


def readiness(acts: pd.DataFrame, race: dict | None = None) -> list[tuple[str, str]]:
    """Gereedheid per discipline als (emoji, tekst), op basis van racevolume."""
    afst = race_distances(race)
    out = []
    swims = acts[acts["sport"] == "swimming"]
    langste_zwem = swims["distance_m"].max() if not swims.empty else 0
    if langste_zwem >= afst["swim_m"]:
        out.append(("✅", f"Zwemmen: langste sessie {langste_zwem:.0f} m — racevolume gehaald."))
    elif langste_zwem > 0:
        out.append(("⚠️", f"Zwemmen: langste sessie {langste_zwem:.0f} m van de "
                          f"{afst['swim_m']:.0f} m — bouw rustig uit."))
    else:
        out.append(("⚠️", "Zwemmen: nog geen sessies."))

    rides = acts[acts["sport"] == "cycling"]
    langste_rit = rides["distance_m"].max() if not rides.empty else 0
    out.append(("✅" if langste_rit >= afst["bike_m"] else "⚠️",
                f"Fietsen: langste rit {langste_rit / 1000:.0f} km "
                f"({'ruim boven' if langste_rit >= afst['bike_m'] else 'onder'} "
                f"de {afst['bike_m'] / 1000:.0f} km van de race)."))

    runs = acts[acts["sport"] == "running"]
    langste_loop = runs["distance_m"].max() if not runs.empty else 0
    out.append(("✅" if langste_loop >= afst["run_m"] else "⚠️",
                f"Lopen: langste loop {langste_loop / 1000:.1f} km "
                f"({'ruim boven' if langste_loop >= afst['run_m'] else 'onder'} "
                f"de {afst['run_m'] / 1000:.0f} km van de race)."))
    return out


def css_estimate(conn: sqlite3.Connection, acts: pd.DataFrame) -> dict | None:
    """Critical Swim Speed (CSS) geschat uit de trainingsdata.

    Zoekt over alle zwemsessies de snelste écht aaneengesloten crawlreeks
    van 400 m en van 200 m. Een reeks telt alleen als de banen direct op
    elkaar volgen: een andere slag of een rustpauze (> ~15 s gat tussen het
    einde van een baan en de start van de volgende) breekt hem. De tijd is
    kloktijd van start eerste tot einde laatste baan — keerpunten tellen dus
    mee, rust aan de kant niet (die breekt de reeks). CSS-tempo per 100 m =
    (t400 − t200) / 2 — het tempo dat je in theorie lang kunt volhouden en
    de basis voor zwemzones.

    Vereist de per-baan-starttijd (kolom ``start_time``, opgeslagen sinds
    juli 2026): zonder die tijden zijn rustpauzes tussen banen onzichtbaar en
    zou de schatting veel te snel uitvallen. Sessies zonder starttijden
    worden overgeslagen; geeft None zolang er geen bruikbare reeks is.
    """
    max_gat_s = 15.0
    swims = acts[acts["sport"] == "swimming"]
    beste: dict[int, float] = {}
    for _, act in swims.iterrows():
        pool = act.get("pool_length")
        if pool is None or pd.isna(pool) or pool <= 0:
            pool = 25.0
        lengths = load_lengths(conn, act["activity_key"])
        if lengths.empty or "start_time" not in lengths:
            continue
        lengths = lengths.dropna(subset=["start_time", "total_timer_time"])
        lengths = lengths[lengths["start_time"].astype(str).str.lower() != "none"]
        if lengths.empty:
            continue
        starts = pd.to_datetime(lengths["start_time"]).tolist()
        duren = lengths["total_timer_time"].astype(float).tolist()
        slagen = lengths["swim_stroke"].tolist()
        for afstand in (400, 200):
            n = int(round(afstand / pool))
            reeks: list[tuple[pd.Timestamp, pd.Timestamp]] = []  # (start, eind)
            for start, duur, slag in zip(starts, duren, slagen):
                eind = start + pd.Timedelta(seconds=duur)
                onderbroken = (
                    slag != "freestyle"
                    or (reeks and (start - reeks[-1][1]).total_seconds() > max_gat_s)
                )
                if onderbroken:
                    reeks = []
                if slag != "freestyle":
                    continue
                reeks.append((start, eind))
                if len(reeks) >= n:
                    venster = reeks[-n:]
                    kloktijd = (venster[-1][1] - venster[0][0]).total_seconds()
                    if afstand not in beste or kloktijd < beste[afstand]:
                        beste[afstand] = kloktijd
    if 400 not in beste or 200 not in beste:
        return None
    dt = beste[400] - beste[200]
    if dt <= 0:
        return None
    return {"t400": beste[400], "t200": beste[200], "css_per_100m": dt / 2}


def race_prediction_history(conn: sqlite3.Connection, acts: pd.DataFrame,
                            race: dict | None = None) -> pd.DataFrame:
    """De racevoorspelling per week, herberekend met alleen de sessies tot en
    met die week — zo zie je of de voorspelde racetijd de goede kant op gaat.

    Geeft per weekeinde (zondag): datum, zwem, fiets, loop en totaal in
    seconden (None waar toen nog te weinig data was).
    """
    if acts.empty:
        return pd.DataFrame()
    df = acts.sort_values("start_time")
    weken = pd.date_range(
        df["start_time"].min().normalize(), df["start_time"].max(),
        freq="W-SUN", tz=df["start_time"].iloc[0].tz,
    )
    weken = weken.append(pd.DatetimeIndex([df["start_time"].max()]))
    rows = []
    for w in weken:
        sub = df[df["start_time"] <= w]
        if sub.empty:
            continue
        p = race_prediction(conn, sub, race)
        rows.append({"datum": w.normalize(), "zwem": p["zwem"], "fiets": p["fiets"],
                     "loop": p["loop"], "totaal": p["totaal"]})
    out = pd.DataFrame(rows).drop_duplicates(subset="datum")
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
        # Tempo uit afstand en zwemtijd, zodat ook sessies zonder avg_speed_ms
        # meetellen; snelste = hoogste afgeleide snelheid.
        met_tempo = [(swim_speed_ms(conn, act), act) for _, act in swims.iterrows()]
        met_tempo = [(sp, act) for sp, act in met_tempo if sp]
        if met_tempo:
            snelste_speed, snelste = max(met_tempo, key=lambda p: p[0])
            sec = 100 / snelste_speed
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
