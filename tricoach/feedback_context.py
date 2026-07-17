"""Rijke context-assemblage voor de coaching-feedback na een upload.

De kwaliteit van de feedback staat of valt met de context die het model
meekrijgt — niet met het model alleen. Deze module bouwt die context uit
vijf bronnen:

1. **De huidige sessie, volledig**: kerncijfers (incl. calorieën en
   hoogtemeters), zoneverdeling, HR-drift (eerste vs tweede helft), en een
   sport-specifieke detailanalyse — km-splits bij lopen, hoogteprofiel bij
   fietsen, baan-voor-baan-tempo en slagritme bij zwemmen — plus de
   opmerking van de atleet en de externe omstandigheden (wind/temperatuur
   uit Open-Meteo).
2. **Vergelijkbare eerdere sessies**: dezelfde sport en vergelijkbare
   intensiteit, met tempo-bij-gelijke-hartslag als belangrijkste
   voortgangsmaat (lopen/fietsen) of afstand, tempo per 100 m en slagritme
   (zwemmen).
3. **Het atletenprofiel** (memory/doelen.md): zones, racedoelen en het
   bekende aandachtspunt (te hard trainen bij lopen/fietsen).
4. **De recente belasting** (laatste ~10 dagen) voor herstelbewust advies,
   en recente opmerkingen van de atleet (ziekte, vermoeidheid, materiaal).
5. **Continuïteit**: de vorige feedback en het vorige advies, zodat de
   coach kan terugkoppelen en toetsen of een advies is opgevolgd.

Alle historie wordt gefilterd op ``start_time < huidige sessie``, zodat de
context ook klopt wanneer oudere sessies achteraf (of in tests door elkaar)
worden geanalyseerd. Er gebeurt hier géén LLM- of netwerkcall; alles komt
uit SQLite en de memory-bestanden.
"""

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from tricoach import power as pwr
from tricoach.analysis import pace_at_hr
from tricoach.combos import combo_block
from tricoach.fit_parser import ParsedActivity
from tricoach.formatting import (
    derive_speed_ms,
    fmt_duration,
    fmt_pace_per_100m,
    fmt_pace_per_km,
    fmt_speed_kmh,
    local_time,
    sport_label,
)
from tricoach.removal import section_is_deleted
from tricoach.rundynamics import dynamics_block
from tricoach.schedule import schedule_as_text
from tricoach.storage import load_activities, load_lengths, load_records
from tricoach.swim import SWOLF_FILTER_LABEL, crawl_swolf
from tricoach.trainingslog import kerncijfers, zone_regel
from tricoach.zones import intensity_category, zone_bounds

# Hoeveel vergelijkbare eerdere sessies er maximaal meegaan.
MAX_SIMILAR = 5
# Venster (dagen) voor het belastingsoverzicht en voor recente opmerkingen.
LOAD_DAYS = 10
NOTES_DAYS = 14
# Het vorige weekadvies gaat ingekort mee (het volledige advies is lang en
# het weekschema-blok dekt de planning al).
MAX_ADVICE_CHARS = 1500


# Lokale weergavetijd komt uit de gedeelde helper; hier alleen een korte alias.
_local = local_time


def _note(row: pd.Series) -> str | None:
    """De opmerking van de atleet uit een activities-rij, of None.

    Pandas maakt van een ontbrekende opmerking NaN (een float, en truthy!),
    dus een kale truthiness-check zou "nan" in de context laten lekken.
    """
    note = row.get("user_note")
    if note is None or (isinstance(note, float) and pd.isna(note)):
        return None
    return str(note)


# ------------------------------------------------------------ huidige sessie --

def hr_drift_values(records: pd.DataFrame) -> tuple[float, float] | None:
    """Gemiddelde hartslag van de eerste en tweede helft, als getallenpaar.

    De belangrijkste vermoeidheids-/herstelmaat bij lopen en fietsen: loopt
    de hartslag in de tweede helft op bij gelijk werk, dan sluipt er
    vermoeidheid in. Geeft None bij te weinig data (< ~4 minuten).
    """
    if records.empty or "heart_rate" not in records:
        return None
    df = records.dropna(subset=["heart_rate"]).sort_values("timestamp")
    if len(df) < 240:
        return None
    mid = df["timestamp"].iloc[0] + (df["timestamp"].iloc[-1] - df["timestamp"].iloc[0]) / 2
    first = df.loc[df["timestamp"] <= mid, "heart_rate"].mean()
    second = df.loc[df["timestamp"] > mid, "heart_rate"].mean()
    return float(first), float(second)


def hr_drift(records: pd.DataFrame) -> str | None:
    """Hartslag eerste helft vs tweede helft als tekstregel (LLM-context)."""
    helften = hr_drift_values(records)
    if helften is None:
        return None
    first, second = helften
    return (
        f"eerste helft gem. HR {first:.0f}, tweede helft gem. HR {second:.0f} "
        f"({second - first:+.0f} slagen)"
    )


def run_splits_df(records: pd.DataFrame, max_km: int = 25) -> pd.DataFrame:
    """Kilometersplits uit de seconde-data: km, duur_s en gem_hr per split.

    Leeg DataFrame als er geen volledige kilometer in de sessie zit.
    """
    if records.empty or "distance_m" not in records:
        return pd.DataFrame()
    df = records.dropna(subset=["distance_m", "timestamp"]).sort_values("timestamp")
    if df.empty or df["distance_m"].max() < 1000:
        return pd.DataFrame()

    dist = df["distance_m"].to_numpy(dtype=float)
    times = df["timestamp"].to_numpy()
    hrs = df["heart_rate"].to_numpy(dtype=float) if "heart_rate" in df else None

    rows, prev = [], 0
    for k in range(1, min(int(dist[-1] // 1000), max_km) + 1):
        idx = int(np.searchsorted(dist, k * 1000))
        if idx >= len(dist):
            break
        seg_s = float((times[idx] - times[prev]) / np.timedelta64(1, "s"))
        hr = None
        if hrs is not None:
            seg_hr = pd.Series(hrs[prev:idx + 1]).dropna()
            if not seg_hr.empty:
                hr = float(seg_hr.mean())
        rows.append({"km": k, "duur_s": seg_s, "gem_hr": hr})
        prev = idx
    return pd.DataFrame(rows)


def _run_splits(records: pd.DataFrame, max_km: int = 25) -> str | None:
    """Tempo-splits per kilometer (met HR per split), als tekst (LLM-context)."""
    splits = run_splits_df(records, max_km)
    if splits.empty:
        return None
    regels = []
    for _, r in splits.iterrows():
        regel = f"- km {int(r['km'])}: {fmt_duration(r['duur_s'])}"
        if r["gem_hr"] is not None and not pd.isna(r["gem_hr"]):
            regel += f" bij HR gem {r['gem_hr']:.0f}"
        regels.append(regel)
    return "\n".join(regels)


def _bike_detail(records: pd.DataFrame, summary: dict) -> str | None:
    """Hoogte- en tempoverloop van een fietsrit, compact."""
    regels = []
    if summary.get("total_ascent"):
        regels.append(f"- Hoogtemeters: {summary['total_ascent']:.0f} m klimmen")
    if not records.empty and "altitude_m" in records:
        alt = records["altitude_m"].dropna()
        if not alt.empty:
            regels.append(f"- Hoogteprofiel: van {alt.min():.0f} tot {alt.max():.0f} m")
    if not records.empty and "speed_ms" in records:
        df = records.dropna(subset=["speed_ms"]).sort_values("timestamp")
        df = df[df["speed_ms"] > 0.5]  # stilstand eruit
        if len(df) > 240:
            half = len(df) // 2
            v1, v2 = df["speed_ms"].iloc[:half].mean(), df["speed_ms"].iloc[half:].mean()
            regels.append(
                f"- Snelheid per helft: {fmt_speed_kmh(v1)} → {fmt_speed_kmh(v2)}"
            )
    return "\n".join(regels) if regels else None


def _bike_power_detail(act: ParsedActivity, ftp: float | None) -> str | None:
    """Vermogens- en cadansanalyse van een fietsrit, voor de LLM-context.

    Alles komt uit de seconde-data van de zojuist geparste sessie: NP, EF,
    Pw:Hr-decoupling, de pacing-check (eerste kwartier vs rest — dé test op
    te hard starten) en de cadans incl./excl. nul. Tijd-in-vermogenszones
    alleen als de FTP bekend is. None voor ritten zonder powerdata (oudere
    sessies), dan valt de feedback vanzelf terug op hartslag en snelheid.
    """
    records = act.records
    if records.empty or "power" not in records or not records["power"].notna().any():
        return None
    s = act.summary
    indoor = act.is_indoor
    bron = pwr.power_source(act.devices, indoor)
    np_watt = pwr.normalized_power(records) or s.get("normalized_power")
    avg_watt = s.get("avg_power") or float(records["power"].dropna().mean())

    regel_watt = f"- Vermogen: gem. {avg_watt:.0f} W"
    if np_watt:
        regel_watt += f", NP {np_watt:.0f} W"
    if s.get("max_power"):
        regel_watt += f", max {s['max_power']:.0f} W"
    regel_watt += f" — bron: {bron} ({'indoor' if indoor else 'buiten'})"
    regels = [regel_watt]

    ef = pwr.efficiency_factor(np_watt, s.get("avg_heart_rate"))
    if ef:
        regels.append(
            f"- Efficiency factor (NP/gem. HR): {ef:.2f} W per hartslag — "
            "windonafhankelijke aerobe maat; vergelijk alleen binnen dezelfde bron"
        )

    dec = pwr.power_decoupling(records)
    if dec is not None:
        regels.append(
            f"- Aerobic decoupling (Pw:Hr): {dec:+.1f}% "
            "(EF eerste vs tweede helft; onder ~5% = goed ontwikkelde aerobe basis)"
        )

    pacing = pwr.pacing_split(records)
    if pacing:
        eerste, rest = pacing
        delta = (eerste - rest) / rest * 100 if rest else 0.0
        regels.append(
            f"- Pacing: eerste {pwr.PACING_FIRST_MIN} min gem. {eerste:.0f} W, "
            f"rest van de rit {rest:.0f} W ({delta:+.0f}%) — bekende valkuil van "
            "de atleet is te hard starten; benoem het als de start duidelijk "
            "feller was dan de rest"
        )

    cad = pwr.cadence_stats(records)
    if cad["excl0"] is not None:
        regel = f"- Cadans: gem. {cad['excl0']:.0f} rpm (excl. nul"
        if cad["incl0"] is not None:
            regel += f"; incl. freewheelen {cad['incl0']:.0f}"
        regel += ")"
        if cad["max"] is not None:
            regel += f", max {cad['max']:.0f} rpm"
        regels.append(regel)

    if ftp:
        tip = pwr.time_in_power_zones(records, ftp)
        totaal = sum(tip.values()) or 1
        verdeling = " · ".join(
            f"{z} {fmt_duration(v)} ({v / totaal * 100:.0f}%)"
            for z, v in tip.items() if v > 0
        )
        regels.append(f"- Tijd in vermogenszones (FTP {ftp:.0f} W, Coggan): {verdeling}")
    else:
        regels.append(
            "- Vermogenszones: nog niet beschikbaar (FTP onbekend — geen "
            "zone-oordeel over het vermogen geven)"
        )
    return "\n".join(regels)


def _swim_detail(lengths: pd.DataFrame, pool_length: float | None,
                 distance_m: float | None) -> str | None:
    """Baan-voor-baan-analyse van een zwemsessie: tempo, slagritme, consistentie.

    Het slagritme (slagen per baan) en de consistentie ervan zijn de
    belangrijkste techniekmaat voor een beginnende zwemmer; de laatste banen
    tonen of de techniek onder vermoeidheid overeind blijft.
    """
    if lengths.empty:
        return None
    t = lengths["total_timer_time"].astype(float)
    s = lengths["total_strokes"].dropna().astype(float)
    n = len(lengths)
    regels = [f"- {n} actieve banen, zuivere zwemtijd {fmt_duration(t.sum())}"]

    if distance_m:
        speed = derive_speed_ms(distance_m, t.sum())
        regels.append(f"- Tempo op zuivere zwemtijd: {fmt_pace_per_100m(speed)}")
    regels.append(
        f"- Tijd per baan: gem. {t.mean():.0f}s, snelste {t.min():.0f}s, "
        f"langzaamste {t.max():.0f}s"
    )

    kwart = max(n // 4, 1)
    t_begin, t_eind = t.iloc[:kwart].mean(), t.iloc[-kwart:].mean()
    regels.append(
        f"- Eerste kwart vs laatste kwart: {t_begin:.0f}s → {t_eind:.0f}s per baan"
    )

    if not s.empty:
        binnen = ((s >= s.mean() - 2) & (s <= s.mean() + 2)).mean() * 100
        regels.append(
            f"- Slagritme: gem. {s.mean():.1f} slagen/baan (min {s.min():.0f}, "
            f"max {s.max():.0f}); {binnen:.0f}% van de banen binnen ±2 van het gemiddelde"
        )
        s_begin, s_eind = s.iloc[:kwart].mean(), s.iloc[-kwart:].mean()
        regels.append(
            f"- Slagen per baan, eerste vs laatste kwart: {s_begin:.1f} → {s_eind:.1f}"
        )

    if "swim_stroke" in lengths:
        verdeling = lengths["swim_stroke"].astype(str).value_counts(normalize=True)
        tekst = ", ".join(f"{v * 100:.0f}% {k}" for k, v in verdeling.items())
        regels.append(
            f"- Slagverdeling volgens het horloge: {tekst} "
            "(let op: slaglabels kunnen fout zijn)"
        )
    return "\n".join(regels)


def session_block(
    act: ParsedActivity,
    tiz: dict[str, int],
    observation: str | None,
    user_note: str | None,
    wind: "object | None",
    training_label: str | None = None,
    ftp: float | None = None,
) -> str:
    """Het volledige beeld van de zojuist voltooide sessie.

    ``training_label`` is het sessielabel van de atleet; bij een
    techniek-/cadanssessie gaat de uitleg mee dat een hogere hartslag daar
    verwacht en oké is, zodat de coach de sessie niet als "te hard getraind"
    beoordeelt. ``ftp`` (optioneel) maakt de tijd-in-vermogenszones in het
    fietsdetail mogelijk.
    """
    s = act.summary
    total = sum(tiz.values()) or 1
    shares = " · ".join(f"{z} {v / total * 100:.0f}%" for z, v in tiz.items() if v > 0)

    regels = [
        f"- Sport: {sport_label(act.sport)}",
        f"- Datum: {_local(act.start_time):%A %d-%m-%Y %H:%M} (lokale tijd)",
        f"- Kerncijfers: {kerncijfers(act)}",
    ]
    if s.get("total_calories"):
        regels.append(f"- Calorieën: {s['total_calories']:.0f} kcal")

    zone_noot = ""
    if act.sport == "swimming":
        zone_noot = (
            " (alleen als losse context — polshartslag onder water is "
            "onbetrouwbaar; NIET als beoordelingsmaat gebruiken)"
        )
    regels.append(f"- Tijd in zones: {zone_regel(tiz)} — verdeling: {shares}{zone_noot}")

    if act.sport in ("running", "cycling"):
        drift = hr_drift(act.records)
        if drift:
            regels.append(f"- HR-drift: {drift}")

    detail = None
    if act.sport == "running":
        detail = _run_splits(act.records)
        kop = "Tempo-splits per kilometer"
    elif act.sport == "cycling":
        detail = _bike_detail(act.records, s)
        kop = "Rit-detail"
    elif act.sport == "swimming":
        detail = _swim_detail(act.lengths, s.get("pool_length"), act.distance_m)
        kop = "Baan-voor-baan-analyse"
    if detail:
        regels.append(f"- {kop}:\n{detail}")

    if act.sport == "cycling":
        power_detail = _bike_power_detail(act, ftp)
        if power_detail:
            regels.append(
                "- Vermogen & cadans (primaire intensiteitsmaat voor deze rit):\n"
                + power_detail
            )

    if act.sport == "cycling" and act.is_indoor:
        regels.append(
            "- Omstandigheden: indoorsessie (trainer/Zwift) — wind, verkeer en "
            "hoogteprofiel zijn niet van toepassing; beoordeel puur op "
            "vermogen, duur en zones. Een blokkerig vermogensprofiel duidt op "
            "een gestructureerde (interval)training: toets die aan de "
            "bedoeling, niet aan rustig-blijven."
        )
    else:
        regels.append(
            f"- Omstandigheden (Open-Meteo): "
            f"{wind.as_text() if wind is not None else '(geen weerdata — binnen of geen GPS)'}"
        )
    if observation:
        regels.append(f"- Beschrijvende observatie (lokaal model): {observation}")
    regels.append(f"- Opmerking van de atleet bij de upload: {user_note or '(geen)'}")
    if training_label:
        regel = f"- Sessielabel van de atleet: {training_label}"
        if "techniek" in training_label.lower() or "cadans" in training_label.lower():
            regel += (
                " — bewuste techniek-/cadanssessie. Een hogere hartslag dan "
                "normaal is hierbij VERWACHT (onwennig bewegingspatroon, went "
                "vanzelf) en is géén te hard trainen of slechte zone "
                "2-uitvoering. Beoordeel deze sessie op het techniekdoel, "
                "niet op zone 2-discipline."
            )
        regels.append(regel)
    return "\n".join(regels)


# ------------------------------------------------------- vergelijkbare historie --

def _prev_activities(conn: sqlite3.Connection, act: ParsedActivity) -> pd.DataFrame:
    """Alle sessies van vóór de huidige, nieuwste eerst."""
    acts = load_activities(conn)
    if acts.empty:
        return acts
    start = pd.Timestamp(act.start_time)
    if start.tzinfo is None:
        start = start.tz_localize("UTC")
    return acts[acts["start_time"] < start]


def _row_line(row: pd.Series, sport: str) -> str:
    """Eén eerdere sessie als vergelijkbare tekstregel."""
    datum = _local(row["start_time"]).strftime("%d-%m-%Y")
    delen = [f"duur {fmt_duration(row['duration_s'])}"]
    speed = derive_speed_ms(row["distance_m"], row["duration_s"])
    if sport == "running":
        delen.append(f"{row['distance_m'] / 1000:.2f} km op {fmt_pace_per_km(speed)}")
    else:
        delen.append(f"{row['distance_m'] / 1000:.1f} km, {fmt_speed_kmh(speed)}")
        if row.get("total_ascent") and not pd.isna(row["total_ascent"]):
            delen.append(f"{row['total_ascent']:.0f} hm")
        # Vermogen als de sessie het heeft (sinds juli 2026): NP + EF + bron,
        # zodat de coach binnen dezelfde bron kan vergelijken.
        np_watt = row.get("np_power")
        if np_watt is not None and not pd.isna(np_watt):
            delen.append(f"NP {np_watt:.0f} W")
            ef = row.get("ef_watt")
            if ef is not None and not pd.isna(ef):
                delen.append(f"EF {ef:.2f}")
            bron = row.get("power_source")
            if bron and not (isinstance(bron, float) and pd.isna(bron)):
                delen.append(str(bron))
    if not pd.isna(row.get("avg_hr")):
        delen.append(f"HR gem {row['avg_hr']:.0f}")
    if not pd.isna(row.get("pct_in_zone2")):
        delen.append(f"{row['pct_in_zone2']:.0f}% Z2")
    if not pd.isna(row.get("wind_speed")):
        delen.append(f"wind {row['wind_speed']:.0f} km/h")
    temp = row.get("temperature_c")
    if temp is not None and not pd.isna(temp):
        delen.append(f"{temp:.0f} °C")
    regel = f"- {datum}: " + ", ".join(delen)
    if _note(row):
        regel += f" — opmerking atleet: \"{_note(row)}\""
    return regel


def _swim_row_line(conn: sqlite3.Connection, row: pd.Series) -> str:
    """Eén eerdere zwemsessie: afstand, tempo per 100 m, slagritme, SWOLF."""
    datum = _local(row["start_time"]).strftime("%d-%m-%Y")
    delen = [f"{row['distance_m']:.0f} m in {fmt_duration(row['duration_s'])}"]

    lengths = load_lengths(conn, row["activity_key"])
    if not lengths.empty:
        t = lengths["total_timer_time"].astype(float)
        speed = derive_speed_ms(row["distance_m"], t.sum())
        delen.append(f"tempo {fmt_pace_per_100m(speed)} (zuivere zwemtijd)")
        s = lengths["total_strokes"].dropna().astype(float)
        if not s.empty:
            delen.append(
                f"slagritme gem. {s.mean():.1f} slagen/baan "
                f"(min {s.min():.0f}, max {s.max():.0f})"
            )
        swolf = crawl_swolf(lengths, row.get("pool_length"), row.get("distance_m"))
        if swolf is not None:
            delen.append(f"SWOLF {swolf:.0f} ({SWOLF_FILTER_LABEL})")
    regel = f"- {datum}: " + ", ".join(delen)
    if _note(row):
        regel += f" — opmerking atleet: \"{_note(row)}\""
    return regel


def similar_sessions_block(conn: sqlite3.Connection, act: ParsedActivity,
                           tiz: dict[str, int]) -> str:
    """De laatste vergelijkbare sessies (zelfde sport, vergelijkbare intensiteit).

    Voor lopen/fietsen wordt eerst op dezelfde intensiteitscategorie
    (rustig/intensief) gefilterd; zijn dat er te weinig, dan vallen we terug
    op alle sessies van die sport (met vermelding). Voor zwemmen tellen alle
    eerdere zwemsessies mee — daar zijn afstand, tempo en slagritme de
    vergelijkingsmaten, niet de hartslag.
    """
    prev = _prev_activities(conn, act)
    prev = prev[prev["sport"] == act.sport] if not prev.empty else prev
    if prev.empty:
        return "(nog geen eerdere sessies van deze sport)"

    noot = ""
    if act.sport in ("running", "cycling"):
        cat = intensity_category(
            tiz.get("Z1", 0), tiz.get("Z2", 0), tiz.get("Z3", 0),
            tiz.get("Z4", 0), tiz.get("Z5", 0),
        )
        if cat is not None:
            zelfde = prev[prev.apply(
                lambda r: intensity_category(
                    r["z1_s"], r["z2_s"], r["z3_s"], r["z4_s"], r["z5_s"]) == cat,
                axis=1,
            )]
            if len(zelfde) >= 2:
                prev = zelfde
                noot = f" (zelfde intensiteitscategorie: {cat})"
            else:
                noot = " (te weinig sessies van gelijke intensiteit; alle recente getoond)"

    prev = prev.head(MAX_SIMILAR).iloc[::-1]  # oudste eerst, leest als een trend
    if act.sport == "swimming":
        regels = [_swim_row_line(conn, row) for _, row in prev.iterrows()]
    else:
        regels = [_row_line(row, act.sport) for _, row in prev.iterrows()]
    return f"Laatste vergelijkbare sessies{noot}:\n" + "\n".join(regels)


def pace_history_block(conn: sqlite3.Connection, act: ParsedActivity,
                       z2: tuple[int, int]) -> str | None:
    """Tempo-bij-gelijke-hartslag (zone 2) als trendlijst — lopen/fietsen.

    Dit is de belangrijkste voortgangsmaat: sneller bij dezelfde hartslag =
    aerobe groei. Voor zwemmen niet zinvol (onbetrouwbare pols-HR), dan None.
    """
    if act.sport not in ("running", "cycling"):
        return None
    acts = load_activities(conn)
    trend = pace_at_hr(conn, acts, act.sport, z2)
    if trend.empty:
        return "(nog geen eerdere sessies met genoeg tijd in zone 2)"

    start = pd.Timestamp(act.start_time)
    if start.tzinfo is None:
        start = start.tz_localize("UTC")
    trend = trend[trend["start_time"] < start].tail(6)
    if trend.empty:
        return "(nog geen eerdere sessies met genoeg tijd in zone 2)"

    fmt = fmt_pace_per_km if act.sport == "running" else fmt_speed_kmh
    return "\n".join(
        f"- {_local(r['start_time']):%d-%m-%Y}: {fmt(r['speed_ms'])} "
        f"bij HR {z2[0]}-{z2[1]}"
        for _, r in trend.iterrows()
    )


def power_history_block(conn: sqlite3.Connection, act: ParsedActivity) -> str | None:
    """EF-historie (NP per hartslag) van eerdere fietssessies, per bron.

    De opvolger van tempo-bij-gelijke-hartslag voor het fietsen: vermogen is
    windonafhankelijk, dus een stijgende EF (of dezelfde power bij een lagere
    HR) is zuivere aerobe vooruitgang. Per vermogensbron gegroepeerd —
    pedalen en trainer meten net iets anders. None voor andere sporten of
    zolang er geen eerdere powersessies zijn.
    """
    if act.sport != "cycling":
        return None
    prev = _prev_activities(conn, act)
    if prev.empty or "ef_watt" not in prev:
        return None
    prev = prev[(prev["sport"] == "cycling") & prev["ef_watt"].notna()]
    if prev.empty:
        return None

    regels = []
    for bron, groep in prev.groupby(prev["power_source"].fillna("onbekende bron")):
        laatste = groep.sort_values("start_time").tail(6)
        regels.append(f"Bron: {bron}")
        regels.extend(
            f"- {_local(r['start_time']):%d-%m-%Y}: EF {r['ef_watt']:.2f} "
            f"(NP {r['np_power']:.0f} W bij HR gem {r['avg_hr']:.0f})"
            for _, r in laatste.iterrows()
        )
    regels.append(
        "Leesinstructie: stijgende EF (of gelijke power bij lagere HR) over "
        "weken = aerobe vooruitgang. Vergelijk alleen binnen dezelfde bron; "
        "benoem het expliciet als een vergelijking bronnen mengt."
    )
    return "\n".join(regels)


# ------------------------------------------------- belasting & recente context --

def recent_load_block(conn: sqlite3.Connection, act: ParsedActivity,
                      days: int = LOAD_DAYS) -> str:
    """Alle sessies van de laatste ~10 dagen, voor herstelbewust advies."""
    prev = _prev_activities(conn, act)
    if prev.empty:
        return "(geen eerdere sessies)"
    start = pd.Timestamp(act.start_time)
    if start.tzinfo is None:
        start = start.tz_localize("UTC")
    window = prev[prev["start_time"] >= start - pd.Timedelta(days=days)]
    if window.empty:
        return f"(geen sessies in de {days} dagen hiervoor)"

    regels = []
    for _, row in window.iloc[::-1].iterrows():  # oudste eerst
        cat = intensity_category(
            row["z1_s"], row["z2_s"], row["z3_s"], row["z4_s"], row["z5_s"])
        regels.append(
            f"- {_local(row['start_time']):%a %d-%m}: {sport_label(row['sport'])}, "
            f"{fmt_duration(row['duration_s'])}"
            + (f", {cat}" if cat else "")
        )
    return "\n".join(regels)


def recent_notes_block(conn: sqlite3.Connection, act: ParsedActivity,
                       days: int = NOTES_DAYS) -> str | None:
    """Vrije opmerkingen van de atleet uit recente sessies (ziekte, moe, materiaal).

    Deze kleuren de interpretatie: een verhoogde hartslag vlak na "buikgriep"
    is geen vormverlies. None als er geen opmerkingen in het venster staan.
    """
    prev = _prev_activities(conn, act)
    if prev.empty:
        return None
    start = pd.Timestamp(act.start_time)
    if start.tzinfo is None:
        start = start.tz_localize("UTC")
    window = prev[
        (prev["start_time"] >= start - pd.Timedelta(days=days))
        & prev["user_note"].notna()
    ]
    if window.empty:
        return None
    return "\n".join(
        f"- {_local(row['start_time']):%d-%m} ({sport_label(row['sport'])}): "
        f"\"{row['user_note']}\""
        for _, row in window.iloc[::-1].iterrows()
    )


# ------------------------------------------------------ profiel & continuïteit --

def profile_block(memory_dir: Path, config: dict) -> str:
    """Atletenprofiel: doelen.md volledig + de actuele zonegrenzen uit config."""
    bounds = zone_bounds(config["athlete"])
    zones = (
        f"Actuele zones (%LTHR, LTHR {config['athlete']['lthr']}, "
        f"max HR {config['athlete']['max_hr']}): "
        f"Z2 {bounds[0]}-{bounds[1] - 1} · Z3 {bounds[1]}-{bounds[2] - 1} · "
        f"Z4 {bounds[2]}-{bounds[3] - 1} · Z5 {bounds[3]}+"
    )
    doelen_path = memory_dir / "doelen.md"
    doelen = (
        doelen_path.read_text(encoding="utf-8")
        if doelen_path.exists() else "(geen doelen.md)"
    )
    return f"{zones}\n\n{doelen}"


def _last_md_section(path: Path, heading_contains: str | None = None) -> str | None:
    """De laatste '## '-sectie uit een markdown-bestand, optioneel gefilterd
    op een woord in de kopregel (bijv. alleen 'Weekadvies'-secties).

    Secties die als vervallen zijn gemarkeerd (de sessie is verwijderd, zie
    :mod:`tricoach.removal`) worden overgeslagen, zodat feedback op een
    verwijderde sessie niet meer als continuïteits-context meegaat.
    """
    if not path.exists():
        return None
    parts = path.read_text(encoding="utf-8").split("\n## ")[1:]
    parts = [p for p in parts if not section_is_deleted(p)]
    if heading_contains:
        parts = [p for p in parts if heading_contains in p.splitlines()[0]]
    return "## " + parts[-1] if parts else None


def continuity_block(memory_dir: Path) -> str | None:
    """De vorige feedback en het vorige advies, voor terugkoppeling.

    De coach kan zo verwijzen naar wat er de vorige keer gezegd is ("vorige
    keer zei ik: let op je drift") en toetsen of een advies is opgevolgd.
    """
    delen = []
    vorige_fb = _last_md_section(memory_dir / "feedback.md")
    if vorige_fb:
        delen.append(f"Vorige feedback:\n\n{vorige_fb}")

    advies = _last_md_section(memory_dir / "adviezen.md")
    if advies:
        if len(advies) > MAX_ADVICE_CHARS:
            advies = advies[:MAX_ADVICE_CHARS] + "\n… _[ingekort]_"
        delen.append(f"Laatste advies van de coach:\n\n{advies}")
    return "\n\n".join(delen) if delen else None


# --------------------------------------------------------------- totaalbeeld --

def build_feedback_context(
    conn: sqlite3.Connection,
    memory_dir: Path,
    config: dict,
    act: ParsedActivity,
    tiz: dict[str, int],
    observation: str | None = None,
    user_note: str | None = None,
    wind: "object | None" = None,
    training_label: str | None = None,
) -> str:
    """Bouw de volledige prompt-context voor de feedback op één sessie."""
    bounds = zone_bounds(config["athlete"])
    z2 = (bounds[0], bounds[1] - 1)

    ftp = config["athlete"].get("ftp") or None

    blocks = [
        "# Zojuist voltooide sessie\n\n"
        + session_block(act, tiz, observation, user_note, wind, training_label,
                        ftp=ftp),
    ]
    # Loopdynamiek gaat als apart, expliciet als langetermijn gemarkeerd blok
    # mee: cadans/grondcontacttijd zijn een project voor ná de eerste race en
    # mogen nooit een sessie-oordeel kleuren.
    if act.sport == "running":
        dyn = dynamics_block(act.summary, _prev_activities(conn, act))
        if dyn:
            blocks.append(
                "# Loopdynamiek (alleen langetermijncontext — geen "
                "beoordelingsmaat voor deze sessie)\n\n" + dyn
            )
    # Sluit deze sessie een brick of triatlon-training af (eerdere onderdelen
    # van vandaag, kort ervoor, in racevolgorde)? Dan gaan de wisseltijden en
    # de bakstenen-benen-analyse als eigen blok mee, direct na de sessie zelf.
    combo = combo_block(conn, act, load_activities(conn), config, load_records)
    if combo:
        blocks.append(
            "# Combinatietraining (brick/triatlon) — wissel- en overgangsdata\n\n"
            + combo
        )
    blocks.append(
        "# Vergelijkbare eerdere sessies (like-for-like)\n\n"
        + similar_sessions_block(conn, act, tiz)
    )
    tempo = pace_history_block(conn, act, z2)
    if tempo:
        blocks.append(
            "# Tempo bij gelijke hartslag (zone 2) — belangrijkste voortgangsmaat\n\n"
            + tempo
        )
    # Voor fietsen komt de EF-historie op vermogen erbij: windonafhankelijk en
    # daarmee de zuiverste voortgangsmaat zodra er powerdata is.
    power_hist = power_history_block(conn, act)
    if power_hist:
        blocks.append(
            "# Efficiency factor op vermogen (fietsen) — zuiverste voortgangsmaat\n\n"
            + power_hist
        )
    blocks.append(
        f"# Recente belasting (laatste {LOAD_DAYS} dagen)\n\n"
        + recent_load_block(conn, act)
    )
    noten = recent_notes_block(conn, act)
    if noten:
        blocks.append(
            "# Recente opmerkingen van de atleet (meewegen bij de interpretatie)\n\n"
            + noten
        )
    blocks.append("# Atletenprofiel\n\n" + profile_block(memory_dir, config))
    blocks.append(
        "# Weekschema (voor de bedoeling van de sessie en de volgende training)\n\n"
        + schedule_as_text(memory_dir)
    )
    cont = continuity_block(memory_dir)
    if cont:
        blocks.append("# Continuïteit — vorige feedback en advies\n\n" + cont)

    blocks.append(
        "# Opdracht\n\nGeef de feedback op de zojuist voltooide sessie volgens "
        "je richtlijnen, en sluit af met je oordeel over de volgende training, "
        "in het gevraagde formaat."
    )
    return "\n\n---\n\n".join(blocks)
