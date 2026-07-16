"""Fietsvermogen en -cadans: normalized power, zones, EF en decoupling.

Sinds juli 2026 bevatten de fiets-FIT-bestanden vermogen en cadans: buiten
via Garmin Rally RS110-pedalen (enkelzijdig, Edge 540), binnen via de Wahoo
Kickr Core 2 onder Zwift. Twee bronnen die net iets anders meten — trends
worden daarom bij voorkeur bínnen één bron vergeleken; de bron wordt per
sessie opgeslagen.

Definities (vastgelegd in memory/beslissingen.md):

- **Normalized power (NP)**: 30 s voortschrijdend gemiddelde van het
  vermogen → elk punt tot de 4e macht → gemiddelde → 4e-machtswortel.
  NP weegt pieken zwaarder en is voor wisselend buitenrijden informatiever
  dan het kale gemiddelde.
- **Vermogenszones (Coggan, % van FTP)**: Z1 <55 · Z2 56–75 · Z3 76–90 ·
  Z4 91–105 · Z5 106–120 · Z6 >120. Alleen berekend zodra de FTP bekend is
  (instellingenpagina); zonder FTP tonen we de power-data zonder zone-oordeel.
- **Efficiency factor (EF)**: NP gedeeld door de gemiddelde hartslag
  (W per slag). Windonafhankelijk, dus een zuiverder aerobe trendmaat dan
  snelheid/HR.
- **Aerobic decoupling (Pw:Hr)**: EF van de eerste helft vs de tweede helft
  van een rit; onder de ~5% duidt op een goed ontwikkelde aerobe basis.
- **Nullen**: cadans 0 en power 0 tijdens freewheelen/stilstaan zijn échte
  waarden. Ze tellen mee in NP en de tijdsverdeling; voor cadans tonen we
  zowel het gemiddelde inclusief als exclusief nul (Garmin toont zelf
  exclusief nul).
- **FTP-schatting**: 95% van het beste 20-minutenvermogen over alle sessies —
  een indicatie zolang er geen echte FTP-test is gedaan.

Oudere sessies hebben geen power/cadans; elke functie geeft dan ``None``
(of een leeg DataFrame) terug en de weergave valt terug op "—".
"""

import sqlite3

import numpy as np
import pandas as pd

# Zonenamen en de bovengrenzen van P1..P5 als fractie van de FTP (Coggan).
# Alles boven de laatste grens is P6.
POWER_ZONE_NAMES = ["P1", "P2", "P3", "P4", "P5", "P6"]
COGGAN_UPPER_PCT = [0.55, 0.75, 0.90, 1.05, 1.20]

# NP-venster (s) en de minimale hoeveelheid powerdata (s) voor een zinvolle NP.
NP_WINDOW_S = 30
MIN_POWER_SECONDS = 300

# FTP-schatting: 95% van het beste 20-minutenvermogen.
FTP_EST_WINDOW_S = 1200
FTP_EST_FACTOR = 0.95

# Minimale duur (s) voor een decoupling-berekening (~30 min).
DECOUPLING_MIN_S = 1800

# Pacing-analyse: het eerste kwartier vs de rest van de rit.
PACING_FIRST_MIN = 15

# Sub-sports die een indoorsessie markeren (Zwift exporteert virtual_activity;
# een trainer-rit zonder Zwift meestal indoor_cycling).
INDOOR_SUB_SPORTS = {"virtual_activity", "indoor_cycling"}

# Records liggen normaal ~1 s uit elkaar; grotere gaten (pauze) tellen
# maximaal dit aantal seconden mee — zelfde regel als de HR-zones.
MAX_GAP_S = 10


# ------------------------------------------------------------ bron & indoor --

def is_indoor(sub_sport: str | None, file_manufacturer: str | None = None) -> bool:
    """Is dit een indoorsessie (trainer/Zwift)?

    Beslist op de sub_sport uit het FIT-bestand; als vangnet geldt ook een
    bestand dat door Zwift zelf is geschreven (``file_id.manufacturer``) als
    indoor. Let op: Zwift-bestanden bevatten wél GPS (virtuele coördinaten),
    dus "geen GPS" is hier geen bruikbaar criterium.
    """
    if sub_sport and str(sub_sport) in INDOOR_SUB_SPORTS:
        return True
    return bool(file_manufacturer) and str(file_manufacturer).lower() == "zwift"


def power_source(devices: list[dict], indoor: bool) -> str:
    """Leesbaar label van de vermogensbron, voor opslag bij de sessie.

    Zoekt in de device-info van het FIT-bestand naar de vermogenssensor
    (device_type ``bike_power``); de fabrikant bepaalt het label. Ontbreekt
    die informatie, dan is indoor/outdoor de terugval — binnen meet de
    trainer, buiten de pedalen. Twee bronnen meten net iets anders, dus
    trends horen per bron vergeleken te worden.
    """
    for d in devices or []:
        device_type = str(d.get("antplus_device_type") or d.get("device_type") or "")
        if device_type not in ("bike_power", "11"):
            continue
        fabrikant = str(d.get("manufacturer") or "").lower()
        if "wahoo" in fabrikant:
            return "trainer (Wahoo Kickr)"
        if fabrikant == "garmin":
            return "pedalen (Garmin Rally)"
        if fabrikant:
            return f"vermogensmeter ({fabrikant})"
    return "trainer (indoor)" if indoor else "pedalen (buiten)"


# ------------------------------------------------------------- kerncijfers --

def _power_series(records: pd.DataFrame) -> pd.Series:
    """Vermogen per meetpunt, geïndexeerd op tijd; leeg zonder powerdata.

    Nullen (freewheelen/stilstaan) blijven staan: dat zijn echte metingen die
    in NP en de tijdsverdeling thuishoren. Alleen ontbrekende waarden vallen af.
    """
    if records.empty or "power" not in records:
        return pd.Series(dtype=float)
    df = records.dropna(subset=["timestamp", "power"]).sort_values("timestamp")
    return pd.Series(df["power"].astype(float).values,
                     index=pd.DatetimeIndex(df["timestamp"]))


def normalized_power(records: pd.DataFrame) -> float | None:
    """Normalized power (NP) volgens de standaardmethode.

    30 s voortschrijdend gemiddelde (op kloktijd, dus robuust voor kleine
    gaten) → 4e macht → gemiddelde → 4e-machtswortel. None bij minder dan
    ~5 minuten powerdata.
    """
    serie = _power_series(records)
    if len(serie) < MIN_POWER_SECONDS:
        return None
    rollend = serie.rolling(f"{NP_WINDOW_S}s").mean()
    # De eerste 30 s heeft nog geen vol venster; die punten wegen anders te licht.
    rollend = rollend[serie.index >= serie.index[0] + pd.Timedelta(seconds=NP_WINDOW_S)]
    if rollend.empty:
        return None
    return float(np.mean(rollend.to_numpy() ** 4) ** 0.25)


def cadence_stats(records: pd.DataFrame) -> dict:
    """Cadansstatistiek van een rit: gemiddeld incl./excl. nul en het maximum.

    Cadans 0 tijdens freewheelen is een echte waarde: het gemiddelde
    inclusief nul zegt iets over hoeveel er niet getrapt is, het gemiddelde
    exclusief nul is het trapritme zelf (zo toont Garmin het ook). Geeft
    ``{"incl0": None, "excl0": None, "max": None}`` zonder cadansdata.
    """
    leeg = {"incl0": None, "excl0": None, "max": None}
    if records.empty or "cadence" not in records:
        return leeg
    cad = records["cadence"].dropna().astype(float)
    if cad.empty:
        return leeg
    trappend = cad[cad > 0]
    return {
        "incl0": float(cad.mean()),
        "excl0": float(trappend.mean()) if not trappend.empty else None,
        "max": float(cad.max()),
    }


# ------------------------------------------------------------ vermogenszones --

def power_zone_bounds(ftp: float) -> list[float]:
    """De bovengrenzen (watt) van P1 t/m P5; alles daarboven is P6."""
    return [ftp * p for p in COGGAN_UPPER_PCT]


def zone_for_power(watt: float, bounds: list[float]) -> str:
    """Zonenaam (P1..P6) voor één vermogenswaarde."""
    zone = 0
    for bound in bounds:
        if watt > bound:
            zone += 1
    return POWER_ZONE_NAMES[min(zone, len(POWER_ZONE_NAMES) - 1)]


def time_in_power_zones(records: pd.DataFrame, ftp: float | None) -> dict[str, int]:
    """Seconden per vermogenszone; overal 0 zonder FTP of zonder powerdata.

    Nullen tellen mee (P1): freewheelen is echte tijd. Gaten groter dan
    ``MAX_GAP_S`` (pauze) tellen afgekapt mee, zoals bij de hartslagzones.
    """
    result = dict.fromkeys(POWER_ZONE_NAMES, 0)
    if not ftp or records.empty or "power" not in records:
        return result
    df = records.dropna(subset=["power", "timestamp"]).sort_values("timestamp")
    if len(df) < 2:
        return result
    bounds = power_zone_bounds(float(ftp))
    seconds = df["timestamp"].diff().dt.total_seconds().clip(upper=MAX_GAP_S)
    for watt, sec in zip(df["power"].iloc[1:], seconds.iloc[1:]):
        result[zone_for_power(float(watt), bounds)] += int(sec)
    return result


# ------------------------------------------------- EF, decoupling & pacing --

def efficiency_factor(np_watt: float | None, avg_hr: float | None) -> float | None:
    """Efficiency factor: NP per hartslag (W/slag). None bij ontbrekende invoer.

    Hoger bij vergelijkbare sessies = grotere aerobe basis. Vermogen is
    windonafhankelijk, dus deze maat is zuiverder dan snelheid/HR.
    """
    if not np_watt or not avg_hr or pd.isna(np_watt) or pd.isna(avg_hr):
        return None
    return float(np_watt) / float(avg_hr)


def power_decoupling(records: pd.DataFrame) -> float | None:
    """Aerobic decoupling (Pw:Hr): EF-verlies van de tweede helft, in %.

    Splitst de rit op het tijdsmidden en vergelijkt vermogen/HR van beide
    helften. Positief = de hartslag loopt relatief op (drift); onder de ~5%
    bij een duurrit duidt op een goede aerobe basis. None bij te weinig
    power- of hartslagdata (< ~30 minuten).
    """
    if records.empty or "power" not in records or "heart_rate" not in records:
        return None
    df = records.dropna(subset=["power", "heart_rate", "timestamp"]).sort_values("timestamp")
    if len(df) < 2:
        return None
    duur = (df["timestamp"].iloc[-1] - df["timestamp"].iloc[0]).total_seconds()
    if duur < DECOUPLING_MIN_S:
        return None
    midden = df["timestamp"].iloc[0] + (df["timestamp"].iloc[-1] - df["timestamp"].iloc[0]) / 2
    h1 = df[df["timestamp"] <= midden]
    h2 = df[df["timestamp"] > midden]
    if h1["heart_rate"].mean() <= 0 or h2["heart_rate"].mean() <= 0:
        return None
    ef1 = h1["power"].mean() / h1["heart_rate"].mean()
    ef2 = h2["power"].mean() / h2["heart_rate"].mean()
    if ef1 <= 0:
        return None
    return float((ef1 - ef2) / ef1 * 100)


def pacing_split(records: pd.DataFrame,
                 first_min: int = PACING_FIRST_MIN) -> tuple[float, float] | None:
    """Gemiddeld vermogen van het eerste kwartier vs de rest van de rit.

    Dé check op de bekende valkuil "te hard starten": met power is dat direct
    zichtbaar. None als de rit te kort is (de rest moet minstens even lang
    zijn als het eerste blok) of zonder powerdata.
    """
    serie = _power_series(records)
    if serie.empty:
        return None
    grens = serie.index[0] + pd.Timedelta(minutes=first_min)
    eerste = serie[serie.index <= grens]
    rest = serie[serie.index > grens]
    if eerste.empty or rest.empty \
            or (serie.index[-1] - grens) < pd.Timedelta(minutes=first_min):
        return None
    return float(eerste.mean()), float(rest.mean())


# ------------------------------------------------------------- FTP-schatting --

def best_20min_power(records: pd.DataFrame) -> float | None:
    """Het hoogste 20-minuten-gemiddelde vermogen binnen één rit, of None."""
    serie = _power_series(records)
    if serie.empty:
        return None
    if (serie.index[-1] - serie.index[0]) < pd.Timedelta(seconds=FTP_EST_WINDOW_S):
        return None
    rollend = serie.rolling(f"{FTP_EST_WINDOW_S}s").mean()
    vol = rollend[serie.index >= serie.index[0] + pd.Timedelta(seconds=FTP_EST_WINDOW_S)]
    return float(vol.max()) if not vol.empty else None


def estimate_ftp(conn: sqlite3.Connection, acts: pd.DataFrame) -> dict | None:
    """FTP-schatting uit de trainingsdata: 95% van het beste 20-min vermogen.

    Doorzoekt alle fietssessies met powerdata; geeft ``{"ftp_watt",
    "best20_watt", "datum", "bron"}`` of None zolang geen rit een vol
    20-minutenblok heeft. Kanttekening (hoort bij elke weergave): een echte
    FTP-test is beter — dit is een ondergrens uit gewone trainingen.
    """
    from tricoach.storage import load_records  # hier tegen een circulaire import

    if acts.empty or "np_power" not in acts:
        return None
    rides = acts[acts["sport"] == "cycling"]
    beste: dict | None = None
    for _, act in rides.iterrows():
        if pd.isna(act.get("avg_power")) and pd.isna(act.get("np_power")):
            continue  # geen power op deze sessie; records scannen is dan zinloos
        best20 = best_20min_power(load_records(conn, act["activity_key"]))
        if best20 is None:
            continue
        if beste is None or best20 > beste["best20_watt"]:
            beste = {
                "best20_watt": best20,
                "ftp_watt": best20 * FTP_EST_FACTOR,
                "datum": act["start_time"],
                "bron": act.get("power_source"),
            }
    return beste


# ------------------------------------------------------------------- trends --

def power_trend(acts: pd.DataFrame) -> pd.DataFrame:
    """Power/cadans/EF per fietssessie uit de gecachete kolommen (oudste eerst).

    Kolommen: start_time, avg_power, np_power, ef_watt, cadans (excl. nul,
    terugval op de sessiecadans), indoor (bool) en power_source. Sessies
    zonder power vallen eruit; leeg DataFrame als er nog geen powersessies zijn.
    """
    if acts.empty or "np_power" not in acts:
        return pd.DataFrame()
    rides = acts[acts["sport"] == "cycling"].copy()
    rides = rides[rides["np_power"].notna() | rides["avg_power"].notna()]
    if rides.empty:
        return pd.DataFrame()
    rides["cadans"] = rides["avg_cadence_excl0"].where(
        rides["avg_cadence_excl0"].notna(), rides["avg_cadence"])
    rides["indoor"] = rides["is_indoor"].fillna(0).astype(int).astype(bool)
    kolommen = ["start_time", "avg_power", "np_power", "ef_watt",
                "cadans", "indoor", "power_source"]
    return rides[kolommen].sort_values("start_time")


def power_decoupling_trend(conn: sqlite3.Connection, acts: pd.DataFrame) -> pd.DataFrame:
    """Pw:Hr-decoupling per fietssessie met powerdata (oudste eerst).

    Scant de seconde-data van elke powersessie; sessies korter dan ~30 min of
    zonder hartslag vallen eruit. Kolommen: start_time, decoupling_pct, indoor.
    """
    from tricoach.storage import load_records

    if acts.empty or "np_power" not in acts:
        return pd.DataFrame()
    rides = acts[(acts["sport"] == "cycling") & acts["np_power"].notna()]
    rows = []
    for _, act in rides.iterrows():
        pct = power_decoupling(load_records(conn, act["activity_key"]))
        if pct is None:
            continue
        rows.append({
            "start_time": act["start_time"],
            "decoupling_pct": pct,
            "indoor": bool(act.get("is_indoor") or 0),
        })
    return pd.DataFrame(rows).sort_values("start_time") if rows else pd.DataFrame()
