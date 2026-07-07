"""Formatteerhulpjes: seconden, tempo's en snelheden als leesbare tekst.

Tempo's en snelheden worden bij voorkeur afgeleid uit afstand en duur — twee
velden die altijd in de database staan — in plaats van uit een los
``avg_speed_ms``-veld dat op sommige sessies ontbreekt (bijvoorbeeld een
samengevoegde zwemsessie). Zo tonen ook al geïmporteerde sessies meteen
correct, zonder opnieuw te importeren.

Alle formatters zijn defensief: bij een ontbrekende, NaN, nul of negatieve
invoer geven ze een nette placeholder (:data:`GEEN_WAARDE`) terug in plaats van
te crashen. Eén kapotte sessie mag nooit de hele tabel laten vallen.
"""

import math

import pandas as pd

# Placeholder voor een onbekende/ongeldige waarde; gelijk aan wat de tabel
# elders (% in zone 2, trend) toont, zodat lege cellen er consistent uitzien.
GEEN_WAARDE = "—"

# Garmin/FIT slaat tijden op in UTC; alles wat een gebruiker of de coach te
# zien krijgt (logboek, feedback, UI) hoort in lokale tijd te staan.
TZ = "Europe/Amsterdam"


def local_time(ts) -> pd.Timestamp:
    """Zet een (UTC-)tijdstempel om naar lokale tijd (Europe/Amsterdam).

    Hét conversiepunt voor weergave: FIT-starttijden zijn UTC (soms naïef,
    dan wordt UTC aangenomen). Gebruik dit overal waar een sessietijd wordt
    getoond of naar een memory-bestand geschreven, zodat logboek, feedback en
    dashboard dezelfde (lokale) klok hanteren.
    """
    ts = pd.Timestamp(ts)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.tz_convert(TZ)


def _ongeldig(x) -> bool:
    """True als ``x`` onbruikbaar is als tempo/snelheid/tijd.

    Vangt None, NaN (zo vult pandas een ontbrekend numeriek veld in),
    niet-numerieke waarden en nul/negatief af — precies de gevallen waarop een
    omrekening anders zou crashen (``int(NaN)`` of een deling door nul).
    """
    try:
        return x is None or math.isnan(float(x)) or float(x) <= 0
    except (TypeError, ValueError):
        return True


def derive_speed_ms(distance_m: float | None, duration_s: float | None) -> float | None:
    """Gemiddelde snelheid (m/s) uit afstand (m) en duur (s).

    Geeft None als afstand of duur ontbreekt of onbruikbaar is, zodat de
    formatters er vanzelf een placeholder van maken.
    """
    if _ongeldig(distance_m) or _ongeldig(duration_s):
        return None
    return distance_m / duration_s


def fmt_duration(seconds: float | None) -> str:
    """Seconden -> 'H:MM:SS' of 'MM:SS'."""
    if _ongeldig(seconds):
        return GEEN_WAARDE
    s = int(seconds)
    h, rest = divmod(s, 3600)
    m, sec = divmod(rest, 60)
    return f"{h}:{m:02d}:{sec:02d}" if h else f"{m}:{sec:02d}"


def fmt_hours_hhmm(hours: float | None, signed: bool = False) -> str:
    """Uren als kommagetal -> 'u:mm' (bijv. 4.3 -> '4:18').

    Met ``signed=True`` krijgt een positief verschil een plus ervoor
    (voor delta-kolommen); negatief mag hier wél, anders dan bij tempo's.
    """
    try:
        h = float(hours)
        if math.isnan(h):
            return GEEN_WAARDE
    except (TypeError, ValueError):
        return GEEN_WAARDE
    teken = "-" if h < 0 else ("+" if signed else "")
    totaal_min = int(round(abs(h) * 60))
    return f"{teken}{totaal_min // 60}:{totaal_min % 60:02d}"


def fmt_pace_per_km(speed_ms: float | None) -> str:
    """Snelheid (m/s) -> looptempo 'M:SS/km'."""
    if _ongeldig(speed_ms):
        return GEEN_WAARDE
    sec_per_km = 1000 / speed_ms
    return f"{int(sec_per_km // 60)}:{int(sec_per_km % 60):02d}/km"


def fmt_pace_per_100m(speed_ms: float | None) -> str:
    """Snelheid (m/s) -> zwemtempo 'M:SS/100m'."""
    if _ongeldig(speed_ms):
        return GEEN_WAARDE
    sec = 100 / speed_ms
    return f"{int(sec // 60)}:{int(sec % 60):02d}/100m"


def fmt_speed_kmh(speed_ms: float | None) -> str:
    """Snelheid (m/s) -> 'XX.X km/h'."""
    if _ongeldig(speed_ms):
        return GEEN_WAARDE
    return f"{speed_ms * 3.6:.1f} km/h"


def sessie_tempo(sport: str, distance_m: float | None,
                 duration_s: float | None,
                 active_swim_s: float | None = None) -> str:
    """Tempo/snelheid-cel voor de sessietabel, afgeleid uit afstand en duur.

    Per sport: looptempo (M:SS/km) voor hardlopen, snelheid (km/h) voor
    fietsen en zwemtempo (M:SS/100m) voor de rest. Voor zwemmen wordt bij
    voorkeur de zuivere zwemtijd (``active_swim_s``, de som van de actieve
    banen) als noemer gebruikt; die telt de rust aan de kant niet mee en geeft
    daardoor hetzelfde tempo als Garmins eigen gemiddelde. Ontbreekt die, dan
    valt het terug op ``duration_s`` (de totale timer-duur).

    Leunt bewust niet op ``avg_speed_ms``: dat veld ontbreekt op sommige
    sessies (zoals een samengevoegde zwemsessie), terwijl afstand en duur er
    altijd zijn. Bij een ontbrekende of ongeldige waarde volgt een placeholder.
    """
    if sport == "running":
        return fmt_pace_per_km(derive_speed_ms(distance_m, duration_s))
    if sport == "cycling":
        return fmt_speed_kmh(derive_speed_ms(distance_m, duration_s))
    zwemtijd = duration_s if _ongeldig(active_swim_s) else active_swim_s
    return fmt_pace_per_100m(derive_speed_ms(distance_m, zwemtijd))


def effective_speed_ms(row, active_swim_s: dict | None = None) -> float | None:
    """Effectieve snelheid (m/s) van een sessie(-rij uit de activiteitentabel).

    Gebruikt ``avg_speed_ms`` waar aanwezig en valt anders terug op afstand en
    duur — voor zwemmen bij voorkeur de zuivere zwemtijd uit ``active_swim_s``
    (dict activity_key -> seconden), zodat rust aan de kant niet meetelt. Zo
    valt een sessie zonder ``avg_speed_ms`` (zoals een samengevoegde
    zwemsessie) niet uit de grafieken.
    """
    if not _ongeldig(row["avg_speed_ms"]):
        return float(row["avg_speed_ms"])
    tijd = None
    if row["sport"] == "swimming" and active_swim_s:
        tijd = active_swim_s.get(row["activity_key"])
    return derive_speed_ms(row["distance_m"], tijd or row["duration_s"])


def run_cadence_spm(avg_cadence: float | None) -> float | None:
    """Garmin-loopcadans (rpm, één been) -> stappen per minuut (beide benen)."""
    if _ongeldig(avg_cadence):
        return None
    return float(avg_cadence) * 2


SPORT_NL = {"running": "Hardlopen", "cycling": "Fietsen", "swimming": "Zwemmen"}

STROKE_NL = {
    "breaststroke": "Schoolslag", "freestyle": "Borstcrawl",
    "backstroke": "Rugslag", "butterfly": "Vlinderslag",
    "mixed": "Gemengd", "drill": "Oefening", "im": "Wisselslag",
}


def sport_label(sport: str) -> str:
    """Engelse FIT-sportnaam -> Nederlands label."""
    return SPORT_NL.get(sport, sport.capitalize())


def stroke_label(stroke: str) -> str:
    """Engelse FIT-slagnaam -> Nederlands label."""
    return STROKE_NL.get(stroke, stroke)
