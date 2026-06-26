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

# Placeholder voor een onbekende/ongeldige waarde; gelijk aan wat de tabel
# elders (% in zone 2, trend) toont, zodat lege cellen er consistent uitzien.
GEEN_WAARDE = "—"


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


SPORT_NL = {"running": "Hardlopen", "cycling": "Fietsen", "swimming": "Zwemmen"}


def sport_label(sport: str) -> str:
    """Engelse FIT-sportnaam -> Nederlands label."""
    return SPORT_NL.get(sport, sport.capitalize())
