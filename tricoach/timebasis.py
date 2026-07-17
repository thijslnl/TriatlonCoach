"""Actieve/bewegende tijd als de ene tijdsbasis voor tempo en snelheid.

Tempo op de totale verstreken tijd vertekent: rust aan de kant van het bad,
stilstaan bij een stoplicht en een handmatige pauze horen niet in het tempo
thuis. Daarom rekent de hele app (sessietabel, detail, trends én
feedback-context) met de **actieve tijd**, per sport gedefinieerd als:

- **Zwemmen**: de som van de actieve banen (``lengths.total_timer_time``) —
  rustpauzes tussen banen tellen niet mee. De totale timer-duur blijft apart
  zichtbaar als "sessieduur".
- **Fietsen**: de rijtijd. ``total_timer_time`` is bij auto-pauze al de
  bewegende tijd; zonder auto-pauze telt stilstand mee en wordt de bewegende
  tijd uit de seconde-data berekend (snelheid boven ~3 km/h).
- **Hardlopen**: ``total_timer_time`` — handmatige pauzes zijn daar al uit.

De keuze en de eenmalige herberekening van bestaande sessies zijn vastgelegd
in ``memory/beslissingen.md`` (2026-07-17). In de UI heet dit "actief tempo";
Garmin Connect kan (vooral bij zwemmen) een ander, trager tempo tonen.
"""

import pandas as pd

# Onder deze snelheid geldt een meetpunt als stilstand (~3 km/h): traag
# genoeg om lopen naast de fiets nog mee te tellen, snel genoeg om wachten
# voor een stoplicht uit te sluiten.
MOVING_MIN_SPEED_MS = 3.0 / 3.6
# Gaten tussen meetpunten groter dan dit tellen als (auto)pauze en worden
# afgekapt, zodat een hervatting na een stop niet als één lang "bewegend"
# interval meetelt.
MAX_GAP_S = 10.0
# Zo benoemen we de tijdsbasis in de UI en in prompts.
ACTIEF_LABEL = "actief tempo"


def moving_seconds(records: pd.DataFrame) -> float | None:
    """Bewegende tijd (s) uit de seconde-data: intervallen met snelheid.

    Elk meetpunt telt voor het interval sinds het vorige punt (afgekapt op
    :data:`MAX_GAP_S`) mee wanneer de snelheid boven de stilstanddrempel
    ligt. None als er geen bruikbare snelheids-/tijddata is.
    """
    if records is None or records.empty \
            or "speed_ms" not in records or "timestamp" not in records:
        return None
    df = records.dropna(subset=["timestamp"]).sort_values("timestamp")
    if df.empty or not df["speed_ms"].notna().any():
        return None
    dt = df["timestamp"].diff().dt.total_seconds().clip(upper=MAX_GAP_S).fillna(1.0)
    bewegend = float(dt[df["speed_ms"].fillna(0.0) > MOVING_MIN_SPEED_MS].sum())
    return bewegend if bewegend > 0 else None


def active_seconds(sport: str, duration_s: float | None,
                   records: pd.DataFrame | None = None,
                   lengths: pd.DataFrame | None = None) -> float | None:
    """Actieve tijd (s) van één sessie, volgens de per-sport definitie.

    ``duration_s`` is de totale timer-duur (``total_timer_time``); die geldt
    als terugval wanneer de baan- of seconde-data ontbreekt. Geeft None als
    er helemaal geen bruikbare duur is.
    """
    if sport == "swimming":
        if lengths is not None and not lengths.empty and "total_timer_time" in lengths:
            actief = float(lengths["total_timer_time"].sum())
            if actief > 0:
                return actief
    elif sport == "cycling":
        bewegend = moving_seconds(records) if records is not None else None
        # Alleen gebruiken als hij wezenlijk korter is dan de timer-duur:
        # met auto-pauze zijn beide vrijwel gelijk en is de timer preciezer.
        if bewegend and duration_s and bewegend < 0.98 * float(duration_s):
            return bewegend
    return float(duration_s) if duration_s else None
