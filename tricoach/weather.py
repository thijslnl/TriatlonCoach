"""Automatische winddata via de gratis Open-Meteo Historical Weather API.

Voor fiets- en loopsessies halen we de wind op die er tijdens de training stond,
als objectieve context bij de feedback: een langzamere terugweg met tegenwind is
geen slechte vorm. De bron is gratis en vereist geen API-key of aanmelding
(niet-commercieel, ~10.000 calls/dag, CC BY 4.0 — vermeld "Weather data by
Open-Meteo.com" in de UI/credits).

Werkwijze (zie ``wind_for_activity``):

1. Pak de GPS-startcoördinaat en het startuur uit de geparste sessie. Voor een
   langere rit nemen we ook een meetpunt halverwege mee, voor de kop/tegenwind-
   duiding.
2. Bevraag Open-Meteo één keer voor dat uur en die locatie → windsnelheid,
   -richting en gusts (km/h, graden waar de wind *vandaan* komt).
3. De aanroep wordt gelogd in memory/externe_data_log.md.

Faalt zacht: geen GPS (zwemmen in een bad, loop zonder fix) of geen internet →
``None``, zónder de upload te laten mislukken. De winddata wordt bij de sessie
gecachet (kolommen in de activities-tabel), zodat we de API per sessie maar één
keer aanroepen en nooit bij een gewone page-load.
"""

import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

from tricoach.fit_parser import ParsedActivity

# Open-Meteo: gratis archief-endpoint, geen key nodig.
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
HOURLY_VARS = "wind_speed_10m,wind_direction_10m,wind_gusts_10m"
# Lokale tijdzone als string; pandas regelt de conversie (zoals elders in de
# app), zodat we niet van de losse ``tzdata``-package afhangen.
TIMEZONE = "Europe/Amsterdam"

# Alleen voor sporten waar wind ertoe doet en GPS gangbaar is.
WIND_RELEVANT_SPORTS = ("cycling", "running")

# Wereldwindrichtingen voor een leesbare richting ("ZW", "NNO", ...).
_COMPASS = [
    "N", "NNO", "NO", "ONO", "O", "OZO", "ZO", "ZZO",
    "Z", "ZZW", "ZW", "WZW", "W", "WNW", "NW", "NNW",
]


@dataclass
class WindData:
    """Winddata bij één sessie, opgehaald bij Open-Meteo.

    ``direction_deg`` is de richting waar de wind *vandaan* komt (meteorologische
    conventie). ``headwind_note`` is een optionele, leesbare kop/tegenwind-duiding
    op basis van de bewegingsrichting; ``None`` als die niet te bepalen was.
    """

    speed_kmh: float
    direction_deg: float
    gusts_kmh: float | None = None
    headwind_note: str | None = None

    @property
    def direction_label(self) -> str:
        """Windrichting als kompaslabel, bijv. 'ZW' (zuidwesten)."""
        idx = int((self.direction_deg % 360) / 22.5 + 0.5) % 16
        return _COMPASS[idx]

    def as_text(self) -> str:
        """Compacte regel voor log, trainingslog en prompt-context."""
        parts = [
            f"{self.speed_kmh:.0f} km/h uit het {self.direction_label} "
            f"({self.direction_deg:.0f}°)"
        ]
        if self.gusts_kmh:
            parts.append(f"uitschieters tot {self.gusts_kmh:.0f} km/h")
        if self.headwind_note:
            parts.append(self.headwind_note)
        return ", ".join(parts)


# --------------------------------------------------------------- GPS helpers --

def _to_local(ts: pd.Timestamp) -> pd.Timestamp:
    """Zet een (UTC-)tijdstempel om naar lokale tijd (Europe/Amsterdam).

    FIT-tijden staan in UTC; Open-Meteo geeft met ``timezone=Europe/Amsterdam``
    lokale uren terug, dus we matchen op het lokale startuur.
    """
    ts = pd.Timestamp(ts)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.tz_convert(TIMEZONE)


def _gps_points(act: ParsedActivity) -> list[tuple[float, float]]:
    """Geldige (lat, lon)-punten uit de records, op volgorde van de rit.

    Leeg als er geen GPS in het bestand zat (zwemmen in een bad, loop zonder fix).
    """
    rec = act.records
    if rec.empty or "lat" not in rec or "lon" not in rec:
        return []
    coords = rec[["lat", "lon"]].dropna()
    # (0, 0) is geen geldige fix maar een Garmin-placeholder.
    coords = coords[(coords["lat"] != 0) | (coords["lon"] != 0)]
    return list(coords.itertuples(index=False, name=None))


def start_coordinate(act: ParsedActivity) -> tuple[float, float] | None:
    """De eerste geldige GPS-coördinaat van de sessie, of ``None`` zonder GPS."""
    points = _gps_points(act)
    return points[0] if points else None


def _bearing(p1: tuple[float, float], p2: tuple[float, float]) -> float:
    """Kompaskoers (graden, 0=N) van punt p1 naar p2 over een groot-cirkel."""
    lat1, lon1, lat2, lon2 = map(math.radians, (*p1, *p2))
    dlon = lon2 - lon1
    x = math.sin(dlon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
    return (math.degrees(math.atan2(x, y)) + 360) % 360


def _wind_relation(move_bearing: float, wind_from_deg: float) -> str:
    """Duid de relatie tussen bewegingsrichting en windrichting.

    De windrichting is waar de wind *vandaan* komt; je hebt tegenwind als je
    díe kant op beweegt. Hoek dichtbij 0° = tegenwind, dichtbij 180° = meewind.
    """
    diff = abs((move_bearing - wind_from_deg + 180) % 360 - 180)
    if diff < 45:
        return "tegenwind"
    if diff > 135:
        return "meewind"
    return "zijwind"


def headwind_note(act: ParsedActivity, wind_from_deg: float) -> str | None:
    """Korte kop/tegenwind-duiding voor heen- en terugweg, of ``None``.

    Vergelijkt de gemiddelde bewegingsrichting van de eerste en tweede helft van
    de rit met de windrichting. Bewust grof: bedoeld als context ("terugweg met
    tegenwind"), niet als exacte meting. Geeft ``None`` bij te weinig GPS.
    """
    points = _gps_points(act)
    if len(points) < 8:
        return None
    mid = len(points) // 2
    first, second = _bearing(points[0], points[mid]), _bearing(points[mid], points[-1])
    rel1, rel2 = _wind_relation(first, wind_from_deg), _wind_relation(second, wind_from_deg)
    if rel1 == rel2:
        return f"overwegend {rel1} gedurende de rit"
    return f"heenweg {rel1}, terugweg {rel2}"


# ------------------------------------------------------------ Open-Meteo call --

def _fetch_hourly(lat: float, lon: float, local_day: str) -> dict | None:
    """Haal de uurlijkse winddata voor één lokale dag op. ``None`` bij netwerkfout."""
    params = {
        "latitude": round(lat, 4),
        "longitude": round(lon, 4),
        "start_date": local_day,
        "end_date": local_day,
        "hourly": HOURLY_VARS,
        "timezone": TIMEZONE,
    }
    try:
        resp = requests.get(ARCHIVE_URL, params=params, timeout=15)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException:
        return None


def _pick_hour(payload: dict, target_hour: int) -> dict | None:
    """Kies uit de uurdata de rij die bij het startuur hoort.

    Geeft een dict met snelheid/richting/gusts en het gematchte tijdstip, of
    ``None`` als de waarden voor dat uur ontbreken (Open-Meteo geeft soms
    ``null`` voor heel recente dagen).
    """
    hourly = payload.get("hourly") or {}
    times = hourly.get("time") or []
    if not times:
        return None
    # Index van het uur dat het dichtst bij het startuur ligt.
    hours = [int(t[11:13]) for t in times]
    idx = min(range(len(hours)), key=lambda i: abs(hours[i] - target_hour))

    def at(name: str):
        col = hourly.get(name) or []
        return col[idx] if idx < len(col) else None

    speed, direction, gusts = at("wind_speed_10m"), at("wind_direction_10m"), at("wind_gusts_10m")
    if speed is None or direction is None:
        return None
    return {
        "speed_kmh": float(speed),
        "direction_deg": float(direction),
        "gusts_kmh": float(gusts) if gusts is not None else None,
        "time": times[idx],
    }


def wind_for_activity(act: ParsedActivity, memory_dir: Path | None = None) -> WindData | None:
    """Haal de winddata voor één sessie op bij Open-Meteo.

    Retourneert ``None`` (en logt niets) als wind niet relevant is of er geen GPS
    in het bestand zit; ``None`` (mét log van de mislukte poging) als de API niet
    bereikbaar was of geen waarden voor dat uur had. Faalt dus altijd zacht — de
    aanroeper mag de upload gewoon laten doorgaan.
    """
    if act.sport not in WIND_RELEVANT_SPORTS:
        return None
    coord = start_coordinate(act)
    if coord is None:
        return None

    local = _to_local(act.start_time)
    payload = _fetch_hourly(coord[0], coord[1], local.strftime("%Y-%m-%d"))
    if payload is None:
        if memory_dir:
            _log_call(memory_dir, act, coord, local, result="netwerk-/API-fout (overgeslagen)")
        return None

    picked = _pick_hour(payload, local.hour)
    if picked is None:
        if memory_dir:
            _log_call(memory_dir, act, coord, local, result="geen winddata voor dit uur")
        return None

    wind = WindData(
        speed_kmh=picked["speed_kmh"],
        direction_deg=picked["direction_deg"],
        gusts_kmh=picked["gusts_kmh"],
        headwind_note=headwind_note(act, picked["direction_deg"]),
    )
    if memory_dir:
        url = f"{ARCHIVE_URL}?latitude={coord[0]:.4f}&longitude={coord[1]:.4f}" \
              f"&start_date={local:%Y-%m-%d}&hourly={HOURLY_VARS} (uur {picked['time']})"
        _log_call(memory_dir, act, coord, local, result=wind.as_text(), url=url)
    return wind


# ------------------------------------------------------------------- logging --

HEADER = """# Externe data-log (Open-Meteo)

Elke aanroep van de Open-Meteo Historical Weather API, nieuwste onderaan. Wordt
gebruikt om winddata als objectieve context bij de feedback te geven. Bron:
Weather data by Open-Meteo.com (CC BY 4.0). Automatisch bijgehouden.
"""


def _log_call(
    memory_dir: Path,
    act: ParsedActivity,
    coord: tuple[float, float],
    local: pd.Timestamp,
    result: str,
    url: str | None = None,
) -> None:
    """Leg één Open-Meteo-aanroep vast in memory/externe_data_log.md."""
    path = memory_dir / "externe_data_log.md"
    if not path.exists():
        path.write_text(HEADER, encoding="utf-8")
    entry = (
        f"\n## {datetime.now():%Y-%m-%d %H:%M:%S} — Open-Meteo wind\n\n"
        f"- **Sessie:** {act.sport} {local:%Y-%m-%d %H:%M} (lokaal) · "
        f"sleutel `{act.activity_key}`\n"
        f"- **Locatie/uur:** lat {coord[0]:.4f}, lon {coord[1]:.4f}, uur {local.hour}:00\n"
        f"- **Aanroep:** {url or ARCHIVE_URL}\n"
        f"- **Resultaat:** {result}\n"
    )
    with open(path, "a", encoding="utf-8") as f:
        f.write(entry)
