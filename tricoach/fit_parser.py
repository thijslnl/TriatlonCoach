"""Parsen van Garmin FIT-bestanden (los of in een zip) naar bruikbare data.

Een FIT-bestand bevat verschillende soorten berichten; wij gebruiken er drie:

- ``session``: één samenvattingsbericht per activiteit (afstand, duur,
  gemiddelde hartslag, enz.). Dit wordt onze sessie-samenvatting.
- ``record``: de seconde-voor-seconde metingen (hartslag, snelheid, cadans).
  Hieruit berekenen we o.a. tijd-in-zones en grafieken.
- ``length``: alleen bij banenzwemmen — één bericht per baan, met slagtype,
  aantal slagen en tijd. Hieruit volgt SWOLF per baan.

De unieke sleutel van een activiteit is de starttijd uit het ``file_id``-
bericht (``time_created``): die is per Garmin-activiteit gegarandeerd uniek
en zit altijd in het bestand zelf, dus deduplicatie blijft werken ongeacht
hoe het bestand heet.
"""

import io
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import fitdecode
import pandas as pd

# Kolommen die we uit record-berichten halen (FIT-veldnaam -> onze kolomnaam).
RECORD_FIELDS = {
    "timestamp": "timestamp",
    "heart_rate": "heart_rate",
    "enhanced_speed": "speed_ms",
    "speed": "speed_ms",  # fallback als enhanced_speed ontbreekt
    "distance": "distance_m",
    "cadence": "cadence",
    "enhanced_altitude": "altitude_m",
    "altitude": "altitude_m",
    "position_lat": "lat",
    "position_long": "lon",
}

# FIT slaat GPS-posities op als "semicircles" (gehele getallen), niet als graden.
# Omrekenen: graden = semicircles × 180 / 2³¹. Sessies zonder GPS (zwemmen in
# een bad, loop zonder horloge-fix) hebben deze velden simpelweg niet.
SEMICIRCLE_TO_DEGREES = 180.0 / 2**31

# Velden per baan bij zwemmen.
LENGTH_FIELDS = ["timestamp", "total_timer_time", "total_strokes", "swim_stroke", "length_type"]


@dataclass
class ParsedActivity:
    """Eén geparste Garmin-activiteit: samenvatting + detaildata."""

    activity_key: str            # unieke sleutel (ISO-starttijd uit file_id)
    sport: str                   # running / cycling / swimming / ...
    sub_sport: str | None
    start_time: pd.Timestamp
    summary: dict                # ruwe session-velden (afstand, HR, enz.)
    records: pd.DataFrame        # seconde-data
    lengths: pd.DataFrame        # baandata (alleen zwemmen, anders leeg)
    source_file: str             # bestandsnaam waar dit uit kwam

    @property
    def duration_s(self) -> float:
        """Actieve duur in seconden (timer-tijd, dus zonder pauzes)."""
        return self.summary.get("total_timer_time") or 0.0

    @property
    def distance_m(self) -> float:
        return self.summary.get("total_distance") or 0.0


# Session-velden die we bewaren. Niet elk veld bestaat bij elke sport;
# ontbrekende velden worden None.
SESSION_FIELDS = [
    "sport", "sub_sport", "start_time",
    "total_timer_time", "total_elapsed_time", "total_distance",
    "avg_heart_rate", "max_heart_rate",
    "enhanced_avg_speed", "avg_speed", "enhanced_max_speed", "max_speed",
    "avg_cadence", "max_cadence", "avg_running_cadence",
    "total_ascent", "total_descent",
    "total_calories", "normalized_power", "avg_power",
    # zwemspecifiek
    "pool_length", "num_lengths", "num_active_lengths",
    "total_strokes", "avg_stroke_distance",
]


def _value(frame: fitdecode.FitDataMessage, name: str):
    """Veilig één veldwaarde uit een FIT-bericht halen (None als het ontbreekt)."""
    try:
        return frame.get_value(name)
    except KeyError:
        return None


def parse_fit(stream, source_name: str) -> ParsedActivity | None:
    """Parse één FIT-bestand (bestandsobject of pad) naar een ParsedActivity.

    Geeft None terug als het bestand geen activiteit met session-data is
    (Garmin-exports kunnen ook settings- of monitoringbestanden bevatten).
    """
    summary: dict = {}
    time_created = None
    record_rows: list[dict] = []
    length_rows: list[dict] = []

    with fitdecode.FitReader(stream) as fit:
        for frame in fit:
            if not isinstance(frame, fitdecode.FitDataMessage):
                continue

            if frame.name == "file_id":
                time_created = _value(frame, "time_created")

            elif frame.name == "session":
                for f in SESSION_FIELDS:
                    val = _value(frame, f)
                    if val is not None:
                        summary[f] = val

            elif frame.name == "record":
                row = {}
                for fit_name, col in RECORD_FIELDS.items():
                    if col in row and row[col] is not None:
                        continue  # enhanced-variant had al een waarde
                    val = _value(frame, fit_name)
                    if val is not None:
                        row[col] = val
                if "timestamp" in row:
                    record_rows.append(row)

            elif frame.name == "length":
                row = {f: _value(frame, f) for f in LENGTH_FIELDS}
                # Alleen actieve banen tellen (geen rustpauzes aan de kant).
                if row.get("length_type") == "active":
                    length_rows.append(row)

    if not summary or "sport" not in summary:
        return None

    start = pd.Timestamp(summary.get("start_time") or time_created)
    records = pd.DataFrame(record_rows)
    if not records.empty:
        records["timestamp"] = pd.to_datetime(records["timestamp"])
        # GPS van semicircles naar graden; alleen als het horloge een fix had.
        for col in ("lat", "lon"):
            if col in records:
                records[col] = records[col] * SEMICIRCLE_TO_DEGREES

    return ParsedActivity(
        activity_key=pd.Timestamp(time_created or start).isoformat(),
        sport=str(summary["sport"]),
        sub_sport=str(summary.get("sub_sport")) if summary.get("sub_sport") else None,
        start_time=start,
        summary=summary,
        records=records,
        lengths=pd.DataFrame(length_rows),
        source_file=source_name,
    )


def parse_zip(zip_path: Path | str) -> list[ParsedActivity]:
    """Pak een Garmin-exportzip uit en parse alle FIT-bestanden erin.

    De zip wordt in het geheugen gelezen; er worden geen bestanden op schijf
    uitgepakt. Niet-FIT-bestanden worden overgeslagen.
    """
    activities = []
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            if not info.filename.lower().endswith(".fit"):
                continue
            data = zf.read(info)
            activity = parse_fit(io.BytesIO(data), source_name=info.filename)
            if activity is not None:
                activities.append(activity)
    return activities
