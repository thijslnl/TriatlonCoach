"""De import-pipeline: zip -> parsen -> SQLite -> trainingslog.md.

Dit is de ene plek waar een upload doorheen gaat, zowel vanuit het dashboard
als vanuit de commandline. Deduplicatie gebeurt hier: een activiteit die al
in de database staat wordt overgeslagen (en komt dus ook niet nogmaals in
het trainingslog).
"""

import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from tricoach.fit_parser import ParsedActivity, parse_zip
from tricoach.storage import activity_exists, save_activity
from tricoach.trainingslog import append_entry
from tricoach.weather import WindData
from tricoach.zones import time_in_zones, zone_bounds

# Een functie die op basis van een sessie een observatie-tekst maakt
# (vanaf stap 5 is dit het lokale Ollama-model). Mag None zijn.
ObservationFn = Callable[[ParsedActivity, dict[str, int]], str | None]

# Een functie die voor een sessie de winddata ophaalt (Open-Meteo). Mag None
# zijn (dan wordt geen wind opgehaald) en mag zelf None teruggeven (geen GPS,
# zwemmen, geen internet) — dat mag de import nooit laten mislukken.
WeatherFn = Callable[[ParsedActivity], "WindData | None"]


@dataclass
class ImportResult:
    """Uitkomst van één activiteit in een import.

    ``tiz`` en ``observation`` zijn gemma's voorwerk (tijd-in-zones en de korte
    observatie) en worden meegegeven zodat de feedback-stap erop kan voortbouwen
    zonder Ollama nog eens aan te roepen. ``user_note`` (de opmerking bij de
    upload) en ``wind`` (Open-Meteo) reizen mee zodat de feedback-stap ze als
    context kan meewegen. Bij een duplicaat blijven ze leeg.
    """

    activity: ParsedActivity
    status: str  # "nieuw" of "duplicaat"
    tiz: dict[str, int] = field(default_factory=dict)
    observation: str | None = None
    user_note: str | None = None
    wind: "WindData | None" = None


def import_zip(
    zip_path: Path | str,
    conn: sqlite3.Connection,
    config: dict,
    memory_dir: Path,
    observation_fn: ObservationFn | None = None,
    weather_fn: WeatherFn | None = None,
    user_note: str | None = None,
) -> list[ImportResult]:
    """Importeer alle FIT-activiteiten uit één zip. Geeft per activiteit de status terug.

    ``user_note`` wordt bij elke nieuwe sessie uit deze zip opgeslagen en in het
    trainingslog vermeld. ``weather_fn`` haalt de winddata op (Open-Meteo);
    beide zijn optioneel. Duplicaten worden vóór elke LLM-/API-aanroep
    afgevangen, zodat een tweede upload geen onnodige calls kost.
    """
    bounds = zone_bounds(config["athlete"])
    note = (user_note or "").strip() or None
    results = []

    for act in parse_zip(zip_path):
        # Eerst dedup-check: duplicaten kosten geen Ollama-, weer- of API-call.
        if activity_exists(conn, act.activity_key):
            results.append(ImportResult(act, "duplicaat"))
            continue

        tiz = time_in_zones(act.records, bounds) if not act.records.empty else {}
        observation = observation_fn(act, tiz) if observation_fn else None
        wind = weather_fn(act) if weather_fn else None

        save_activity(conn, act, bounds, user_note=note, wind=wind)
        append_entry(memory_dir, act, tiz, observation, user_note=note, wind=wind)
        results.append(ImportResult(
            act, "nieuw", tiz=tiz, observation=observation, user_note=note, wind=wind,
        ))

    return results
