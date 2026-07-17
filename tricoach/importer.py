"""De import-pipeline: zip -> parsen -> archief -> SQLite -> trainingslog.md.

Dit is de ene plek waar een upload doorheen gaat, zowel vanuit het dashboard
als vanuit de commandline. Deduplicatie gebeurt hier: een activiteit die al
in de database staat wordt overgeslagen (en komt dus ook niet nogmaals in
het trainingslog). Een duplicaat is niet helemaal voor niets: ontbrekende
summary-velden (zoals de loopdynamiek van sessies die vóór die uitbreiding
zijn geïmporteerd) worden dan alsnog aangevuld uit het FIT-bestand.

Elk aangeleverd FIT-bestand — óók van duplicaten — gaat eerst het
origineel-archief in (zie :mod:`tricoach.archive`), zodat er altijd een
onaangetast origineel bewaard blijft voor verificatie en terugrol.
"""

import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from tricoach.archive import archive_and_register
from tricoach.fit_parser import ParsedActivity, parse_zip
from tricoach.storage import (
    activity_exists,
    enrich_activity_summary,
    enrich_power_records,
    is_deleted,
    save_activity,
)
from tricoach.trainingslog import append_entry
from tricoach.transport import suggest_transport
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
    upload), ``training_label`` (het sessielabel van de atleet, bijv.
    "techniek/cadans") en ``wind`` (Open-Meteo) reizen mee zodat de
    feedback-stap ze als context kan meewegen. Bij een duplicaat blijven ze
    leeg; ``enriched`` is dan True als de herimport ontbrekende summary-velden
    heeft aangevuld.

    ``transport_suggested`` betekent: dit lijkt een transport-ritje (korte,
    rustige fietsrit) — de UI hoort dan een bevestigingsvraag te tonen en de
    coach-feedback uit te stellen tot de gebruiker heeft gekozen.
    ``archived`` is het relatieve pad van het gearchiveerde origineel (None
    als archiveren uitstond of de bytes ontbraken).
    """

    activity: ParsedActivity
    status: str  # "nieuw", "duplicaat" of "verwijderd" (blijft verwijderd)
    tiz: dict[str, int] = field(default_factory=dict)
    observation: str | None = None
    user_note: str | None = None
    training_label: str | None = None
    wind: "WindData | None" = None
    enriched: bool = False
    transport_suggested: bool = False
    archived: str | None = None


def import_zip(
    zip_path: Path | str,
    conn: sqlite3.Connection,
    config: dict,
    memory_dir: Path,
    observation_fn: ObservationFn | None = None,
    weather_fn: WeatherFn | None = None,
    user_note: str | None = None,
    training_label: str | None = None,
    uploads_dir: Path | None = None,
) -> list[ImportResult]:
    """Importeer alle FIT-activiteiten uit één zip. Geeft per activiteit de status terug.

    ``user_note`` en ``training_label`` worden bij elke nieuwe sessie uit deze
    zip opgeslagen en in het trainingslog vermeld. ``weather_fn`` haalt de
    winddata op (Open-Meteo); alles is optioneel. Duplicaten worden vóór elke
    LLM-/API-aanroep afgevangen, zodat een tweede upload geen onnodige calls
    kost — wel worden bij een duplicaat ontbrekende summary-velden aangevuld
    (zo krijgt een herupload van oude zips de loopdynamiek alsnog binnen).

    ``uploads_dir`` is de archiefmap voor de originele FIT-bestanden; met
    None (bijv. in tests) wordt er niet gearchiveerd. Archiveren gebeurt ook
    voor duplicaten en verwijderde sessies: elk aangeleverd origineel blijft
    bewaard, byte-identieke heruploads leveren geen nieuwe versie op.
    """
    bounds = zone_bounds(config["athlete"])
    ftp = config["athlete"].get("ftp") or None
    note = (user_note or "").strip() or None
    label = (training_label or "").strip() or None
    results = []

    for act in parse_zip(zip_path):
        # Eerst dedup-check: duplicaten kosten geen Ollama-, weer- of API-call.
        # Een soft-verwijderde sessie telt óók als bekend en blijft verwijderd:
        # de status "verwijderd" laat de UI uitleggen hoe je hem kunt herstellen.
        if activity_exists(conn, act.activity_key):
            status = "verwijderd" if is_deleted(conn, act.activity_key) else "duplicaat"
            enriched = enrich_activity_summary(conn, act)
            # Herupload van een rit die vóór de power-uitbreiding is
            # geïmporteerd: vul de vermogensdata (records + afgeleide
            # kolommen) alsnog aan.
            enriched = enrich_power_records(conn, act, ftp) or enriched
            archived = _archive(conn, uploads_dir, act)
            results.append(ImportResult(act, status, enriched=enriched,
                                        archived=archived))
            continue

        tiz = time_in_zones(act.records, bounds) if not act.records.empty else {}
        observation = observation_fn(act, tiz) if observation_fn else None
        wind = weather_fn(act) if weather_fn else None

        save_activity(conn, act, bounds, user_note=note, wind=wind,
                      training_label=label, ftp=ftp)
        archived = _archive(conn, uploads_dir, act)
        append_entry(memory_dir, act, tiz, observation, user_note=note,
                     wind=wind, training_label=label)
        results.append(ImportResult(
            act, "nieuw", tiz=tiz, observation=observation, user_note=note,
            training_label=label, wind=wind,
            transport_suggested=suggest_transport(act, config),
            archived=archived,
        ))

    return results


def _archive(conn: sqlite3.Connection, uploads_dir: Path | None,
             act: ParsedActivity) -> str | None:
    """Archiveer het origineel van één activiteit; None als dat niet kan.

    Archiveren mag een import nooit laten mislukken: bij een schrijffout
    blijft de sessie gewoon geïmporteerd, alleen zonder geregistreerd
    origineel (en dus zichtbaar in de verificatierun).
    """
    if uploads_dir is None or act.raw_data is None:
        return None
    try:
        return archive_and_register(conn, uploads_dir, act, act.raw_data).rel_path
    except OSError:
        return None
