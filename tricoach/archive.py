"""Onveranderlijk archief van originele FIT-uploads.

Elk geüpload FIT-bestand (uit de Garmin-zip) wordt vóór interpretatie
bewaard onder ``uploads/yyyy/mm/yyyy-mm-dd_HHmm_<activityid>.fit``:

- datum/tijd = de **lokale starttijd** van de sessie (Europe/Amsterdam, via
  :func:`tricoach.formatting.local_time`), niet het uploadmoment, en zonder
  dubbele punten (Windows-onverenigbaar) — vandaar ``HHmm``;
- ``<activityid>`` = de Garmin activity-ID uit de bestandsnaam van de upload
  (bijv. ``23627119348_ACTIVITY.fit``); die koppelt archief ↔ database ↔
  deduplicatie. Ontbreekt hij, dan geldt de starttijd-sleutel als terugval.

Bewust géén submappen per sport: de sport is pas bekend ná het parsen (een
verkeerd herkende sport zou het bestand in de verkeerde map zetten; een datum
is nooit fout), multisport-bestanden bevatten meerdere sporten in één bestand,
en filteren op sport is een database-query — het bestandssysteem is het
chronologische archief, de database de betekenislaag.

Wordt dezelfde activity-ID opnieuw geüpload met ándere inhoud (correctie),
dan wordt niets overschreven maar komt er ``_v2``, ``_v3``, ... bij; de
database (``activities.archived_path``) wijst de actieve versie aan.
Byte-identieke heruploads leveren géén nieuwe versie op. Verificatieruns
(:func:`verify_originals`) parsen de originelen opnieuw en vergelijken het
resultaat met de database; sessies met ``original_missing`` slaan ze over.
"""

import hashlib
import io
import re
import sqlite3
import zipfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from tricoach.config import PROJECT_ROOT
from tricoach.fit_parser import ParsedActivity, parse_fit
from tricoach.formatting import local_time
from tricoach.storage import set_archived_path

# Standaard-archiefmap (relatief aan de projectroot) als de config er geen
# noemt; config.yaml kan hem overschrijven via paths.uploads_dir.
DEFAULT_UPLOADS_DIR = "uploads"


def uploads_root(config: dict) -> Path:
    """De absolute archiefmap uit de config (``paths.uploads_dir``), met
    terugval op ``uploads/`` in de projectroot."""
    rel = (config.get("paths") or {}).get("uploads_dir", DEFAULT_UPLOADS_DIR)
    return PROJECT_ROOT / rel


def activity_id(act: ParsedActivity) -> str:
    """De Garmin activity-ID van een sessie, uit de upload-bestandsnaam.

    Garmin-exports heten ``<activityid>_ACTIVITY.fit``; de eerste lange
    cijferreeks in de naam is de ID. Ontbreekt die (hernoemd bestand), dan
    geldt de starttijd-sleutel (cijfers van de activity_key) als stabiele
    terugval — die is per activiteit even uniek.
    """
    naam = Path(act.source_file).name
    m = re.search(r"\d{6,}", naam)
    if m:
        return m.group()
    return re.sub(r"\D", "", act.activity_key)


@dataclass
class ArchiveResult:
    """Uitkomst van één archivering: het pad (relatief aan de projectroot),
    het versienummer en of er daadwerkelijk een nieuw bestand is geschreven."""

    rel_path: str
    version: int
    is_new_file: bool


def _versions(root: Path, aid: str) -> list[Path]:
    """Alle gearchiveerde versies van deze activity-ID, over het hele archief."""
    if not root.exists():
        return []
    return sorted(
        p for p in root.glob(f"*/*/*_{aid}*.fit")
        if re.fullmatch(rf".*_{re.escape(aid)}(_v\d+)?\.fit", p.name)
    )


def _version_number(path: Path) -> int:
    """Versienummer uit een archiefbestandsnaam (zonder suffix = versie 1)."""
    m = re.search(r"_v(\d+)\.fit$", path.name)
    return int(m.group(1)) if m else 1


def archive_original(root: Path, act: ParsedActivity,
                     data: bytes) -> ArchiveResult:
    """Bewaar het originele FIT-bestand van een sessie in het archief.

    Byte-identiek aan een al gearchiveerde versie? Dan komt er niets bij en
    wijst het resultaat naar die bestaande versie. Nieuwe of gewijzigde
    inhoud krijgt het eerstvolgende versienummer (``_v2``, ``_v3``, ...);
    bestaande bestanden worden nooit overschreven.
    """
    aid = activity_id(act)
    lokaal = local_time(act.start_time)
    bestaand = _versions(root, aid)

    digest = hashlib.sha256(data).hexdigest()
    for p in bestaand:
        if hashlib.sha256(p.read_bytes()).hexdigest() == digest:
            return ArchiveResult(str(p.relative_to(PROJECT_ROOT)),
                                 _version_number(p), False)

    versie = max((_version_number(p) for p in bestaand), default=0) + 1
    suffix = "" if versie == 1 else f"_v{versie}"
    doel = (root / f"{lokaal:%Y}" / f"{lokaal:%m}"
            / f"{lokaal:%Y-%m-%d_%H%M}_{aid}{suffix}.fit")
    doel.parent.mkdir(parents=True, exist_ok=True)
    doel.write_bytes(data)
    return ArchiveResult(str(doel.relative_to(PROJECT_ROOT)), versie, True)


def archive_and_register(conn: sqlite3.Connection, root: Path,
                         act: ParsedActivity, data: bytes) -> ArchiveResult:
    """Archiveer een origineel én registreer de actieve versie in de database.

    Een nieuw geschreven versie wordt altijd de actieve (``archived_path``);
    bij een byte-identieke herupload wordt alleen een nog lege registratie
    gevuld, zodat een eerder aangewezen versie actief blijft.
    """
    res = archive_original(root, act, data)
    if res.is_new_file:
        set_archived_path(conn, act.activity_key, res.rel_path)
    else:
        row = conn.execute(
            "SELECT archived_path FROM activities WHERE activity_key = ?",
            (act.activity_key,)).fetchone()
        if row is not None and row[0] is None:
            set_archived_path(conn, act.activity_key, res.rel_path)
    return res


# -------------------------------------------------------------------- migratie --

def migrate_originals(conn: sqlite3.Connection, root: Path,
                      search_dirs: list[Path]) -> int:
    """Eenmalige inhaalslag: sluimerende originelen alsnog archiveren.

    Doorzoekt ``search_dirs`` (oude importmappen, tijdelijke mappen) op losse
    FIT-bestanden en zips, archiveert alles wat bij een al geïmporteerde
    sessie hoort en registreert het pad. Sessies waarvoor géén origineel
    opduikt houden hun ``original_missing``-markering (gezet bij de
    schemamigratie), zodat verificatieruns ze overslaan in plaats van falen.
    Idempotent en goedkoop bij lege mappen. Geeft het aantal sessies terug
    waarvoor een origineel is gearchiveerd.
    """
    bekend = {row[0] for row in conn.execute("SELECT activity_key FROM activities")}
    n = 0
    for map_ in search_dirs:
        if not map_.exists():
            continue
        for pad in sorted(map_.rglob("*")):
            if pad.suffix.lower() == ".fit":
                kandidaten = [(pad.name, pad.read_bytes())]
            elif pad.suffix.lower() == ".zip":
                try:
                    with zipfile.ZipFile(pad) as zf:
                        kandidaten = [
                            (info.filename, zf.read(info))
                            for info in zf.infolist()
                            if info.filename.lower().endswith(".fit")
                        ]
                except zipfile.BadZipFile:
                    continue
            else:
                continue
            for naam, data in kandidaten:
                try:
                    act = parse_fit(io.BytesIO(data), source_name=naam)
                except Exception:
                    continue  # geen geldig FIT-bestand: overslaan
                if act is None or act.activity_key not in bekend:
                    continue
                archive_and_register(conn, root, act, data)
                n += 1
    return n


# ----------------------------------------------------------------- verificatie --

def verify_originals(conn: sqlite3.Connection,
                     parser=parse_fit) -> pd.DataFrame:
    """Verificatierun: parse elk gearchiveerd origineel opnieuw en vergelijk
    met de database.

    Per niet-verwijderde sessie: status ``ok`` (opnieuw geparst en de
    kernvelden — sleutel, sport, duur, afstand, gem. hartslag — komen
    overeen), ``afwijking`` (met detail), ``bestand_weg`` (pad geregistreerd
    maar bestand verdwenen) of ``overgeslagen`` (``original_missing``: er is
    geen origineel, dat is bekend en geen fout). ``parser`` is injecteerbaar
    voor tests. Geeft een DataFrame met kolommen activity_key, status, detail.
    """
    rows = conn.execute(
        "SELECT activity_key, sport, duration_s, distance_m, avg_hr, "
        "archived_path, original_missing FROM activities "
        "WHERE deleted_at IS NULL").fetchall()
    uit = []
    for key, sport, duur, afstand, avg_hr, pad, missing in rows:
        if pad is None:
            uit.append({"activity_key": key, "status": "overgeslagen",
                        "detail": "origineel ontbreekt (van vóór het archief)"
                        if missing else "geen archiefpad geregistreerd"})
            continue
        bestand = PROJECT_ROOT / pad
        if not bestand.exists():
            uit.append({"activity_key": key, "status": "bestand_weg",
                        "detail": pad})
            continue
        act = parser(io.BytesIO(bestand.read_bytes()), source_name=bestand.name)
        if act is None:
            uit.append({"activity_key": key, "status": "afwijking",
                        "detail": "bestand is niet (meer) als activiteit te parsen"})
            continue
        fouten = _compare(act, key, sport, duur, afstand, avg_hr)
        uit.append({
            "activity_key": key,
            "status": "ok" if not fouten else "afwijking",
            "detail": "; ".join(fouten),
        })
    return pd.DataFrame(uit)


def _compare(act: ParsedActivity, key: str, sport: str, duur: float | None,
             afstand: float | None, avg_hr: float | None) -> list[str]:
    """Vergelijk de kernvelden van een herparse met de databaserij."""
    fouten = []
    if act.activity_key != key:
        fouten.append(f"sleutel: {act.activity_key} ≠ {key}")
    if act.sport != sport:
        fouten.append(f"sport: {act.sport} ≠ {sport}")
    s = act.summary
    # Duur/afstand met een kleine marge: floats uit het FIT-bestand vs SQLite.
    if duur is not None and s.get("total_timer_time") is not None \
            and abs(s["total_timer_time"] - duur) > 1.0:
        fouten.append(f"duur: {s['total_timer_time']:.0f}s ≠ {duur:.0f}s")
    if afstand is not None and s.get("total_distance") is not None \
            and abs(s["total_distance"] - afstand) > 1.0:
        fouten.append(f"afstand: {s['total_distance']:.0f}m ≠ {afstand:.0f}m")
    if avg_hr is not None and s.get("avg_heart_rate") is not None \
            and int(s["avg_heart_rate"]) != int(avg_hr):
        fouten.append(f"gem. HR: {s['avg_heart_rate']} ≠ {avg_hr}")
    return fouten
