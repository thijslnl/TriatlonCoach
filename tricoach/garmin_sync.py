"""Synchronisatie met Garmin Connect: wellness-data en activiteiten.

Gebruikt ``python-garminconnect`` (cyberjunky) — een **onofficiële** wrapper om
de Connect-sessie, niet de officiële Health API. Daarom is alles hier op zacht
falen gebouwd: elke netwerk- of API-fout wordt een :class:`GarminSyncError`
(of een overgeslagen dag/activiteit), nooit een crash van het dashboard.

Inloggen:

- Credentials komen uit environment variables ``GARMIN_EMAIL`` en
  ``GARMIN_PASSWORD`` (via ``.env``, dat buiten git blijft) — nooit uit code
  of config.yaml.
- Na de eerste login worden OAuth-tokens bewaard in ``data/garmin_tokens/``
  (binnen de al-geïgnoreerde ``data/``-map); daarna is maandenlang geen
  wachtwoord of MFA meer nodig.
- MFA wordt ondersteund zonder blocking prompt: :func:`connect_client` geeft
  dan status ``"mfa"`` terug en de UI vraagt de code op, waarna
  :func:`complete_mfa` de login afmaakt.

Activiteiten-sync:

- Nieuwe activiteiten worden als origineel FIT-bestand gedownload en door de
  bestaande import-pipeline (:func:`tricoach.importer.import_zip`) gehaald —
  dus mét archief (``uploads/yyyy/mm/...``), trainingslog en dedup.
- Deduplicatie werkt dubbel: vóór het downloaden op de Garmin activity-ID
  (kolom ``activities.garmin_activity_id``, ook teruggevuld uit de
  bestandsnamen van handmatige uploads), en in de pipeline zelf op de
  starttijd-sleutel. Soft-verwijderde sessies blijven daardoor verwijderd.
- Handmatige upload blijft gewoon bestaan (terugvaloptie en voor
  gecorrigeerde/samengevoegde bestanden).
"""

import io
import os
import re
import sqlite3
import zipfile
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path

from tricoach import wellness
from tricoach.config import PROJECT_ROOT
from tricoach.importer import ImportResult, import_zip

# OAuth-tokens van garth; in data/ zodat ze buiten git blijven en een
# container-herstart overleven (data/ is een bind-mount).
TOKEN_DIR = PROJECT_ROOT / "data" / "garmin_tokens"

# Dagen die bij een wellness-sync altijd opnieuw worden opgehaald, ook als ze
# al gevuld zijn: Garmin werkt recente dagen gedurende de dag nog bij.
REFRESH_DAYS = 2


class GarminSyncError(Exception):
    """Sync kon niet (verder): duidelijke melding voor de UI, geen crash."""


def credentials() -> tuple[str | None, str | None]:
    """GARMIN_EMAIL en GARMIN_PASSWORD uit de environment (.env)."""
    return os.getenv("GARMIN_EMAIL"), os.getenv("GARMIN_PASSWORD")


def has_tokens() -> bool:
    """Staan er bewaarde login-tokens? (Zegt niet of ze nog geldig zijn.)"""
    return TOKEN_DIR.exists() and any(TOKEN_DIR.iterdir())


def connect_client(tokens_only: bool = False):
    """Maak verbinding met Garmin Connect.

    Geeft ``("ok", client)`` bij succes of ``("mfa", (client, state))`` als
    Garmin een MFA-code vraagt — de UI toont dan een invoerveld en rondt af
    met :func:`complete_mfa`. Probeert eerst de bewaarde tokens; pas daarna
    (tenzij ``tokens_only``, voor de stille automatische sync) een verse
    login met de credentials uit de environment.

    Raises:
        GarminSyncError: geen tokens/credentials, of het inloggen mislukt
            (verkeerd wachtwoord, Garmin-wijziging, geen netwerk).
    """
    try:
        from garminconnect import Garmin
    except ImportError as e:
        raise GarminSyncError(
            "Package 'garminconnect' is niet geïnstalleerd — draai "
            "'docker compose up -d --build' na de requirements-wijziging."
        ) from e

    if has_tokens():
        try:
            client = Garmin()
            client.login(str(TOKEN_DIR))
            return "ok", client
        except Exception:
            pass  # tokens verlopen/ongeldig: door naar een verse login

    if tokens_only:
        raise GarminSyncError(
            "Geen geldige Garmin-tokens — log eenmalig in via de sync-knop "
            "in de zijbalk."
        )

    email, password = credentials()
    if not email or not password:
        raise GarminSyncError(
            "Geen Garmin-inloggegevens: zet GARMIN_EMAIL en GARMIN_PASSWORD "
            "in .env (naast de ANTHROPIC_API_KEY) en herstart de container."
        )
    try:
        client = Garmin(email=email, password=password, return_on_mfa=True)
        result1, result2 = client.login()
    except Exception as e:
        raise GarminSyncError(f"Inloggen bij Garmin mislukt: {e}") from e

    if result1 == "needs_mfa":
        return "mfa", (client, result2)
    _save_tokens(client)
    # Bij return_on_mfa keert login() vroeg terug ZONDER het profiel te laden;
    # endpoints die de display name in de URL zetten (o.a. de dagsamenvatting
    # met rustpols/stress/body battery) falen dan stilletjes. Opnieuw inloggen
    # via de zojuist bewaarde tokens laadt het profiel wél.
    return "ok", _relogin_via_tokens(client)


def complete_mfa(client, client_state: dict, mfa_code: str):
    """Rond een MFA-login af met de code van de gebruiker en bewaar tokens."""
    try:
        client.resume_login(client_state, mfa_code.strip())
    except Exception as e:
        raise GarminSyncError(f"MFA-code niet geaccepteerd: {e}") from e
    _save_tokens(client)
    return client


def _garth(client):
    """Het interne garth-object: heet ``garth`` (<=0.2.x) of ``client`` (0.3.x)."""
    return getattr(client, "garth", None) or getattr(client, "client", None)


def _save_tokens(client) -> None:
    """Bewaar de OAuth-tokens; mislukken is niet fataal (dan volgende keer
    opnieuw inloggen)."""
    try:
        TOKEN_DIR.mkdir(parents=True, exist_ok=True)
        _garth(client).dump(str(TOKEN_DIR))
    except Exception:
        pass


def _relogin_via_tokens(client):
    """Verse client via de bewaarde tokens (mét profiel); anders de oude."""
    from garminconnect import Garmin

    if not has_tokens():
        return client
    try:
        vers = Garmin()
        vers.login(str(TOKEN_DIR))
        return vers
    except Exception:
        return client


def logout() -> None:
    """Vergeet de bewaarde tokens (koppeling verwijderen)."""
    if TOKEN_DIR.exists():
        for p in TOKEN_DIR.iterdir():
            p.unlink()


# --------------------------------------------------------------- sync-status --

SYNC_STATE_SCHEMA = """
CREATE TABLE IF NOT EXISTS sync_state (
    key   TEXT PRIMARY KEY,
    value TEXT
);
"""


def _set_state(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.executescript(SYNC_STATE_SCHEMA)
    conn.execute(
        "INSERT INTO sync_state (key, value) VALUES (?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (key, value),
    )
    conn.commit()


def get_state(conn: sqlite3.Connection, key: str) -> str | None:
    conn.executescript(SYNC_STATE_SCHEMA)
    row = conn.execute(
        "SELECT value FROM sync_state WHERE key = ?", (key,)).fetchone()
    return row[0] if row else None


def last_sync(conn: sqlite3.Connection) -> datetime | None:
    """Wanneer is er voor het laatst (succesvol) gesynct?"""
    val = get_state(conn, "last_sync")
    try:
        return datetime.fromisoformat(val) if val else None
    except ValueError:
        return None


def mark_synced(conn: sqlite3.Connection, samenvatting: str) -> None:
    """Leg het sync-moment en een korte samenvatting vast (voor de UI)."""
    _set_state(conn, "last_sync",
               datetime.now().isoformat(timespec="seconds"))
    _set_state(conn, "last_sync_summary", samenvatting)


# ------------------------------------------------------------- wellness-sync --

@dataclass
class WellnessSyncResult:
    """Uitkomst van één wellness-sync, voor de melding in de UI."""

    days: int = 0            # dagen bekeken
    fetched: int = 0         # dagen daadwerkelijk opgehaald
    rhr_filled: int = 0      # dagen met rustpols
    hrv_filled: int = 0      # dagen met HRV
    errors: list[str] = field(default_factory=list)


def _get(d, *pad, default=None):
    """Veilig genest uitlezen van de (wisselvallige) Garmin-antwoorden."""
    for sleutel in pad:
        if isinstance(d, list):
            if not isinstance(sleutel, int) or sleutel >= len(d):
                return default
            d = d[sleutel]
        elif isinstance(d, dict):
            d = d.get(sleutel)
        else:
            return default
        if d is None:
            return default
    return d


def _fetch_day(client, dag: date) -> dict:
    """Haal de wellness-waarden van één dag op; elk deel mag los mislukken."""
    iso = dag.isoformat()
    waarden: dict = {}

    try:
        stats = client.get_stats(iso) or {}
        waarden.update({
            "resting_hr": stats.get("restingHeartRate"),
            "stress_avg": stats.get("averageStressLevel"),
            "body_battery_high": stats.get("bodyBatteryHighestValue"),
            "body_battery_low": stats.get("bodyBatteryLowestValue"),
        })
    except Exception:
        pass

    if waarden.get("resting_hr") is None:
        # De dagsamenvatting kan falen of de rustpols weglaten; het aparte
        # rustpols-endpoint is dan de terugval.
        try:
            rhr = client.get_rhr_day(iso)
            waarden["resting_hr"] = _get(
                rhr, "allMetrics", "metricsMap",
                "WELLNESS_RESTING_HEART_RATE", 0, "value")
        except Exception:
            pass

    try:
        hrv = client.get_hrv_data(iso) or {}
        s = hrv.get("hrvSummary") or {}
        waarden.update({
            "hrv_last_night": s.get("lastNightAvg"),
            "hrv_weekly_avg": s.get("weeklyAvg"),
            "hrv_status": s.get("status"),
            "hrv_baseline_low": _get(s, "baseline", "balancedLow"),
            "hrv_baseline_high": _get(s, "baseline", "balancedUpper"),
        })
    except Exception:
        pass

    try:
        slaap = client.get_sleep_data(iso) or {}
        dto = slaap.get("dailySleepDTO") or {}
        waarden.update({
            "sleep_s": dto.get("sleepTimeSeconds"),
            "deep_s": dto.get("deepSleepSeconds"),
            "light_s": dto.get("lightSleepSeconds"),
            "rem_s": dto.get("remSleepSeconds"),
            "awake_s": dto.get("awakeSleepSeconds"),
            "sleep_score": _get(dto, "sleepScores", "overall", "value"),
        })
    except Exception:
        pass

    try:
        readiness = client.get_training_readiness(iso)
        waarden.update({
            "training_readiness": _get(readiness, 0, "score"),
            "readiness_level": _get(readiness, 0, "level"),
        })
    except Exception:
        pass

    try:
        metrics = client.get_max_metrics(iso)
        # Antwoordvorm wisselt per accounttype: soms een lijst, soms een dict.
        eerste = metrics[0] if isinstance(metrics, list) and metrics else metrics
        waarden.update({
            "vo2max_run": _get(eerste, "generic", "vo2MaxPreciseValue")
            or _get(eerste, "generic", "vo2MaxValue"),
            "vo2max_bike": _get(eerste, "cycling", "vo2MaxPreciseValue")
            or _get(eerste, "cycling", "vo2MaxValue"),
        })
    except Exception:
        pass

    return waarden


def sync_wellness(client, conn: sqlite3.Connection,
                  days: int = 30) -> WellnessSyncResult:
    """Haal de wellness-data van de laatste ``days`` dagen op en sla ze op.

    Dagen die al compleet in de database staan (rustpols én HRV) worden
    overgeslagen, behalve de laatste :data:`REFRESH_DAYS` — die werkt Garmin
    gedurende de dag nog bij. Een dag die mislukt wordt overgeslagen, nooit
    fataal.
    """
    res = WellnessSyncResult(days=days)
    vandaag = date.today()
    for terug in range(days):
        dag = vandaag - timedelta(days=terug)
        if terug >= REFRESH_DAYS and wellness.day_is_complete(conn, dag):
            continue
        try:
            waarden = _fetch_day(client, dag)
        except Exception as e:  # vangnet; _fetch_day vangt zelf al per deel
            res.errors.append(f"{dag}: {e}")
            continue
        if not any(v is not None for v in waarden.values()):
            continue
        wellness.upsert_day(conn, dag, waarden)
        res.fetched += 1
        if waarden.get("resting_hr") is not None:
            res.rhr_filled += 1
        if waarden.get("hrv_last_night") is not None:
            res.hrv_filled += 1
    return res


# ---------------------------------------------------------- activiteiten-sync --

@dataclass
class ActivitySyncResult:
    """Uitkomst van één activiteiten-sync."""

    new: list[ImportResult] = field(default_factory=list)  # vers geïmporteerd
    skipped_known: int = 0    # al bekend (activity-ID), niet gedownload
    duplicates: int = 0       # gedownload maar dubbel op starttijd-sleutel
    deleted_kept: int = 0     # soft-verwijderd en verwijderd gebleven
    errors: list[str] = field(default_factory=list)


def ensure_garmin_ids(conn: sqlite3.Connection) -> int:
    """Vul ``garmin_activity_id`` terug uit de bestandsnamen van eerdere imports.

    Garmin-exports heten ``<activityid>_ACTIVITY.fit`` (dezelfde conventie
    als :func:`tricoach.archive.activity_id`), dus voor handmatig geüploade
    sessies is de ID uit ``source_file`` te halen. Zo herkent de sync een
    handmatig geüploade activiteit vóór het downloaden — en andersom. Geeft
    het aantal aangevulde rijen terug; idempotent.
    """
    rows = conn.execute(
        "SELECT activity_key, source_file FROM activities "
        "WHERE garmin_activity_id IS NULL AND source_file IS NOT NULL"
    ).fetchall()
    n = 0
    for key, source_file in rows:
        m = re.search(r"\d{6,}", Path(source_file).name)
        if not m:
            continue
        conn.execute(
            "UPDATE activities SET garmin_activity_id = ? WHERE activity_key = ?",
            (m.group(), key),
        )
        n += 1
    conn.commit()
    return n


def known_garmin_ids(conn: sqlite3.Connection) -> set[str]:
    """Alle bekende Garmin activity-ID's, inclusief soft-verwijderde sessies —
    een verwijderde activiteit mag de sync niet opnieuw binnenhalen."""
    return {row[0] for row in conn.execute(
        "SELECT garmin_activity_id FROM activities "
        "WHERE garmin_activity_id IS NOT NULL")}


def _download_as_zip(client, aid: str) -> bytes:
    """Download het originele bestand en lever het als zip met de
    standaard-bestandsnaam ``<activityid>_ACTIVITY.fit``.

    Garmin's 'origineel'-download is normaal al een zip, maar de naam van het
    FIT-bestand daarin varieert; we herverpakken zodat archief en dedup
    dezelfde naamconventie zien als bij een handmatige upload.
    """
    from garminconnect import Garmin

    data = client.download_activity(
        aid, dl_fmt=Garmin.ActivityDownloadFormat.ORIGINAL)

    fit_bytes = None
    if data[:2] == b"PK":  # zip
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            for info in zf.infolist():
                if info.filename.lower().endswith(".fit"):
                    fit_bytes = zf.read(info)
                    break
    else:  # los FIT-bestand
        fit_bytes = data
    if not fit_bytes:
        raise GarminSyncError(f"Download van activiteit {aid} bevat geen FIT-bestand")

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(f"{aid}_ACTIVITY.fit", fit_bytes)
    return buf.getvalue()


def sync_activities(
    client,
    conn: sqlite3.Connection,
    config: dict,
    memory_dir: Path,
    days: int = 14,
    observation_fn=None,
    weather_fn=None,
    uploads_dir: Path | None = None,
    tmp_dir: Path | None = None,
) -> ActivitySyncResult:
    """Haal nieuwe activiteiten van de laatste ``days`` dagen op en importeer ze.

    Elke onbekende activiteit gaat als origineel FIT-bestand door
    :func:`tricoach.importer.import_zip` — precies dezelfde route als een
    handmatige upload (archief, trainingslog, observatie, wind, dedup).
    De aanroeper beslist wat er met de nieuwe sessies gebeurt (feedback
    genereren, tonen); een mislukte download of import van één activiteit
    wordt genoteerd en overgeslagen.
    """
    res = ActivitySyncResult()
    ensure_garmin_ids(conn)
    bekend = known_garmin_ids(conn)

    vandaag = date.today()
    try:
        lijst = client.get_activities_by_date(
            (vandaag - timedelta(days=days)).isoformat(), vandaag.isoformat())
    except Exception as e:
        raise GarminSyncError(f"Activiteitenlijst ophalen mislukt: {e}") from e

    tmp_dir = tmp_dir or (PROJECT_ROOT / "data" / "sync_tmp")
    tmp_dir.mkdir(parents=True, exist_ok=True)

    for activiteit in lijst or []:
        aid = str(activiteit.get("activityId") or "")
        if not aid:
            continue
        if aid in bekend:
            res.skipped_known += 1
            continue
        try:
            zip_bytes = _download_as_zip(client, aid)
            tmp_zip = tmp_dir / f"{aid}.zip"
            tmp_zip.write_bytes(zip_bytes)
            try:
                results = import_zip(
                    tmp_zip, conn, config, memory_dir,
                    observation_fn=observation_fn, weather_fn=weather_fn,
                    uploads_dir=uploads_dir,
                )
            finally:
                tmp_zip.unlink(missing_ok=True)
        except GarminSyncError as e:
            res.errors.append(str(e))
            continue
        except Exception as e:
            res.errors.append(f"Activiteit {aid}: {e}")
            continue

        for r in results:
            # ID registreren, óók bij duplicaten en verwijderde sessies:
            # dan slaat de volgende sync de download meteen over.
            conn.execute(
                "UPDATE activities SET garmin_activity_id = ? "
                "WHERE activity_key = ? AND garmin_activity_id IS NULL",
                (aid, r.activity.activity_key),
            )
            conn.commit()
            if r.status == "nieuw":
                res.new.append(r)
            elif r.status == "verwijderd":
                res.deleted_kept += 1
            else:
                res.duplicates += 1
        bekend.add(aid)

    return res
