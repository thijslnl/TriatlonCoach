"""SQLite-opslag voor geparste activiteiten.

Drie tabellen:

- ``activities``: één rij per sessie met de samenvatting én de vooraf
  berekende tijd-in-zones (z1_s..z5_s), zodat het dashboard snel kan tekenen.
- ``records``: de seconde-data (hartslag, snelheid, ...) per sessie.
- ``lengths``: baandata bij zwemmen (slagtype, slagen, tijd per baan).

De primaire sleutel van ``activities`` is ``activity_key`` (de starttijd uit
het FIT-bestand); een tweede import van dezelfde activiteit wordt daardoor
genegeerd in plaats van gedupliceerd.

Verwijderen is een *soft delete*: ``deleted_at`` wordt gevuld en
``load_activities`` filtert die rijen weg, maar de rij (en de records/banen)
blijft staan. Zo blijft de dedup werken — een verwijderde sessie komt bij een
herimport niet stilletjes terug — en is herstellen altijd mogelijk. Definitief
wissen kan met ``purge_activity``.
"""

import json
import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd

from tricoach.fit_parser import ParsedActivity
from tricoach.power import (
    POWER_ZONE_NAMES,
    cadence_stats,
    efficiency_factor,
    normalized_power,
    power_source,
    time_in_power_zones,
)
from tricoach.timebasis import active_seconds
from tricoach.zones import pct_in_zone2, time_in_zones

SCHEMA = """
CREATE TABLE IF NOT EXISTS activities (
    activity_key  TEXT PRIMARY KEY,
    sport         TEXT NOT NULL,
    sub_sport     TEXT,
    start_time    TEXT NOT NULL,
    duration_s    REAL,
    distance_m    REAL,
    avg_hr        INTEGER,
    max_hr        INTEGER,
    avg_speed_ms  REAL,
    avg_cadence   REAL,
    total_ascent  REAL,
    calories      REAL,
    pool_length   REAL,
    num_lengths   INTEGER,
    total_strokes INTEGER,
    z1_s INTEGER, z2_s INTEGER, z3_s INTEGER, z4_s INTEGER, z5_s INTEGER,
    pct_in_zone2       REAL,
    aerobic_efficiency REAL,
    avg_power          REAL,
    np_power           REAL,
    avg_cadence_excl0  REAL,
    max_cadence        REAL,
    is_indoor          INTEGER,
    power_source       TEXT,
    ef_watt            REAL,
    p1_s INTEGER, p2_s INTEGER, p3_s INTEGER,
    p4_s INTEGER, p5_s INTEGER, p6_s INTEGER,
    user_note      TEXT,
    training_label TEXT,
    wind_speed     REAL,
    wind_direction REAL,
    wind_gusts     REAL,
    temperature_c  REAL,
    excluded_reason  TEXT,
    archived_path    TEXT,
    original_missing INTEGER,
    summary_json  TEXT,
    source_file   TEXT,
    imported_at   TEXT,
    deleted_at    TEXT
);
CREATE TABLE IF NOT EXISTS records (
    activity_key TEXT NOT NULL REFERENCES activities(activity_key),
    timestamp    TEXT NOT NULL,
    heart_rate   INTEGER,
    speed_ms     REAL,
    distance_m   REAL,
    cadence      REAL,
    power        REAL,
    altitude_m   REAL
);
CREATE INDEX IF NOT EXISTS idx_records_key ON records(activity_key);
CREATE TABLE IF NOT EXISTS lengths (
    activity_key     TEXT NOT NULL REFERENCES activities(activity_key),
    timestamp        TEXT,
    start_time       TEXT,
    total_timer_time REAL,
    total_strokes    INTEGER,
    swim_stroke      TEXT
);
CREATE INDEX IF NOT EXISTS idx_lengths_key ON lengths(activity_key);
"""


def connect(db_path: Path | str) -> sqlite3.Connection:
    """Open (en maak zo nodig) de database, inclusief schema."""
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.executescript(SCHEMA)
    _migrate(conn)
    return conn


# Kolommen die later zijn toegevoegd; bestaande databases krijgen ze via een
# ALTER TABLE (CREATE TABLE IF NOT EXISTS voegt geen kolommen toe aan een
# bestaande tabel). De waarden blijven NULL tot een import of recompute ze vult.
_ADDED_COLUMNS = {
    "pct_in_zone2": "REAL",
    "aerobic_efficiency": "REAL",
    # Vrije opmerking bij de upload en de automatisch opgehaalde winddata
    # (Open-Meteo). Blijven NULL bij oude sessies, bij zwemmen en bij sessies
    # zonder GPS of internet.
    "user_note": "TEXT",
    "wind_speed": "REAL",
    "wind_direction": "REAL",
    "wind_gusts": "REAL",
    # Temperatuur tijdens de sessie (Open-Meteo), voor de feedback-context.
    "temperature_c": "REAL",
    # Sessielabel van de atleet, bijv. "techniek/cadans": een bewuste
    # cadans-oefensessie waarbij een hogere hartslag verwacht en oké is.
    # De feedback weegt dit mee; NULL = gewoon een normale sessie.
    "training_label": "TEXT",
    # Soft delete: gevuld = de sessie is verwijderd en telt nergens meer mee,
    # maar de rij blijft staan zodat de dedup op activity_key intact blijft
    # en herstellen mogelijk is.
    "deleted_at": "TEXT",
    # Fietsvermogen en -cadans (sinds de Rally-pedalen/Kickr-trainer, juli
    # 2026): gemiddeld en genormaliseerd vermogen, cadans exclusief nul
    # (Garmin-conventie) en het maximum, indoor-markering (Zwift/trainer),
    # de vermogensbron (pedalen vs trainer — trends per bron vergelijken) en
    # de efficiency factor (NP per hartslag). Oude sessies blijven NULL.
    "avg_power": "REAL",
    "np_power": "REAL",
    "avg_cadence_excl0": "REAL",
    "max_cadence": "REAL",
    "is_indoor": "INTEGER",
    "power_source": "TEXT",
    "ef_watt": "REAL",
    # Tijd per vermogenszone (Coggan, % van FTP); NULL zolang de FTP onbekend
    # is en herrekend via recompute_power_zones zodra hij wijzigt.
    "p1_s": "INTEGER", "p2_s": "INTEGER", "p3_s": "INTEGER",
    "p4_s": "INTEGER", "p5_s": "INTEGER", "p6_s": "INTEGER",
    # Actieve/bewegende tijd (zie tricoach.timebasis): de ene tijdsbasis voor
    # tempo, snelheid en EF. Bestaande sessies worden bij het toevoegen van
    # de kolom eenmalig herrekend (zie _migrate/recompute_active_time), zodat
    # trends niet halverwege van definitie wisselen.
    "active_s": "REAL",
    # Uitsluiting van trainingsanalyses (zie tricoach.transport): "transport"
    # = geen training (ritje naar het zwembad, boodschappen). De sessie blijft
    # zichtbaar en telt mee in weektotalen/belasting, maar niet in trends,
    # records, zone-statistieken, brick-detectie en de feedback-vergelijking.
    # NULL = gewone trainingssessie.
    "excluded_reason": "TEXT",
    # Origineel-archief (zie tricoach.archive): het pad (relatief aan de
    # projectroot) van de actieve archiefversie van het originele FIT-bestand,
    # en de markering dat er géén origineel meer bestaat (sessies van vóór het
    # archief) zodat verificatieruns die sessies kunnen overslaan.
    "archived_path": "TEXT",
    "original_missing": "INTEGER",
    # Garmin activity-ID (zie tricoach.garmin_sync): dedupliceert de
    # automatische sync tegen handmatige uploads vóór het downloaden. Wordt
    # gevuld door de sync en teruggevuld uit de upload-bestandsnamen.
    "garmin_activity_id": "TEXT",
}


def _migrate(conn: sqlite3.Connection) -> None:
    """Voeg ontbrekende kolommen toe aan al bestaande tabellen."""
    have = {row[1] for row in conn.execute("PRAGMA table_info(activities)")}
    for name, decl in _ADDED_COLUMNS.items():
        if name not in have:
            conn.execute(f"ALTER TABLE activities ADD COLUMN {name} {decl}")
    have_lengths = {row[1] for row in conn.execute("PRAGMA table_info(lengths)")}
    if have_lengths and "start_time" not in have_lengths:
        # Per-baan-starttijd (sinds de CSS-schatting); oude rijen blijven NULL.
        conn.execute("ALTER TABLE lengths ADD COLUMN start_time TEXT")
    have_records = {row[1] for row in conn.execute("PRAGMA table_info(records)")}
    if have_records and "power" not in have_records:
        # Vermogen per meetpunt (sinds de vermogensmeters); oude rijen NULL.
        conn.execute("ALTER TABLE records ADD COLUMN power REAL")
    conn.commit()
    # Eenmalige datamigratie: sessies van vóór de actieve-tijd-definitie
    # (active_s nog NULL) worden herrekend, zodat álle sessies dezelfde
    # tijdsbasis hebben. Idempotent: daarna is er niets meer te doen.
    if have and "active_s" not in have:
        recompute_active_time(conn)
    # Datamigratie voor het origineel-archief, idempotent op de data (niet op
    # het schema): elke sessie zonder geregistreerd origineel — geïmporteerd
    # vóór het archief bestond, of waarvan het archiveren ooit mislukte —
    # krijgt de markering "origineel ontbreekt", zodat verificatieruns haar
    # overslaan in plaats van falen. Duikt het origineel later alsnog op
    # (herupload, of tricoach.archive.migrate_originals), dan wist
    # set_archived_path de markering weer. Invariant na elke connect: een rij
    # heeft óf een archived_path, óf original_missing = 1.
    conn.execute(
        "UPDATE activities SET original_missing = 1 "
        "WHERE archived_path IS NULL AND original_missing IS NULL")
    conn.commit()


def recompute_active_time(conn: sqlite3.Connection) -> int:
    """Herbereken de actieve tijd (en afgeleiden) van sessies zonder ``active_s``.

    Per sessie wordt de actieve tijd bepaald volgens
    :func:`tricoach.timebasis.active_seconds` en worden ``avg_speed_ms``
    (afstand / actieve tijd) en ``aerobic_efficiency`` daarop herrekend —
    de definitie waarmee ook nieuwe imports worden opgeslagen. Soft-verwijderde
    sessies doen mee, zodat een herstelde sessie dezelfde basis heeft.
    Geeft het aantal bijgewerkte sessies terug.
    """
    rows = conn.execute(
        "SELECT activity_key, sport, duration_s, distance_m, avg_hr "
        "FROM activities WHERE active_s IS NULL").fetchall()
    n = 0
    for key, sport, duration_s, distance_m, avg_hr in rows:
        records = load_records(conn, key) if sport == "cycling" else None
        lengths = load_lengths(conn, key) if sport == "swimming" else None
        actief = active_seconds(sport, duration_s, records, lengths)
        if actief is None:
            continue
        speed = (distance_m / actief) if distance_m and actief > 0 else None
        eff = aerobic_efficiency(sport, speed, avg_hr)
        conn.execute(
            "UPDATE activities SET active_s = ?, "
            "avg_speed_ms = COALESCE(?, avg_speed_ms), "
            "aerobic_efficiency = COALESCE(?, aerobic_efficiency) "
            "WHERE activity_key = ?",
            (actief, speed, eff, key))
        n += 1
    conn.commit()
    return n


def aerobic_efficiency(sport: str, avg_speed_ms: float | None,
                       avg_hr: float | None) -> float | None:
    """Aerobe efficiëntie = gemiddelde snelheid (m/s) per hartslag.

    Hoger = meer snelheid bij dezelfde hartslag. Alleen zinvol voor lopen en
    fietsen; zwemmen valt af omdat de pols-HR onder water onbetrouwbaar is.
    Geeft None als de snelheid of hartslag ontbreekt.
    """
    if sport not in ("running", "cycling"):
        return None
    if not avg_speed_ms or not avg_hr:
        return None
    return avg_speed_ms / avg_hr


def _power_fields(act: ParsedActivity, ftp: float | None) -> dict:
    """De power-/cadanskolommen voor één fietssessie, uit de seconde-data.

    Voor andere sporten (en fietssessies zonder powerdata) blijft alles None —
    het loopvermogen dat het horloge schat blijft bewust in ``summary_json``
    (zie :func:`tricoach.analysis.run_power`), zodat de fietskolommen zuiver
    één bron per sport bevatten. De tijd-in-vermogenszones (p1..p6) blijft
    None zolang de FTP onbekend is.
    """
    velden = dict.fromkeys(
        ["avg_power", "np_power", "avg_cadence_excl0", "max_cadence",
         "is_indoor", "power_source", "ef_watt"], None)
    velden.update(dict.fromkeys([f"p{i}_s" for i in range(1, 7)], None))
    if act.sport != "cycling":
        return velden

    indoor = act.is_indoor
    velden["is_indoor"] = int(indoor)

    heeft_power = (not act.records.empty and "power" in act.records
                   and act.records["power"].notna().any())
    if not heeft_power:
        return velden

    s = act.summary
    np_watt = normalized_power(act.records) or s.get("normalized_power")
    cad = cadence_stats(act.records)
    velden.update({
        "avg_power": s.get("avg_power"),
        "np_power": np_watt,
        "avg_cadence_excl0": cad["excl0"],
        "max_cadence": cad["max"] if cad["max"] is not None else s.get("max_cadence"),
        "power_source": power_source(act.devices, indoor),
        "ef_watt": efficiency_factor(np_watt, s.get("avg_heart_rate")),
    })
    if velden["avg_power"] is None and not act.records.empty:
        velden["avg_power"] = float(act.records["power"].dropna().mean())
    if ftp:
        tip = time_in_power_zones(act.records, ftp)
        velden.update({f"p{i}_s": tip[naam]
                       for i, naam in enumerate(POWER_ZONE_NAMES, start=1)})
    return velden


def activity_exists(conn: sqlite3.Connection, activity_key: str) -> bool:
    """Is deze activiteit al eens geïmporteerd?

    Telt bewust óók soft-verwijderde sessies mee: de dedup moet een verwijderde
    activiteit bij een herimport blijven herkennen, anders komt hij stilletjes
    terug.
    """
    row = conn.execute(
        "SELECT 1 FROM activities WHERE activity_key = ?", (activity_key,)
    ).fetchone()
    return row is not None


def is_deleted(conn: sqlite3.Connection, activity_key: str) -> bool:
    """Is deze activiteit soft-verwijderd? (False als hij niet bestaat.)"""
    row = conn.execute(
        "SELECT deleted_at FROM activities WHERE activity_key = ?", (activity_key,)
    ).fetchone()
    return row is not None and row[0] is not None


def save_activity(
    conn: sqlite3.Connection,
    act: ParsedActivity,
    zone_bounds: list[int] | None,
    user_note: str | None = None,
    wind: "WindData | None" = None,
    training_label: str | None = None,
    ftp: float | None = None,
) -> bool:
    """Sla één activiteit op. Geeft False terug als hij al bestond (dedup).

    ``zone_bounds`` zijn de hartslagzonegrenzen die vóór déze sport gelden —
    de aanroeper haalt ze op met
    :func:`tricoach.sportzones.hr_zone_bounds`, want de drempel verschilt per
    sport. ``None`` betekent "geen zone-oordeel" (zwemmen): de z1..z5-kolommen
    blijven dan op 0 en ``pct_in_zone2`` op NULL.

    ``user_note`` is de vrije opmerking die de gebruiker bij de upload typte
    (optioneel). ``wind`` is de automatisch opgehaalde Open-Meteo-winddata
    (optioneel; ``None`` bij zwemmen, geen GPS of geen internet).
    ``training_label`` is het sessielabel van de atleet (bijv.
    "techniek/cadans"); None = normale sessie. ``ftp`` (optioneel) is nodig
    voor de tijd-in-vermogenszones van fietssessies met powerdata; zonder FTP
    blijven die kolommen NULL (zie ``recompute_power_zones``).
    """
    if activity_exists(conn, act.activity_key):
        return False

    s = act.summary
    tiz = time_in_zones(act.records, zone_bounds)
    # Tempo/snelheid op actieve tijd (zie tricoach.timebasis): rust aan de
    # kant en stilstand tellen niet mee. Terugval op het Garmin-veld als de
    # afstand of actieve tijd ontbreekt.
    actief_s = active_seconds(act.sport, s.get("total_timer_time"),
                              act.records, act.lengths)
    avg_speed = (s.get("total_distance") / actief_s
                 if s.get("total_distance") and actief_s
                 else s.get("enhanced_avg_speed") or s.get("avg_speed"))
    avg_hr = s.get("avg_heart_rate")
    # Vooraf berekend en gecachet, zodat de tabel niet bij elke render alle
    # FIT-records opnieuw hoeft te parsen.
    pct_z2 = pct_in_zone2(tiz["Z1"], tiz["Z2"], tiz["Z3"], tiz["Z4"], tiz["Z5"])
    eff = aerobic_efficiency(act.sport, avg_speed, avg_hr)
    power = _power_fields(act, ftp)

    conn.execute(
        """INSERT INTO activities (
               activity_key, sport, sub_sport, start_time, duration_s, distance_m,
               avg_hr, max_hr, avg_speed_ms, avg_cadence, total_ascent, calories,
               pool_length, num_lengths, total_strokes,
               z1_s, z2_s, z3_s, z4_s, z5_s,
               pct_in_zone2, aerobic_efficiency,
               avg_power, np_power, avg_cadence_excl0, max_cadence,
               is_indoor, power_source, ef_watt,
               p1_s, p2_s, p3_s, p4_s, p5_s, p6_s,
               user_note, training_label,
               wind_speed, wind_direction, wind_gusts, temperature_c,
               active_s,
               summary_json, source_file, imported_at
           ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,
                     ?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            act.activity_key, act.sport, act.sub_sport, act.start_time.isoformat(),
            s.get("total_timer_time"), s.get("total_distance"),
            avg_hr, s.get("max_heart_rate"),
            avg_speed,
            s.get("avg_running_cadence") or s.get("avg_cadence"),
            s.get("total_ascent"), s.get("total_calories"),
            s.get("pool_length"),
            s.get("num_active_lengths") or s.get("num_lengths"),
            s.get("total_strokes"),
            tiz["Z1"], tiz["Z2"], tiz["Z3"], tiz["Z4"], tiz["Z5"],
            pct_z2, eff,
            power["avg_power"], power["np_power"],
            power["avg_cadence_excl0"], power["max_cadence"],
            power["is_indoor"], power["power_source"], power["ef_watt"],
            power["p1_s"], power["p2_s"], power["p3_s"],
            power["p4_s"], power["p5_s"], power["p6_s"],
            ((user_note or "").strip() or None),
            ((training_label or "").strip() or None),
            wind.speed_kmh if wind else None,
            wind.direction_deg if wind else None,
            wind.gusts_kmh if wind else None,
            wind.temperature_c if wind else None,
            actief_s,
            json.dumps(s, default=str), act.source_file,
            datetime.now().isoformat(timespec="seconds"),
        ),
    )

    if not act.records.empty:
        df = act.records.copy()
        df["activity_key"] = act.activity_key
        df["timestamp"] = df["timestamp"].astype(str)
        # Zorg dat alle schemakolommen bestaan (zwemrecords missen bijv. cadans).
        for col in ["heart_rate", "speed_ms", "distance_m", "cadence", "power",
                    "altitude_m"]:
            if col not in df:
                df[col] = None
        df[["activity_key", "timestamp", "heart_rate", "speed_ms",
            "distance_m", "cadence", "power", "altitude_m"]].to_sql(
            "records", conn, if_exists="append", index=False)

    if not act.lengths.empty:
        df = act.lengths.copy()
        df["activity_key"] = act.activity_key
        df["timestamp"] = df["timestamp"].astype(str)
        if "start_time" not in df:
            df["start_time"] = None
        else:
            df["start_time"] = df["start_time"].map(
                lambda v: str(v) if v is not None else None)
        df["swim_stroke"] = df["swim_stroke"].astype(str)
        df[["activity_key", "timestamp", "start_time", "total_timer_time",
            "total_strokes", "swim_stroke"]].to_sql(
            "lengths", conn, if_exists="append", index=False)

    conn.commit()
    return True


def enrich_activity_summary(conn: sqlite3.Connection, act: ParsedActivity) -> bool:
    """Vul ontbrekende summary-velden van een al geïmporteerde sessie aan.

    Nodig voor velden die pas later aan ``SESSION_FIELDS`` zijn toegevoegd
    (zoals de loopdynamiek): oude sessies missen die in hun ``summary_json``,
    en de ruwe FIT-bestanden worden niet bewaard. Bij een herupload van
    dezelfde Garmin-zip vult deze functie alléén de ontbrekende sleutels aan —
    bestaande waarden (en alle andere kolommen) blijven onaangeroerd, zodat
    een herimport nooit iets kan overschrijven. Geeft True als er iets is
    aangevuld. Werkt ook op soft-verwijderde sessies (onschadelijk: alleen
    summary_json).
    """
    row = conn.execute(
        "SELECT summary_json FROM activities WHERE activity_key = ?",
        (act.activity_key,),
    ).fetchone()
    if row is None:
        return False
    try:
        bestaand = json.loads(row[0] or "{}")
    except (TypeError, ValueError):
        bestaand = {}
    # Dezelfde normalisatie als bij save_activity (datetimes -> strings),
    # zodat de aangevulde velden er hetzelfde uitzien als bij een verse import.
    vers = json.loads(json.dumps(act.summary, default=str))
    nieuw = {k: v for k, v in vers.items() if k not in bestaand}
    if not nieuw:
        return False
    bestaand.update(nieuw)
    conn.execute(
        "UPDATE activities SET summary_json = ? WHERE activity_key = ?",
        (json.dumps(bestaand), act.activity_key),
    )
    conn.commit()
    return True


def enrich_power_records(conn: sqlite3.Connection, act: ParsedActivity,
                         ftp: float | None = None) -> bool:
    """Vul de powerdata van een al geïmporteerde fietssessie alsnog in.

    Voor sessies die vóór de power-uitbreiding zijn geïmporteerd: de
    ``power``-kolom in ``records`` bestond toen nog niet, dus bij een
    herupload van dezelfde zip zetten we de seconde-data opnieuw (zelfde
    bron, zelfde FIT-bestand) en vullen we de afgeleide kolommen (NP, EF,
    cadans, bron, zones). Doet niets — en overschrijft dus nooit iets — als
    de sessie al powerdata heeft of het geparste bestand geen power bevat.
    Geeft True als er is aangevuld. Werkt ook op soft-verwijderde sessies.
    """
    if act.sport != "cycling" or act.records.empty \
            or "power" not in act.records or not act.records["power"].notna().any():
        return False
    row = conn.execute(
        "SELECT 1 FROM activities WHERE activity_key = ?", (act.activity_key,)
    ).fetchone()
    if row is None:
        return False
    heeft_al = conn.execute(
        "SELECT 1 FROM records WHERE activity_key = ? AND power IS NOT NULL LIMIT 1",
        (act.activity_key,),
    ).fetchone()
    if heeft_al is not None:
        return False

    conn.execute("DELETE FROM records WHERE activity_key = ?", (act.activity_key,))
    df = act.records.copy()
    df["activity_key"] = act.activity_key
    df["timestamp"] = df["timestamp"].astype(str)
    for col in ["heart_rate", "speed_ms", "distance_m", "cadence", "power",
                "altitude_m"]:
        if col not in df:
            df[col] = None
    df[["activity_key", "timestamp", "heart_rate", "speed_ms",
        "distance_m", "cadence", "power", "altitude_m"]].to_sql(
        "records", conn, if_exists="append", index=False)

    power = _power_fields(act, ftp)
    conn.execute(
        "UPDATE activities SET avg_power=?, np_power=?, avg_cadence_excl0=?, "
        "max_cadence=?, is_indoor=?, power_source=?, ef_watt=?, "
        "p1_s=?, p2_s=?, p3_s=?, p4_s=?, p5_s=?, p6_s=? WHERE activity_key=?",
        (power["avg_power"], power["np_power"], power["avg_cadence_excl0"],
         power["max_cadence"], power["is_indoor"], power["power_source"],
         power["ef_watt"], power["p1_s"], power["p2_s"], power["p3_s"],
         power["p4_s"], power["p5_s"], power["p6_s"], act.activity_key),
    )
    conn.commit()
    return True


def recompute_power_zones(conn: sqlite3.Connection, ftp: float | None) -> int:
    """Herreken de tijd-in-vermogenszones van alle sessies met powerdata.

    Nodig wanneer de FTP wordt gezet of gewijzigd: de p1..p6-kolommen zijn
    (net als de HR-zones) bij import berekend met de toenmalige waarde.
    ``ftp=None`` wist de zonetijden weer (terug naar "FTP onbekend"). Geeft
    het aantal bijgewerkte sessies terug.
    """
    keys = [row[0] for row in conn.execute(
        "SELECT DISTINCT activity_key FROM records WHERE power IS NOT NULL")]
    for key in keys:
        tip = time_in_power_zones(load_records(conn, key), ftp)
        waarden = ([tip[naam] for naam in POWER_ZONE_NAMES]
                   if ftp else [None] * len(POWER_ZONE_NAMES))
        conn.execute(
            "UPDATE activities SET p1_s=?, p2_s=?, p3_s=?, p4_s=?, p5_s=?, p6_s=? "
            "WHERE activity_key=?", (*waarden, key),
        )
    conn.commit()
    return len(keys)


def set_training_label(conn: sqlite3.Connection, activity_key: str,
                       training_label: str | None) -> bool:
    """Zet of wis het sessielabel van een bestaande sessie (bijv. achteraf
    alsnog als "techniek/cadans" markeren). Geeft True als er een rij is geraakt."""
    cur = conn.execute(
        "UPDATE activities SET training_label = ? WHERE activity_key = ?",
        (((training_label or "").strip() or None), activity_key),
    )
    conn.commit()
    return cur.rowcount > 0


def set_excluded_reason(conn: sqlite3.Connection, activity_key: str,
                        reason: str | None) -> bool:
    """Zet of wis de uitsluitingsreden van een sessie (bijv. "transport").

    Een gezette reden haalt de sessie uit trends, records, zone-statistieken,
    brick-detectie en de feedback-vergelijking (zie :func:`training_activities`);
    de sessie blijft zichtbaar en telt mee in weektotalen en belasting.
    ``None`` maakt er weer een gewone training van. Geeft True als er een rij
    is geraakt.
    """
    cur = conn.execute(
        "UPDATE activities SET excluded_reason = ? WHERE activity_key = ?",
        (((reason or "").strip() or None), activity_key),
    )
    conn.commit()
    return cur.rowcount > 0


def set_archived_path(conn: sqlite3.Connection, activity_key: str,
                      archived_path: str) -> bool:
    """Registreer de actieve archiefversie van het originele FIT-bestand.

    ``archived_path`` is relatief aan de projectroot (bijv.
    ``uploads/2026/07/2026-07-17_0822_23627119348.fit``). Zet meteen
    ``original_missing`` terug op NULL: er ís nu een origineel. Geeft True
    als er een rij is geraakt (de sessie kan ook soft-verwijderd zijn).
    """
    cur = conn.execute(
        "UPDATE activities SET archived_path = ?, original_missing = NULL "
        "WHERE activity_key = ?",
        (str(archived_path), activity_key),
    )
    conn.commit()
    return cur.rowcount > 0


def training_activities(acts: pd.DataFrame) -> pd.DataFrame:
    """Alleen de échte trainingen: sessies met een ``excluded_reason``
    (transport/casual) eruit gefilterd.

    Dit is het ene filterpunt voor de transport-markering, zoals
    :func:`load_activities` dat is voor soft deletes: trends, records,
    zone-statistieken, brick-detectie en de feedback-vergelijking rekenen op
    deze subset. Weektotalen en belastingsoverzichten gebruiken juist de
    volledige set (compleet belastingsbeeld).
    """
    if acts.empty or "excluded_reason" not in acts:
        return acts
    return acts[acts["excluded_reason"].isna()]


def load_activities(conn: sqlite3.Connection) -> pd.DataFrame:
    """Alle niet-verwijderde sessies als DataFrame, nieuwste eerst.

    Dit is het ene filterpunt voor soft deletes: alles wat sessies toont of
    meerekent (tabellen, grafieken, trends, weekvolume, feedback-context)
    leest via deze functie en ziet verwijderde sessies dus niet. Sessies met
    een ``excluded_reason`` (transport) zitten er wél in; filter die waar
    nodig weg met :func:`training_activities`.
    """
    df = pd.read_sql_query(
        "SELECT * FROM activities WHERE deleted_at IS NULL "
        "ORDER BY start_time DESC", conn)
    if not df.empty:
        df["start_time"] = pd.to_datetime(df["start_time"])
    return df


def load_deleted_activities(conn: sqlite3.Connection) -> pd.DataFrame:
    """Alleen de soft-verwijderde sessies, nieuwste eerst (voor herstel-UI)."""
    df = pd.read_sql_query(
        "SELECT * FROM activities WHERE deleted_at IS NOT NULL "
        "ORDER BY start_time DESC", conn)
    if not df.empty:
        df["start_time"] = pd.to_datetime(df["start_time"])
    return df


def soft_delete_activity(conn: sqlite3.Connection, activity_key: str) -> bool:
    """Markeer een sessie als verwijderd (soft delete).

    De rij blijft staan; alleen ``deleted_at`` wordt gezet. Geeft True terug
    als er daadwerkelijk een (nog niet verwijderde) sessie is gemarkeerd.
    """
    cur = conn.execute(
        "UPDATE activities SET deleted_at = ? "
        "WHERE activity_key = ? AND deleted_at IS NULL",
        (datetime.now().isoformat(timespec="seconds"), activity_key),
    )
    conn.commit()
    return cur.rowcount > 0


def restore_activity(conn: sqlite3.Connection, activity_key: str) -> bool:
    """Maak een soft delete ongedaan. Geeft True als er iets hersteld is."""
    cur = conn.execute(
        "UPDATE activities SET deleted_at = NULL "
        "WHERE activity_key = ? AND deleted_at IS NOT NULL",
        (activity_key,),
    )
    conn.commit()
    return cur.rowcount > 0


def purge_activity(conn: sqlite3.Connection, activity_key: str) -> bool:
    """Wis een sessie definitief uit alle tabellen (activities, records, lengths).

    Onomkeerbaar — en de dedup vergeet deze activiteit, dus een herimport van
    dezelfde zip voegt hem gewoon opnieuw toe. Geeft True als er een rij is
    gewist.
    """
    conn.execute("DELETE FROM records WHERE activity_key = ?", (activity_key,))
    conn.execute("DELETE FROM lengths WHERE activity_key = ?", (activity_key,))
    cur = conn.execute(
        "DELETE FROM activities WHERE activity_key = ?", (activity_key,))
    conn.commit()
    return cur.rowcount > 0


def load_records(conn: sqlite3.Connection, activity_key: str) -> pd.DataFrame:
    """Seconde-data van één sessie."""
    df = pd.read_sql_query(
        "SELECT * FROM records WHERE activity_key = ? ORDER BY timestamp",
        conn, params=(activity_key,))
    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def recompute_zones(conn: sqlite3.Connection, athlete: dict) -> int:
    """Herreken de tijd-in-zones én de afgeleide maten van álle activiteiten.

    Nodig wanneer een drempel (en dus de zones) wijzigt: de z1_s..z5_s-kolommen
    en het daaruit afgeleide ``pct_in_zone2`` zijn bij import berekend met de
    toenmalige grenzen. Elke sessie wordt herrekend met de grenzen van háár
    eigen sport (zie :func:`tricoach.sportzones.hr_zone_bounds`) — hardlopen op
    de loop-LTHR, fietsen op de fiets-LTHR, zwemmen zonder zones. Zo blijven
    trends over de hele historie op één definitie staan in plaats van
    halverwege om te schakelen.

    ``aerobic_efficiency`` hangt niet van de zones af, maar wordt hier
    meegenomen zodat één herberekening alle gecachete maten dekt (bijv. ook
    handig om een bestaande database eenmalig te vullen). De seconde-data staat
    in ``records``, dus herrekenen kan altijd. Geeft het aantal bijgewerkte
    activiteiten terug.
    """
    from tricoach.sportzones import hr_zone_bounds  # hier tegen een circulaire import

    # Grenzen per sport één keer bepalen; dat scheelt werk per sessie.
    bounds_per_sport: dict[str, list[int] | None] = {}
    rows = conn.execute(
        "SELECT activity_key, sport, avg_speed_ms, avg_hr FROM activities"
    ).fetchall()
    for key, sport, avg_speed_ms, avg_hr in rows:
        if sport not in bounds_per_sport:
            bounds_per_sport[sport] = hr_zone_bounds(athlete, sport)
        records = load_records(conn, key)
        tiz = time_in_zones(records, bounds_per_sport[sport])
        pct_z2 = pct_in_zone2(tiz["Z1"], tiz["Z2"], tiz["Z3"], tiz["Z4"], tiz["Z5"])
        eff = aerobic_efficiency(sport, avg_speed_ms, avg_hr)
        conn.execute(
            "UPDATE activities SET z1_s=?, z2_s=?, z3_s=?, z4_s=?, z5_s=?, "
            "pct_in_zone2=?, aerobic_efficiency=? WHERE activity_key=?",
            (tiz["Z1"], tiz["Z2"], tiz["Z3"], tiz["Z4"], tiz["Z5"],
             pct_z2, eff, key),
        )
    conn.commit()
    return len(rows)


def load_lengths(conn: sqlite3.Connection, activity_key: str) -> pd.DataFrame:
    """Baandata van één zwemsessie."""
    return pd.read_sql_query(
        "SELECT * FROM lengths WHERE activity_key = ? ORDER BY timestamp",
        conn, params=(activity_key,))


def swim_active_seconds(conn: sqlite3.Connection) -> dict[str, float]:
    """Actieve zwemtijd (s) per activiteit: de som van de actieve banen.

    De ``lengths``-tabel bevat alleen actieve banen — rustpauzes aan de kant
    zijn bij het parsen al weggefilterd — dus de som van ``total_timer_time``
    is de zuivere zwemtijd zonder rust. Dat is een betrouwbaardere noemer voor
    het tempo per 100 m dan de totale timer-duur (``duration_s``), die de rust
    aan de kant meetelt en het tempo dus te traag laat lijken.

    Geeft een dict ``{activity_key: seconden}``; sessies zonder baandata
    (bijv. open water of een loop/fietssessie) ontbreken simpelweg, zodat de
    aanroeper kan terugvallen op de totale duur.
    """
    rows = conn.execute(
        "SELECT activity_key, SUM(total_timer_time) FROM lengths "
        "GROUP BY activity_key"
    ).fetchall()
    return {key: total for key, total in rows if total}
