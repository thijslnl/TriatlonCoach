"""Persoonlijke heatmap van alle GPS-tracks (fietsen, hardlopen, open water).

Een Strava-achtige kaart: op een donkere ondergrond alle routes die ooit zijn
gefietst, gelopen of open water gezwommen, waarbij vaker bezochte paden feller
oplichten. Dit is een kijk-feature, geen trainingsanalyse — transport-ritjes
tellen hier dus standaard mee (het gaat om *waar* je komt, niet om de prikkel).

De keten bestaat uit vier stappen:

1. **Extractie** — de GPS-punten komen uit de originele FIT-bestanden in het
   archief (:mod:`tricoach.archive`); de ``records``-tabel bewaart geen
   coördinaten. :func:`parse_fit` rekent semicircles al om naar graden
   (``graden = semicircles × 180 / 2³¹``).
2. **Herbemonstering op afstand** — dit is de kern. Een horloge logt per
   seconde, dus waar je langzaam gaat (stoplicht, klim, oversteek) stapelen de
   punten zich op. Een heatmap die ruwe punten telt licht daar fel op terwijl
   je er niet vaker was. Elke track wordt daarom vóór het tellen herbemonsterd
   naar punten op een **vaste afstand** (:data:`RESAMPLE_M`, 10 m), met
   interpolatie tussen de gelogde punten. Zo weegt elke gereden meter even
   zwaar, ongeacht snelheid. Tien minuten stilstaan levert precies één punt op.
3. **Cache** — de herbemonsterde punten gaan in ``track_points``, met per
   activiteit een regel in ``track_extract`` (ook voor sessies zonder GPS).
   Daardoor wordt elk FIT-bestand precies één keer geparst en is een render
   een simpele SQLite-query. :func:`refresh_track_cache` werkt alleen nieuwe
   activiteiten bij.
4. **Dichtheid per rastercel** — de punten worden op een raster gelegd
   (:data:`DEFAULT_CELL_M`) en per cel wordt het aantal **passages** geteld:
   opeenvolgende punten in dezelfde cel binnen één track gelden als één
   passage, een latere terugkomst als een nieuwe. Dat maakt de telling ook
   ongevoelig voor de rasterhoek: één keer door een cel is één keer, of je
   er nu drie of zeven herbemonsterde punten in had.

De kleurschaal (:func:`scale_counts`) is percentiel- of logaritmisch, nooit
lineair: één dagelijkse woon-werkroute met tientallen passages zou anders alle
andere routes onzichtbaar maken.

Privacy: :func:`apply_privacy_zone` laat punten binnen een instelbare straal
rond het huisadres weg, zodat een screenshot de voordeur niet verraadt.
"""

import json
import math
import sqlite3
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from tricoach.config import PROJECT_ROOT, resolve_path
from tricoach.fit_parser import parse_fit
from tricoach.sportzones import CYCLING, RUNNING, SWIMMING

# ------------------------------------------------------------------ constanten --

# Herbemonsteringsafstand: elke 10 m één punt. Fijn genoeg voor een raster van
# 15 m en groter, en het maakt de cache meteen compacter dan de ruwe seconde-data.
RESAMPLE_M = 10.0

# Een sprong groter dan dit is geen meetgat maar een verplaatsing (pauze en
# verderop hervat, of het horloge aan tijdens een autorit): daartussen wordt
# niet geïnterpoleerd, anders trekt de kaart een kaarsrechte spooklijn.
MAX_GAP_M = 500.0

# Rastercel voor de dichtheid. 20 m is ongeveer een wegbreedte: fijn genoeg om
# een fietspad naast een weg te onderscheiden, grof genoeg om GPS-ruis (±5 m)
# op dezelfde route in dezelfde cel te laten vallen.
DEFAULT_CELL_M = 20.0

# Privacyzone rond het huisadres; 400 m is ruim genoeg om de straat niet te
# verraden en klein genoeg om de routes intact te laten.
DEFAULT_PRIVACY_RADIUS_M = 400.0

# Eén graad breedte in meters. De lengtegraad krimpt met cos(breedte); voor
# tracks binnen één land is deze vlakke benadering nauwkeurig tot ruim onder
# de meter — precies genoeg voor een raster van tientallen meters.
METERS_PER_DEG_LAT = 111_320.0

OPEN_WATER = "open_water"

# Sportkeuzes in de UI: label -> (FIT-sport, vereiste sub_sport of None).
# Zwemmen alleen als open water; banenzwemmen heeft geen GPS.
SPORT_CATEGORIES: dict[str, tuple[str, str | None]] = {
    "Hardlopen": (RUNNING, None),
    "Fietsen": (CYCLING, None),
    "Open water zwemmen": (SWIMMING, OPEN_WATER),
}

# Sub-sporten die per definitie geen bruikbare GPS hebben: banenzwemmen en
# indoor/virtueel fietsen (Zwift). Die bestanden worden niet eens geparst.
NO_GPS_SUB_SPORTS = {"lap_swimming", "virtual_activity", "indoor_cycling",
                     "treadmill", "indoor_running"}

# Statussen in track_extract.
STATUS_OK = "ok"
STATUS_NO_GPS = "geen_gps"
STATUS_NO_FILE = "geen_origineel"
STATUS_ERROR = "fout"

SCHEMA = """
CREATE TABLE IF NOT EXISTS track_points (
    activity_key TEXT NOT NULL,
    sport        TEXT NOT NULL,
    sub_sport    TEXT,
    start_time   TEXT NOT NULL,
    seq          INTEGER NOT NULL,
    timestamp    TEXT,
    lat          REAL NOT NULL,
    lon          REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_track_points_key ON track_points(activity_key);
CREATE TABLE IF NOT EXISTS track_extract (
    activity_key TEXT PRIMARY KEY,
    status       TEXT NOT NULL,
    n_points     INTEGER NOT NULL,
    resample_m   REAL NOT NULL,
    extracted_at TEXT NOT NULL
);
"""


def ensure_tables(conn: sqlite3.Connection) -> None:
    """Maak de cachetabellen aan als ze nog niet bestaan (idempotent)."""
    conn.executescript(SCHEMA)
    conn.commit()


def category_of(sport: str, sub_sport: str | None) -> str | None:
    """Het UI-sportlabel voor deze sessie, of None als hij niet op de kaart hoort."""
    for label, (want_sport, want_sub) in SPORT_CATEGORIES.items():
        if sport == want_sport and (want_sub is None or sub_sport == want_sub):
            return label
    return None


# ----------------------------------------------------- herbemonstering op afstand --

def _local_meters(lat: np.ndarray, lon: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Projecteer graden naar lokale meters (equirectangulair rond het midden).

    Goed genoeg voor één track: over de lengte van een fietsrit blijft de fout
    ruim onder een meter, en we meten hier afstanden van tientallen meters.
    """
    lat0 = float(np.mean(lat))
    meters_per_deg_lon = METERS_PER_DEG_LAT * math.cos(math.radians(lat0))
    return lon * meters_per_deg_lon, lat * METERS_PER_DEG_LAT


def _valid_positions(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Masker van bruikbare coördinaten: eindig, binnen bereik en niet (0, 0).

    Nulpunten komen voor als het horloge het bericht al schrijft vóór de
    eerste satellietfix; die liggen in de Golf van Guinee en zouden de
    bounding box van de hele kaart opblazen.
    """
    return (np.isfinite(lat) & np.isfinite(lon)
            & (np.abs(lat) <= 90.0) & (np.abs(lon) <= 180.0)
            & ~((lat == 0.0) & (lon == 0.0)))


def resample_track(lat, lon, t=None, interval_m: float = RESAMPLE_M,
                   max_gap_m: float = MAX_GAP_M) -> pd.DataFrame:
    """Herbemonster één track naar punten op vaste **afstand** langs de lijn.

    Tussen opeenvolgende GPS-punten wordt lineair geïnterpoleerd, zodat er
    elke ``interval_m`` meter precies één punt ligt — ongeacht hoe hard je
    daar ging. Dat is het verschil tussen lijndichtheid en puntdichtheid: een
    stoplicht levert één punt op in plaats van negentig.

    Sprongen groter dan ``max_gap_m`` breken de track in stukken; daartussen
    wordt niet geïnterpoleerd. Een stuk dat korter is dan één interval houdt
    zijn beginpunt, zodat een kort ritje niet volledig verdwijnt.

    ``t`` is optioneel (tijdstippen per gelogd punt) en wordt mee-geïnterpoleerd.
    Geeft een DataFrame met kolommen ``lat``, ``lon`` en ``t`` (NaT zonder ``t``).
    """
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    tijd = (pd.to_datetime(pd.Series(t)).astype("int64").to_numpy(dtype=float)
            if t is not None else None)

    houd = _valid_positions(lat, lon)
    lat, lon = lat[houd], lon[houd]
    if tijd is not None:
        tijd = tijd[houd]
    if lat.size == 0:
        return pd.DataFrame({"lat": [], "lon": [], "t": pd.to_datetime([])})

    x, y = _local_meters(lat, lon)
    stap = np.hypot(np.diff(x), np.diff(y))
    # Elke te grote sprong start een nieuw segment.
    grenzen = [0, *(np.flatnonzero(stap > max_gap_m) + 1), lat.size]

    uit_lat, uit_lon, uit_t = [], [], []
    for begin, eind in zip(grenzen, grenzen[1:]):
        seg = slice(begin, eind)
        s_lat, s_lon = lat[seg], lon[seg]
        if s_lat.size == 1:
            uit_lat.append(s_lat)
            uit_lon.append(s_lon)
            if tijd is not None:
                uit_t.append(tijd[seg])
            continue
        # Afgelegde weg langs het segment; daarop bemonsteren we op vaste stap.
        weg = np.concatenate(([0.0], np.cumsum(stap[begin:eind - 1])))
        doel = np.arange(0.0, max(weg[-1], 1e-9), interval_m)
        uit_lat.append(np.interp(doel, weg, s_lat))
        uit_lon.append(np.interp(doel, weg, s_lon))
        if tijd is not None:
            uit_t.append(np.interp(doel, weg, tijd[seg]))

    frame = {
        "lat": np.concatenate(uit_lat),
        "lon": np.concatenate(uit_lon),
    }
    frame["t"] = (pd.to_datetime(np.concatenate(uit_t).astype("int64"))
                  if tijd is not None else pd.NaT)
    return pd.DataFrame(frame)


# --------------------------------------------------------------------- extractie --

def gps_from_fit(path: Path) -> pd.DataFrame | None:
    """Lees de GPS-punten uit één (gearchiveerd) FIT-bestand.

    Geeft None als het bestand geen activiteit is; een leeg DataFrame als de
    activiteit geen positiedata heeft (banenzwemmen, indoortrainer, sessie
    zonder satellietfix). :func:`tricoach.fit_parser.parse_fit` heeft de
    semicircles dan al naar graden omgerekend.
    """
    with open(path, "rb") as f:
        act = parse_fit(f, source_name=path.name)
    if act is None:
        return None
    rec = act.records
    if rec.empty or "lat" not in rec or "lon" not in rec:
        return pd.DataFrame(columns=["lat", "lon", "timestamp"])
    gps = rec[["lat", "lon", "timestamp"]].dropna(subset=["lat", "lon"])
    return gps[_valid_positions(gps["lat"].to_numpy(dtype=float),
                                gps["lon"].to_numpy(dtype=float))]


def pending_activities(conn: sqlite3.Connection,
                       resample_m: float = RESAMPLE_M) -> pd.DataFrame:
    """Activiteiten waarvoor de trackcache nog moet worden gevuld.

    Dat zijn de sessies die nog helemaal niet zijn ingelezen én de sessies die
    met een andere herbemonsteringsafstand zijn ingelezen (dan is de cache
    verouderd en wordt hij opnieuw opgebouwd). Soft-deleted sessies gaan mee:
    ze zijn met één vinkje weer op de kaart te zetten.
    """
    ensure_tables(conn)
    return pd.read_sql_query(
        "SELECT a.activity_key, a.sport, a.sub_sport, a.start_time, "
        "       a.archived_path "
        "FROM activities a "
        "LEFT JOIN track_extract e ON e.activity_key = a.activity_key "
        "WHERE e.activity_key IS NULL OR e.resample_m != ? "
        "ORDER BY a.start_time",
        conn, params=(resample_m,))


def refresh_track_cache(conn: sqlite3.Connection,
                        resample_m: float = RESAMPLE_M,
                        progress=None) -> dict[str, int]:
    """Vul de trackcache bij voor nieuwe activiteiten (idempotent).

    Per activiteit wordt het gearchiveerde origineel één keer geparst, de
    track herbemonsterd op vaste afstand (:func:`resample_track`) en in
    ``track_points`` gezet. Ook een sessie *zonder* GPS krijgt een regel in
    ``track_extract`` — met status ``geen_gps`` — zodat een zwembadsessie niet
    bij elke verversing opnieuw wordt opengetrokken.

    ``progress`` is een optionele callback ``(gedaan, totaal, label)`` voor een
    voortgangsbalk. Geeft een telling per status terug, plus het aantal
    toegevoegde punten.
    """
    ensure_tables(conn)
    todo = pending_activities(conn, resample_m)
    telling = {STATUS_OK: 0, STATUS_NO_GPS: 0, STATUS_NO_FILE: 0,
               STATUS_ERROR: 0, "punten": 0}
    totaal = len(todo)
    for n, rij in enumerate(todo.itertuples(index=False), start=1):
        if progress is not None:
            progress(n, totaal, str(rij.start_time)[:10])
        status, punten = _extract_one(conn, rij, resample_m)
        telling[status] += 1
        telling["punten"] += punten
    conn.commit()
    return telling


def _extract_one(conn: sqlite3.Connection, rij, resample_m: float) -> tuple[str, int]:
    """Lees één activiteit in en schrijf het resultaat naar de cache."""
    # Oude punten weg: bij een gewijzigde herbemonsteringsafstand of een
    # herupload van hetzelfde origineel bouwen we de track opnieuw op.
    conn.execute("DELETE FROM track_points WHERE activity_key = ?",
                 (rij.activity_key,))

    if category_of(rij.sport, rij.sub_sport) is None \
            or rij.sub_sport in NO_GPS_SUB_SPORTS:
        return _register(conn, rij.activity_key, STATUS_NO_GPS, 0, resample_m), 0
    # Let op de NaN-check: een lege archived_path-kolom komt uit pandas als
    # float('nan') terug, en die is truthy.
    if not rij.archived_path or pd.isna(rij.archived_path):
        return _register(conn, rij.activity_key, STATUS_NO_FILE, 0, resample_m), 0
    pad = PROJECT_ROOT / str(rij.archived_path)
    if not pad.exists():
        return _register(conn, rij.activity_key, STATUS_NO_FILE, 0, resample_m), 0

    try:
        gps = gps_from_fit(pad)
    except Exception:
        # Een onleesbaar of afgekapt bestand mag de hele verversing niet slopen.
        return _register(conn, rij.activity_key, STATUS_ERROR, 0, resample_m), 0
    if gps is None or gps.empty:
        return _register(conn, rij.activity_key, STATUS_NO_GPS, 0, resample_m), 0

    punten = resample_track(gps["lat"], gps["lon"], gps["timestamp"],
                            interval_m=resample_m)
    if punten.empty:
        return _register(conn, rij.activity_key, STATUS_NO_GPS, 0, resample_m), 0

    sub = None if pd.isna(rij.sub_sport) else str(rij.sub_sport)
    conn.executemany(
        "INSERT INTO track_points (activity_key, sport, sub_sport, start_time, "
        "seq, timestamp, lat, lon) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        [(rij.activity_key, rij.sport, sub, str(rij.start_time),
          i, None if pd.isna(t) else pd.Timestamp(t).isoformat(),
          float(la), float(lo))
         for i, (la, lo, t) in enumerate(zip(punten["lat"], punten["lon"],
                                             punten["t"]))])
    return (_register(conn, rij.activity_key, STATUS_OK, len(punten), resample_m),
            len(punten))


def _register(conn: sqlite3.Connection, key: str, status: str, n_points: int,
              resample_m: float) -> str:
    """Leg de uitkomst van één extractie vast; geeft de status terug."""
    conn.execute(
        "INSERT INTO track_extract (activity_key, status, n_points, resample_m, "
        "extracted_at) VALUES (?, ?, ?, ?, ?) "
        "ON CONFLICT(activity_key) DO UPDATE SET status = excluded.status, "
        "n_points = excluded.n_points, resample_m = excluded.resample_m, "
        "extracted_at = excluded.extracted_at",
        (key, status, n_points, resample_m,
         datetime.now().isoformat(timespec="seconds")))
    return status


def prune_track_cache(conn: sqlite3.Connection) -> int:
    """Ruim cacheregels op van definitief gewiste sessies (na een purge).

    Geeft het aantal opgeruimde activiteiten terug.
    """
    ensure_tables(conn)
    weg = conn.execute(
        "SELECT activity_key FROM track_extract WHERE activity_key NOT IN "
        "(SELECT activity_key FROM activities)").fetchall()
    for (key,) in weg:
        conn.execute("DELETE FROM track_points WHERE activity_key = ?", (key,))
        conn.execute("DELETE FROM track_extract WHERE activity_key = ?", (key,))
    conn.commit()
    return len(weg)


def cache_stats(conn: sqlite3.Connection) -> dict:
    """Stand van de trackcache: aantal sessies per status en totaal aantal punten.

    Dient ook als cachesleutel voor de UI: verandert de stand, dan moeten de
    afgeleide berekeningen opnieuw.
    """
    ensure_tables(conn)
    rijen = conn.execute(
        "SELECT status, COUNT(*), COALESCE(SUM(n_points), 0) "
        "FROM track_extract GROUP BY status").fetchall()
    stats = {status: aantal for status, aantal, _ in rijen}
    stats["punten"] = sum(punten for _, _, punten in rijen)
    stats["laatst"] = conn.execute(
        "SELECT MAX(extracted_at) FROM track_extract").fetchone()[0]
    return stats


# ------------------------------------------------------------------- inlezen/filters --

def load_track_points(conn: sqlite3.Connection, tz: str = "Europe/Amsterdam"
                      ) -> pd.DataFrame:
    """Alle gecachede trackpunten, met de sessiecontext die de filters nodig hebben.

    Kolommen: ``activity_key``, ``sport``, ``sub_sport``, ``seq``, ``lat``,
    ``lon``, ``start_time`` (lokale tijd), ``datum`` (lokale datum),
    ``categorie`` (UI-sportlabel), ``is_transport`` en ``is_deleted``.
    Gesorteerd op track en volgorde binnen de track — dat is de aanname van
    :func:`density_grid`, die opeenvolgende punten in dezelfde cel samenvouwt.
    """
    ensure_tables(conn)
    df = pd.read_sql_query(
        "SELECT p.activity_key, p.sport, p.sub_sport, p.seq, p.lat, p.lon, "
        "       p.start_time, "
        "       a.excluded_reason IS NOT NULL AS is_transport, "
        "       a.deleted_at IS NOT NULL AS is_deleted "
        "FROM track_points p "
        "JOIN activities a ON a.activity_key = p.activity_key "
        "ORDER BY p.activity_key, p.seq", conn)
    if df.empty:
        return df.assign(datum=pd.Series(dtype="object"),
                         categorie=pd.Series(dtype="object"))
    lokaal = pd.to_datetime(df["start_time"], utc=True).dt.tz_convert(tz)
    df["start_time"] = lokaal
    df["datum"] = lokaal.dt.date
    df["categorie"] = [category_of(s, ss) for s, ss
                       in zip(df["sport"], df["sub_sport"])]
    df["is_transport"] = df["is_transport"].astype(bool)
    df["is_deleted"] = df["is_deleted"].astype(bool)
    return df


def filter_points(points: pd.DataFrame, categories=None, start=None, end=None,
                  include_transport: bool = True,
                  include_deleted: bool = False) -> pd.DataFrame:
    """Filter de trackpunten op sport, periode en sessiemarkeringen.

    ``categories`` is een lijst UI-sportlabels (None = alles); ``start``/``end``
    zijn lokale datums (inclusief). Transport telt standaard mee — voor een
    heatmap gaat het om waar je komt, niet om de trainingsprikkel — en
    soft-deleted sessies standaard niet.
    """
    if points.empty:
        return points
    masker = pd.Series(True, index=points.index)
    if categories is not None:
        masker &= points["categorie"].isin(list(categories))
    if start is not None:
        masker &= points["datum"] >= start
    if end is not None:
        masker &= points["datum"] <= end
    if not include_transport:
        masker &= ~points["is_transport"]
    if not include_deleted:
        masker &= ~points["is_deleted"]
    return points[masker]


# ------------------------------------------------------------------------ privacy --

PRIVACY_FILE = "heatmap_privacy.json"


def privacy_path(config: dict) -> Path:
    """Pad van het privacyzone-bestand: naast de database in ``data/``.

    Bewust **niet** in config.yaml: dat bestand staat in versiebeheer, en het
    middelpunt van de privacyzone *is* het huisadres. Dat in een repository
    zetten haalt precies weg wat de zone moet beschermen. ``data/`` staat in
    .gitignore, net als de FIT-bestanden waar de coördinaten uit komen.
    """
    return resolve_path(config, "database").parent / PRIVACY_FILE


def privacy_settings(config: dict) -> dict:
    """De opgeslagen privacyzone, met standaardwaarden.

    Vorm van het bestand (zie :func:`privacy_path`)::

        {"enabled": true, "lat": 52.2, "lon": 5.2, "radius_m": 400}

    Zonder ``lat``/``lon`` is de zone niet toepasbaar; de UI doet dan een
    voorstel op basis van de startpunten (:func:`suggest_home`). Standaard staat
    de zone aan, zodat een screenshot niet per ongeluk de voordeur toont.
    """
    pad = privacy_path(config)
    zone: dict = {}
    if pad.exists():
        try:
            zone = json.loads(pad.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            zone = {}  # onleesbaar bestand: terug naar de standaarden
    return {
        "enabled": bool(zone.get("enabled", True)),
        "lat": zone.get("lat"),
        "lon": zone.get("lon"),
        "radius_m": float(zone.get("radius_m") or DEFAULT_PRIVACY_RADIUS_M),
    }


def store_privacy_settings(config: dict, enabled: bool, lat: float | None,
                           lon: float | None, radius_m: float) -> Path:
    """Sla de privacyzone op naast de database; geeft het gebruikte pad terug."""
    pad = privacy_path(config)
    pad.parent.mkdir(parents=True, exist_ok=True)
    pad.write_text(json.dumps({
        "enabled": bool(enabled),
        "lat": None if lat is None else round(float(lat), 6),
        "lon": None if lon is None else round(float(lon), 6),
        "radius_m": float(radius_m),
    }, indent=2), encoding="utf-8")
    return pad


def distance_to_m(points: pd.DataFrame, lat: float, lon: float) -> np.ndarray:
    """Afstand (m) van elk punt tot (lat, lon), vlak benaderd rond dat punt."""
    meters_per_deg_lon = METERS_PER_DEG_LAT * math.cos(math.radians(lat))
    dx = (points["lon"].to_numpy(dtype=float) - lon) * meters_per_deg_lon
    dy = (points["lat"].to_numpy(dtype=float) - lat) * METERS_PER_DEG_LAT
    return np.hypot(dx, dy)


def apply_privacy_zone(points: pd.DataFrame, lat: float | None, lon: float | None,
                       radius_m: float = DEFAULT_PRIVACY_RADIUS_M) -> pd.DataFrame:
    """Laat de punten binnen de privacyzone weg.

    Tracks beginnen en eindigen bij de voordeur; zonder deze zone wijst de
    kaart precies het huisadres aan. De overgebleven track houdt gewoon op bij
    de rand van de zone. Zonder middelpunt gebeurt er niets.
    """
    if points.empty or lat is None or lon is None:
        return points
    return points[distance_to_m(points, float(lat), float(lon)) > radius_m]


def suggest_home(points: pd.DataFrame) -> tuple[float, float] | None:
    """Schat het huisadres uit de startpunten van alle tracks (mediaan).

    De meeste sessies beginnen bij de voordeur, dus de mediaan van alle
    startpunten ligt daar dicht bij — handig om het middelpunt van de
    privacyzone in te vullen zonder coördinaten op te zoeken. De mediaan (niet
    het gemiddelde) omdat een handvol sessies elders begint.
    """
    if points.empty:
        return None
    eerste = points.sort_values(["activity_key", "seq"]) \
                   .groupby("activity_key").first()
    if eerste.empty:
        return None
    return float(eerste["lat"].median()), float(eerste["lon"].median())


# ---------------------------------------------------------------------- dichtheid --

def density_grid(points: pd.DataFrame, cell_m: float = DEFAULT_CELL_M
                 ) -> pd.DataFrame:
    """Tel per rastercel hoe vaak er langs is gekomen.

    De herbemonsterde punten worden op een raster van ``cell_m`` gelegd.
    Geteld worden **passages**, niet punten: opeenvolgende punten in dezelfde
    cel binnen dezelfde track vormen samen één passage, en een latere
    terugkomst (rondje, heen-en-terug) telt weer als een nieuwe. Daarmee is de
    telling ongevoelig voor snelheid (dat deed de herbemonstering al) én voor
    de hoek waaronder een cel gekruist wordt.

    Verwacht ``points`` gesorteerd op ``activity_key``/``seq`` (zoals
    :func:`load_track_points` teruggeeft). Geeft per cel het middelpunt terug
    met ``count`` (passages) en ``sessies`` (aantal verschillende sessies).
    """
    kolommen = ["lat", "lon", "count", "sessies"]
    if points.empty:
        return pd.DataFrame(columns=kolommen)

    lat0 = float(points["lat"].mean())
    dlat = cell_m / METERS_PER_DEG_LAT
    dlon = cell_m / (METERS_PER_DEG_LAT * math.cos(math.radians(lat0)))

    ix = np.floor(points["lon"].to_numpy(dtype=float) / dlon).astype("int64")
    iy = np.floor(points["lat"].to_numpy(dtype=float) / dlat).astype("int64")
    cel = pd.DataFrame({"key": points["activity_key"].to_numpy(),
                        "ix": ix, "iy": iy})

    # Eén passage per aaneengesloten reeks punten in dezelfde cel.
    zelfde = ((cel["ix"] == cel["ix"].shift())
              & (cel["iy"] == cel["iy"].shift())
              & (cel["key"] == cel["key"].shift()))
    passages = cel[~zelfde.to_numpy()]

    tellen = passages.groupby(["ix", "iy"]).size().rename("count")
    sessies = (passages.drop_duplicates(["ix", "iy", "key"])
               .groupby(["ix", "iy"]).size().rename("sessies"))
    raster = pd.concat([tellen, sessies], axis=1).reset_index()
    # Middelpunt van de cel, zodat de stip midden op het pad valt.
    # Zes decimalen is ruim 10 cm: nauwkeuriger dan de cel én dan GPS, en het
    # scheelt een kwart in de JSON die naar de browser gaat.
    raster["lat"] = ((raster["iy"] + 0.5) * dlat).round(6)
    raster["lon"] = ((raster["ix"] + 0.5) * dlon).round(6)
    return raster[kolommen]


# ------------------------------------------------------------------- kleurschaal --

# Kleurverloop van dof donkerrood (één passage) via oranje en geel naar wit
# (het vaakst gereden pad). Bewust géén blauw/groen erin: op een donkere kaart
# leest een warme ramp als gloed, en de sportkleuren van het dashboard blijven
# daarmee onderscheidend.
GRADIENT: list[tuple[float, tuple[int, int, int]]] = [
    (0.00, (110, 22, 22)),
    (0.30, (196, 44, 30)),
    (0.55, (240, 112, 26)),
    (0.78, (250, 202, 48)),
    (1.00, (255, 255, 240)),
]

# Doorzichtigheid loopt mee met de intensiteit: zelden bereden paden blijven
# ingetogen aanwezig, veelgebruikte lichten vol op.
ALPHA_RANGE = (170, 255)

SCALE_LOG = "logaritmisch"
SCALE_PERCENTILE = "percentiel"


class IntensityScale:
    """Omzetting van passage-aantallen naar een intensiteit 0..1.

    Nooit lineair: een woon-werkroute die tientallen keren is gereden zou alle
    andere routes tot vlak boven zwart platdrukken. Twee schalen, die dezelfde
    data een andere vraag laten beantwoorden:

    - ``logaritmisch`` (standaard): ``log(n) / log(top)``, met ``top`` op het
      99,5e percentiel van de celtellingen, zodat één uitschieter de schaal
      niet leegtrekt. Houdt de verhoudingen tussen aantallen herkenbaar: een
      route die je tien keer reed is duidelijk feller dan een route van twee
      keer, en de dagelijkse route loopt vol uit naar wit.
    - ``percentiel``: de intensiteit is het aandeel cellen met minder passages.
      Omdat de meeste cellen precies één passage hebben, springt álles wat je
      meer dan eens deed meteen ver naar boven. Antwoordt dus op "waar kom ik
      vaker dan eens", ten koste van het onderscheid binnen de top.

    Beide schalen leggen het laagste aantal op 0 (dof donkerrood). Komt er maar
    één niveau voor — bijvoorbeeld bij een selectie van één sessie — dan krijgt
    alles de middenintensiteit: er is dan niets te vergelijken, en "overal dof
    rood" zou een onterechte uitspraak zijn.

    De schaal wordt éénmaal op de volledige celverdeling gebouwd en daarna op
    willekeurige aantallen toegepast (``scale(waarden)``), zodat de legenda
    exact dezelfde omzetting gebruikt als de kaart.
    """

    def __init__(self, counts, method: str = SCALE_LOG,
                 clip_quantile: float = 0.995):
        self.method = method
        c = np.asarray(counts, dtype=float)
        self._niveaus = np.unique(c) if c.size else np.array([])
        self._plat = self._niveaus.size <= 1
        if self._plat:
            return
        self._onder = float(self._niveaus[0])
        if method == SCALE_LOG:
            # Minstens tot het tweede voorkomende niveau, anders is de deler 0.
            self._top = max(float(np.quantile(c, clip_quantile)),
                            float(self._niveaus[1]))
        else:
            self._gesorteerd = np.sort(c)
            # Het hoogste voorkomende aantal moet op 1,0 uitkomen (fel wit).
            # Het aandeel cellen onder dat aantal is daarom de noemer.
            self._noemer = (np.searchsorted(self._gesorteerd, self._niveaus[-1],
                                            side="left") / c.size) or 1.0

    def __call__(self, values) -> np.ndarray:
        v = np.asarray(values, dtype=float)
        if v.size == 0:
            return v
        if self._plat:
            return np.full(v.shape, 0.5)
        if self.method == SCALE_LOG:
            t = np.log1p(v - self._onder) / np.log1p(self._top - self._onder)
        else:
            t = (np.searchsorted(self._gesorteerd, v, side="left")
                 / self._gesorteerd.size / self._noemer)
        return np.clip(t, 0.0, 1.0)


def scale_counts(counts, method: str = SCALE_LOG) -> np.ndarray:
    """Intensiteit 0..1 per passage-aantal (zie :class:`IntensityScale`)."""
    return IntensityScale(counts, method)(counts)


def gradient_colors(t) -> np.ndarray:
    """Intensiteiten 0..1 -> RGBA-kleuren (n×4, uint8) volgens :data:`GRADIENT`."""
    t = np.clip(np.asarray(t, dtype=float), 0.0, 1.0)
    stops = np.array([p for p, _ in GRADIENT])
    kleuren = np.array([c for _, c in GRADIENT], dtype=float)
    rgba = np.empty((t.size, 4), dtype=np.uint8)
    for kanaal in range(3):
        rgba[:, kanaal] = np.round(np.interp(t, stops, kleuren[:, kanaal]))
    rgba[:, 3] = np.round(ALPHA_RANGE[0]
                          + t * (ALPHA_RANGE[1] - ALPHA_RANGE[0]))
    return rgba


def legend_stops(counts, method: str = SCALE_LOG, n: int = 6
                 ) -> list[tuple[int, str]]:
    """Legenda-stappen: ``n`` passage-aantallen uit het bereik met hun kleur (hex).

    Toont onder de kaart wat "dof rood" en "fel wit" in aantallen betekenen.
    Gebruikt dezelfde :class:`IntensityScale` als de kaart, opgebouwd op de
    volledige celverdeling — anders zou de legenda andere kleuren noemen dan er
    getekend staan. Aantallen die op dezelfde kleur uitkomen (boven het punt
    waar de schaal verzadigt) worden samengevouwen tot de eerste; de legenda
    belooft dan geen onderscheid dat de kaart niet maakt.
    """
    c = np.asarray(counts, dtype=float)
    if c.size == 0:
        return []
    schaal = IntensityScale(c, method)
    # Gelijkmatig over de vóórkomende aantallen, zodat de legenda geen niveaus
    # noemt die niet op de kaart staan.
    niveaus = np.unique(c)
    keuze = np.unique(np.round(np.quantile(niveaus, np.linspace(0, 1, n))))
    stappen: list[tuple[int, str]] = []
    for aantal, kleur in zip(keuze, gradient_colors(schaal(keuze))):
        hex_kleur = "#%02x%02x%02x" % tuple(kleur[:3])
        if not stappen or stappen[-1][1] != hex_kleur:
            stappen.append((int(aantal), hex_kleur))
    return stappen


def heatmap_cells(points: pd.DataFrame, cell_m: float = DEFAULT_CELL_M,
                  method: str = SCALE_LOG) -> pd.DataFrame:
    """Volledige stap van gefilterde trackpunten naar tekenbare rastercellen.

    Geeft per cel ``lat``, ``lon``, ``count``, ``sessies``, de intensiteit
    ``t`` en de kleurkanalen ``r``, ``g``, ``b``, ``a`` — de vorm die de
    pydeck-laag verwacht (``get_fill_color="[r, g, b, a]"``).
    """
    raster = density_grid(points, cell_m)
    if raster.empty:
        return raster.assign(t=[], r=[], g=[], b=[], a=[])
    raster["t"] = scale_counts(raster["count"], method)
    rgba = gradient_colors(raster["t"])
    # Als gewone int, niet uint8: de JSON-serialisatie van pydeck struikelt
    # over numpy-schaaltypen.
    for i, kanaal in enumerate("rgba"):
        raster[kanaal] = rgba[:, i].astype(int)
    # Fel bovenop: deck.gl tekent in rijvolgorde, dus de drukste cellen laatst.
    return raster.sort_values("count").reset_index(drop=True)


# ------------------------------------------------------------------- kaartpositie --

DEFAULT_VIEW_COVERAGE = 0.90


def view_bounds(points: pd.DataFrame, coverage: float = DEFAULT_VIEW_COVERAGE,
                step: float = 0.01) -> tuple[float, float, float, float]:
    """De strakste box die nog minstens ``coverage`` van de cellen omvat.

    Op álles passen is niet wat je wilt: één hardloopje in het buitenland zou
    de kaart naar landniveau uitzoomen en het gebied waar je 99% van je
    kilometers maakt tot een vlekje maken. Die uitschieters blijven wél op de
    kaart staan — ze vallen alleen buiten het *startbeeld*, en uitzoomen brengt
    ze terug.

    Symmetrisch quantielen afknippen werkt hier slecht: ligt een groep verre
    cellen net iets boven de toegestane fractie, dan schuift de grens er
    middenin en blijft de kaart even ver uitgezoomd. In plaats daarvan wordt
    per stap de rand ingetrokken die de **meeste kaartbreedte per opgegeven cel**
    oplevert, tot het budget (``1 - coverage`` van de cellen) op is of geen rand
    nog wat oplevert. Zo verdwijnt een compacte verre groep in één keer, terwijl
    een gelijkmatig uitgesmeerde spreiding grotendeels in beeld blijft.

    Geeft ``(lat_min, lat_max, lon_min, lon_max)``.
    """
    lat = points["lat"].to_numpy(dtype=float)
    lon = points["lon"].to_numpy(dtype=float)
    box = [float(lat.min()), float(lat.max()),
           float(lon.min()), float(lon.max())]
    n = lat.size
    max_buiten = int((1.0 - coverage) * n)
    # Onder een handvol cellen (of zonder budget) valt er niets te winnen.
    if coverage >= 1.0 or n < 20 or max_buiten < 1:
        return tuple(box)

    def binnen(b: list[float]) -> np.ndarray:
        return ((lat >= b[0]) & (lat <= b[1]) & (lon >= b[2]) & (lon <= b[3]))

    def span_m(b: list[float]) -> float:
        """De grootste zijde van de box in meters — die bepaalt de zoom."""
        mid = math.radians((b[0] + b[1]) / 2)
        return max((b[1] - b[0]) * METERS_PER_DEG_LAT,
                   (b[3] - b[2]) * METERS_PER_DEG_LAT * math.cos(mid))

    masker = binnen(box)
    for _ in range(200):
        beste = None
        for rand in range(4):  # 0/1 = lat onder/boven, 2/3 = lon links/rechts
            waarden = lat[masker] if rand < 2 else lon[masker]
            # Onderranden schuiven omhoog, bovenranden omlaag.
            q = step if rand in (0, 2) else 1.0 - step
            kandidaat = list(box)
            kandidaat[rand] = float(np.quantile(waarden, q))
            k_masker = binnen(kandidaat)
            if n - int(k_masker.sum()) > max_buiten:
                continue  # past niet in het budget
            winst = span_m(box) - span_m(kandidaat)
            if winst <= 0:
                continue  # deze rand bepaalt de zoom niet
            verlies = max(int(masker.sum()) - int(k_masker.sum()), 1)
            score = winst / verlies
            if beste is None or score > beste[0]:
                beste = (score, kandidaat, k_masker)
        if beste is None:
            break
        _, box, masker = beste
    return tuple(box)


def fit_view(points: pd.DataFrame, width_px: int = 1100, height_px: int = 640,
             padding: float = 0.15,
             coverage: float = DEFAULT_VIEW_COVERAGE) -> dict:
    """Beginpositie van de kaart: middelpunt en zoom op het kerngebied.

    Rekent de box van :func:`view_bounds` om naar een Web-Mercator-zoomniveau
    (512 px per tegel, zoals deck.gl); de kleinste van de horizontale en
    verticale fit wint, zodat het kerngebied volledig in beeld valt.
    """
    if points.empty:
        # Midden van Nederland als er niets te tonen is.
        return {"latitude": 52.15, "longitude": 5.35, "zoom": 7.0}
    lat_min, lat_max, lon_min, lon_max = view_bounds(points, coverage)

    def merc_y(lat: float) -> float:
        lat = max(min(lat, 85.0), -85.0)
        return math.log(math.tan(math.pi / 4 + math.radians(lat) / 2))

    lon_span = max(lon_max - lon_min, 1e-4) * (1 + padding)
    lat_span = max(merc_y(lat_max) - merc_y(lat_min), 1e-6) * (1 + padding)
    zoom_lon = math.log2(width_px / 512 * 360 / lon_span)
    zoom_lat = math.log2(height_px / 512 * 2 * math.pi / lat_span)
    return {
        "latitude": (lat_min + lat_max) / 2,
        "longitude": (lon_min + lon_max) / 2,
        "zoom": max(1.0, min(15.0, min(zoom_lon, zoom_lat))),
    }


def covered_km(cells: pd.DataFrame, cell_m: float = DEFAULT_CELL_M) -> float:
    """Ruwe schatting van het aantal unieke kilometers weg/pad op de kaart.

    Elke bezochte cel staat voor ongeveer ``cell_m`` meter route; dubbel
    gereden stukken tellen één keer. Bedoeld als "actieradius"-getal, niet als
    kilometerstand — daarvoor zijn de sessie-afstanden.
    """
    if cells.empty:
        return 0.0
    return len(cells) * cell_m / 1000.0
