"""Transport-/casual-markering: ritjes die geen training zijn.

Sommige geregistreerde sessies zijn verplaatsingen — naar het zwembad
fietsen, boodschappen doen — en horen wél in het logboek (compleet
belastingsbeeld) maar niet in de trainingsanalyses. De markering werkt via
``activities.excluded_reason = "transport"`` (zie :mod:`tricoach.storage`):

- **telt NIET mee** in trends (EF, tempo-bij-HR), records/PR's,
  zone-statistieken, brick-detectie en de vergelijkingscontext van de
  feedback — overal waar :func:`tricoach.storage.training_activities`
  het filterpunt is;
- **telt WEL mee** in weektotalen en het belastingsoverzicht (CTL/ATL/TRIMP),
  desgewenst als aparte categorie "transport";
- blijft zichtbaar in de sessielijst, gedimd en met label.

Markeren gebeurt nooit automatisch: bij import doet :func:`suggest_transport`
hooguit een voorstel (korte, rustige fietsrit) dat de gebruiker met één klik
bevestigt of negeert.
"""

import sqlite3
from datetime import date
from pathlib import Path

import pandas as pd

from tricoach.fit_parser import ParsedActivity
from tricoach.removal import note_in_trainingslog
from tricoach.storage import set_excluded_reason
from tricoach.zones import zone_bounds

# De ene uitsluitingsreden die de UI nu kent; het kolomtype laat ruimte voor
# latere redenen (bijv. "test" of "materiaalcheck").
TRANSPORT_REASON = "transport"

# Suggestiedrempel: een fietsrit korter dan dit is een transport-kandidaat.
MAX_TRANSPORT_KM = 10.0


def is_transport(row: pd.Series) -> bool:
    """Is deze activities-rij als transport gemarkeerd? (NaN-veilig.)"""
    reden = row.get("excluded_reason")
    if reden is None or (isinstance(reden, float) and pd.isna(reden)):
        return False
    return str(reden) == TRANSPORT_REASON


def suggest_transport(act: ParsedActivity, config: dict) -> bool:
    """Lijkt deze sessie een transport-ritje? (Alleen een suggestie.)

    Criteria: een fietsrit buiten, korter dan ~10 km, met een gemiddelde
    hartslag onder de zone 2-ondergrens — dus onder trainingsintensiteit.
    Een echte woon-werkrit in zone 2 (37 km) valt af op de afstand; een
    korte harde intervalrit valt af op de hartslag. Zonder hartslagdata
    geen suggestie: dan valt er niets te beoordelen.
    """
    if act.sport != "cycling" or act.is_indoor:
        return False
    if not act.distance_m or act.distance_m >= MAX_TRANSPORT_KM * 1000:
        return False
    avg_hr = act.summary.get("avg_heart_rate")
    if not avg_hr:
        return False
    z2_onder = zone_bounds(config["athlete"])[0]
    return avg_hr < z2_onder


def _drop_transport_proposals(conn: sqlite3.Connection, activity_key: str) -> None:
    """Trek nog openstaande brick-voorstellen met deze sessie in.

    De brick-detectie kan een transport-ritje (bijv. terugfietsen van het
    zwembad) al als brick hebben voorgesteld vóórdat de gebruiker de sessie
    als transport markeerde. Alleen *voorstellen* worden verwijderd; een door
    de gebruiker bevestigde combo blijft diens expliciete keuze.
    """
    from tricoach.combos import ensure_tables
    ensure_tables(conn)
    rows = conn.execute(
        "SELECT DISTINCT c.combo_id FROM combos c "
        "JOIN combo_members m ON m.combo_id = c.combo_id "
        "WHERE c.status = 'voorgesteld' AND m.activity_key = ?",
        (activity_key,),
    ).fetchall()
    for (combo_id,) in rows:
        conn.execute("DELETE FROM combo_members WHERE combo_id = ?", (combo_id,))
        conn.execute("DELETE FROM combos WHERE combo_id = ?", (combo_id,))
    conn.commit()


def mark_transport(conn: sqlite3.Connection, memory_dir: Path,
                   activity_key: str) -> bool:
    """Markeer een sessie als transport en noteer dat in het trainingslog.

    Trekt ook eventuele openstaande brick-voorstellen met deze sessie in.
    Geeft False als de sleutel onbekend is.
    """
    if not set_excluded_reason(conn, activity_key, TRANSPORT_REASON):
        return False
    _drop_transport_proposals(conn, activity_key)
    note_in_trainingslog(
        memory_dir, activity_key,
        f"gemarkeerd als transport op {date.today():%d-%m-%Y} — telt mee in "
        "weektotalen en belasting, maar niet in trends, records en "
        "feedback-vergelijkingen")
    return True


def unmark_transport(conn: sqlite3.Connection, memory_dir: Path,
                     activity_key: str) -> bool:
    """Haal de transport-markering weer weg (het was toch een training)."""
    if not set_excluded_reason(conn, activity_key, None):
        return False
    note_in_trainingslog(
        memory_dir, activity_key,
        f"transport-markering verwijderd op {date.today():%d-%m-%Y} — telt "
        "weer gewoon mee als training")
    return True
