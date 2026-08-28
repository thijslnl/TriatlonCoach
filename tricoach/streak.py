"""Pure streak-/freeze-/consistentielogica voor gewoontevorming.

Werkt uitsluitend op ``set[date]``/``dict[date, int]`` — geen ``sqlite3``,
geen pandas nodig. Herbruikbaar voor elke "iets elke dag doen"-tracker; hier
gebruikt door :mod:`tricoach.coretraining` voor de core-/mobiliteitsstreak.

**Kernkeuze, bevestigd door de atleet:** een streak freeze houdt de reeks
intact (de teller springt niet terug naar 0) maar telt zelf niet mee in het
getal — het streakgetal blijft het aantal dagen dat je *echt* iets deed.
"""

from __future__ import annotations

from datetime import date, timedelta

DEFAULT_THRESHOLD = 1
FULL_DAY_THRESHOLD = 6
MAX_FREEZES_PER_MONTH = 2
MILESTONES = (7, 14, 30, 60, 100, 365)

NEVER_TWICE_TEKST = "Gisteren overgeslagen. Vandaag weer oppakken is het enige dat telt."


def is_dag_compleet(counts: dict[date, int], day: date, threshold: int = DEFAULT_THRESHOLD) -> bool:
    """Is deze dag compleet? (aantal gelogde oefeningen >= drempel)."""
    return counts.get(day, 0) >= threshold


def is_volledige_dag(counts: dict[date, int], day: date) -> bool:
    """Puur cosmetische vlag: >= 6 oefeningen op deze dag."""
    return counts.get(day, 0) >= FULL_DAY_THRESHOLD


def compute_freezes(complete_days: set[date], today: date) -> set[date]:
    """Welke dagen worden door een streak freeze gered?

    Vooruitlopend, chronologisch, deterministisch: van de vroegste dag in
    ``complete_days`` tot en met ``today``. Een dag ``d`` krijgt een freeze
    precies als:

    1. ``d`` zelf NIET compleet is — er valt iets te redden.
    2. de dag ervóór WEL compleet is. Dit implementeert "nooit twee dagen
       achter elkaar redden" vanzelf: een bevroren dag zit per definitie
       niet in ``complete_days``, dus een bevroren ``d-1`` diskwalificeert
       ``d`` automatisch (conditie 2 faalt dan voor ``d``).
    3. de dag erná WEL compleet is — "precies één dag ontbreekt tussen twee
       complete dagen". Hierdoor kan ``today`` zelf nooit bevroren worden
       (er is nog geen "dag erna" om aan te toetsen), wat correct is:
       vandaag is nog niet gemist, het is nog bezig.
    4. de maandquota (:data:`MAX_FREEZES_PER_MONTH`) voor die kalendermaand
       nog niet op is. Omdat de doorloop chronologisch is, krijgen de
       vroegste gaten in een maand de freeze.
    """
    if not complete_days:
        return set()
    start = min(complete_days)
    bevroren: set[date] = set()
    gebruikt: dict[tuple[int, int], int] = {}
    d = start
    while d <= today:
        if d not in complete_days:
            vorige_compleet = (d - timedelta(days=1)) in complete_days
            volgende_compleet = (d + timedelta(days=1)) in complete_days
            maand = (d.year, d.month)
            if (vorige_compleet and volgende_compleet
                    and gebruikt.get(maand, 0) < MAX_FREEZES_PER_MONTH):
                bevroren.add(d)
                gebruikt[maand] = gebruikt.get(maand, 0) + 1
        d += timedelta(days=1)
    return bevroren


def current_streak(complete_days: set[date], freezes: set[date], today: date) -> int:
    """De actuele streak (in echt-gelogde dagen; freezes tellen niet mee).

    Ankerpunt: ``today`` als die compleet is, anders ``today - 1 dag`` (zo
    lijkt de streak overdag niet nul, terwijl je de dag nog niet hebt
    afgesloten). Loopt daarna achteruit zolang een dag compleet of bevroren
    is; een bevroren dag overbrugt zonder zelf de teller te verhogen.
    """
    anker = today if today in complete_days else today - timedelta(days=1)
    streak = 0
    d = anker
    while d in complete_days or d in freezes:
        if d in complete_days:
            streak += 1
        d -= timedelta(days=1)
    return streak


def longest_streak(complete_days: set[date], freezes: set[date]) -> int:
    """De langste aaneengesloten reeks ooit (freezes overbruggen, tellen
    zelf niet mee in het getal)."""
    if not complete_days:
        return 0
    alle_relevante = complete_days | freezes
    start, eind = min(alle_relevante), max(alle_relevante)
    langste = huidige = 0
    d = start
    while d <= eind:
        if d in complete_days:
            huidige += 1
            langste = max(langste, huidige)
        elif d not in freezes:
            huidige = 0
        d += timedelta(days=1)
    return langste


def consistency_pct(complete_days: set[date], today: date, days: int = 30) -> float:
    """Percentage complete dagen (freezes tellen NIET mee) over de laatste
    ``days`` kalenderdagen tot en met vandaag.

    Bewust de eerlijke, vergevingsgezinde maat náást de streak: een freeze
    mag de streak redden maar blaast dit percentage niet op.
    """
    if days <= 0:
        return 0.0
    venster = [today - timedelta(days=i) for i in range(days)]
    compleet = sum(1 for d in venster if d in complete_days)
    return round(compleet / days * 100, 1)


def milestone_reached(streak: int) -> int | None:
    """Alleen bij een exacte mijlpaal (7/14/30/60/100/365), anders None."""
    return streak if streak in MILESTONES else None


def never_twice_message(complete_days: set[date], today: date) -> str | None:
    """"Nooit twee keer missen"-melding: gisteren gemist, vandaag nog niks.

    Geen melding bij een compleet lege log — dat zou een nieuwe gebruiker
    zonder enige historie een verwijt geven dat nergens op slaat.
    """
    if not complete_days:
        return None
    if today in complete_days:
        return None
    if (today - timedelta(days=1)) in complete_days:
        return None
    return NEVER_TWICE_TEKST


def heatmap_matrix(complete_days: set[date], freezes: set[date],
                   start: date, end: date) -> tuple[list, list, list, list]:
    """``(z, x_labels, y_labels, hovertext)`` voor een 7-rijen (ma..zo) x
    N-weken kalenderheatmap.

    ``z``-waarden: 0 = leeg, 1 = gedaan, 2 = bevroren; ``None`` voor cellen
    buiten ``[start, end]`` (de rand van de eerste/laatste weekkolom valt
    zelden precies op een maandag/zondag). Puur, dus testbaar zonder Plotly.
    """
    start_maandag = start - timedelta(days=start.weekday())
    eind_zondag = end + timedelta(days=6 - end.weekday())
    n_weken = (eind_zondag - start_maandag).days // 7 + 1

    z = [[None] * n_weken for _ in range(7)]
    hover = [[""] * n_weken for _ in range(7)]
    x_labels = []
    for week_idx in range(n_weken):
        week_maandag = start_maandag + timedelta(weeks=week_idx)
        x_labels.append(week_maandag.isoformat())
        for dag_idx in range(7):
            d = week_maandag + timedelta(days=dag_idx)
            if d < start or d > end:
                continue
            if d in complete_days:
                waarde = 1
            elif d in freezes:
                waarde = 2
            else:
                waarde = 0
            z[dag_idx][week_idx] = waarde
            hover[dag_idx][week_idx] = d.isoformat()
    y_labels = ["ma", "di", "wo", "do", "vr", "za", "zo"]
    return z, x_labels, y_labels, hover
