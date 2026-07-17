"""Memory-onderhoud: versheid van doelen.md bewaken en reviewdata leveren.

De feedback- en adviesprompts lezen ``memory/doelen.md`` integraal mee; als
daar verouderde informatie staat, sijpelt die in élke feedback door. Dit
module doet daar twee dingen tegen:

1. **Wijzigingsdatum per regel** (:func:`track_line_dates`): memory/ staat
   niet in git, dus we houden zelf een snapshot bij (een verborgen
   JSON-bestand naast doelen.md). Bij elke aanroep worden de huidige regels
   met difflib tegen de vorige snapshot gelegd: ongewijzigde regels houden
   hun datum, nieuwe of gewijzigde regels krijgen vandaag. De instellingen-tab
   toont dit als reviewtabel, zodat periodiek opschonen makkelijk is.
2. **Versheidswaarschuwingen** (:func:`freshness_warnings`): is het
   handgeschreven deel ouder dan ~8 weken, of conflicteert het aantoonbaar
   met recente sessies (bijv. "net begonnen met zwemmen" terwijl de laatste
   sessies vrijwel volledig crawl zijn), dan gaan er waarschuwingsregels mee
   de prompt in zodat de coach de context niet klakkeloos overneemt.

Alleen het handgeschreven deel telt: het profielblok (tussen de
PROFIEL-markers) wordt door het dashboard beheerd en de changelog groeit bij
elke logregel — beide zouden de reviewdata alleen maar vervuilen. Doordat de
changelog buiten de tracking valt, kan het loggen van een review-wijziging
zelf nooit een nieuwe wijziging uitlokken.
"""

import json
from dataclasses import dataclass
from datetime import date, datetime
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd

from tricoach.profile import CHANGELOG_HEADER, START as PROFIEL_START

# Ouder dan dit geldt het handgeschreven profiel als "mogelijk verouderd".
MAX_LEEFTIJD_WEKEN = 8
# Sidecar met de vorige snapshot van doelen.md (regels + datums).
STATE_BESTAND = ".doelen_regels.json"
# Vanaf dit crawl-aandeel in recente zwemsessies geldt "crawl is de
# hoofdslag" als aangetoond, en botst beginnerstaal in doelen.md daarmee.
CRAWL_HOOFDSLAG_PCT = 80.0
# Tekstsignalen in doelen.md die op een zwem-beginnersfase duiden. Bewust
# zonder het ambigue "leren zwemmen" (dat matcht ook op "níet leren zwemmen").
BEGINNER_SIGNALEN = ("net begonnen", "cursus van 10 lessen gepland",
                     "komt later", "moet nog starten")


@dataclass
class RegelDatum:
    """Eén regel uit doelen.md met de datum waarop hij voor het laatst wijzigde."""
    regel: str
    datum: date


def _handgeschreven_deel(text: str) -> list[str]:
    """De regels vóór het beheerde profielblok (of de changelog, wat eerder komt)."""
    for marker in (PROFIEL_START, CHANGELOG_HEADER):
        if marker in text:
            text = text.split(marker, 1)[0]
    return text.rstrip().splitlines()


def _lees_state(state_path: Path) -> list[dict] | None:
    """De vorige snapshot: [{"text": regel, "date": "YYYY-MM-DD"}, ...] of None."""
    if not state_path.exists():
        return None
    try:
        data = json.loads(state_path.read_text(encoding="utf-8"))
        return data.get("lines")
    except (ValueError, OSError):
        return None


def track_line_dates(path: Path, state_path: Path | None = None,
                     ) -> tuple[list[RegelDatum], dict[str, int] | None]:
    """Wijzigingsdatum per regel van het handgeschreven deel van ``path``.

    Legt de huidige regels tegen de vorige snapshot: ongewijzigde regels
    houden hun opgeslagen datum, nieuwe/gewijzigde regels krijgen vandaag.
    De snapshot wordt daarna bijgewerkt. Bij de allereerste aanroep (geen
    snapshot) geldt de bestandsdatum (mtime) voor alle regels.

    Geeft ``(regels, wijzigingen)`` terug; ``wijzigingen`` is None als er
    niets veranderde, anders ``{"toegevoegd": n, "verwijderd": n}`` —
    een gewijzigde regel telt als verwijderd + toegevoegd.
    """
    if state_path is None:
        state_path = path.parent / STATE_BESTAND
    if not path.exists():
        return [], None

    regels = _handgeschreven_deel(path.read_text(encoding="utf-8"))
    vorige = _lees_state(state_path)
    vandaag = date.today()

    if vorige is None:
        mtime = date.fromtimestamp(path.stat().st_mtime)
        datums = [mtime] * len(regels)
        wijzigingen = None
    else:
        oude_regels = [e["text"] for e in vorige]
        datums = [vandaag] * len(regels)
        toegevoegd = len(regels)
        verwijderd = len(oude_regels)
        sm = SequenceMatcher(a=oude_regels, b=regels, autojunk=False)
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag != "equal":
                continue
            for offset in range(j2 - j1):
                datums[j1 + offset] = date.fromisoformat(vorige[i1 + offset]["date"])
            toegevoegd -= j2 - j1
            verwijderd -= i2 - i1
        wijzigingen = ({"toegevoegd": toegevoegd, "verwijderd": verwijderd}
                       if toegevoegd or verwijderd else None)

    state_path.write_text(json.dumps({
        "updated": datetime.now().isoformat(timespec="seconds"),
        "lines": [{"text": r, "date": d.isoformat()} for r, d in zip(regels, datums)],
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    return [RegelDatum(r, d) for r, d in zip(regels, datums)], wijzigingen


def refresh_doelen_tracking(memory_dir: Path) -> list[RegelDatum]:
    """Ververs de regeldatums van doelen.md en log gevonden wijzigingen.

    Het ene punt waarlangs alle lezers (reviewpagina én feedback-context) de
    tracking bijwerken: wie de wijziging ook als eerste ziet, hij komt één
    keer in de changelog terecht. Handmatige edits krijgen als datum de dag
    waarop ze voor het eerst zijn geconstateerd.
    """
    regels, wijzigingen = track_line_dates(memory_dir / "doelen.md")
    if wijzigingen:
        delen = []
        if wijzigingen["toegevoegd"]:
            delen.append(f"{wijzigingen['toegevoegd']} regel(s) nieuw of gewijzigd")
        if wijzigingen["verwijderd"]:
            delen.append(f"{wijzigingen['verwijderd']} regel(s) vervallen")
        log_review_change(memory_dir, "doelen.md bewerkt — " + ", ".join(delen))
    return regels


def review_dataframe(memory_dir: Path) -> pd.DataFrame:
    """doelen.md als reviewtabel: regel + laatste wijzigingsdatum + leeftijd.

    Voor de instellingen-tab; lege regels blijven weg zodat de tabel compact
    is.
    """
    regels = refresh_doelen_tracking(memory_dir)
    vandaag = date.today()
    rows = [{
        "Regel": r.regel,
        "Laatst gewijzigd": r.datum,
        "Weken oud": (vandaag - r.datum).days // 7,
    } for r in regels if r.regel.strip()]
    return pd.DataFrame(rows)


def log_review_change(memory_dir: Path, beschrijving: str) -> None:
    """Voeg een memory-review-entry toe aan de changelog in doelen.md.

    De changelog is (door :func:`tricoach.profile.update_doelen`) altijd de
    laatste sectie, dus een entry toevoegen is een append; ontbreekt de kop
    nog, dan komt die er eerst bij.
    """
    path = memory_dir / "doelen.md"
    if not path.exists():
        return
    text = path.read_text(encoding="utf-8")
    entry = f"- **{date.today():%Y-%m-%d}** (memory-review): {beschrijving}"
    if CHANGELOG_HEADER not in text:
        text = text.rstrip() + f"\n\n{CHANGELOG_HEADER}\n"
    path.write_text(text.rstrip() + "\n" + entry + "\n", encoding="utf-8")


def _recent_crawl_pct(acts: pd.DataFrame, conn) -> float | None:
    """Gemiddeld crawl-aandeel (%) over de laatste 4 zwemsessies met baandata."""
    swims = acts[acts["sport"] == "swimming"].sort_values("start_time").tail(4)
    pcts = []
    for _, act in swims.iterrows():
        lengths = pd.read_sql_query(
            "SELECT swim_stroke FROM lengths WHERE activity_key = ?",
            conn, params=(act["activity_key"],))
        if not lengths.empty:
            pcts.append((lengths["swim_stroke"] == "freestyle").mean() * 100)
    return sum(pcts) / len(pcts) if pcts else None


def freshness_warnings(memory_dir: Path, conn, acts: pd.DataFrame,
                       max_age_weeks: int = MAX_LEEFTIJD_WEKEN) -> list[str]:
    """Waarschuwingen over mogelijk verouderde doelen.md-context, voor de prompt.

    Twee lichte checks:

    - **Leeftijd**: is de jongste regel van het handgeschreven deel ouder dan
      ``max_age_weeks`` weken, dan is het hele profiel aan review toe.
    - **Zwemconflict**: bevat de tekst beginnerssignalen ("net begonnen",
      "cursus moet nog starten") terwijl de recente zwemsessies vrijwel
      volledig crawl zijn, dan is dat aantoonbaar achterhaald.

    Lege lijst = niets aan de hand. De regels zijn geformuleerd als
    instructie aan de coach ("behandel als mogelijk verouderd"), niet als
    feit, zodat een vals alarm geen schade doet.
    """
    path = memory_dir / "doelen.md"
    if not path.exists():
        return []
    warnings = []

    regels = refresh_doelen_tracking(memory_dir)
    if regels:
        jongste = max(r.datum for r in regels)
        weken = (date.today() - jongste).days // 7
        if weken >= max_age_weeks:
            warnings.append(
                f"doelen.md is voor het laatst inhoudelijk gewijzigd op "
                f"{jongste:%d-%m-%Y} ({weken} weken geleden). Behandel details "
                "die botsen met recente sessies of opmerkingen als mogelijk "
                "verouderd en benoem dat expliciet in plaats van ze over te nemen."
            )

    tekst = "\n".join(r.regel for r in regels).lower()
    if any(signaal in tekst for signaal in BEGINNER_SIGNALEN):
        crawl_pct = _recent_crawl_pct(acts, conn)
        if crawl_pct is not None and crawl_pct >= CRAWL_HOOFDSLAG_PCT:
            warnings.append(
                "doelen.md bevat nog zwem-beginnerstaal, maar de recente "
                f"zwemsessies zijn gemiddeld {crawl_pct:.0f}% crawl — de crawl "
                "is aantoonbaar de hoofdslag. Baseer het zwemoordeel op de "
                "recente sessies en markeer de profieltekst als verouderd."
            )
    return warnings
