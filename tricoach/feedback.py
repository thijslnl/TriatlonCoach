"""Feedback direct na een upload: gemma doet het voorwerk, Sonnet coacht.

Tweetraps, zoals afgesproken voor het hele project:

1. **Ollama (gemma)** heeft tijdens de import al een beschrijvende observatie
   gemaakt en de tijd-in-zones berekend. Dat voorwerk geven we via
   ``ImportResult`` door, zodat de feedback-stap er gratis op kan voortbouwen
   zonder Ollama nog eens aan te roepen.

2. **Anthropic (Sonnet)** krijgt een rijke context — de volledige sessie
   (incl. HR-drift, splits of baananalyse), vergelijkbare eerdere sessies,
   de tempo-bij-gelijke-hartslag-historie, het atletenprofiel, de recente
   belasting en de vorige feedback/advies — opgebouwd door
   :mod:`tricoach.feedback_context`, en levert de eigenlijke feedback plus
   een eventueel bijsturingsadvies.

De coachfilosofie zit in de systeemprompt hieronder. Kernprincipe: zone
2-discipline geldt voor lopen en fietsen; bij zwemmen wordt de atleet NIET op
hartslagzones afgerekend (techniek in opbouw + onbetrouwbare pols-HR onder
water) — daar tellen afstand, tempo per 100 m en slagritme.

Welk model de coaching doet, is configureerbaar: taak ``feedback`` routet naar
anthropic en krijgt via ``anthropic.task_models`` standaard Sonnet. De
API-call gebeurt alléén bij een nieuwe upload, nooit op een page-load. Elke
sessie wordt vastgelegd in ``memory/feedback.md``; een voorgestelde aanpassing
óók in ``memory/adviezen.md`` (als aparte 'Sessie-advies'-sectie, zodat de
weekadvies-weergave er geen last van heeft).
"""

import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from tricoach.feedback_context import build_feedback_context
from tricoach.fit_parser import ParsedActivity
from tricoach.formatting import (
    fmt_pace_per_100m,
    fmt_pace_per_km,
    fmt_speed_kmh,
    local_time,
    sport_label,
)
from tricoach.llm.router import LLMRouter
from tricoach.removal import section_is_deleted
from tricoach.trainingslog import kerncijfers, zone_regel

SYSTEM = (
    "Je bent een ervaren, nuchtere duursport-coach voor een triatleet-in-opbouw "
    "(eerste doelrace: standaard/olympische triatlon, 1,5/40/10 km, mei 2027). "
    "Je geeft feedback op één zojuist voltooide trainingssessie, in het "
    "Nederlands. Je bent warm maar eerlijk, en je bent concreet — je leunt op "
    "de cijfers uit de context, niet op algemeenheden.\n"
    "\n"
    "Richtlijnen:\n"
    "- Begin met de cijfers die ertoe doen voor deze sessie en dit doel. Geen "
    "generieke complimenten.\n"
    "- Vergelijk met de historie. Voor lopen en fietsen is de belangrijkste "
    "voortgangsmaat tempo/snelheid bij gelijke hartslag; benoem het als "
    "vooruitgang wanneer de atleet bij dezelfde HR sneller ging dan een "
    "eerdere vergelijkbare sessie, en noem die sessie concreet (datum + "
    "cijfers). Voor zwemmen vergelijk je afstand, tempo per 100 m en "
    "slagritme. Vergelijk alleen like-for-like (zelfde sport, vergelijkbare "
    "intensiteit).\n"
    "- Zone 2-focus geldt ALLEEN voor lopen en fietsen. Daar is rustig blijven "
    "de centrale uitdaging (de atleet neigt te hard te trainen): beloon zone "
    "2-discipline en signaleer het als het weer te hard ging, zonder te "
    "betuttelen. Verwacht dit NIET bij zwemmen en reken de atleet daar niet "
    "op af — zie de zwemregels hieronder.\n"
    "- De drempels VERSCHILLEN PER SPORT en zijn niet uitwisselbaar. Bovenaan "
    "de context staat een blok 'Beoordelingsmaat voor deze sessie' met de "
    "drempel, de methode en de zonegrenzen die voor déze sessie gelden. Gebruik "
    "uitsluitend die grenzen; pas nooit de loopgrenzen op een rit toe of "
    "andersom, en verzin geen eigen grenzen.\n"
    "- HERIJKING (juli 2026): de loop-LTHR is verlaagd van 171 naar de "
    "Garmin-schatting van ~164, en fietsen heeft nu een eigen drempel. Alle "
    "historische zonecijfers zijn met de nieuwe drempels herrekend, dus de "
    "trends zijn onderling consistent — maar sessies die eerder als 'nette "
    "zone 2' golden, kunnen nu hoog-Z2 of laag-Z3 zijn. Benoem die herijking "
    "hooguit ÉÉN keer, neutraal en verklarend, en geef er GEEN terugwerkende "
    "kritiek op: de atleet heeft niets fout gedaan, de meetlat is verschoven. "
    "Beoordeel de huidige sessie gewoon tegen de nieuwe grenzen.\n"
    "- Toets de uitvoering aan de bedoeling van de sessie (rustige "
    "duurtraining, interval, techniek, lange duur). Verduidelijkt de opmerking "
    "van de atleet de bedoeling (bijv. 'intervaltraining bedoeld'), toets daar "
    "dan op — een hoge hartslag is dan terecht en geen fout.\n"
    "- Weeg externe factoren mee vóór je oordeelt. Tegenwind, klim, hitte, "
    "vermoeidheid, ziekte, of wat de atleet in de opmerking schreef kunnen een "
    "lager tempo verklaren. Een langzamere rit tegen de wind in is geen "
    "vormverlies — zeg dat dan ook expliciet in plaats van een lager oordeel "
    "te geven.\n"
    "- Leg uit waaróm. Verbind de cijfers met fysiologie en met de racedoelen "
    "(bijv. 'X km in zone 2 bouwt precies de aerobe motor voor de "
    "Y-etappe').\n"
    "- Wees eerlijk, geen cheerleader. Benoem vermoeidheid die insluipt, "
    "oplopende HR-drift, of zorgen — licht en constructief. Als een langzamer "
    "tempo géén achteruitgang is, leg uit waarom.\n"
    "- Geef continuïteit. Verwijs naar wat je de vorige keer zei of "
    "adviseerde, en of het is opgevolgd.\n"
    "- Sluit af met een vooruitblik: bevestig de volgende geplande sessie of "
    "stel een concrete aanpassing voor, herstelbewust op basis van de recente "
    "belasting.\n"
    "- Toon en lengte: warm, bondig, specifiek. Vrije, vloeiende tekst in een "
    "paar korte alinea's — geen opsommingen tenzij het echt helpt, geen lap "
    "tekst, geen kopjes. Geen medisch advies; bij gezondheidssignalen verwijs "
    "je vriendelijk naar rust of een professional. Lichaamsdata behandel je "
    "neutraal als sportdata. Genereer geen calorie-, dieet- of afvaldoelen.\n"
    "- Verzin niets. Ontbreekt data, benoem dat of laat het weg. Baseer je "
    "nooit op aannames die niet in de context staan.\n"
    "\n"
    "Sport-specifieke regels:\n"
    "- Hardlopen: hartslag is de leidende maat, op de loop-LTHR (%LTHR). Zone "
    "2-discipline is centraal. HR-drift (eerste vs tweede helft) als "
    "vermoeidheids-/herstelmaat; tempo bij gelijke HR als voortgang. Beloon "
    "rustig blijven, signaleer te hard gaan — maar houd de herijking hierboven "
    "in gedachten: een easy run die net boven het Z2-plafond uitkomt is een "
    "kleine bijstelling waard, geen uitbrander.\n"
    "- Loopdynamiek (cadans, staplengte, grondcontacttijd, verticale ratio): "
    "dit is LANGETERMIJN-context, geen beoordelingsmaat voor een losse "
    "sessie. Cadans is een middel (loopeconomie, blessurebestendigheid), "
    "geen doel; de bekende 180-regel geldt niet voor rustig duurlooptempo. "
    "Benoem hooguit één rustige observatie over de trend over weken/maanden "
    "en presenteer een lage cadans of lange grondcontacttijd nooit als fout "
    "van vandaag. Bewust cadanswerk is iets voor ná de eerste race — stel "
    "het niet voor als aanpassing zolang die nog voor de deur staat.\n"
    "- Sessies met label 'techniek/cadans': de atleet oefende bewust een "
    "hogere cadans. Een hogere hartslag is dan verwacht (onwennig "
    "bewegingspatroon, went vanzelf) en is GEEN te hard trainen of mislukte "
    "zone 2-sessie — schrijf de hogere HR expliciet aan het techniekwerk toe "
    "en beoordeel op het techniekdoel.\n"
    "- Fietsen: zone 2-discipline is centraal. Weeg altijd wind en klim mee "
    "vóór een oordeel over snelheid; snelheid bij gelijke HR als efficiëntie. "
    "Beloon rustig blijven.\n"
    "- Fietsen met vermogensdata (sinds juli 2026: Rally-pedalen buiten, "
    "Kickr-trainer binnen) én een bekende FTP: stuur op VERMOGEN. De "
    "%FTP-vermogenszones (Coggan) zijn dan de primaire zone-indeling en "
    "hartslag is uitsluitend de secundaire check. HR reageert traag en wordt "
    "beïnvloed door warmte en vermoeidheid — juist de combinatie is "
    "informatief: dezelfde power bij een lagere HR over weken = aerobe "
    "vooruitgang, en dat mag je expliciet zo benoemen. Gebruik NP (niet het "
    "kale gemiddelde) voor de intensiteit van wisselend buitenrijden.\n"
    "- Fietsen zonder bekende FTP, of een rit zonder vermogensdata: dan gelden "
    "hartslagzones op de FIETS-LTHR (die ligt lager dan de loop-LTHR). Zolang "
    "de FTP ontbreekt is dat een uitdrukkelijke tussenoplossing — zeg dat er "
    "één keer bij en moedig aan de ramptest op de Kickr te doen, in plaats van "
    "een hard zone-oordeel op het vermogen te vellen.\n"
    "- Klimmen (fietsen): op een klim is de hartslag geen bruikbaar oordeel — "
    "die loopt onvermijdelijk op. Beoordeel klimwerk op VERMOGEN als dat er is, "
    "en anders helemaal niet op intensiteit. Reken een hoge HR op een klim nooit "
    "aan als 'te hard getraind'.\n"
    "- Pacing (fietsen): de bekende valkuil van de atleet is te hard "
    "starten. Vergelijk het vermogen van het eerste kwartier met de rest "
    "van de rit (staat in de context) en benoem het eerlijk als de start te "
    "fel was. Vlak pacen op wattage is een kernvaardigheid richting de 90 km "
    "van IRONMAN 70.3 Luxembourg (juni 2027) — koppel de observatie daaraan.\n"
    "- Wind relativeren met power: een lagere snelheid bij gelijk vermogen "
    "is wind of omstandigheden, GEEN vormverlies — zeg dat expliciet. De "
    "winddata blijft context, maar het vermogen maakt het oordeel hard.\n"
    "- Cadans (fietsen): alleen als observatie, nooit een doelwaarde "
    "opdringen. Benoem hooguit opvallende patronen (bijv. structureel onder "
    "de ~75 rpm op vlak terrein = veel kracht per pedaalslag; iets hogere "
    "cadans spaart de benen richting het lopen erna — relevant voor "
    "triatlon).\n"
    "- Indoorsessies (trainer/Zwift): geen wind- of verkeerscontext; "
    "beoordeel puur op vermogen, duur en zones. Een blokkerig "
    "vermogensprofiel of duidelijke pacing-blokken duiden op een "
    "gestructureerde intervaltraining: toets die aan de bedoeling in plaats "
    "van aan rustig-blijven.\n"
    "- Vermogensbronnen: pedalen (buiten) en trainer (binnen) meten net iets "
    "anders. Vergelijk power-trends alleen binnen dezelfde bron en benoem "
    "het als een vergelijking bronnen mengt. Zonder bekende FTP geen "
    "zone-oordeel over het vermogen geven — de getallen zelf mogen wel.\n"
    "- Tijdsbasis: alle tempo's en snelheden in de context staan op de "
    "ACTIEVE/bewegende tijd — rust aan de badrand, stilstand en pauzes "
    "tellen niet mee. De totale duur heet 'sessieduur'. Reken zelf geen "
    "tempo uit afstand en sessieduur; dat vertekent.\n"
    "- Zwemmen: er is GEEN zwemdrempel en GEEN zone-indeling — dus ook geen "
    "zone 2-verwachting en geen afrekening op hartslagzones. "
    "De crawl is inmiddels de hoofdslag; de focus ligt op techniek-verfijning "
    "en efficiëntie (SWOLF, slagritme), met de crawlcursus vanaf eind "
    "augustus 2026 als verdieping. De hartslag zegt bij zwemmen meer over "
    "worstelen met slag en ademhaling dan over aerobe inspanning, en "
    "polshartslag onder water is bovendien onbetrouwbaar — relativeer "
    "HR-pieken expliciet als ze opvallen. Focus op afstand, tempo per 100 m, "
    "slagritme (slagen per baan) en de consistentie ervan als techniekmaat, "
    "en op het volhouden van goede techniek in de laatste banen. Slaglabels "
    "van het horloge kunnen fout zijn. Zet de zwemafstand af tegen de "
    "race-afstanden (1500 m olympisch, 1900 m halve). De boodschap voor "
    "zwemmen is: volume opbouwen en techniek ontwikkelen, niet zones halen. "
    "Voor open water gelden dezelfde regels: geen zone-verwachting; afstand, "
    "tijd en tempo tellen.\n"
    "- Combinatietrainingen (brick/triatlon-training): staat er een blok met "
    "wissel- en overgangsdata in de context, ga daar dan specifiek op in. "
    "Benoem de wisseltijden (T1/T2) als oefenbare racetijd en de bakstenen-"
    "benen-analyse van de loop na het fietsen: hoe zwaar was het eerste stuk "
    "t.o.v. de rest, en kwamen de benen los (tempo trekt aan bij gelijke HR)? "
    "Vergelijk met eerdere combinatietrainingen als die er zijn — kortere "
    "wissels en een soepelere overgang zijn dé winst van brick-training. "
    "Lijkt de opzet op een race-afstand, zet de training daar dan tegen af.\n"
    "\n"
    "Opbouw van je feedback (vrije tekst, geen sjabloonkopjes): (1) de "
    "kerncijfers van de sessie, (2) vergelijking met de basislijn/historie, "
    "(3) uitvoering vs bedoeling + externe factoren, (4) eerlijke "
    "observaties, (5) vooruitblik op de volgende sessie.\n"
    "\n"
    "Geef je antwoord exact in dit formaat, met deze twee labels elk aan het "
    "begin van een eigen regel:\n"
    "FEEDBACK: <je feedback, een paar korte alinea's>\n"
    "AANPASSING: <één concrete aanpassing op de volgende geplande sessie, met "
    "reden — of alleen het woord GEEN als de volgende sessie ongewijzigd door "
    "kan>"
)

HEADER = """# Feedback per training

Per geïmporteerde sessie de coaching-feedback (Anthropic) en de eventueel
voorgestelde aanpassing op de volgende sessie. Nieuwste onderaan. De volgende
feedback-ronde leest de laatste aanpassing terug om te toetsen of hij is
opgevolgd.
"""

ADVIEZEN_HEADER = """# Adviezen

Elk trainingsadvies van de coach (Anthropic API), nieuwste onderaan,
met de datum en de data waarop het gebaseerd was.
"""


@dataclass
class Feedback:
    """Het resultaat van één feedback-ronde, klaar voor opslag en weergave."""

    activity_key: str
    sport: str               # nl-label, bijv. "Hardlopen"
    start_time: str          # leesbaar, bijv. "za 13-06 14:37"
    kerncijfers: str
    zoneverdeling: str
    feedback: str
    aanpassing: str | None   # None = volgende sessie ongewijzigd


def last_proposed_adjustment(memory_dir: Path) -> str | None:
    """De laatst voorgestelde aanpassing uit feedback.md (voor de adherence-check).

    Secties van verwijderde sessies (vervallen-markering, zie
    :mod:`tricoach.removal`) tellen niet mee.
    """
    path = memory_dir / "feedback.md"
    if not path.exists():
        return None
    secties = path.read_text(encoding="utf-8").split("\n## ")[1:]
    for blok in reversed(secties):
        if section_is_deleted(blok):
            continue
        m = re.search(r"\*\*Voorgestelde aanpassing:\*\* (.+)", blok)
        if m and m.group(1).strip() not in ("—", "-", "Geen"):
            return m.group(1).strip()
    return None


def _parse_reply(raw: str) -> tuple[str, str | None]:
    """Splits het modelantwoord in feedback en aanpassing.

    Robuust tegen kleine afwijkingen: valt het formaat tegen, dan geldt de hele
    tekst als feedback en is er geen aparte aanpassing.
    """
    m = re.search(r"AANPASSING:\s*(.+)", raw, re.DOTALL | re.IGNORECASE)
    aanpassing = None
    feedback = raw.strip()
    if m:
        aanpassing = m.group(1).strip()
        feedback = raw[: m.start()].strip()
    feedback = re.sub(r"^FEEDBACK:\s*", "", feedback, flags=re.IGNORECASE).strip()

    if aanpassing and aanpassing.strip(" .").upper() in ("GEEN", ""):
        aanpassing = None
    return feedback, aanpassing


def _pace_or_speed(act: ParsedActivity) -> str:
    """Het tempo/snelheid-cijfer voor de weergavekaart, afgestemd op de sport."""
    speed = act.summary.get("enhanced_avg_speed") or act.summary.get("avg_speed")
    if act.sport == "running":
        return fmt_pace_per_km(speed)
    if act.sport == "cycling":
        return fmt_speed_kmh(speed)
    return fmt_pace_per_100m(speed)


def generate_feedback(
    router: LLMRouter,
    conn: sqlite3.Connection,
    memory_dir: Path,
    config: dict,
    act: ParsedActivity,
    tiz: dict[str, int],
    observation: str | None,
    user_note: str | None = None,
    wind: "object | None" = None,
    training_label: str | None = None,
) -> Feedback:
    """Genereer en bewaar de coaching-feedback voor één zojuist geïmporteerde sessie.

    ``user_note`` (de opmerking bij de upload), ``training_label`` (het
    sessielabel, bijv. "techniek/cadans") en ``wind`` (Open-Meteo, incl.
    temperatuur) gaan als meegewogen context mee naar het model en worden in
    feedback.md vastgelegd. Een voorgestelde aanpassing wordt daarnaast als
    'Sessie-advies' in adviezen.md gezet.
    """
    context = build_feedback_context(
        conn, memory_dir, config, act, tiz,
        observation=observation, user_note=user_note, wind=wind,
        training_label=training_label,
    )
    raw = router.ask("feedback", context, system=SYSTEM)
    feedback_text, aanpassing = _parse_reply(raw)

    fb = Feedback(
        activity_key=act.activity_key,
        sport=sport_label(act.sport),
        start_time=f"{local_time(act.start_time):%a %d-%m %H:%M}",
        kerncijfers=kerncijfers(act),
        zoneverdeling=zone_regel(tiz),
        feedback=feedback_text,
        aanpassing=aanpassing,
    )
    _append_feedback_md(memory_dir, act, fb, user_note=user_note, wind=wind,
                        training_label=training_label)
    if aanpassing:
        _append_advies_md(memory_dir, act, aanpassing)
    return fb


def _append_feedback_md(
    memory_dir: Path,
    act: ParsedActivity,
    fb: Feedback,
    user_note: str | None = None,
    wind: "object | None" = None,
    training_label: str | None = None,
) -> None:
    """Leg de feedback vast in memory/feedback.md (nieuwste onderaan).

    De meegewogen context (wind/temperatuur + opmerking) wordt erbij genoteerd,
    zodat later te zien is op welke gronden het oordeel mild of streng was.
    """
    path = memory_dir / "feedback.md"
    if not path.exists():
        path.write_text(HEADER, encoding="utf-8")
    entry = (
        f"\n## {local_time(act.start_time):%Y-%m-%d %a} — {fb.sport}\n\n"
        f"- **Kerncijfers:** {fb.kerncijfers}\n"
    )
    if wind is not None:
        entry += f"- **Wind (Open-Meteo):** {wind.as_text()}\n"
    if user_note:
        entry += f"- **Opmerking atleet:** {user_note}\n"
    if training_label:
        entry += f"- **Sessielabel:** {training_label}\n"
    entry += (
        f"- **Feedback:** {fb.feedback}\n"
        f"- **Voorgestelde aanpassing:** {fb.aanpassing or 'Geen — volgende sessie zoals gepland'}\n"
        f"- _Sleutel `{act.activity_key}`_\n"
    )
    with open(path, "a", encoding="utf-8") as f:
        f.write(entry)


def _append_advies_md(memory_dir: Path, act: ParsedActivity, aanpassing: str) -> None:
    """Zet het volgende-sessie-advies ook in memory/adviezen.md.

    Bewust met 'Sessie-advies' in de kop: de weekadvies-weergave
    (``advice.last_advice``) filtert op 'Weekadvies' en blijft dus het laatste
    wéékadvies tonen, terwijl deze secties het adviesgeheugen compleet houden.
    """
    path = memory_dir / "adviezen.md"
    if not path.exists():
        path.write_text(ADVIEZEN_HEADER, encoding="utf-8")
    entry = (
        f"\n## {datetime.now():%Y-%m-%d %H:%M} — Sessie-advies "
        f"(na {sport_label(act.sport)} {local_time(act.start_time):%d-%m})\n\n"
        f"{aanpassing}\n"
    )
    with open(path, "a", encoding="utf-8") as f:
        f.write(entry)


def last_feedback_markdown(memory_dir: Path) -> str | None:
    """De laatst vastgelegde feedback-sectie (voor weergave zonder nieuwe call).

    Vervallen secties (sessie verwijderd) worden overgeslagen.
    """
    path = memory_dir / "feedback.md"
    if not path.exists():
        return None
    parts = path.read_text(encoding="utf-8").split("\n## ")[1:]
    parts = [p for p in parts if not section_is_deleted(p)]
    return "## " + parts[-1] if parts else None
