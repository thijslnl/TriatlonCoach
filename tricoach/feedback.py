"""Feedback direct na een upload: gemma doet het voorwerk, Haiku coacht.

Tweetraps, zoals afgesproken voor het hele project:

1. **Ollama (gemma)** heeft tijdens de import al een beschrijvende observatie
   gemaakt en de tijd-in-zones berekend. Dat voorwerk geven we via
   ``ImportResult`` door, zodat de feedback-stap er gratis op kan voortbouwen
   zonder Ollama nog eens aan te roepen.

2. **Anthropic (Haiku)** krijgt die samenvatting plus de relevante
   memory-context — de zones en het bekende aandachtspunt, het weekschema (voor
   de bedoeling van de sessie én de volgende geplande training), de
   tempo-bij-gelijke-hartslag-historie en de vórige voorgestelde aanpassing —
   en levert de eigenlijke feedback plus een eventueel bijsturingsadvies.

Welk model de coaching doet, is configureerbaar: taak ``feedback`` routet naar
anthropic en krijgt via ``anthropic.task_models`` standaard Haiku. De API-call
gebeurt alléén bij een nieuwe upload, nooit op een page-load. Elke sessie wordt
vastgelegd in ``memory/feedback.md``.
"""

import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from tricoach.analysis import pace_at_hr
from tricoach.fit_parser import ParsedActivity
from tricoach.formatting import (
    fmt_pace_per_100m,
    fmt_pace_per_km,
    fmt_speed_kmh,
    sport_label,
)
from tricoach.llm.router import LLMRouter
from tricoach.schedule import schedule_as_text
from tricoach.storage import load_activities
from tricoach.trainingslog import kerncijfers, zone_regel
from tricoach.zones import zone_bounds

SYSTEM = (
    "Je bent een nuchtere, ervaren triatloncoach. Je geeft feedback op één "
    "zojuist voltooide training van een beginnende triatleet die in mei 2027 "
    "een standaard/olympische triatlon (1,5/40/10 km) goed wil finishen. Schrijf "
    "in het Nederlands, maximaal "
    "5 à 6 korte zinnen, geen lap-voor-lap-analyse. Houd je aan deze eisen:\n"
    "- Concreet en kort: noem de cijfers die ertoe doen (tempo, hartslag, "
    "zoneverdeling), geen algemeenheden.\n"
    "- Toets de uitvoering aan de bedoeling van de sessie: was dit bedoeld als "
    "rustige zone 2-duurtraining? Bleef de atleet in zone 2? Zo niet, waar liep "
    "het mis? Als de opmerking de bedoeling verduidelijkt (bijv. 'intervaltraining "
    "bedoeld'), pas je toetsing daarop aan — een hoge hartslag is dan terecht en "
    "geen fout.\n"
    "- Weeg externe factoren (wind, en wat de atleet in de opmerking schreef) "
    "mee VOORDAT je een oordeel velt over tempo of efficiëntie. Een langzamere "
    "(terug)weg met tegenwind is geen slechte vorm. Als een lagere efficiëntie- "
    "of tempo-uitkomst verklaarbaar is door wind of een andere genoemde "
    "omstandigheid, benoem dat expliciet in plaats van het als achteruitgang te "
    "presenteren.\n"
    "- Vergelijk met de historie waar dat zinvol is, vooral tempo-bij-gelijke-"
    "hartslag als progressiemaat.\n"
    "- Bekend aandachtspunt: de atleet traint structureel te hard (te veel "
    "Z3/Z4). Beloon het expliciet als hij netjes in Z2 bleef; signaleer het "
    "scherp als hij weer te hard ging — tenzij de opmerking aangeeft dat hard "
    "trainen juist de bedoeling was.\n"
    "- Was er een vorige voorgestelde aanpassing? Benoem kort of hij zich eraan "
    "hield.\n"
    "- Genereer geen calorie-, dieet- of afvaldoelen.\n"
    "Sluit af met een oordeel over de VOLGENDE training. Geef je antwoord exact "
    "in dit formaat, met deze twee labels elk op een eigen regel:\n"
    "FEEDBACK: <je feedback van maximaal 5 à 6 zinnen>\n"
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

# Zoveel recente trainingslog-entries gaan mee als historie-context.
MAX_LOG_ENTRIES = 6


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


def _pace_history(conn: sqlite3.Connection, sport: str, z2: tuple[int, int]) -> str:
    """Tempo-bij-gelijke-hartslag (zone 2) per eerdere sessie, als tekstregels.

    Dit is de belangrijkste progressiemaat. Zwemmen heeft te weinig zone 2-tijd
    om hier zinvol te zijn; daar geven we dat eerlijk aan.
    """
    if sport == "swimming":
        return "(voor zwemmen nog niet zinvol — te weinig zone 2-tijd)"

    acts = load_activities(conn)
    trend = pace_at_hr(conn, acts, sport, z2)
    if trend.empty:
        return "(nog geen eerdere sessies met genoeg tijd in zone 2)"

    regels = []
    for _, r in trend.iterrows():
        datum = r["start_time"].strftime("%d-%m-%Y")
        if sport == "running":
            regels.append(f"- {datum}: {fmt_pace_per_km(r['speed_ms'])} bij HR {z2[0]}-{z2[1]}")
        else:  # cycling
            regels.append(f"- {datum}: {fmt_speed_kmh(r['speed_ms'])} bij HR {z2[0]}-{z2[1]}")
    return "\n".join(regels)


def _recent_log(memory_dir: Path, exclude_key: str, n: int = MAX_LOG_ENTRIES) -> str:
    """De laatste n trainingslog-entries, exclusief de zojuist geïmporteerde."""
    path = memory_dir / "trainingslog.md"
    if not path.exists():
        return "(nog geen logboek)"
    parts = path.read_text(encoding="utf-8").split("\n## ")[1:]
    parts = [p for p in parts if exclude_key not in p]
    return "\n\n".join("## " + p for p in parts[-n:]) or "(nog geen eerdere sessies)"


def last_proposed_adjustment(memory_dir: Path) -> str | None:
    """De laatst voorgestelde aanpassing uit feedback.md (voor de adherence-check)."""
    path = memory_dir / "feedback.md"
    if not path.exists():
        return None
    secties = path.read_text(encoding="utf-8").split("\n## ")[1:]
    for blok in reversed(secties):
        m = re.search(r"\*\*Voorgestelde aanpassing:\*\* (.+)", blok)
        if m and m.group(1).strip() not in ("—", "-", "Geen"):
            return m.group(1).strip()
    return None


def _build_context(
    conn: sqlite3.Connection,
    memory_dir: Path,
    config: dict,
    act: ParsedActivity,
    tiz: dict[str, int],
    observation: str | None,
    user_note: str | None = None,
    wind: "object | None" = None,
) -> str:
    """Bouw de compacte prompt-context voor de coaching-feedback.

    ``user_note`` (de opmerking bij de upload) en ``wind`` (Open-Meteo) gaan als
    expliciete context mee, zodat de coach externe factoren kan meewegen vóór een
    oordeel over tempo of efficiëntie.
    """
    bounds = zone_bounds(config["athlete"])
    z2 = (bounds[0], bounds[1] - 1)
    total = sum(tiz.values()) or 1
    z2_share = tiz.get("Z2", 0) / total * 100
    hard_share = (tiz.get("Z4", 0) + tiz.get("Z5", 0)) / total * 100

    sessie = (
        "# Zojuist voltooide sessie\n\n"
        f"- Sport: {sport_label(act.sport)}\n"
        f"- Datum: {act.start_time:%A %d-%m-%Y %H:%M}\n"
        f"- Kerncijfers: {kerncijfers(act)}\n"
        f"- Tijd in zones: {zone_regel(tiz)}\n"
        f"- Aandeel zone 2: {z2_share:.0f}% · aandeel zone 4+5: {hard_share:.0f}%\n"
        f"- Gemma-observatie: {observation or '(geen)'}\n"
        f"- Wind tijdens de sessie (Open-Meteo): "
        f"{wind.as_text() if wind is not None else '(geen winddata)'}\n"
        f"- Opmerking van de atleet: {user_note or '(geen)'}"
    )

    vorige = last_proposed_adjustment(memory_dir)
    blocks = [
        sessie,
        "# Externe context — meewegen vóór een oordeel\n\n"
        "- Wind en de opmerking van de atleet zijn objectieve/subjectieve "
        "context. Een langzamere terugweg met tegenwind, of een sessie die de "
        "atleet als zwaar omschreef, is geen bewijs van slechte vorm. Verklaar "
        "een lagere efficiëntie- of tempo-uitkomst eerst hiermee voordat je hem "
        "als achteruitgang duidt. Verduidelijkt de opmerking de bedoeling van de "
        "sessie (bijv. intervallen), toets daar dan op.",
        "# Zones en aandachtspunt\n\n"
        f"- Zones (%LTHR, LTHR {config['athlete']['lthr']}): "
        f"Z2 {bounds[0]}-{bounds[1] - 1} · Z3 {bounds[1]}-{bounds[2] - 1} · "
        f"Z4 {bounds[2]}-{bounds[3] - 1} · Z5 {bounds[3]}+\n"
        "- Aandachtspunt: de atleet traint structureel te hard; meer zone 2 is "
        "de grootste verbeterkans.",
        f"# Weekschema (voor bedoeling sessie + volgende training)\n\n{schedule_as_text(memory_dir)}",
        f"# Tempo bij gelijke hartslag (zone 2), historie\n\n"
        f"{_pace_history(conn, act.sport, z2)}",
        f"# Recente sessies (logboek)\n\n{_recent_log(memory_dir, act.activity_key)}",
    ]
    if vorige:
        blocks.append(
            "# Vorige voorgestelde aanpassing (toets of hij is opgevolgd)\n\n"
            f"{vorige}"
        )
    blocks.append(
        "# Opdracht\n\nGeef de feedback op de zojuist voltooide sessie en sluit "
        "af met je oordeel over de volgende training, in het gevraagde formaat."
    )
    return "\n\n---\n\n".join(blocks)


def _parse_reply(raw: str) -> tuple[str, str | None]:
    """Splits het Haiku-antwoord in feedback en aanpassing.

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
) -> Feedback:
    """Genereer en bewaar de coaching-feedback voor één zojuist geïmporteerde sessie.

    ``user_note`` (de opmerking bij de upload) en ``wind`` (Open-Meteo) gaan als
    meegewogen context mee naar het model en worden in feedback.md vastgelegd.
    """
    context = _build_context(conn, memory_dir, config, act, tiz, observation, user_note, wind)
    raw = router.ask("feedback", context, system=SYSTEM)
    feedback_text, aanpassing = _parse_reply(raw)

    fb = Feedback(
        activity_key=act.activity_key,
        sport=sport_label(act.sport),
        start_time=f"{act.start_time:%a %d-%m %H:%M}",
        kerncijfers=kerncijfers(act),
        zoneverdeling=zone_regel(tiz),
        feedback=feedback_text,
        aanpassing=aanpassing,
    )
    _append_feedback_md(memory_dir, act, fb, user_note=user_note, wind=wind)
    return fb


def _append_feedback_md(
    memory_dir: Path,
    act: ParsedActivity,
    fb: Feedback,
    user_note: str | None = None,
    wind: "object | None" = None,
) -> None:
    """Leg de feedback vast in memory/feedback.md (nieuwste onderaan).

    De meegewogen context (wind + opmerking) wordt erbij genoteerd, zodat later
    te zien is op welke gronden het oordeel mild of streng was.
    """
    path = memory_dir / "feedback.md"
    if not path.exists():
        path.write_text(HEADER, encoding="utf-8")
    entry = (
        f"\n## {act.start_time:%Y-%m-%d %a} — {fb.sport}\n\n"
        f"- **Kerncijfers:** {fb.kerncijfers}\n"
    )
    if wind is not None:
        entry += f"- **Wind (Open-Meteo):** {wind.as_text()}\n"
    if user_note:
        entry += f"- **Opmerking atleet:** {user_note}\n"
    entry += (
        f"- **Feedback:** {fb.feedback}\n"
        f"- **Voorgestelde aanpassing:** {fb.aanpassing or 'Geen — volgende sessie zoals gepland'}\n"
        f"- _Sleutel `{act.activity_key}`_\n"
    )
    with open(path, "a", encoding="utf-8") as f:
        f.write(entry)


def last_feedback_markdown(memory_dir: Path) -> str | None:
    """De laatst vastgelegde feedback-sectie (voor weergave zonder nieuwe call)."""
    path = memory_dir / "feedback.md"
    if not path.exists():
        return None
    parts = path.read_text(encoding="utf-8").split("\n## ")
    return "## " + parts[-1] if len(parts) > 1 else None
