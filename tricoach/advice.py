"""Trainingsadvies via de Anthropic API, gevoed met de memory-bestanden.

De adviesfunctie bouwt een prompt uit:
- de doelen en voorkeuren (memory/doelen.md)
- het weekschema (memory/weekschema.md)
- de recente trainingen (laatste entries uit memory/trainingslog.md)
- het vorige advies (laatste entry uit memory/adviezen.md), zodat de coach
  voortbouwt in plaats van zichzelf te herhalen
- weekstatistieken uit de database (volume, zoneverdeling)

Elk gegenereerd advies wordt vastgelegd in memory/adviezen.md. Het dashboard
toont altijd het laatst opgeslagen advies en vraagt alleen een nieuw advies
aan als de gebruiker daar expliciet om vraagt (zuinig met API-calls).
"""

import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd

from tricoach.llm.router import LLMRouter
from tricoach.schedule import schedule_as_text
from tricoach.storage import load_activities

SYSTEM = (
    "Je bent een ervaren, nuchtere triatloncoach voor een beginnende triatleet "
    "met een jong gezin. Je adviseert in het Nederlands. Elk advies bevat per "
    "voorgestelde sessie: (1) de sport en de dag, (2) duur en/of afstand die "
    "binnen het beschikbare tijdblok past, (3) een concreet hartslagdoel met "
    "de zone erbij, (4) een kort voedingsadvies voor tijdens de sessie — "
    "concreet in grammen koolhydraten en milliliters vocht per uur; bij "
    "sessies tot ~75 minuten volstaat meestal water, daarboven koolhydraten "
    "(en train het race-eten op de lange duurtrainingen) — en (5) een korte "
    "motivatie waarom deze training nu past in de opbouw. Belangrijkste "
    "coachingsprincipe: de atleet traint structureel te hard (te veel Z3/Z4); "
    "stuur actief op meer Z2-volume. Houd je strikt aan het weekschema en de "
    "beschikbare tijd. Wees concreet en beknopt; geen algemene "
    "trainingsleer-verhandelingen."
)

ADVIEZEN_HEADER = """# Adviezen

Elk trainingsadvies van de coach (Anthropic API), nieuwste onderaan,
met de datum en de data waarop het gebaseerd was.
"""

# Hoeveel recente trainingslog-entries er in de prompt meegaan.
MAX_LOG_ENTRIES = 10


def _last_md_section(path: Path) -> str | None:
    """De laatste '## '-sectie uit een markdown-bestand (of None)."""
    if not path.exists():
        return None
    parts = path.read_text(encoding="utf-8").split("\n## ")
    return "## " + parts[-1] if len(parts) > 1 else None


def _recent_log_entries(memory_dir: Path, n: int = MAX_LOG_ENTRIES) -> str:
    """De laatste n entries uit het trainingslog."""
    path = memory_dir / "trainingslog.md"
    if not path.exists():
        return "(nog geen trainingen geïmporteerd)"
    parts = path.read_text(encoding="utf-8").split("\n## ")
    return "\n\n".join("## " + p for p in parts[1:][-n:])


def _week_stats(conn: sqlite3.Connection) -> str:
    """Volume en zoneverdeling van de afgelopen 7 en 28 dagen, als tekst."""
    acts = load_activities(conn)
    if acts.empty:
        return "(geen data)"

    lines = []
    now = pd.Timestamp.now(tz="UTC")
    for label, days in [("afgelopen 7 dagen", 7), ("afgelopen 28 dagen", 28)]:
        recent = acts[acts["start_time"] >= now - pd.Timedelta(days=days)]
        if recent.empty:
            lines.append(f"- {label}: geen trainingen")
            continue
        per_sport = recent.groupby("sport")["duration_s"].sum() / 3600
        sports = ", ".join(f"{s}: {h:.1f}u" for s, h in per_sport.items())
        total = recent["duration_s"].sum()
        z2 = recent["z2_s"].sum() / total * 100 if total else 0
        hard = (recent["z4_s"].sum() + recent["z5_s"].sum()) / total * 100 if total else 0
        lines.append(
            f"- {label}: {len(recent)} sessies, {total / 3600:.1f}u "
            f"({sports}); {z2:.0f}% in Z2, {hard:.0f}% in Z4+Z5"
        )
    return "\n".join(lines)


def build_context(conn: sqlite3.Connection, memory_dir: Path) -> str:
    """Bouw de volledige prompt-context uit memory en database."""
    doelen = (memory_dir / "doelen.md").read_text(encoding="utf-8")
    vorige = _last_md_section(memory_dir / "adviezen.md")

    blocks = [
        f"# Doelen en voorkeuren van de atleet\n\n{doelen}",
        f"# Weekschema\n\n{schedule_as_text(memory_dir)}",
        f"# Belastingsoverzicht\n\n{_week_stats(conn)}",
        f"# Recente trainingen (logboek)\n\n{_recent_log_entries(memory_dir)}",
    ]
    if vorige:
        blocks.append(f"# Vorige advies (bouw hierop voort, herhaal het niet)\n\n{vorige}")
    blocks.append(
        f"# Vraag\n\nHet is vandaag {datetime.now():%A %d-%m-%Y}. "
        "Geef een concreet advies voor de komende trainingsweek volgens het weekschema."
    )
    return "\n\n---\n\n".join(blocks)


def generate_advice(router: LLMRouter, conn: sqlite3.Connection, memory_dir: Path) -> str:
    """Genereer een nieuw advies via de API en leg het vast in memory/adviezen.md."""
    context = build_context(conn, memory_dir)
    advies = router.ask("advice", context, system=SYSTEM)

    path = memory_dir / "adviezen.md"
    if not path.exists():
        path.write_text(ADVIEZEN_HEADER, encoding="utf-8")
    entry = (
        f"\n## {datetime.now():%Y-%m-%d %H:%M} — Weekadvies\n\n"
        f"_Gebaseerd op: doelen.md, weekschema.md, de laatste "
        f"{MAX_LOG_ENTRIES} logentries en het belastingsoverzicht._\n\n"
        f"{advies}\n"
    )
    with open(path, "a", encoding="utf-8") as f:
        f.write(entry)
    return advies


def last_advice(memory_dir: Path) -> str | None:
    """Het laatst opgeslagen advies (voor weergave zonder nieuwe API-call)."""
    return _last_md_section(memory_dir / "adviezen.md")


# ----------------------------------------------------------- inzichten --

INSIGHTS_SYSTEM = (
    "Je bent een analytische triatloncoach. Je krijgt de doelen, het logboek "
    "en voortgangsstatistieken van een beginnende triatleet. Zoek naar "
    "lángetermijnpatronen: wordt het tempo bij gelijke hartslag sneller, "
    "verandert de zoneverdeling, hoe ontwikkelt het zwemmen (crawl-aandeel, "
    "SWOLF), is de belastingsopbouw gezond? Schrijf in het Nederlands 3 tot 6 "
    "bondige inzichten als bullets, elk met de data waarop het inzicht stoelt. "
    "Benoem het eerlijk als er voor iets nog te weinig data is. Geen advies "
    "voor volgende trainingen — alleen patronen en observaties."
)


def generate_insights(
    router: LLMRouter,
    conn: sqlite3.Connection,
    memory_dir: Path,
    progress_text: str = "",
) -> str:
    """Laat de cloud-coach trends analyseren en leg de inzichten vast.

    ``progress_text`` is een vooraf berekende samenvatting (TRIMP, EF,
    decoupling, records) zodat het model met harde cijfers werkt in plaats
    van ze zelf te moeten afleiden.
    """
    context = build_context(conn, memory_dir)
    if progress_text:
        context += f"\n\n---\n\n# Voortgangsstatistieken\n\n{progress_text}"
    context += (
        "\n\n---\n\n# Vraag\n\nAnalyseer bovenstaande data op "
        "langetermijnpatronen en geef je inzichten."
    )
    inzichten = router.ask("trends", context, system=INSIGHTS_SYSTEM)

    path = memory_dir / "inzichten.md"
    bestaand = path.read_text(encoding="utf-8") if path.exists() else "# Inzichten\n"
    # De placeholder-regel van de lege versie verwijderen zodra er echte inzichten zijn.
    bestaand = bestaand.replace(
        "_Nog geen inzichten — er is pas ~2 weken aan data (5 sessies, juni 2026)._\n", "")
    entry = f"\n## {datetime.now():%Y-%m-%d} — Trendanalyse\n\n{inzichten}\n"
    path.write_text(bestaand + entry, encoding="utf-8")
    return inzichten


def last_insights(memory_dir: Path) -> str | None:
    """De laatst vastgelegde trendanalyse."""
    return _last_md_section(memory_dir / "inzichten.md")
