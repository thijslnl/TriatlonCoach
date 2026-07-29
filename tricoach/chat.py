"""Vraag & antwoord over de trainingsdata, met hybride routing.

Standaard gaan vragen naar het lokale Ollama-model (taak ``qa_simple``,
gratis en onbeperkt). De gebruiker kan een vraag escaleren naar de
Anthropic API (taak ``qa_complex``) als het lokale antwoord tekortschiet
of de vraag echt redeneerwerk vraagt.
"""

import sqlite3
from pathlib import Path

from tricoach.advice import _recent_log_entries, _week_stats
from tricoach.llm.router import LLMRouter
from tricoach.schedule import schedule_as_text
from tricoach.sportzones import zone_overview_text

SYSTEM = (
    "Je bent de assistent van een triatlon-trainingsdashboard. Je beantwoordt "
    "vragen van de atleet over zijn eigen trainingsdata, in het Nederlands. "
    "Baseer je uitsluitend op de meegeleverde data; als iets er niet in staat, "
    "zeg dat dan eerlijk. Wees beknopt en concreet. Context over de atleet: "
    "beginnende triatleet; bekende valkuil is te hard trainen (te weinig zone "
    "2). De drempels en zonegrenzen VERSCHILLEN PER SPORT en staan in de "
    "context onder 'Zone-indeling per sport' — gebruik uitsluitend die "
    "grenzen en verzin er geen. Zwemmen kent bewust geen zones. "
    "Lichaamssamenstelling is "
    "neutrale prestatiedata: geef geen calorie-, dieet- of afvaldoelen en geen "
    "streefgewichten; vraagt de atleet daar toch om, verwijs dan vriendelijk naar "
    "een sportdiëtist of arts."
)


def answer_question(
    router: LLMRouter,
    conn: sqlite3.Connection,
    memory_dir: Path,
    question: str,
    escalate: bool = False,
    config: dict | None = None,
) -> str:
    """Beantwoord één vraag. ``escalate=True`` stuurt hem naar de Anthropic API.

    ``config`` (optioneel) levert de actuele drempels per sport; zonder config
    krijgt het model geen zonegrenzen mee en hoort het er dus ook geen te
    noemen.
    """
    zones = ""
    if config:
        zones = ("# Zone-indeling per sport\n\n"
                 + zone_overview_text(config["athlete"]) + "\n\n")
    context = (
        f"{zones}"
        f"# Belastingsoverzicht\n\n{_week_stats(conn)}\n\n"
        f"# Weekschema\n\n{schedule_as_text(memory_dir)}\n\n"
        f"# Recente trainingen\n\n{_recent_log_entries(memory_dir)}\n\n"
        f"# Vraag van de atleet\n\n{question}"
    )
    task = "qa_complex" if escalate else "qa_simple"
    return router.ask(task, context, system=SYSTEM)
