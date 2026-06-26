"""Observaties bij trainingen, gegenereerd door het lokale Ollama-model.

Dit is bewust een 'eenvoudige' LLM-taak (routing: session_summary -> ollama):
één of twee zinnen die de kerncijfers van een sessie duiden, met speciale
aandacht voor zonediscipline (de bekende valkuil: te hard trainen).
"""

from tricoach.fit_parser import ParsedActivity
from tricoach.formatting import fmt_duration, sport_label
from tricoach.llm.router import LLMRouter
from tricoach.trainingslog import kerncijfers, zone_regel

SYSTEM = (
    "Je bent een nuchtere triatloncoach. Je krijgt de kerncijfers van één "
    "trainingssessie. Schrijf in het Nederlands één tot twee korte zinnen met "
    "een observatie over deze sessie. Let vooral op zonediscipline: de atleet "
    "traint structureel te hard (te veel Z3/Z4); veel tijd in Z2 is juist goed. "
    "Geen opsomming van de cijfers zelf, geen advies voor volgende keer, "
    "alleen een observatie. Geen aanhalingstekens."
)


def session_observation(router: LLMRouter, act: ParsedActivity, tiz: dict[str, int]) -> str | None:
    """Vraag Ollama om een korte observatie. Geeft None bij fouten (import gaat door)."""
    prompt = (
        f"Sessie: {sport_label(act.sport)}, {act.start_time:%A %d-%m-%Y}\n"
        f"Kerncijfers: {kerncijfers(act)}\n"
        f"Tijd in hartslagzones: {zone_regel(tiz)}\n"
        f"Totale duur: {fmt_duration(act.duration_s)}"
    )
    try:
        return router.ask("session_summary", prompt, system=SYSTEM)
    except Exception:
        # Ollama onbereikbaar of traag: de import mag hier nooit op stuklopen.
        return None
