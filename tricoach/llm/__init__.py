"""LLM-laag: routing tussen het lokale Ollama-model en de Anthropic API.

Gebruik: maak één ``LLMRouter`` (zie ``router.py``) en stel vragen per taak::

    router = LLMRouter(config, memory_dir)
    antwoord = router.ask("session_summary", prompt)

Welke taak naar welk model gaat staat in config.yaml onder ``llm.routing``.
Elke aanroep wordt gelogd in memory/llm_log.md.
"""

from tricoach.llm.router import LLMRouter

__all__ = ["LLMRouter"]
