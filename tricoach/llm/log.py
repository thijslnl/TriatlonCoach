"""Loggen van alle LLM-communicatie naar memory/llm_log.md.

Verplicht onderdeel van het ontwerp: élke aanroep (Ollama én Anthropic)
komt hier langs, zodat altijd terug te lezen is welk model wat gevraagd
en geantwoord heeft, en wat het kostte aan tokens.
"""

import re
from datetime import datetime
from pathlib import Path

import pandas as pd

HEADER = """# LLM-log

Alle communicatie tussen de tool en de taalmodellen, nieuwste onderaan.
Automatisch bijgehouden; niet handmatig bewerken.
"""

# Prompts langer dan dit worden ingekort in het log (de volledige prompt
# is reconstrueerbaar uit de data; het log moet leesbaar blijven).
MAX_PROMPT_CHARS = 1200


def _shorten(text: str, limit: int = MAX_PROMPT_CHARS) -> str:
    if len(text) <= limit:
        return text
    return f"{text[:limit]}\n... _[ingekort; volledige prompt was {len(text)} tekens]_"


def log_call(
    memory_dir: Path,
    provider: str,
    model: str,
    task: str,
    prompt: str,
    response: str,
    prompt_tokens: int | None = None,
    completion_tokens: int | None = None,
    routing_note: str = "",
) -> None:
    """Schrijf één LLM-aanroep weg naar memory/llm_log.md."""
    log_path = memory_dir / "llm_log.md"
    if not log_path.exists():
        log_path.write_text(HEADER, encoding="utf-8")

    tokens = (
        f"prompt {prompt_tokens}, antwoord {completion_tokens}"
        if prompt_tokens is not None else "n.v.t."
    )
    entry = (
        f"\n## {datetime.now():%Y-%m-%d %H:%M:%S} — {provider} ({model}) — taak: {task}\n\n"
        f"- **Routing:** {task} → {provider}{f' ({routing_note})' if routing_note else ''}\n"
        f"- **Tokens:** {tokens}\n"
        f"- **Prompt:**\n\n```\n{_shorten(prompt)}\n```\n\n"
        f"- **Antwoord:**\n\n```\n{_shorten(response)}\n```\n"
    )
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(entry)


# Herkent de kopregel van een log-entry, bijv.:
# "2026-06-12 14:24:54 — ollama (gemma4:12b) — taak: session_summary"
_HEADER_RE = re.compile(r"^(.+?) — (\w+) \((.+?)\) — taak: (.+)$")
_TOKENS_RE = re.compile(r"prompt (\d+), antwoord (\d+)")


def usage_summary(memory_dir: Path) -> pd.DataFrame:
    """Lees memory/llm_log.md terug als DataFrame voor het verbruiksoverzicht.

    Kolommen: tijdstip, provider, model, taak, prompt_tokens, completion_tokens.
    Het markdown-log is de enige bron; zo blijft alles op één plek staan.
    """
    log_path = memory_dir / "llm_log.md"
    if not log_path.exists():
        return pd.DataFrame()

    rows = []
    for block in log_path.read_text(encoding="utf-8").split("\n## ")[1:]:
        header = _HEADER_RE.match(block.splitlines()[0])
        if not header:
            continue
        tokens = _TOKENS_RE.search(block)
        rows.append({
            "tijdstip": header.group(1),
            "provider": header.group(2),
            "model": header.group(3),
            "taak": header.group(4),
            "prompt_tokens": int(tokens.group(1)) if tokens else 0,
            "completion_tokens": int(tokens.group(2)) if tokens else 0,
        })
    return pd.DataFrame(rows)
