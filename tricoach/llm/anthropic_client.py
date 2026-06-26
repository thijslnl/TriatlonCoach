"""Client voor de Anthropic API (het 'echte coachwerk').

De API-key komt uit de environment variable ANTHROPIC_API_KEY — nooit
hardcoden. De SDK leest die zelf; wij geven alleen een duidelijke foutmelding
als hij ontbreekt.
"""

import os

import anthropic

from tricoach.llm.ollama_client import LLMReply


class AnthropicError(RuntimeError):
    """Anthropic API niet beschikbaar of aanroep mislukt."""


def chat(
    model: str,
    prompt: str,
    system: str | None = None,
    max_tokens: int = 2000,
) -> LLMReply:
    """Stuur één prompt naar de Anthropic API en geef het antwoord terug."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise AnthropicError(
            "ANTHROPIC_API_KEY is niet gezet. Zet de environment variable en "
            "herstart de app (PowerShell: $env:ANTHROPIC_API_KEY = '...')."
        )

    client = anthropic.Anthropic()
    kwargs = {"system": system} if system else {}
    try:
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )
    except anthropic.APIError as e:
        raise AnthropicError(f"Anthropic API-aanroep mislukt: {e}") from e

    text = "".join(b.text for b in response.content if b.type == "text")
    return LLMReply(
        text=text.strip(),
        prompt_tokens=response.usage.input_tokens,
        completion_tokens=response.usage.output_tokens,
    )
