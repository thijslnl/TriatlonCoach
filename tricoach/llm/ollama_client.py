"""Client voor het lokale Ollama-model (op de Unraid-server).

Gebruikt de standaard Ollama REST-API (/api/chat). Geen streaming: de
antwoorden zijn kort en we willen tokenaantallen uit het slotbericht.
"""

from dataclasses import dataclass

import requests


@dataclass
class LLMReply:
    """Antwoord van een model, met tokenaantallen voor het log."""

    text: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


class OllamaError(RuntimeError):
    """Ollama is onbereikbaar of gaf een foutstatus terug."""


def chat(
    host: str,
    model: str,
    prompt: str,
    system: str | None = None,
    images: list[str] | None = None,
    timeout_s: int = 120,
) -> LLMReply:
    """Stuur één prompt naar Ollama en geef het antwoord terug.

    ``images`` is een optionele lijst base64-strings; gemma is multimodaal, dus
    hiermee kan een screenshot mee (zie de screenshot-extractie bij
    lichaamssamenstelling). Bij tekst-only laat je het weg.
    """
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    user_msg: dict = {"role": "user", "content": prompt}
    if images:
        user_msg["images"] = images
    messages.append(user_msg)

    try:
        resp = requests.post(
            f"{host}/api/chat",
            json={"model": model, "messages": messages, "stream": False},
            timeout=timeout_s,
        )
        resp.raise_for_status()
    except requests.RequestException as e:
        raise OllamaError(f"Ollama-aanroep naar {host} mislukt: {e}") from e

    data = resp.json()
    return LLMReply(
        text=data["message"]["content"].strip(),
        prompt_tokens=data.get("prompt_eval_count"),
        completion_tokens=data.get("eval_count"),
    )
