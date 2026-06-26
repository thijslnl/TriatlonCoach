"""Routing van LLM-taken: lokaal (Ollama) eerst, cloud (Anthropic) voor het zware werk.

Welke taak naar welk model gaat staat in config.yaml onder ``llm.routing``,
bijvoorbeeld::

    routing:
      session_summary: ollama
      advice: anthropic

Elke aanroep — succesvol of niet — wordt gelogd in memory/llm_log.md.
"""

from pathlib import Path

from tricoach.llm import anthropic_client, ollama_client
from tricoach.llm.log import log_call


class LLMRouter:
    """Eén aanspreekpunt voor alle LLM-vragen, met taakgebaseerde routing."""

    def __init__(self, config: dict, memory_dir: Path):
        self.config = config["llm"]
        self.memory_dir = memory_dir

    def provider_for(self, task: str) -> str:
        """Welke provider hoort bij deze taak? (ollama of anthropic)"""
        provider = self.config["routing"].get(task)
        if provider not in ("ollama", "anthropic"):
            raise ValueError(
                f"Onbekende routing voor taak '{task}': {provider!r}. "
                "Controleer llm.routing in config.yaml."
            )
        return provider

    def ask(self, task: str, prompt: str, system: str | None = None) -> str:
        """Stuur een prompt naar het model dat bij de taak hoort en log alles."""
        provider = self.provider_for(task)

        if provider == "ollama":
            cfg = self.config["ollama"]
            model = cfg["model"]
            reply = ollama_client.chat(
                host=cfg["host"], model=model, prompt=prompt,
                system=system, timeout_s=cfg.get("timeout_s", 120),
            )
        else:
            cfg = self.config["anthropic"]
            model = self._anthropic_model(task)
            reply = anthropic_client.chat(
                model=model, prompt=prompt, system=system,
                max_tokens=cfg.get("max_tokens", 2000),
            )

        log_call(
            self.memory_dir, provider, model, task,
            prompt=(f"[system] {system}\n\n[user] {prompt}" if system else prompt),
            response=reply.text,
            prompt_tokens=reply.prompt_tokens,
            completion_tokens=reply.completion_tokens,
            routing_note="volgens config.yaml",
        )
        return reply.text

    def _anthropic_model(self, task: str) -> str:
        """Het anthropic-model voor deze taak: een per-taak override
        (``anthropic.task_models``) of anders het standaardmodel."""
        cfg = self.config["anthropic"]
        return cfg.get("task_models", {}).get(task, cfg["model"])

    def ask_with_images(
        self, task: str, prompt: str, images: list[str], system: str | None = None
    ) -> str:
        """Zoals ``ask``, maar met afbeeldingen erbij (multimodaal).

        Alleen ondersteund voor Ollama (gemma is multimodaal); gebruikt voor
        het uitlezen van een weegschaal-screenshot. Wordt net als elke aanroep
        gelogd in memory/llm_log.md.
        """
        provider = self.provider_for(task)
        if provider != "ollama":
            raise ValueError(
                f"Afbeeldingen worden alleen door Ollama ondersteund; taak "
                f"'{task}' routet naar {provider}."
            )
        cfg = self.config["ollama"]
        model = cfg["model"]
        reply = ollama_client.chat(
            host=cfg["host"], model=model, prompt=prompt, system=system,
            images=images, timeout_s=cfg.get("timeout_s", 120),
        )
        log_call(
            self.memory_dir, provider, model, task,
            prompt=(f"[system] {system}\n\n[user] {prompt}\n\n[+{len(images)} afbeelding(en)]"
                    if system else f"{prompt}\n\n[+{len(images)} afbeelding(en)]"),
            response=reply.text,
            prompt_tokens=reply.prompt_tokens,
            completion_tokens=reply.completion_tokens,
            routing_note="multimodaal (screenshot)",
        )
        return reply.text
