"""Laden van de projectconfiguratie uit config.yaml.

De config wordt als gewone dict doorgegeven; voor de paar plekken waar
we structuur willen (zones, paden) zijn hier helpers.
"""

from pathlib import Path

import yaml

# De projectroot is de map waarin config.yaml staat (één niveau boven dit bestand).
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"


def load_config(path: Path = CONFIG_PATH) -> dict:
    """Lees config.yaml en geef de inhoud als dict terug."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(config: dict, key: str) -> Path:
    """Maak van een relatief pad uit de config een absoluut pad binnen het project.

    Voorbeeld: resolve_path(cfg, "database") -> C:/.../data/training.db
    """
    return PROJECT_ROOT / config["paths"][key]
