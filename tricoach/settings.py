"""Opslaan van instellingen naar config.yaml (vanuit de instellingen-tab).

Let op: de instellingen-tab herschrijft config.yaml volledig via yaml.dump,
waardoor handmatige commentaarregels in dat bestand verloren gaan. De uitleg
van de instellingen staat daarom hier en in de UI, niet in de yaml zelf.
"""

from pathlib import Path

import yaml

from tricoach.config import CONFIG_PATH

HEADER = (
    "# Triatlon Coach - configuratie\n"
    "# Dit bestand wordt beheerd via de instellingen-tab in het dashboard;\n"
    "# handmatig aanpassen kan ook (uitleg per veld: zie README.md).\n"
)


def save_config(config: dict, path: Path = CONFIG_PATH) -> None:
    """Schrijf de (aangepaste) configuratie terug naar config.yaml."""
    text = HEADER + yaml.dump(
        config, allow_unicode=True, sort_keys=False, default_flow_style=False
    )
    path.write_text(text, encoding="utf-8")
