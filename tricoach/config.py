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
    """Lees config.yaml en geef de inhoud als dict terug.

    De athlete-sectie wordt meteen genormaliseerd naar de sport-afhankelijke
    drempelstructuur (zie :func:`normalize_athlete`), zodat de rest van de app
    nooit met twee vormen te maken heeft.
    """
    with open(path, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if isinstance(config, dict) and isinstance(config.get("athlete"), dict):
        normalize_athlete(config["athlete"])
    return config


def normalize_athlete(athlete: dict) -> dict:
    """Zet een oude, platte athlete-config om naar drempels per sport.

    Vóór de sport-afhankelijke drempels stonden er één ``lthr`` (voor álle
    sporten) en één ``ftp`` los in de athlete-sectie. Die worden hier verplaatst
    naar ``thresholds.running.lthr`` respectievelijk ``thresholds.cycling.ftp``;
    de fiets-LTHR wordt geschat als hij ontbreekt (zie
    :data:`tricoach.sportzones.BIKE_LTHR_OFFSET`). Al genormaliseerde configs
    blijven onaangeroerd — de functie is idempotent. Past ``athlete`` ter
    plekke aan en geeft het terug.
    """
    from tricoach.sportzones import bike_lthr, ftp, run_lthr, set_thresholds

    thresholds = athlete.get("thresholds") or {}
    heeft_run = bool((thresholds.get("running") or {}).get("lthr"))
    heeft_bike = bool((thresholds.get("cycling") or {}).get("lthr"))
    if heeft_run and heeft_bike:
        athlete.pop("lthr", None)
        athlete.pop("ftp", None)
        return athlete

    # De accessors lezen zowel de nieuwe als de oude vorm; daarmee halen we de
    # actuele waarden op vóór we ze in de nieuwe structuur wegschrijven.
    return set_thresholds(athlete, run_lthr(athlete), bike_lthr(athlete),
                          ftp(athlete))


def resolve_path(config: dict, key: str) -> Path:
    """Maak van een relatief pad uit de config een absoluut pad binnen het project.

    Voorbeeld: resolve_path(cfg, "database") -> C:/.../data/training.db
    """
    return PROJECT_ROOT / config["paths"][key]
