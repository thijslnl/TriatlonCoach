"""LTHR-geschiedenis, opgeslagen als markdown-tabel in memory/lthr_geschiedenis.md.

De LTHR (drempelhartslag) verandert naarmate de fitheid groeit; de zones
worden ervan afgeleid (%LTHR). Elke wijziging — via de instellingen-tab of
een nieuwe Garmin-detectie — krijgt hier een regel, zodat de ontwikkeling
zichtbaar blijft en oude trainingen tegen de juiste zones gelezen kunnen worden.
"""

from datetime import date
from pathlib import Path

import pandas as pd

HEADER = """# LTHR-geschiedenis

De drempelhartslag (LTHR) door de tijd. De hartslagzones worden hiervan
afgeleid; bij een wijziging worden de zonetijden in de database herrekend.

| Datum | LTHR | Opmerking |
|---|---|---|
"""


def _path(memory_dir: Path) -> Path:
    return memory_dir / "lthr_geschiedenis.md"


def load_history(memory_dir: Path, initial_lthr: int) -> pd.DataFrame:
    """Lees de geschiedenis; maak het bestand aan met de startwaarde als het ontbreekt."""
    path = _path(memory_dir)
    if not path.exists():
        path.write_text(
            HEADER + f"| {date.today():%Y-%m-%d} | {initial_lthr} | Startwaarde (Garmin auto-detectie) |\n",
            encoding="utf-8",
        )

    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if cells[0] in ("Datum", "") or set(cells[0]) <= {"-", " ", ":"}:
            continue
        rows.append({
            "datum": pd.to_datetime(cells[0]).date(),
            "lthr": int(cells[1]),
            "opmerking": cells[2] if len(cells) > 2 else "",
        })
    return pd.DataFrame(rows).sort_values("datum").reset_index(drop=True)


def append_entry(memory_dir: Path, lthr: int, note: str) -> None:
    """Voeg een nieuwe LTHR-waarde toe aan de geschiedenis."""
    with open(_path(memory_dir), "a", encoding="utf-8") as f:
        f.write(f"| {date.today():%Y-%m-%d} | {lthr} | {note} |\n")
