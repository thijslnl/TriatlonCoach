"""Het aanpasbare weekschema, opgeslagen als markdown-tabel in memory/weekschema.md.

Het schema is kennis (welke dag, welke sport, hoeveel tijd) en hoort dus in
markdown thuis. De UI bewerkt het als tabel; hier zit het lezen en schrijven.
"""

from datetime import date
from pathlib import Path

import pandas as pd

COLUMNS = ["Dag", "Sport", "Duur", "Opmerking"]

# Vertrekpunt uit het intakegesprek (juni 2026).
DEFAULT_ROWS = [
    ["Maandag óf vrijdagochtend", "Zwemmen", "30-45 min", "1x per week; crawlcursus komt later apart"],
    ["Vrijdagavond", "Kort fietsen of lopen", "30-45 min", "optioneel; de sport die zondag niet aan bod komt"],
    ["Zondag", "Lange duurtraining", "1,5-2 uur", "fiets of loop, afwisselend"],
]

HEADER = """# Weekschema

Het geplande trainingsritme. Aanpasbaar in het dashboard (tab Coach);
het trainingsadvies wordt op dit schema gebaseerd.

Laatst bijgewerkt: {date}

"""


def _path(memory_dir: Path) -> Path:
    return memory_dir / "weekschema.md"


def load_schedule(memory_dir: Path) -> pd.DataFrame:
    """Lees het weekschema; maak het default-schema aan als het nog niet bestaat."""
    path = _path(memory_dir)
    if not path.exists():
        df = pd.DataFrame(DEFAULT_ROWS, columns=COLUMNS)
        save_schedule(df, memory_dir)
        return df

    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if cells[0] in ("Dag", "") or set(cells[0]) <= {"-", " ", ":"}:
            continue  # kopregel en scheidingsregel overslaan
        rows.append((cells + [""] * len(COLUMNS))[: len(COLUMNS)])
    return pd.DataFrame(rows, columns=COLUMNS)


def save_schedule(df: pd.DataFrame, memory_dir: Path) -> None:
    """Schrijf het weekschema terug naar memory/weekschema.md."""
    lines = ["| " + " | ".join(COLUMNS) + " |", "|" + "---|" * len(COLUMNS)]
    for _, row in df.fillna("").iterrows():
        if not str(row["Dag"]).strip():
            continue  # lege regels uit de editor overslaan
        lines.append("| " + " | ".join(str(row[c]).strip() for c in COLUMNS) + " |")

    text = HEADER.format(date=date.today().isoformat()) + "\n".join(lines) + "\n"
    _path(memory_dir).write_text(text, encoding="utf-8")


def add_note_row(memory_dir: Path, note: str) -> None:
    """Voeg een gedateerde notitieregel aan het weekschema toe.

    Gebruikt door de knop 'Aanpassing overnemen in planning': de door de coach
    voorgestelde aanpassing landt zo zichtbaar in het schema, dat de gebruiker
    in de Coach-tab verder kan bijschaven en dat het trainingsadvies meeneemt.
    """
    df = load_schedule(memory_dir)
    nieuwe_rij = pd.DataFrame(
        [[f"Aanpassing {date.today():%d-%m}", "", "", note]], columns=COLUMNS
    )
    save_schedule(pd.concat([df, nieuwe_rij], ignore_index=True), memory_dir)


def schedule_as_text(memory_dir: Path) -> str:
    """Het schema als platte tekst, voor in LLM-prompts."""
    df = load_schedule(memory_dir)
    return "\n".join(
        f"- {r['Dag']}: {r['Sport']} ({r['Duur']})" + (f" — {r['Opmerking']}" if r["Opmerking"] else "")
        for _, r in df.iterrows()
    )
