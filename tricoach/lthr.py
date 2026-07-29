"""Drempelgeschiedenis per sport, als markdown-tabel in memory/lthr_geschiedenis.md.

De drempels veranderen naarmate de fitheid groeit, en ze verschillen per sport:
de loop-LTHR en de fiets-LTHR in slagen per minuut, de FTP in watt. De zones
worden ervan afgeleid, dus elke wijziging — via de instellingen-tab, een nieuwe
Garmin-detectie of een FTP-test — krijgt hier een regel. Zo blijft de
ontwikkeling zichtbaar en zijn oude trainingen tegen de juiste zones te lezen.

De tabel heeft sinds de sport-afhankelijke drempels een kolom **Drempel**
(bijv. "Hardlopen LTHR" of "Fietsen FTP"). Regels van vóór die uitbreiding
hebben die kolom niet; die worden gelezen als loop-LTHR, want dat was destijds
de enige drempel die de app kende.
"""

from datetime import date
from pathlib import Path

import pandas as pd

# De drempelsoorten die de app kent, met hun eenheid. De sleutel is wat er in
# de kolom "Drempel" komt te staan.
RUN_LTHR = "Hardlopen LTHR"
BIKE_LTHR = "Fietsen LTHR"
BIKE_FTP = "Fietsen FTP"

UNITS = {RUN_LTHR: "bpm", BIKE_LTHR: "bpm", BIKE_FTP: "W"}

HEADER = """# Drempelgeschiedenis per sport

De drempelwaarden door de tijd: de loop-LTHR en fiets-LTHR (drempelhartslag,
bpm) en de FTP (fietsvermogen, watt). De zones worden hiervan afgeleid; bij een
wijziging worden de zonetijden in de database herrekend. Regels zonder
drempelsoort zijn van vóór de sport-afhankelijke drempels en gelden als
loop-LTHR.

| Datum | Drempel | Waarde | Opmerking |
|---|---|---|---|
"""


def _path(memory_dir: Path) -> Path:
    return memory_dir / "lthr_geschiedenis.md"


def _parse_rows(text: str) -> list[dict]:
    """Lees de tabelregels; zowel de nieuwe als de oude (3-koloms) vorm.

    De oude vorm was ``| Datum | LTHR | Opmerking |`` zonder drempelsoort —
    destijds bestond er maar één LTHR, die voor alle sporten gold. Die regels
    krijgen hier :data:`RUN_LTHR` toegewezen, want de loop-LTHR is de directe
    voortzetting ervan.
    """
    rows = []
    for line in text.splitlines():
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if cells[0] in ("Datum", "") or set(cells[0]) <= {"-", " ", ":"}:
            continue
        if len(cells) >= 4:
            drempel, waarde, opmerking = cells[1], cells[2], cells[3]
        else:  # oude vorm: | Datum | LTHR | Opmerking |
            drempel, waarde = RUN_LTHR, cells[1]
            opmerking = cells[2] if len(cells) > 2 else ""
        rows.append({
            "datum": pd.to_datetime(cells[0]).date(),
            "drempel": drempel,
            "waarde": int(float(waarde)),
            "opmerking": opmerking,
        })
    return rows


def _upgrade_format(path: Path) -> None:
    """Herschrijf een bestand in de oude 3-koloms vorm naar de nieuwe tabel.

    Idempotent: een bestand dat de kolom "Drempel" al heeft blijft ongemoeid.
    De bestaande regels blijven behouden en krijgen :data:`RUN_LTHR` als
    drempelsoort — precies wat ze destijds betekenden.
    """
    if not path.exists():
        return
    tekst = path.read_text(encoding="utf-8")
    if "| Datum | Drempel |" in tekst:
        return
    regels = [
        f"| {r['datum']:%Y-%m-%d} | {r['drempel']} | {r['waarde']} | {r['opmerking']} |"
        for r in _parse_rows(tekst)
    ]
    path.write_text(HEADER + "\n".join(regels) + ("\n" if regels else ""),
                    encoding="utf-8")


def load_history(memory_dir: Path, initial_lthr: int,
                 kind: str | None = None) -> pd.DataFrame:
    """Lees de geschiedenis; maak het bestand aan met de startwaarde als het ontbreekt.

    ``kind`` filtert op één drempelsoort (bijv. :data:`RUN_LTHR`); zonder
    filter komt alles terug. Kolommen: datum, drempel, waarde, opmerking —
    plus ``lthr`` als alias van ``waarde``, zodat bestaande grafiekcode blijft
    werken.
    """
    path = _path(memory_dir)
    if not path.exists():
        path.write_text(
            HEADER + f"| {date.today():%Y-%m-%d} | {RUN_LTHR} | {initial_lthr} | "
                     "Startwaarde (Garmin auto-detectie) |\n",
            encoding="utf-8",
        )
    _upgrade_format(path)

    rows = _parse_rows(path.read_text(encoding="utf-8"))
    df = pd.DataFrame(rows, columns=["datum", "drempel", "waarde", "opmerking"])
    if kind:
        df = df[df["drempel"] == kind]
    df = df.sort_values("datum").reset_index(drop=True)
    df["lthr"] = df["waarde"]
    return df


def append_entry(memory_dir: Path, value: int, note: str,
                 kind: str = RUN_LTHR) -> None:
    """Voeg een nieuwe drempelwaarde toe aan de geschiedenis.

    ``kind`` is de drempelsoort (:data:`RUN_LTHR`, :data:`BIKE_LTHR` of
    :data:`BIKE_FTP`); ``note`` legt de aanleiding vast (bijv. "ramptest op de
    Kickr" of "Aangepast via instellingen-tab").
    """
    path = _path(memory_dir)
    if not path.exists():
        path.write_text(HEADER, encoding="utf-8")
    _upgrade_format(path)
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"| {date.today():%Y-%m-%d} | {kind} | {value} | {note} |\n")
