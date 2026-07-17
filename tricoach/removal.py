"""Sessies verwijderen, herstellen en definitief wissen — mét memory-boekhouding.

De database-kant (soft delete via ``deleted_at``) zit in :mod:`tricoach.storage`;
deze module orkestreert daaromheen wat er in de memory-bestanden moet gebeuren,
zodat de geschiedenis blijft kloppen:

- **trainingslog.md**: bij de entry van de sessie komt een statusregel
  ("verwijderd op ...", "hersteld op ...", "definitief gewist op ..."), zodat
  het leesbare logboek vertelt wat er met de sessie is gebeurd.
- **feedback.md**: de feedback-sectie van een verwijderde sessie krijgt een
  vervallen-markering. De feedback-context (:func:`feedback_context
  .continuity_block`) en de adherence-check slaan gemarkeerde secties over,
  zodat oude feedback van een verwijderde sessie niet meer als "vorige
  feedback" aan nieuwe rondes wordt meegegeven. Bij herstel gaat de markering
  er weer af.

De app roept alleen :func:`remove_session`, :func:`restore_session` en
:func:`purge_session` aan; de losse helpers zijn testbaar op zichzelf.
"""

from datetime import date
from pathlib import Path

from tricoach.storage import purge_activity, restore_activity, soft_delete_activity

# Herkenbare markering in feedback.md-secties van verwijderde sessies.
# Secties met deze tekst worden door de feedback-context overgeslagen.
FEEDBACK_VERVALLEN = "**Vervallen:**"


def section_is_deleted(section: str) -> bool:
    """Is deze markdown-sectie gemarkeerd als vervallen (sessie verwijderd)?"""
    return FEEDBACK_VERVALLEN in section


# ------------------------------------------------------------- trainingslog --

def note_in_trainingslog(memory_dir: Path, activity_key: str, note: str) -> None:
    """Zet een statusregel bij de trainingslog-entry van deze sessie.

    De entry wordt herkend aan zijn sleutelregel (``_Sleutel `<key>` ...``);
    de statusregel komt daar direct boven, als gewoon opsommingsitem. Bestaat
    er geen entry (oude of handmatig opgeschoonde logs), dan komt de notitie
    als losse regel onderaan het bestand, met de sleutel erbij.
    """
    log_path = memory_dir / "trainingslog.md"
    if not log_path.exists():
        return
    lines = log_path.read_text(encoding="utf-8").splitlines(keepends=True)
    marker = f"_Sleutel `{activity_key}`"
    for i, line in enumerate(lines):
        if marker in line:
            lines.insert(i, f"- **Status:** {note}\n")
            log_path.write_text("".join(lines), encoding="utf-8")
            return
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"\n- **Status** (sessie `{activity_key}`): {note}\n")


# ----------------------------------------------------------------- feedback --

def _mark_feedback(memory_dir: Path, activity_key: str) -> None:
    """Markeer de feedback-sectie van deze sessie als vervallen.

    De regel komt direct boven de sleutelregel van de sectie. Heeft de sessie
    geen feedback-sectie (bijv. feedback destijds overgeslagen), dan gebeurt
    er niets. Dubbel markeren wordt voorkomen.
    """
    path = memory_dir / "feedback.md"
    if not path.exists():
        return
    text = path.read_text(encoding="utf-8")
    marker = f"_Sleutel `{activity_key}`_"
    idx = text.find(marker)
    if idx == -1:
        return
    # Alleen de sectie rond deze sleutel bekijken, niet het hele bestand.
    sectie_start = text.rfind("\n## ", 0, idx)
    if section_is_deleted(text[sectie_start:idx]):
        return
    regel_start = text.rfind("\n", 0, idx) + 1
    vervallen = (
        f"- {FEEDBACK_VERVALLEN} sessie verwijderd op {date.today():%d-%m-%Y} — "
        "niet meer gebruiken als context\n"
    )
    path.write_text(text[:regel_start] + vervallen + text[regel_start:],
                    encoding="utf-8")


def _unmark_feedback(memory_dir: Path, activity_key: str) -> None:
    """Haal de vervallen-markering weer weg (bij herstel van de sessie).

    Zoekt binnen de sectie van deze sessie — vanaf de sleutelregel terug tot
    de sectiekop — naar de markeringsregel en verwijdert die.
    """
    path = memory_dir / "feedback.md"
    if not path.exists():
        return
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    sleutel_idx = next(
        (i for i, line in enumerate(lines) if f"_Sleutel `{activity_key}`_" in line),
        None)
    if sleutel_idx is None:
        return
    for i in range(sleutel_idx, -1, -1):
        if lines[i].startswith("## "):
            return  # sectiekop bereikt zonder markering: niets te doen
        if FEEDBACK_VERVALLEN in lines[i]:
            del lines[i]
            path.write_text("".join(lines), encoding="utf-8")
            return


# -------------------------------------------------------------- orkestratie --

def remove_session(conn, memory_dir: Path, activity_key: str,
                   reason: str | None = None) -> bool:
    """Soft delete één sessie en werk de memory-bestanden bij.

    ``reason`` is de optionele toelichting van de gebruiker (bijv. "verkeerd
    bestand"); die komt in het trainingslog te staan. Geeft False als er
    niets te verwijderen viel (onbekende sleutel of al verwijderd).
    """
    if not soft_delete_activity(conn, activity_key):
        return False
    note = f"verwijderd op {date.today():%d-%m-%Y}"
    if reason and reason.strip():
        note += f": {reason.strip()}"
    note += " — telt niet meer mee in trends, volume en feedback-context"
    note_in_trainingslog(memory_dir, activity_key, note)
    _mark_feedback(memory_dir, activity_key)
    return True


def restore_session(conn, memory_dir: Path, activity_key: str) -> bool:
    """Herstel een soft-verwijderde sessie en draai de memory-boekhouding terug."""
    if not restore_activity(conn, activity_key):
        return False
    note_in_trainingslog(
        memory_dir, activity_key,
        f"hersteld op {date.today():%d-%m-%Y} — telt weer gewoon mee")
    _unmark_feedback(memory_dir, activity_key)
    return True


def purge_session(conn, memory_dir: Path, activity_key: str) -> bool:
    """Wis een sessie definitief (onomkeerbaar) en noteer dat in het trainingslog.

    De vervallen-markering in feedback.md blijft staan: de feedback hoort bij
    een sessie die er niet meer is en mag geen context meer zijn.
    """
    if not purge_activity(conn, activity_key):
        return False
    note_in_trainingslog(
        memory_dir, activity_key,
        f"definitief gewist op {date.today():%d-%m-%Y} — een herimport van "
        "dezelfde zip voegt deze sessie als nieuw toe")
    return True
