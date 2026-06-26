"""Integratietest voor de upload-feedback + lichaamssamenstelling + profiel.

Draait zonder API-kosten: de Anthropic-coaching wordt gestubd met een nep-router
die een vast antwoord teruggeeft. De rest (FIT-import, context-opbouw,
body-opslag, profiel-changelog) draait tegen echte data in een temp-omgeving.

    .venv/Scripts/python.exe test_uitbreiding.py
"""

import shutil
import sqlite3
import tempfile
from datetime import date
from pathlib import Path

from tricoach import body, profile as profile_mod
from tricoach.config import load_config
from tricoach.feedback import (
    _parse_reply,
    generate_feedback,
    last_proposed_adjustment,
)
from tricoach.importer import import_zip
from tricoach.schedule import add_note_row, load_schedule
from tricoach.storage import connect

CONFIG = load_config()
ZIP = Path("garmin_import/23219243873.zip")  # zwemsessie


class FakeRouter:
    """Nep-router: geeft een vast Haiku-achtig antwoord, doet geen API-call."""

    def __init__(self):
        self.calls = []

    def ask(self, task, prompt, system=None):
        self.calls.append((task, prompt))
        return (
            "FEEDBACK: Nette rustige sessie, je bleef grotendeels in zone 2 — goed gedaan.\n"
            "AANPASSING: Maak de volgende loop 10 minuten korter en strikt zone 2."
        )


def test_parse_reply():
    fb, aanp = _parse_reply("FEEDBACK: Goed bezig.\nAANPASSING: GEEN")
    assert fb == "Goed bezig.", fb
    assert aanp is None, aanp
    fb, aanp = _parse_reply("FEEDBACK: Te hard.\nAANPASSING: Volgende keer korter.")
    assert aanp == "Volgende keer korter.", aanp
    # Formaat dat tegenvalt: alles wordt feedback, geen aanpassing.
    fb, aanp = _parse_reply("Gewoon wat losse tekst zonder labels.")
    assert "losse tekst" in fb and aanp is None
    print("OK  _parse_reply")


def test_feedback_pipeline(tmp: Path):
    mem = tmp / "memory"
    mem.mkdir()
    conn = connect(tmp / "t.db")
    results = import_zip(ZIP, conn, CONFIG, mem)  # geen observation_fn (geen Ollama nodig)
    assert results and results[0].status == "nieuw"
    r = results[0]

    router = FakeRouter()
    fb = generate_feedback(router, conn, mem, CONFIG, r.activity, r.tiz, r.observation)
    assert "zone 2" in fb.feedback
    assert fb.aanpassing and "10 minuten" in fb.aanpassing
    assert router.calls[0][0] == "feedback"  # juiste taak-routing

    # Vastgelegd in feedback.md en terug te lezen voor de adherence-check.
    tekst = (mem / "feedback.md").read_text(encoding="utf-8")
    assert "Voorgestelde aanpassing" in tekst
    assert last_proposed_adjustment(mem) == fb.aanpassing
    conn.close()
    print("OK  feedback-pipeline (FIT-import + context + feedback.md)")


def test_body(tmp: Path):
    mem = tmp / "memory_b"
    mem.mkdir()
    conn = connect(tmp / "b.db")
    body.save_measurement(conn, date(2026, 6, 14), {
        "weight_kg": 108.7, "fat_pct": 21.3, "muscle_mass_kg": 79.8, "visceral_fat": 8.0,
        "bmr_kcal": 2218, "bmi": None,
    })
    body.save_measurement(conn, date(2026, 6, 21), {
        "weight_kg": 107.9, "fat_pct": 20.8, "muscle_mass_kg": 79.9, "visceral_fat": 7.8,
    })
    df = body.load_measurements(conn)
    assert len(df) == 2 and df["weight_kg"].iloc[0] == 108.7
    # Upsert: zelfde datum overschrijft.
    body.save_measurement(conn, date(2026, 6, 21), {"weight_kg": 107.5})
    assert len(body.load_measurements(conn)) == 2

    norm = body.normalized_trends(df, body.TREND_FIELDS)
    assert not norm.empty and abs(norm["index"].iloc[0] - 100.0) < 1e-9

    body.log_measurement(mem, date(2026, 6, 14), {"weight_kg": 108.7, "fat_pct": 21.3})
    log = (mem / "lichaamssamenstelling.md").read_text(encoding="utf-8")
    assert "Gewicht 108.7kg" in log

    extracted = body._parse_extracted('rommel {"weight_kg": "108.7", "fat_pct": 21.3} extra')
    assert extracted["weight_kg"] == 108.7 and extracted["fat_pct"] == 21.3
    conn.close()
    print("OK  body (opslag, upsert, trends, log, screenshot-parser)")


def test_profile(tmp: Path):
    mem = tmp / "memory_p"
    mem.mkdir()
    shutil.copy("memory/doelen.md", mem / "doelen.md")

    import copy
    new = copy.deepcopy(CONFIG)
    new["athlete"]["lthr"] = 174
    new["athlete"]["training_days"] = "ma/wo/vr/zo"

    changes = profile_mod.update_doelen(mem, CONFIG, new, note="test")
    assert any("LTHR" in c for c in changes), changes
    doelen = (mem / "doelen.md").read_text(encoding="utf-8")
    assert profile_mod.START in doelen and profile_mod.END in doelen
    assert "Wijzigingslog" in doelen and "171 → 174" in doelen
    assert "# Doelen & Voorkeuren" in doelen  # handgeschreven inhoud behouden

    # Tweede wijziging: changelog groeit, blok blijft uniek.
    new2 = copy.deepcopy(new)
    new2["athlete"]["max_hr"] = 195
    profile_mod.update_doelen(mem, new, new2, note="test2")
    doelen = (mem / "doelen.md").read_text(encoding="utf-8")
    assert doelen.count(profile_mod.START) == 1
    assert "193 → 195" in doelen and "171 → 174" in doelen
    print("OK  profile (beheerd blok + changelog, idempotent)")


def test_schedule(tmp: Path):
    mem = tmp / "memory_s"
    mem.mkdir()
    add_note_row(mem, "Volgende loop 10 min korter, strikt zone 2.")
    df = load_schedule(mem)
    assert (df["Opmerking"].str.contains("strikt zone 2")).any()
    print("OK  schedule add_note_row")


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        test_parse_reply()
        test_feedback_pipeline(tmp)
        test_body(tmp)
        test_profile(tmp)
        test_schedule(tmp)
    print("\nAlle tests geslaagd.")
