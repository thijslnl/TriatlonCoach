"""Testscript voor de loopdynamiek: eenheden, richtbereiken, verrijking en label.

Gebruik:  python test_rundynamics.py

Toetst met de cijfers van de 10 km van 12 juli (Garmin Connect: 137 spm,
grondcontacttijd ~356 ms, verticale ratio 7,9%):

1. cadans-omrekening: FIT (rpm één been + fractie) -> spm zoals Garmin Connect;
2. dynamics_from_summary: alle velden in de juiste weergave-eenheid;
3. verticale ratio 7,9% valt in het "prima"-bereik;
4. verrijking: een herimport (duplicaat) vult ontbrekende summary-velden aan
   zonder bestaande waarden te overschrijven;
5. sessielabel "techniek/cadans": opslag, achteraf zetten, en de
   feedback-context legt uit dat een hogere hartslag verwacht en oké is.
"""

# De tests staan in tests/; zet de projectroot op sys.path zodat
# `python tests/test_<naam>.py` het tricoach-package kan importeren.
import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))


import json
import sqlite3

import pandas as pd

from tricoach.fit_parser import ParsedActivity
from tricoach.feedback_context import session_block
from tricoach.rundynamics import (
    cadence_spm,
    dynamics_block,
    dynamics_from_summary,
    dynamics_trend,
    vertical_ratio_is_good,
)
from tricoach.storage import (
    connect,
    enrich_activity_summary,
    load_activities,
    save_activity,
    set_training_label,
)

BOUNDS = [137, 152, 160, 172]

# Ruwe FIT-samenvatting zoals de Forerunner 265 hem levert (na fitdecode-
# schaling): cadans in rpm van één been, staplengte in mm, GCT in ms.
SUMMARY_12_JULI = {
    "sport": "running",
    "start_time": "2026-07-12 17:39:31+00:00",
    "total_timer_time": 3953.528,
    "total_distance": 10013.61,
    "avg_heart_rate": 144,
    "avg_cadence": 68,
    "avg_running_cadence": 68,
    "avg_fractional_cadence": 0.5,
    "avg_step_length": 1103.0,
    "avg_stance_time": 356.0,
    "avg_vertical_oscillation": 87.1,
    "avg_vertical_ratio": 7.9,
    "normalized_power": 370,
}


def act_met(summary: dict, key: str = "2026-07-12T17:39:31+00:00") -> ParsedActivity:
    """Bouw een minimale ParsedActivity rond een summary-dict."""
    return ParsedActivity(
        activity_key=key, sport="running", sub_sport=None,
        start_time=pd.Timestamp(summary["start_time"]),
        summary=summary, records=pd.DataFrame(), lengths=pd.DataFrame(),
        source_file="test.fit",
    )


def test_cadans_eenheid():
    """FIT-cadans (één been) -> totale spm, exact het Garmin Connect-getal."""
    assert cadence_spm(68, 0.5) == 137.0, "68.5 rpm één been moet 137 spm zijn"
    assert cadence_spm(68) == 136.0, "zonder fractie: 68 rpm -> 136 spm"
    assert cadence_spm(None) is None
    assert cadence_spm(0) is None, "cadans 0 is onbruikbaar, geen 0 spm tonen"
    print("✅ cadans: 68(.5) rpm één been -> 136/137 spm (geen dubbeltelling)")


def test_dynamics_eenheden():
    dyn = dynamics_from_summary(SUMMARY_12_JULI)
    assert dyn["cadans_spm"] == 137.0, dyn
    assert abs(dyn["staplengte_m"] - 1.103) < 1e-9, "staplengte mm -> m"
    assert dyn["gct_ms"] == 356.0
    assert dyn["vert_ratio_pct"] == 7.9
    assert dyn["vermogen_w"] == 370.0
    # Ontbrekende velden (oude import) geven None, geen crash.
    kaal = dynamics_from_summary({"avg_cadence": 68})
    assert kaal["cadans_spm"] == 136.0 and kaal["gct_ms"] is None
    print("✅ dynamics_from_summary: 137 spm, 1.10 m, 356 ms, 7.9%, 370 W")


def test_verticale_ratio_prima():
    assert vertical_ratio_is_good(7.9) is True, "7,9% moet als prima gelden"
    assert vertical_ratio_is_good(8.6) is False
    assert vertical_ratio_is_good(None) is None
    blok = dynamics_block(SUMMARY_12_JULI, pd.DataFrame())
    assert "prima" in blok, "context moet expliciet zeggen dat 7,9% prima is"
    assert "geen norm" in blok and "LANGETERMIJN" in blok
    print("✅ verticale ratio 7,9% = prima; contextblok is langetermijn-geframed")


def test_verrijking_bij_herimport():
    conn = connect(":memory:")

    # Eerste import zoals vóór de dynamics-uitbreiding: kale summary.
    oud = {k: SUMMARY_12_JULI[k] for k in
           ("sport", "start_time", "total_timer_time", "total_distance",
            "avg_heart_rate", "avg_cadence")}
    oud["total_distance"] = 9999.0  # bewust anders: mag NIET overschreven worden
    assert save_activity(conn, act_met(oud), BOUNDS) is True

    # Herimport van dezelfde activiteit, nu met alle dynamics-velden.
    assert enrich_activity_summary(conn, act_met(SUMMARY_12_JULI)) is True
    rij = conn.execute("SELECT summary_json FROM activities").fetchone()
    summary = json.loads(rij[0])
    assert summary["avg_vertical_ratio"] == 7.9, "nieuw veld moet zijn aangevuld"
    assert summary["avg_stance_time"] == 356.0
    assert summary["total_distance"] == 9999.0, "bestaande waarde mag niet wijzigen"
    # Nogmaals verrijken: niets meer aan te vullen.
    assert enrich_activity_summary(conn, act_met(SUMMARY_12_JULI)) is False

    # De trend leest de aangevulde velden terug.
    trend = dynamics_trend(load_activities(conn))
    assert len(trend) == 1
    assert trend.iloc[0]["cadans_spm"] == 137.0
    assert trend.iloc[0]["vert_ratio_pct"] == 7.9
    conn.close()
    print("✅ verrijking: duplicaat-import vult dynamics aan, overschrijft niets")


def test_sessielabel():
    conn = connect(":memory:")
    save_activity(conn, act_met(SUMMARY_12_JULI), BOUNDS,
                  training_label="techniek/cadans")
    acts = load_activities(conn)
    assert acts.iloc[0]["training_label"] == "techniek/cadans"

    # Achteraf wisselen en wissen.
    assert set_training_label(conn, acts.iloc[0]["activity_key"], None) is True
    assert load_activities(conn).iloc[0]["training_label"] is None
    conn.close()

    # De feedback-context legt bij het label uit dat hogere HR verwacht is,
    # zodat de coach de sessie niet als "te hard getraind" beoordeelt.
    blok = session_block(act_met(SUMMARY_12_JULI), {"Z2": 1800, "Z3": 600},
                         None, None, None, training_label="techniek/cadans")
    assert "techniek/cadans" in blok
    assert "VERWACHT" in blok and "te hard trainen" in blok
    print("✅ sessielabel: opslag, achteraf aanpassen, en HR-nuance in de context")


if __name__ == "__main__":
    test_cadans_eenheid()
    test_dynamics_eenheden()
    test_verticale_ratio_prima()
    test_verrijking_bij_herimport()
    test_sessielabel()
    print("\nAlle loopdynamiek-tests geslaagd.")
