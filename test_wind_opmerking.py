"""Tests voor de opmerking-bij-upload (deel A) en de Open-Meteo-winddata (deel B).

Draait zonder API-kosten: de Anthropic-coaching wordt gestubd met een nep-router.
De winddata-test roept wél de échte (gratis) Open-Meteo-API aan; faalt het
netwerk, dan wordt die test overgeslagen in plaats van rood.

    .venv/Scripts/python.exe test_wind_opmerking.py
"""

import tempfile
from pathlib import Path

import pandas as pd

from tricoach.config import load_config
from tricoach.fit_parser import ParsedActivity, SEMICIRCLE_TO_DEGREES
from tricoach.feedback import generate_feedback
from tricoach.storage import connect, load_activities, save_activity
from tricoach.trainingslog import append_entry
from tricoach import weather
from tricoach.weather import WindData, wind_for_activity
from tricoach.zones import zone_bounds

CONFIG = load_config()


def _bike_activity(with_gps: bool = True) -> ParsedActivity:
    """Een synthetische fietsrit van 14-06-2026 19:15 lokaal (= 17:15 UTC).

    Met GPS: een 'heenweg' naar het oosten en een 'terugweg' naar het westen,
    zodat de kop/tegenwind-duiding iets te bepalen heeft.
    """
    start = pd.Timestamp("2026-06-14T17:15:50+00:00")
    n = 12
    times = [start + pd.Timedelta(minutes=5 * i) for i in range(n)]
    rec = {
        "timestamp": times,
        "heart_rate": [138 + i for i in range(n)],
        "speed_ms": [7.5] * n,
    }
    if with_gps:
        # Eerst oostwaarts (lon stijgt), dan terug westwaarts.
        lons = [5.10, 5.12, 5.14, 5.16, 5.18, 5.20, 5.18, 5.16, 5.14, 5.12, 5.11, 5.10]
        rec["lat"] = [52.09] * n
        rec["lon"] = lons
    return ParsedActivity(
        activity_key=start.isoformat(),
        sport="cycling", sub_sport=None, start_time=start,
        summary={
            "sport": "cycling", "total_timer_time": 3300.0, "total_distance": 22000.0,
            "enhanced_avg_speed": 7.5, "avg_heart_rate": 144, "max_heart_rate": 162,
            "total_ascent": 120.0,
        },
        records=pd.DataFrame(rec),
        lengths=pd.DataFrame(),
        source_file="synthetic_bike.fit",
    )


def _swim_activity() -> ParsedActivity:
    """Een synthetische zwemsessie zonder GPS (zoals banenzwemmen)."""
    start = pd.Timestamp("2026-06-15T05:39:43+00:00")
    return ParsedActivity(
        activity_key=start.isoformat(),
        sport="swimming", sub_sport="lap_swimming", start_time=start,
        summary={"sport": "swimming", "total_timer_time": 1800.0,
                 "total_distance": 1500.0, "avg_speed": 0.83, "pool_length": 25.0},
        records=pd.DataFrame(),
        lengths=pd.DataFrame(),
        source_file="synthetic_swim.fit",
    )


class FakeRouter:
    """Nep-router: vangt de prompt op, doet geen API-call."""

    def __init__(self):
        self.calls = []

    def ask(self, task, prompt, system=None):
        self.calls.append((task, prompt, system))
        return ("FEEDBACK: De tragere terugweg viel samen met de tegenwind, dus dat "
                "is geen vormverlies.\nAANPASSING: GEEN")


def test_semicircle_constant():
    # 2^31 semicircles = 180 graden.
    assert abs(2**31 * SEMICIRCLE_TO_DEGREES - 180.0) < 1e-9
    print("OK  semicircle→graden-constante")


def test_winddata_text():
    w = WindData(speed_kmh=23.4, direction_deg=225.0, gusts_kmh=41.0)
    assert w.direction_label == "ZW", w.direction_label
    assert "uit het ZW" in w.as_text() and "41 km/h" in w.as_text()
    # Zonder gusts blijft de regel netjes.
    assert "uitschieters" not in WindData(10.0, 90.0).as_text()
    print("OK  WindData.as_text + kompaslabel")


def test_wind_skips_swim_and_no_gps(tmp: Path):
    mem = tmp / "m_swim"; mem.mkdir()
    # Zwemmen: niet wind-relevant → None, en geen log geschreven.
    assert wind_for_activity(_swim_activity(), mem) is None
    assert not (mem / "externe_data_log.md").exists()
    # Fietsen maar zonder GPS → ook None, geen log.
    assert wind_for_activity(_bike_activity(with_gps=False), mem) is None
    assert not (mem / "externe_data_log.md").exists()
    print("OK  wind: zwemmen + geen-GPS falen zacht (geen fout, geen log)")


def test_wind_softfail_on_network_error(tmp: Path, monkeypatch_target=weather):
    """Bij een netwerkfout: None terug én een log-regel, geen exception."""
    mem = tmp / "m_neterr"; mem.mkdir()
    orig = weather._fetch_hourly
    weather._fetch_hourly = lambda *a, **k: None  # simuleer offline
    try:
        assert wind_for_activity(_bike_activity(), mem) is None
    finally:
        weather._fetch_hourly = orig
    log = (mem / "externe_data_log.md").read_text(encoding="utf-8")
    assert "overgeslagen" in log
    print("OK  wind: netwerkfout → None + log, upload blijft heel")


def test_wind_live(tmp: Path):
    """Echte Open-Meteo-aanroep voor de rit van 14-06-2026 ~19:15 lokaal."""
    mem = tmp / "m_live"; mem.mkdir()
    try:
        w = wind_for_activity(_bike_activity(), mem)
    except Exception as e:  # netwerk kan in een testomgeving ontbreken
        print(f"SKIP wind-live (geen netwerk?): {e}")
        return
    if w is None:
        print("SKIP wind-live (Open-Meteo gaf geen waarde voor dit uur)")
        return
    assert 0 <= w.speed_kmh < 200 and 0 <= w.direction_deg <= 360
    assert w.headwind_note  # met 12 GPS-punten te bepalen
    log = (mem / "externe_data_log.md").read_text(encoding="utf-8")
    assert "Open-Meteo wind" in log and "archive-api.open-meteo.com" in log
    print(f"OK  wind-live 14-06 19:15 → {w.as_text()}")


def test_storage_note_and_wind(tmp: Path):
    conn = connect(tmp / "s.db")
    bounds = zone_bounds(CONFIG["athlete"])
    wind = WindData(speed_kmh=24.0, direction_deg=210.0, gusts_kmh=38.0)
    assert save_activity(conn, _bike_activity(), bounds,
                         user_note="meewind heen, tegenwind terug", wind=wind)
    df = load_activities(conn)
    row = df.iloc[0]
    assert row["user_note"] == "meewind heen, tegenwind terug"
    assert row["wind_speed"] == 24.0 and row["wind_direction"] == 210.0 and row["wind_gusts"] == 38.0
    # Lege opmerking → NULL, niet "".
    conn2 = connect(tmp / "s2.db")
    save_activity(conn2, _bike_activity(), bounds, user_note="   ", wind=None)
    assert load_activities(conn2).iloc[0]["user_note"] is None
    conn.close(); conn2.close()
    print("OK  storage: user_note + winddata-kolommen (incl. lege opmerking → NULL)")


def test_trainingslog_note_and_wind(tmp: Path):
    mem = tmp / "m_log"; mem.mkdir()
    wind = WindData(speed_kmh=24.0, direction_deg=210.0, gusts_kmh=38.0)
    append_entry(mem, _bike_activity(), {"Z2": 600, "Z3": 1200},
                 observation="stevige rit", user_note="nieuwe schoenen", wind=wind)
    log = (mem / "trainingslog.md").read_text(encoding="utf-8")
    assert "**Wind (Open-Meteo):**" in log and "uit het ZZW" in log
    assert "**Opmerking:** nieuwe schoenen" in log
    print("OK  trainingslog: wind- en opmerkingregel")


def test_feedback_weegt_context_mee(tmp: Path):
    mem = tmp / "m_fb"; mem.mkdir()
    conn = connect(tmp / "f.db")
    bounds = zone_bounds(CONFIG["athlete"])
    wind = WindData(speed_kmh=26.0, direction_deg=225.0, gusts_kmh=40.0)
    act = _bike_activity()
    save_activity(conn, act, bounds, user_note="tegenwind op de terugweg", wind=wind)

    router = FakeRouter()
    fb = generate_feedback(router, conn, mem, CONFIG, act, {"Z2": 1800, "Z3": 1200},
                           observation="rustige rit", user_note="tegenwind op de terugweg",
                           wind=wind)
    # De prompt moet zowel de wind als de opmerking bevatten.
    _, prompt, system = router.calls[0]
    assert "uit het ZW" in prompt and "tegenwind op de terugweg" in prompt
    assert "Weeg externe factoren" in system
    # En het wordt vastgelegd in feedback.md.
    md = (mem / "feedback.md").read_text(encoding="utf-8")
    assert "**Wind (Open-Meteo):**" in md and "**Opmerking atleet:** tegenwind" in md
    assert "tegenwind" in fb.feedback
    conn.close()
    print("OK  feedback: wind + opmerking in prompt, systeemprompt en feedback.md")


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        test_semicircle_constant()
        test_winddata_text()
        test_wind_skips_swim_and_no_gps(tmp)
        test_wind_softfail_on_network_error(tmp)
        test_wind_live(tmp)
        test_storage_note_and_wind(tmp)
        test_trainingslog_note_and_wind(tmp)
        test_feedback_weegt_context_mee(tmp)
    print("\nAlle tests geslaagd.")
