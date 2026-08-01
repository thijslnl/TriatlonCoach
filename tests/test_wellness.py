"""Testscript voor de wellness-opslag en de herstelcontext (tricoach.wellness).

Gebruik:  python tests/test_wellness.py

Draait volledig op een tijdelijke database; raakt de echte data niet aan.
Controleert:

1. upsert: één rij per dag, en een her-sync met ontbrekende velden (None)
   wist eerder opgehaalde waarden niet (COALESCE-gedrag);
2. day_is_complete: pas True als rustpols én HRV binnen zijn;
3. with_rolling: het 7-daags voortschrijdend gemiddelde op kalenderdagen;
4. recovery_snapshot: de feedback-context met 7-daagse gemiddelden, en None
   wanneer de data te oud (>2 dagen) of afwezig is;
5. sync_wellness (met een fake client): deels falende endpoints zijn geen
   probleem, en al complete oude dagen worden overgeslagen.
"""

# De tests staan in tests/; zet de projectroot op sys.path zodat
# `python tests/test_<naam>.py` het tricoach-package kan importeren.
import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))


import sqlite3
import tempfile
from datetime import date, timedelta
from pathlib import Path

from tricoach import garmin_sync, wellness


class FakeWellnessClient:
    """Doet zich voor als de Garmin-client; sommige endpoints falen bewust."""

    def __init__(self):
        self.stats_calls = 0

    def get_stats(self, iso):
        self.stats_calls += 1
        return {"restingHeartRate": 52, "averageStressLevel": 28,
                "bodyBatteryHighestValue": 88, "bodyBatteryLowestValue": 21}

    def get_hrv_data(self, iso):
        return {"hrvSummary": {"lastNightAvg": 48, "weeklyAvg": 51,
                               "status": "BALANCED",
                               "baseline": {"balancedLow": 42,
                                            "balancedUpper": 60}}}

    def get_sleep_data(self, iso):
        raise RuntimeError("slaap-endpoint doet het even niet")

    def get_training_readiness(self, iso):
        return [{"score": 61, "level": "MODERATE"}]

    def get_max_metrics(self, iso):
        return [{"generic": {"vo2MaxPreciseValue": 41.8}, "cycling": None}]


def main() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="tricoach_wellness_"))
    conn = sqlite3.connect(tmp / "test.db")
    vandaag = date.today()

    # 1. upsert + niet-klobberen ---------------------------------------------
    wellness.upsert_day(conn, vandaag, {"resting_hr": 50, "hrv_last_night": 45})
    wellness.upsert_day(conn, vandaag, {"resting_hr": None, "sleep_score": 80})
    rij = conn.execute(
        "SELECT resting_hr, hrv_last_night, sleep_score FROM wellness").fetchone()
    assert rij == (50, 45, 80), f"COALESCE-upsert faalt: {rij}"
    assert conn.execute("SELECT COUNT(*) FROM wellness").fetchone()[0] == 1
    print("1. upsert bewaart bestaande waarden bij een her-sync met gaten OK")

    # 2. day_is_complete ------------------------------------------------------
    assert wellness.day_is_complete(conn, vandaag)
    wellness.upsert_day(conn, vandaag - timedelta(days=1), {"resting_hr": 51})
    assert not wellness.day_is_complete(conn, vandaag - timedelta(days=1))
    print("2. day_is_complete vraagt rustpols én HRV OK")

    # 3. voortschrijdend gemiddelde ------------------------------------------
    conn.execute("DELETE FROM wellness")
    for i in range(10):
        wellness.upsert_day(conn, vandaag - timedelta(days=9 - i),
                            {"resting_hr": 50 + i, "hrv_last_night": 40 + i})
    df = wellness.with_rolling(wellness.load_wellness(conn))
    laatste_ma = df["resting_hr_ma"].iloc[-1]
    # Kalendervenster '7D' = de laatste 7 dagen: rustpols 53..59 -> gem. 56.
    assert abs(laatste_ma - 56.0) < 1e-9, f"7d-gemiddelde klopt niet: {laatste_ma}"
    print("3. with_rolling rekent het 7-daags gemiddelde op kalenderdagen OK")

    # 4. recovery_snapshot ----------------------------------------------------
    snap = wellness.recovery_snapshot(conn, vandaag)
    assert snap is not None and snap["rustpols"] == 59
    # 7 dagen vóór de laatste dag: 52..58 -> gemiddelde 55.
    assert abs(snap["rustpols_7d"] - 55.0) < 1e-9, snap
    assert wellness.recovery_snapshot(conn, vandaag + timedelta(days=5)) is None, \
        "een snapshot ouder dan 2 dagen hoort None te zijn"
    print("4. recovery_snapshot geeft dag + 7d-gemiddelde, en None bij oude data OK")

    # 5. sync_wellness met deels falende endpoints ---------------------------
    conn.execute("DELETE FROM wellness")
    conn.commit()
    client = FakeWellnessClient()
    res = garmin_sync.sync_wellness(client, conn, days=5)
    assert res.fetched == 5 and res.rhr_filled == 5 and res.hrv_filled == 5, vars(res)
    rij = conn.execute(
        "SELECT resting_hr, hrv_last_night, sleep_s, training_readiness, vo2max_run "
        "FROM wellness ORDER BY day DESC LIMIT 1").fetchone()
    assert rij == (52, 48, None, 61, 41.8), f"dagwaarden kloppen niet: {rij}"
    # Her-sync: oude complete dagen worden overgeslagen, alleen de laatste
    # REFRESH_DAYS worden opnieuw opgehaald.
    client.stats_calls = 0
    garmin_sync.sync_wellness(client, conn, days=5)
    assert client.stats_calls == garmin_sync.REFRESH_DAYS, \
        f"verwachtte {garmin_sync.REFRESH_DAYS} verversingen, kreeg {client.stats_calls}"
    print("5. sync_wellness: falende endpoints zacht, complete dagen overgeslagen OK")

    print("\nAlle wellness-tests geslaagd.")


if __name__ == "__main__":
    main()
