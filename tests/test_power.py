"""Testscript voor fietsvermogen & cadans: NP, zones, EF, decoupling en opslag.

Gebruik:  python test_power.py

Toetst de hele power-keten met synthetische ritten (geen FIT-parsen nodig):

1. NP: constant vermogen -> NP = gemiddelde; blokken (interval) -> NP > gemiddelde;
2. cadansstatistiek incl./excl. nul en het maximum;
3. tijd-in-vermogenszones (Coggan) bij een bekende FTP;
4. Pw:Hr-decoupling en de pacing-check (eerste kwartier vs rest);
5. indoor-detectie (Zwift/virtual_activity) en de vermogensbron uit device_info,
   plus: geen Open-Meteo-call voor een indoorrit (Zwift heeft virtuele GPS!);
6. opslag: powervelden in de database, p-zones pas na een FTP (recompute),
   FTP-schatting, en een oude rit zónder power die overal netjes None blijft;
7. verrijkende herimport: een vóór de power-uitbreiding geïmporteerde rit
   krijgt bij een herupload de powerdata alsnog;
8. feedback-context: powerblok met NP/pacing/zones, indoor zonder windregel,
   en de EF-historie per bron.
"""

# De tests staan in tests/; zet de projectroot op sys.path zodat
# `python tests/test_<naam>.py` het tricoach-package kan importeren.
import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))


import numpy as np
import pandas as pd

from tricoach.feedback_context import _bike_power_detail, power_history_block, session_block
from tricoach.fit_parser import ParsedActivity
from tricoach.power import (
    cadence_stats,
    estimate_ftp,
    is_indoor,
    normalized_power,
    pacing_split,
    power_decoupling,
    power_source,
    power_trend,
    time_in_power_zones,
)
from tricoach.storage import (
    connect,
    enrich_power_records,
    load_activities,
    load_records,
    recompute_power_zones,
    save_activity,
)
from tricoach.weather import wind_for_activity

BOUNDS = [137, 152, 162, 171]
FTP = 200


def maak_rit(minuten: int, power, hr=140, cadence=85, start="2026-07-14 17:00:00+00:00",
             sub_sport="road", devices=None, file_manufacturer=None,
             key_suffix="") -> ParsedActivity:
    """Synthetische fietsrit: 1 record per seconde, power/hr/cadans als scalar
    of array van dezelfde lengte."""
    n = minuten * 60
    ts = pd.date_range(start, periods=n, freq="s")
    records = pd.DataFrame({
        "timestamp": ts,
        "power": np.resize(np.asarray(power, dtype=float), n),
        "heart_rate": np.resize(np.asarray(hr, dtype=float), n),
        "cadence": np.resize(np.asarray(cadence, dtype=float), n),
        "speed_ms": 8.0,
        "distance_m": np.arange(n) * 8.0,
    })
    summary = {
        "sport": "cycling", "sub_sport": sub_sport, "start_time": start,
        "total_timer_time": float(n), "total_distance": float(n * 8.0),
        "avg_heart_rate": float(np.mean(records["heart_rate"])),
        "avg_power": float(np.mean(records["power"])),
    }
    return ParsedActivity(
        activity_key=pd.Timestamp(start).isoformat() + key_suffix,
        sport="cycling", sub_sport=sub_sport,
        start_time=pd.Timestamp(start), summary=summary,
        records=records, lengths=pd.DataFrame(), source_file="test.fit",
        devices=devices or [], file_manufacturer=file_manufacturer,
    )


print("1) Normalized power")
vlak = maak_rit(30, power=200)
np_vlak = normalized_power(vlak.records)
assert np_vlak is not None and abs(np_vlak - 200) < 1, np_vlak
# Blokken van 2 min 300 W / 2 min 100 W: gemiddelde 200, NP flink hoger.
blok = np.concatenate([np.full(120, 300.0), np.full(120, 100.0)])
interval = maak_rit(40, power=blok)
np_interval = normalized_power(interval.records)
assert np_interval is not None and np_interval > 220, np_interval
assert normalized_power(pd.DataFrame()) is None
assert normalized_power(maak_rit(2, power=200).records) is None  # te kort
print(f"   constant 200 W -> NP {np_vlak:.1f}; blokken 100/300 -> NP {np_interval:.0f} ✓")

print("2) Cadansstatistiek (nul = freewheelen, echte waarde)")
cad = np.concatenate([np.full(1200, 90.0), np.full(600, 0.0)])  # 20 min trappen, 10 min rollen
rit_cad = maak_rit(30, power=200, cadence=cad)
stats = cadence_stats(rit_cad.records)
assert abs(stats["excl0"] - 90) < 0.5 and abs(stats["incl0"] - 60) < 0.5
assert stats["max"] == 90
assert cadence_stats(pd.DataFrame()) == {"incl0": None, "excl0": None, "max": None}
print(f"   incl. nul {stats['incl0']:.0f}, excl. nul {stats['excl0']:.0f}, max {stats['max']:.0f} ✓")

print("3) Tijd in vermogenszones (FTP 200)")
# 10 min 100 W (P1, <=110), 10 min 130 W (P2, 111-150), 10 min 220 W (P5, 211-240).
zp = np.concatenate([np.full(600, 100.0), np.full(600, 130.0), np.full(600, 220.0)])
rit_z = maak_rit(30, power=zp)
tip = time_in_power_zones(rit_z.records, FTP)
assert abs(tip["P1"] - 600) <= 5 and abs(tip["P2"] - 600) <= 5 and abs(tip["P5"] - 600) <= 5, tip
assert sum(time_in_power_zones(rit_z.records, None).values()) == 0  # zonder FTP geen zones
print(f"   P1 {tip['P1']}s · P2 {tip['P2']}s · P5 {tip['P5']}s, zonder FTP alles 0 ✓")

print("4) Pw:Hr-decoupling en pacing")
hr_drift = np.concatenate([np.full(1200, 140.0), np.full(1200, 150.0)])
rit_drift = maak_rit(40, power=200, hr=hr_drift)
dec = power_decoupling(rit_drift.records)
assert dec is not None and abs(dec - (1 - 140 / 150) * 100) < 0.5, dec  # ~6,7%
assert power_decoupling(maak_rit(10, power=200).records) is None  # te kort
felle_start = np.concatenate([np.full(900, 250.0), np.full(1800, 200.0)])
pacing = pacing_split(maak_rit(45, power=felle_start).records)
assert pacing is not None and abs(pacing[0] - 250) < 1 and abs(pacing[1] - 200) < 1
print(f"   decoupling {dec:+.1f}% (verwacht ~+6,7), pacing {pacing[0]:.0f} -> {pacing[1]:.0f} W ✓")

print("5) Indoor-detectie, bron en wind-overslag")
assert is_indoor("virtual_activity") and is_indoor("indoor_cycling")
assert is_indoor("road", file_manufacturer="zwift")
assert not is_indoor("road") and not is_indoor(None)
rally = [{"manufacturer": "garmin", "antplus_device_type": "bike_power"}]
kickr = [{"manufacturer": "wahoo_fitness", "antplus_device_type": "bike_power"}]
assert power_source(rally, indoor=False) == "pedalen (Garmin Rally)"
assert power_source(kickr, indoor=True) == "trainer (Wahoo Kickr)"
assert power_source([], indoor=True) == "trainer (indoor)"
# Zwift-rit mét (virtuele) GPS: wind moet worden overgeslagen, geen netwerk-call.
zwift = maak_rit(30, power=200, sub_sport="virtual_activity", devices=kickr)
zwift.records["lat"] = -11.64
zwift.records["lon"] = 166.95
assert zwift.is_indoor and wind_for_activity(zwift) is None
print("   Zwift herkend als indoor, bronlabels kloppen, geen windophaling ✓")

print("6) Opslag: powervelden, FTP-recompute, schatting en oude sessie")
conn = connect(":memory:")
buiten = maak_rit(40, power=felle_start[:2400], hr=145, devices=rally,
                  start="2026-07-10 09:00:00+00:00")
save_activity(conn, buiten, BOUNDS)  # nog zonder FTP
binnen = maak_rit(35, power=180, hr=135, sub_sport="virtual_activity",
                  devices=kickr, start="2026-07-12 19:00:00+00:00")
save_activity(conn, binnen, BOUNDS, ftp=FTP)
oud = maak_rit(30, power=200, start="2026-06-01 09:00:00+00:00")
oud.records = oud.records.drop(columns=["power", "cadence"])
oud.summary.pop("avg_power")
save_activity(conn, oud, BOUNDS)  # rit van vóór de vermogensmeter

acts = load_activities(conn)
rij_buiten = acts[acts["activity_key"] == buiten.activity_key].iloc[0]
assert rij_buiten["np_power"] > 200 and rij_buiten["power_source"] == "pedalen (Garmin Rally)"
assert rij_buiten["is_indoor"] == 0 and pd.isna(rij_buiten["p1_s"])  # zonder FTP geen zones
assert rij_buiten["ef_watt"] > 0
rij_binnen = acts[acts["activity_key"] == binnen.activity_key].iloc[0]
assert rij_binnen["is_indoor"] == 1 and rij_binnen["p3_s"] > 0  # FTP bekend bij import (180 W = 90% FTP -> P3)
rij_oud = acts[acts["activity_key"] == oud.activity_key].iloc[0]
assert pd.isna(rij_oud["avg_power"]) and pd.isna(rij_oud["np_power"]) \
    and pd.isna(rij_oud["power_source"])

n_pw = recompute_power_zones(conn, FTP)
assert n_pw == 2  # alleen de twee ritten mét powerdata
rij_buiten = load_activities(conn).set_index("activity_key").loc[buiten.activity_key]
assert rij_buiten["p4_s"] > 0  # 200 W = 100% FTP -> P4 (drempel)
assert recompute_power_zones(conn, None) == 2  # FTP gewist -> zones weer leeg
assert pd.isna(load_activities(conn).set_index("activity_key")
               .loc[buiten.activity_key, "p2_s"])
recompute_power_zones(conn, FTP)

schatting = estimate_ftp(conn, load_activities(conn))
assert schatting is not None and schatting["ftp_watt"] > 150
trend = power_trend(load_activities(conn))
assert len(trend) == 2 and set(trend["indoor"]) == {True, False}
print(f"   kolommen gevuld, zones na recompute, FTP-schatting ~{schatting['ftp_watt']:.0f} W, "
      f"oude rit overal leeg ✓")

print("7) Verrijkende herimport (rit van vóór de power-uitbreiding)")
kaal = maak_rit(30, power=200, hr=142, devices=rally, start="2026-07-08 09:00:00+00:00")
zonder_power = maak_rit(30, power=200, hr=142, start="2026-07-08 09:00:00+00:00")
zonder_power.records = zonder_power.records.drop(columns=["power"])
zonder_power.summary.pop("avg_power")
save_activity(conn, zonder_power, BOUNDS)  # zo stond hij er vóór de uitbreiding
assert pd.isna(load_activities(conn).set_index("activity_key")
               .loc[kaal.activity_key, "np_power"])
assert enrich_power_records(conn, kaal, ftp=FTP)  # herupload van dezelfde zip
rij = load_activities(conn).set_index("activity_key").loc[kaal.activity_key]
assert rij["np_power"] > 0 and rij["p2_s"] >= 0 and rij["power_source"] == "pedalen (Garmin Rally)"
assert load_records(conn, kaal.activity_key)["power"].notna().all()
assert not enrich_power_records(conn, kaal, ftp=FTP)  # tweede keer: niets te doen
print("   powerdata aangevuld zonder dubbel werk of overschrijven ✓")

print("8) Feedback-context")
detail = _bike_power_detail(buiten, ftp=FTP)
assert detail is not None
assert "NP" in detail and "Pacing" in detail and "vermogenszones" in detail.lower()
assert "pedalen (Garmin Rally)" in detail
detail_zonder_ftp = _bike_power_detail(buiten, ftp=None)
assert "FTP onbekend" in detail_zonder_ftp
assert _bike_power_detail(oud, ftp=FTP) is None  # oude rit: geen powerblok

tiz = {"Z1": 600, "Z2": 1500, "Z3": 0, "Z4": 0, "Z5": 0}
blok_indoor = session_block(binnen, tiz, None, None, None, ftp=FTP)
assert "indoorsessie" in blok_indoor and "Open-Meteo" not in blok_indoor
blok_buiten = session_block(buiten, tiz, None, None, None, ftp=FTP)
assert "Open-Meteo" in blok_buiten

latere_rit = maak_rit(30, power=210, hr=140, devices=rally,
                      start="2026-07-15 09:00:00+00:00")
historie = power_history_block(conn, latere_rit)
assert historie is not None and "pedalen (Garmin Rally)" in historie \
    and "trainer (Wahoo Kickr)" in historie and "dezelfde bron" in historie
print("   powerblok, indoor zonder windregel en EF-historie per bron ✓")

conn.close()
print("\nAlle power-tests geslaagd ✓")
