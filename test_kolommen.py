"""Verificatie van de nieuwe sessiekolommen (% in zone 2 + aerobe-efficiëntie-trend).

Draait de migratie + backfill op de bestaande database en print de drie
controlesessies uit de opdracht, zodat de berekening te bevestigen is voordat
het in het dashboard komt.
"""

import sys

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")  # arrows/em-dash op een cp1252-console

from tricoach.analysis import aerobic_efficiency_trend
from tricoach.config import load_config, resolve_path
from tricoach.storage import connect, load_activities, recompute_zones
from tricoach.zones import zone_bounds

pd.set_option("display.width", 200)

config = load_config()
bounds = zone_bounds(config["athlete"])
conn = connect(resolve_path(config, "database"))

# Migratie draait al in connect(); vul de nieuwe kolommen voor bestaande rijen.
n = recompute_zones(conn, bounds)
print(f"Backfill: {n} sessies herberekend (zones={bounds}).\n")

acts = load_activities(conn)
trend = aerobic_efficiency_trend(acts)


def regel(datum: str) -> None:
    """Print de berekende waarden voor de sessie op een datum (yyyy-mm-dd)."""
    row = acts[acts["start_time"].dt.strftime("%Y-%m-%d") == datum].iloc[0]
    t = trend[row["activity_key"]]
    pct = row["pct_in_zone2"]
    eff = row["aerobic_efficiency"]
    ref = t["ref"].strftime("%d %b") if t["ref"] is not None else "—"
    delta = f"{t['delta_pct']:+.1f}%" if t["delta_pct"] is not None else "—"
    print(f"{datum}  {row['sport']:<9} "
          f"%Z2={'—' if pd.isna(pct) else f'{pct:.0f}%':>4}  "
          f"eff={'—' if pd.isna(eff) else f'{eff:.5f}'}  "
          f"trend={t['symbol']} {delta} (vs {ref}, "
          f"{'gelijke intensiteit' if t['exact'] else 'dichtstbijzijnde ⚠'})")
    if t["note"]:
        print(f"           note: {t['note']}")


print("Controlesessies uit de opdracht:")
regel("2026-06-15")  # zwemmen: %Z2 indien HR, geen pijl
regel("2026-06-14")  # fietsen 29 km/h, HR 144
regel("2026-06-05")  # fietsen 30 km/h, HR 155 (referentie van 14 jun in de opdracht)
regel("2026-06-13")  # loop 6:36/km, HR 138 -> ~82% Z2

print("\nAlle sessies (oplopend):")
for d in sorted(acts["start_time"].dt.strftime("%Y-%m-%d")):
    regel(d)

conn.close()
