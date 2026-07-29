"""Migratie naar de sport-afhankelijke drempels en zones (juli 2026).

Vóór deze migratie draaide de hele app op één LTHR voor alle sporten (171,
later 170). Vanaf nu heeft elke sport zijn eigen drempel: de loop-LTHR voor
hardlopen, de fiets-LTHR (of de FTP) voor fietsen, en zwemmen krijgt bewust
geen zones meer.

Bestaande sessies zijn geïmporteerd met de oude, uniforme grenzen. Zonder
herberekening zou de "tijd in zone 2"-trend halverwege van definitie wisselen
en een sprong tonen die niets met de conditie te maken heeft. Dit script
herrekent daarom álle sessies met de nieuwe, sport-afhankelijke drempels:

- **hardlopen** → hartslagzones op de loop-LTHR uit ``config.yaml``;
- **fietsen** → hartslagzones op de fiets-LTHR, plus de tijd in
  vermogenszones zodra er een FTP is;
- **zwemmen** → alle zonetijden op 0 en ``pct_in_zone2`` op NULL.

Draaien vanuit de projectroot::

    python -m tricoach.migrate_sportzones            # herberekenen
    python -m tricoach.migrate_sportzones --dry-run  # alleen tonen wat er zou gebeuren

Het script is idempotent: nog eens draaien levert dezelfde uitkomst op. De
herijking zelf (loop-LTHR 171 → 164) staat beschreven in
``memory/beslissingen.md``, zodat de sprong in de trend verklaarbaar blijft.
"""

import argparse
import sqlite3

import pandas as pd

from tricoach.config import load_config, resolve_path
from tricoach.sportzones import (
    CYCLING,
    RUNNING,
    SWIMMING,
    bike_lthr,
    ftp as athlete_ftp,
    hr_zone_bounds,
    run_lthr,
    zone_overview_text,
)
from tricoach.storage import connect, recompute_power_zones, recompute_zones


def zone_snapshot(conn: sqlite3.Connection) -> pd.DataFrame:
    """Het aandeel zone 2 per sport, om vóór en ná te kunnen vergelijken.

    Geeft per sport het aantal sessies met zonetijd en het gemiddelde
    ``pct_in_zone2``. Zo is meteen zichtbaar hoe groot de verschuiving door de
    herijking is — precies het getal dat in ``memory/beslissingen.md`` hoort.
    """
    df = pd.read_sql_query(
        "SELECT sport, pct_in_zone2, z1_s, z2_s, z3_s, z4_s, z5_s "
        "FROM activities WHERE deleted_at IS NULL", conn)
    if df.empty:
        return pd.DataFrame(columns=["sport", "sessies", "gem_pct_z2"])
    df["heeft_zones"] = df[["z1_s", "z2_s", "z3_s", "z4_s", "z5_s"]].sum(axis=1) > 0
    return (df.groupby("sport")
            .agg(sessies=("heeft_zones", "sum"),
                 gem_pct_z2=("pct_in_zone2", "mean"))
            .reset_index())


def _print_snapshot(titel: str, snap: pd.DataFrame) -> None:
    """Druk een momentopname leesbaar af."""
    print(f"\n{titel}")
    if snap.empty:
        print("  (geen sessies)")
        return
    for _, r in snap.iterrows():
        pct = "—" if pd.isna(r["gem_pct_z2"]) else f"{r['gem_pct_z2']:.1f}%"
        print(f"  {r['sport']:<10} {int(r['sessies']):>3} sessies met zonetijd, "
              f"gemiddeld {pct} in zone 2")


def migrate(dry_run: bool = False) -> None:
    """Voer de migratie uit (of toon alleen wat er zou gebeuren)."""
    config = load_config()
    athlete = config["athlete"]
    conn = connect(resolve_path(config, "database"))

    print("Sport-afhankelijke drempels uit config.yaml:")
    print(f"  loop-LTHR   {run_lthr(athlete)} bpm → zones "
          f"{hr_zone_bounds(athlete, RUNNING)}")
    print(f"  fiets-LTHR  {bike_lthr(athlete)} bpm → zones "
          f"{hr_zone_bounds(athlete, CYCLING)}")
    ftp = athlete_ftp(athlete)
    print(f"  FTP         {f'{ftp:.0f} W' if ftp else 'nog onbekend'}")
    print(f"  zwemmen     geen zones (bounds={hr_zone_bounds(athlete, SWIMMING)})")
    print("\nZone-indeling per sport:")
    print(zone_overview_text(athlete))

    voor = zone_snapshot(conn)
    _print_snapshot("VOOR de herberekening:", voor)

    if dry_run:
        aantal = conn.execute("SELECT COUNT(*) FROM activities").fetchone()[0]
        print(f"\n[dry-run] {aantal} sessies zouden worden herrekend. "
              "Er is niets gewijzigd.")
        return

    n = recompute_zones(conn, athlete)
    n_pw = recompute_power_zones(conn, ftp)
    print(f"\nHerrekend: {n} sessies (hartslagzones), "
          f"{n_pw} ritten (vermogenszones).")

    na = zone_snapshot(conn)
    _print_snapshot("NA de herberekening:", na)

    # Het verschil is de trendbreuk waar memory/beslissingen.md over gaat.
    if not voor.empty and not na.empty:
        samen = voor.merge(na, on="sport", suffixes=("_voor", "_na"))
        print("\nVerschuiving in het gemiddelde zone 2-aandeel "
              "(dit is de gedocumenteerde herijking, geen conditieverandering):")
        for _, r in samen.iterrows():
            if pd.isna(r["gem_pct_z2_voor"]) and pd.isna(r["gem_pct_z2_na"]):
                continue
            v = "—" if pd.isna(r["gem_pct_z2_voor"]) else f"{r['gem_pct_z2_voor']:.1f}%"
            w = "—" if pd.isna(r["gem_pct_z2_na"]) else f"{r['gem_pct_z2_na']:.1f}%"
            print(f"  {r['sport']:<10} {v} → {w}")

    print("\nKlaar. Zie memory/beslissingen.md voor de onderbouwing van de "
          "herijking en de te verwachten sprong in de trends.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dry-run", action="store_true",
                        help="toon alleen wat er zou gebeuren")
    migrate(dry_run=parser.parse_args().dry_run)


if __name__ == "__main__":
    main()
