"""Testscript voor stap 2: parse alle zips in garmin_import/ en print samenvattingen.

Gebruik:  python test_parse.py
"""

from pathlib import Path

from tricoach.config import load_config, resolve_path
from tricoach.fit_parser import ParsedActivity, parse_zip
from tricoach.zones import time_in_zones, zone_bounds


def fmt_duration(seconds: float) -> str:
    """Seconden -> 'H:MM:SS' of 'MM:SS'."""
    s = int(seconds)
    h, rest = divmod(s, 3600)
    m, sec = divmod(rest, 60)
    return f"{h}:{m:02d}:{sec:02d}" if h else f"{m}:{sec:02d}"


def fmt_pace_per_km(speed_ms: float) -> str:
    """Snelheid (m/s) -> looptempo 'M:SS/km'."""
    if not speed_ms:
        return "-"
    sec_per_km = 1000 / speed_ms
    return f"{int(sec_per_km // 60)}:{int(sec_per_km % 60):02d}/km"


def fmt_pace_per_100m(speed_ms: float) -> str:
    """Snelheid (m/s) -> zwemtempo 'M:SS/100m'."""
    if not speed_ms:
        return "-"
    sec = 100 / speed_ms
    return f"{int(sec // 60)}:{int(sec % 60):02d}/100m"


def describe(act: ParsedActivity, bounds: list[int]) -> None:
    """Print een leesbare samenvatting van één activiteit."""
    s = act.summary
    speed = s.get("enhanced_avg_speed") or s.get("avg_speed")

    print(f"\n=== {act.sport}{f' ({act.sub_sport})' if act.sub_sport else ''} — "
          f"{act.start_time:%a %d-%m-%Y %H:%M} ===")
    print(f"  bron: {act.source_file}   sleutel: {act.activity_key}")
    print(f"  duur: {fmt_duration(act.duration_s)}   afstand: {act.distance_m/1000:.2f} km")

    if s.get("avg_heart_rate"):
        print(f"  hartslag: gem {s['avg_heart_rate']} / max {s.get('max_heart_rate', '-')}")

    if act.sport == "running":
        print(f"  tempo: {fmt_pace_per_km(speed)}   cadans: "
              f"{s.get('avg_running_cadence') or s.get('avg_cadence', '-')}")
    elif act.sport == "cycling":
        kmh = speed * 3.6 if speed else 0
        print(f"  snelheid: {kmh:.1f} km/h   hoogtemeters: {s.get('total_ascent', '-')}")
    elif act.sport == "swimming":
        n = s.get("num_active_lengths") or s.get("num_lengths")
        print(f"  tempo: {fmt_pace_per_100m(speed)}   banen: {n} x {s.get('pool_length', '?')}m"
              f"   slagen: {s.get('total_strokes', '-')}")
        if not act.lengths.empty:
            per_stroke = act.lengths.groupby("swim_stroke").size().to_dict()
            print(f"  slagtypes (banen): {per_stroke}")
            swolf = (act.lengths["total_timer_time"] + act.lengths["total_strokes"]).mean()
            print(f"  gem. SWOLF: {swolf:.0f}")

    if not act.records.empty and "heart_rate" in act.records:
        tiz = time_in_zones(act.records, bounds)
        parts = [f"{z}: {fmt_duration(t)}" for z, t in tiz.items() if t > 0]
        print(f"  tijd in zones: {'  '.join(parts)}")
    print(f"  records: {len(act.records)} meetpunten")


def main() -> None:
    config = load_config()
    bounds = zone_bounds(config["athlete"])
    import_dir = resolve_path(config, "import_dir")

    zips = sorted(import_dir.glob("*.zip"))
    print(f"{len(zips)} zip(s) gevonden in {import_dir}")

    for zip_path in zips:
        for act in parse_zip(zip_path):
            describe(act, bounds)


if __name__ == "__main__":
    main()
