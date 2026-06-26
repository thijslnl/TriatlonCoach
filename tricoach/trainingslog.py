"""Schrijven van trainingslog-entries naar memory/trainingslog.md.

Elke geïmporteerde sessie krijgt één markdown-entry met datum, sport,
kerncijfers en (zodra de LLM-integratie er is) een observatie. Het bestand
is bedoeld als leesbaar, doorzoekbaar geheugen — de ruwe data staat in SQLite.
"""

from pathlib import Path

from tricoach.fit_parser import ParsedActivity
from tricoach.formatting import (
    fmt_duration,
    fmt_pace_per_100m,
    fmt_pace_per_km,
    fmt_speed_kmh,
    sport_label,
)

HEADER = """# Trainingslog

Automatisch bijgehouden door de tool: één entry per geïmporteerde sessie,
nieuwste onderaan. Kerncijfers komen uit het FIT-bestand; observaties van
het lokale LLM (of handmatig).
"""


def kerncijfers(act: ParsedActivity) -> str:
    """Bouw de kerncijfer-regel voor een sessie, afgestemd op de sport."""
    s = act.summary
    speed = s.get("enhanced_avg_speed") or s.get("avg_speed")
    parts = [f"duur {fmt_duration(act.duration_s)}"]

    if act.sport == "running":
        parts.append(f"{act.distance_m / 1000:.2f} km op {fmt_pace_per_km(speed)}")
    elif act.sport == "cycling":
        parts.append(f"{act.distance_m / 1000:.1f} km, {fmt_speed_kmh(speed)}")
        if s.get("total_ascent"):
            parts.append(f"{s['total_ascent']:.0f} hm")
    elif act.sport == "swimming":
        parts.append(f"{act.distance_m:.0f} m op {fmt_pace_per_100m(speed)}")
        n = s.get("num_active_lengths") or s.get("num_lengths")
        if n and s.get("pool_length"):
            parts.append(f"{n} banen à {s['pool_length']:.0f}m")
        if not act.lengths.empty:
            swolf = (act.lengths["total_timer_time"] + act.lengths["total_strokes"]).mean()
            parts.append(f"SWOLF {swolf:.0f}")

    if s.get("avg_heart_rate"):
        parts.append(f"HR gem {s['avg_heart_rate']} / max {s.get('max_heart_rate', '-')}")
    return ", ".join(parts)


def zone_regel(tiz: dict[str, int]) -> str:
    """Tijd-in-zones als compacte regel, lege zones weggelaten."""
    parts = [f"{z} {fmt_duration(t)}" for z, t in tiz.items() if t > 0]
    return " · ".join(parts) if parts else "-"


def append_entry(
    memory_dir: Path,
    act: ParsedActivity,
    tiz: dict[str, int],
    observation: str | None = None,
    user_note: str | None = None,
    wind: "object | None" = None,
) -> None:
    """Voeg één sessie-entry toe aan memory/trainingslog.md.

    ``user_note`` is de vrije opmerking bij de upload en ``wind`` de
    automatisch opgehaalde Open-Meteo-winddata (een ``WindData`` met
    ``as_text()``); beide optioneel en worden alleen vermeld als ze er zijn.
    """
    log_path = memory_dir / "trainingslog.md"
    if not log_path.exists():
        log_path.write_text(HEADER, encoding="utf-8")

    entry = (
        f"\n## {act.start_time:%Y-%m-%d %a %H:%M} — {sport_label(act.sport)}\n\n"
        f"- **Kerncijfers:** {kerncijfers(act)}\n"
        f"- **Tijd in zones:** {zone_regel(tiz)}\n"
        f"- **Observaties:** {observation or '_nog geen (LLM volgt)_'}\n"
    )
    if wind is not None:
        entry += f"- **Wind (Open-Meteo):** {wind.as_text()}\n"
    if user_note:
        entry += f"- **Opmerking:** {user_note}\n"
    entry += f"- _Sleutel `{act.activity_key}` · bron `{act.source_file}`_\n"

    with open(log_path, "a", encoding="utf-8") as f:
        f.write(entry)
