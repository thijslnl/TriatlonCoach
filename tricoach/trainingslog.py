"""Schrijven van trainingslog-entries naar memory/trainingslog.md.

Elke geïmporteerde sessie krijgt één markdown-entry met datum, sport,
kerncijfers en (zodra de LLM-integratie er is) een observatie. Het bestand
is bedoeld als leesbaar, doorzoekbaar geheugen — de ruwe data staat in SQLite.
"""

from pathlib import Path

from tricoach.fit_parser import ParsedActivity
from tricoach.formatting import (
    derive_speed_ms,
    fmt_duration,
    fmt_pace_per_100m,
    fmt_pace_per_km,
    fmt_speed_kmh,
    local_time,
    sport_label,
)
from tricoach.swim import SWOLF_FILTER_LABEL, crawl_swolf
from tricoach.timebasis import active_seconds

HEADER = """# Trainingslog

Automatisch bijgehouden door de tool: één entry per geïmporteerde sessie,
nieuwste onderaan. Kerncijfers komen uit het FIT-bestand; observaties van
het lokale LLM (of handmatig).
"""


def kerncijfers(act: ParsedActivity) -> str:
    """Bouw de kerncijfer-regel voor een sessie, afgestemd op de sport.

    Tempo en snelheid staan op de actieve tijd (zie :mod:`tricoach.timebasis`):
    rust aan de badrand en stilstand tellen niet mee. Bij zwemmen staat de
    totale timer-duur er apart bij als "sessieduur".
    """
    s = act.summary
    actief = active_seconds(act.sport, act.duration_s, act.records, act.lengths)
    speed = (derive_speed_ms(act.distance_m, actief)
             or s.get("enhanced_avg_speed") or s.get("avg_speed"))
    parts = [f"duur {fmt_duration(act.duration_s)}"]

    if act.sport == "running":
        parts.append(f"{act.distance_m / 1000:.2f} km op {fmt_pace_per_km(speed)} (actief)")
    elif act.sport == "cycling":
        if actief and actief < 0.99 * act.duration_s:
            parts[0] = (f"duur {fmt_duration(act.duration_s)} "
                        f"(rijtijd {fmt_duration(actief)})")
        parts.append(f"{act.distance_m / 1000:.1f} km, {fmt_speed_kmh(speed)} (actief)")
        if s.get("total_ascent"):
            parts.append(f"{s['total_ascent']:.0f} hm")
        # Vermogen (sinds de Rally/Kickr): gemiddeld + NP uit de samenvatting;
        # oudere sessies hebben deze velden niet en slaan de regel gewoon over.
        if s.get("avg_power"):
            watt = f"gem {s['avg_power']:.0f} W"
            if s.get("normalized_power"):
                watt += f" (NP {s['normalized_power']:.0f} W)"
            parts.append(watt)
        if act.is_indoor:
            parts.append("indoor (trainer)")
    elif act.sport == "swimming":
        parts[0] = (f"actieve zwemtijd {fmt_duration(actief)} "
                    f"(sessieduur {fmt_duration(act.duration_s)})")
        parts.append(f"{act.distance_m:.0f} m op {fmt_pace_per_100m(speed)} (actief tempo)")
        n = s.get("num_active_lengths") or s.get("num_lengths")
        if n and s.get("pool_length"):
            parts.append(f"{n} banen à {s['pool_length']:.0f}m")
        swolf = crawl_swolf(act.lengths, s.get("pool_length"), act.distance_m)
        if swolf is not None:
            parts.append(f"SWOLF {swolf:.0f} ({SWOLF_FILTER_LABEL})")

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
    training_label: str | None = None,
) -> None:
    """Voeg één sessie-entry toe aan memory/trainingslog.md.

    ``user_note`` is de vrije opmerking bij de upload, ``wind`` de automatisch
    opgehaalde Open-Meteo-winddata (een ``WindData`` met ``as_text()``) en
    ``training_label`` het sessielabel (bijv. "techniek/cadans"); alle drie
    optioneel en worden alleen vermeld als ze er zijn.
    """
    log_path = memory_dir / "trainingslog.md"
    if not log_path.exists():
        log_path.write_text(HEADER, encoding="utf-8")

    entry = (
        f"\n## {local_time(act.start_time):%Y-%m-%d %a %H:%M} — {sport_label(act.sport)}\n\n"
        f"- **Kerncijfers:** {kerncijfers(act)}\n"
        f"- **Tijd in zones:** {zone_regel(tiz)}\n"
        f"- **Observaties:** {observation or '_nog geen (LLM volgt)_'}\n"
    )
    if wind is not None:
        entry += f"- **Wind (Open-Meteo):** {wind.as_text()}\n"
    if user_note:
        entry += f"- **Opmerking:** {user_note}\n"
    if training_label:
        entry += f"- **Label:** {training_label}\n"
    entry += f"- _Sleutel `{act.activity_key}` · bron `{act.source_file}`_\n"

    with open(log_path, "a", encoding="utf-8") as f:
        f.write(entry)
