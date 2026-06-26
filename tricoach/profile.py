"""Profielwaarden spiegelen naar memory/doelen.md (bron van waarheid) + changelog.

De technische waarden (max HR, LTHR, %LTHR-zones) blijven in ``config.yaml``,
omdat de hele app daar al op draait (zone-berekening, herrekening bij een
LTHR-wijziging, enz.). Maar de gebruiker beheert zijn profiel op de
instellingenpagina, en ``doelen.md`` is de leesbare bron die in álle advies- en
feedback-prompts meegaat. Daarom houden we in ``doelen.md`` een door het
dashboard beheerd profielblok bij, plus een changelog die élke wijziging
vastlegt — zo blijven eerdere zones reproduceerbaar.

Het beheerde blok staat tussen twee markers; de rest van ``doelen.md`` (de met
de hand geschreven achtergrond) blijft onaangeroerd.
"""

from datetime import date
from pathlib import Path

from tricoach.zones import bounds_from_lthr, zone_bounds

START = "<!-- PROFIEL:START — beheerd door het dashboard, niet handmatig bewerken -->"
END = "<!-- PROFIEL:END -->"
CHANGELOG_HEADER = "## Wijzigingslog profielwaarden"


def _zones_text(athlete: dict) -> str:
    """De afgeleide %LTHR-zones als één regel."""
    b = zone_bounds(athlete)
    return (f"Z2 {b[0]}–{b[1] - 1} · Z3 {b[1]}–{b[2] - 1} · "
            f"Z4 {b[2]}–{b[3] - 1} · Z5 {b[3]}+")


def profile_block(config: dict) -> str:
    """Het beheerde profielblok als markdown (zonder de markers)."""
    a = config["athlete"]
    regels = [
        "## Profiel (actueel — beheerd door het dashboard)",
        "",
        f"- **Max hartslag:** {a['max_hr']}",
        f"- **LTHR:** {a['lthr']} → zones (%LTHR): {_zones_text(a)}",
        f"- **Trainingsdagen:** {a.get('training_days', '—') or '—'}",
        f"- **Beschikbare tijd per sessie:** {a.get('session_time', '—') or '—'}",
        "- **Racedoelen en streeftijden:**",
    ]
    races = config.get("races", [])
    if races:
        for r in races:
            streef = str(r.get("target_time") or "").strip() or "geen streeftijd"
            regels.append(
                f"  - {r.get('name', '?')} ({r.get('date', '?')}): "
                f"{r.get('distances', '')} — {streef}"
            )
    else:
        regels.append("  - (geen races ingesteld)")
    return "\n".join(regels)


def _profile_values(config: dict) -> dict[str, str]:
    """De profielwaarden plat, voor het vergelijken bij de changelog."""
    a = config["athlete"]
    vals = {
        "Max hartslag": str(a.get("max_hr")),
        "LTHR": str(a.get("lthr")),
        "Trainingsdagen": str(a.get("training_days") or ""),
        "Beschikbare tijd per sessie": str(a.get("session_time") or ""),
    }
    for r in config.get("races", []):
        naam = r.get("name", "?")
        vals[f"Race · {naam} · datum"] = str(r.get("date") or "")
        vals[f"Race · {naam} · streeftijd"] = str(r.get("target_time") or "")
    return vals


def diff_profiles(old: dict, new: dict) -> list[str]:
    """Welke profielwaarden zijn gewijzigd? Geeft leesbare 'oud → nieuw'-regels."""
    ov, nv = _profile_values(old), _profile_values(new)
    regels = []
    for sleutel in nv:
        was = ov.get(sleutel, "")
        wordt = nv[sleutel]
        if was != wordt:
            regels.append(f"{sleutel}: {was or '—'} → {wordt or '—'}")
    return regels


def _split_managed(text: str) -> tuple[str, str]:
    """Splits doelen.md in (deel vóór het beheerde blok, changelog-entries).

    Het beheerde blok en de changelog-kop worden eruit gehaald; de losse
    changelog-regels blijven behouden zodat de historie niet verloren gaat.
    """
    # Beheerd blok eruit knippen.
    if START in text and END in text:
        voor, rest = text.split(START, 1)
        _, na = rest.split(END, 1)
        text = voor.rstrip() + "\n" + na.lstrip()

    # Bestaande changelog-entries afsplitsen.
    entries = ""
    if CHANGELOG_HEADER in text:
        text, changelog = text.split(CHANGELOG_HEADER, 1)
        entries = "\n".join(
            line for line in changelog.splitlines() if line.strip().startswith("- ")
        )
    return text.rstrip(), entries


def update_doelen(
    memory_dir: Path, old_config: dict, new_config: dict, note: str = ""
) -> list[str]:
    """Herschrijf het beheerde profielblok in doelen.md en log de wijzigingen.

    Geeft de lijst gewijzigde waarden terug (leeg als er niets veranderde). De
    met de hand geschreven inhoud van doelen.md blijft staan; alleen het blok
    tussen de markers en de changelog worden door het dashboard beheerd.
    """
    path = memory_dir / "doelen.md"
    bestaand = path.read_text(encoding="utf-8") if path.exists() else "# Doelen & Voorkeuren\n"
    voor, oude_entries = _split_managed(bestaand)

    wijzigingen = diff_profiles(old_config, new_config)
    nieuwe_entries = oude_entries
    if wijzigingen:
        regel = f"- **{date.today():%Y-%m-%d}**" + (f" ({note})" if note else "") + ": " + "; ".join(wijzigingen)
        nieuwe_entries = (oude_entries + "\n" + regel).strip()

    blok = f"{START}\n\n{profile_block(new_config)}\n\n{END}"
    changelog = f"{CHANGELOG_HEADER}\n\n{nieuwe_entries}\n" if nieuwe_entries else ""
    nieuw = f"{voor.rstrip()}\n\n{blok}\n\n{changelog}".rstrip() + "\n"
    path.write_text(nieuw, encoding="utf-8")
    return wijzigingen
