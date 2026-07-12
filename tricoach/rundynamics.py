"""Loopdynamiek (running dynamics) uit de Garmin FIT-samenvattingen.

De Forerunner 265 meet per loopsessie cadans, staplengte, grondcontacttijd
en verticale ratio (plus een polsgeschat loopvermogen). Deze module haalt
die velden uit de sessie-samenvatting (``summary_json``), rekent ze om naar
weergave-eenheden en levert de trend-DataFrames en de feedback-context.

Eenheden (besluit, vastgelegd in memory/beslissingen.md):

- **Cadans = totale stappen per minuut (spm, beide benen).** FIT slaat
  loopcadans op als omwentelingen per minuut van één been (~68 rpm), met het
  fractionele deel apart in ``avg_fractional_cadence`` (bijv. 0.5). De
  weergavewaarde is dus ``(cadans + fractie) × 2`` — hetzelfde getal dat
  Garmin Connect toont (~137 spm), dat is de referentie. Geen dubbeltelling:
  we verdubbelen precies één keer.
- **Staplengte**: FIT geeft millimeters, wij tonen meters.
- **Grondcontacttijd**: milliseconden (zoals FIT en Garmin Connect).
- **Verticale ratio**: procenten (verticale oscillatie ÷ staplengte).

Richtbereiken (bewust zacht geformuleerd):

- **Cadans**: 150–165 spm als richtlijn voor rustig duurlooptempo. De bekende
  "180-regel" komt van elite-lopers op wedstrijdtempo en is geen universeel
  doel; cadans hangt bovendien af van tempo en beenlengte. De eigen trend is
  belangrijker dan het bereik — een geleidelijke stijging (~+5% per paar
  weken) helpt loopeconomie en blessurebestendigheid, een getal najagen niet.
- **Grondcontacttijd**: 200–300 ms is het bereik van geoefende lopers; puur
  referentie. Daalt vanzelf mee wanneer de cadans stijgt.
- **Verticale ratio**: onder ~8% is prima; hier valt dan niets te verbeteren.
- **Loopvermogen**: onrijpe, merk-afhankelijke maat — tonen mag, sturen doen
  we op hartslag.
"""

import json

import pandas as pd

from tricoach.formatting import run_cadence_spm

# Zacht richtbereik (spm) voor cadans op rustig duurlooptempo — een richtlijn
# om de eigen trend tegen af te zetten, nadrukkelijk geen norm.
CADENCE_GUIDE_SPM = (150, 165)
# Grondcontacttijd (ms) van geoefende lopers, als referentie in de grafiek.
GCT_TRAINED_MS = (200, 300)
# Verticale ratio (%) tot deze grens geldt als prima (efficiënt, weinig stuiteren).
VERTICAL_RATIO_GOOD_PCT = 8.0


def cadence_spm(avg_cadence: float | None,
                fractional: float | None = None) -> float | None:
    """FIT-loopcadans (rpm één been, + fractie) -> totale stappen per minuut.

    ``avg_cadence`` is het gehele deel (bijv. 68), ``fractional`` het
    fractionele deel uit ``avg_fractional_cadence`` (bijv. 0.5); samen ×2
    geeft het spm-getal van Garmin Connect (137). Zonder fractie wijkt het
    hooguit 1 spm af. None bij ontbrekende/onbruikbare invoer.
    """
    base = run_cadence_spm(avg_cadence)
    if base is None:
        return None
    if fractional is not None and not pd.isna(fractional):
        base += float(fractional) * 2
    return base


def vertical_ratio_is_good(ratio_pct: float | None) -> bool | None:
    """Valt de verticale ratio in het 'prima'-bereik? None als onbekend."""
    if ratio_pct is None or pd.isna(ratio_pct):
        return None
    return float(ratio_pct) <= VERTICAL_RATIO_GOOD_PCT


def dynamics_from_summary(summary: dict) -> dict:
    """De loopdynamiek-velden uit één sessie-samenvatting, in weergave-eenheden.

    Geeft een dict met ``cadans_spm``, ``staplengte_m``, ``gct_ms``,
    ``vert_ratio_pct`` en ``vermogen_w``; velden die het horloge (of een
    oudere import) niet heeft zijn None. Werkt op de ruwe FIT-eenheden
    zoals ``save_activity`` ze in ``summary_json`` bewaart.
    """
    def _num(key: str) -> float | None:
        val = summary.get(key)
        try:
            val = float(val)
        except (TypeError, ValueError):
            return None
        return None if pd.isna(val) else val

    staplengte_mm = _num("avg_step_length")
    return {
        "cadans_spm": cadence_spm(
            _num("avg_running_cadence") or _num("avg_cadence"),
            _num("avg_fractional_cadence"),
        ),
        "staplengte_m": staplengte_mm / 1000 if staplengte_mm else None,
        "gct_ms": _num("avg_stance_time"),
        "vert_ratio_pct": _num("avg_vertical_ratio"),
        "vermogen_w": _num("normalized_power") or _num("avg_power"),
    }


def dynamics_trend(activities: pd.DataFrame) -> pd.DataFrame:
    """Loopdynamiek per loopsessie als trend-DataFrame (oudste eerst).

    Leest ``summary_json`` van elke loopsessie; kolommen: ``start_time`` plus
    de velden van :func:`dynamics_from_summary`. Sessies die vóór de
    dynamics-uitbreiding zijn geïmporteerd missen de meeste velden (NaN) tot
    de zip opnieuw wordt geüpload (de import vult ze dan aan) — de cadans is
    er via ``avg_cadence`` altijd.
    """
    if activities.empty:
        return pd.DataFrame()
    runs = activities[activities["sport"] == "running"].sort_values("start_time")
    rows = []
    for _, act in runs.iterrows():
        try:
            summary = json.loads(act.get("summary_json") or "{}")
        except (TypeError, ValueError):
            summary = {}
        # Vangnet: de kolom avg_cadence is er ook als summary_json kaal is.
        summary.setdefault("avg_cadence", act.get("avg_cadence"))
        rows.append({"start_time": act["start_time"], **dynamics_from_summary(summary)})
    return pd.DataFrame(rows)


# ------------------------------------------------------------ feedback-context --

def _fmt(value: float | None, template: str) -> str | None:
    """Format een waarde, of None als hij ontbreekt (regel wordt weggelaten)."""
    if value is None or pd.isna(value):
        return None
    return template.format(value)


def dynamics_block(summary: dict, activities: pd.DataFrame) -> str | None:
    """Loopdynamiek als contextblok voor de coaching-feedback.

    Bevat de waarden van de huidige sessie plus de trend van de laatste
    loopsessies, mét de leesinstructie dat dit langetermijn-context is: de
    coach mag er hooguit een observatie over maken, geen sessie-oordeel op
    baseren. None wanneer de sessie geen enkel dynamics-veld heeft.
    """
    dyn = dynamics_from_summary(summary)
    regels = [t for t in (
        _fmt(dyn["cadans_spm"], "- Cadans: {:.0f} spm (totale stappen per minuut)"),
        _fmt(dyn["staplengte_m"], "- Staplengte: {:.2f} m"),
        _fmt(dyn["gct_ms"], "- Grondcontacttijd: {:.0f} ms"),
        _fmt(dyn["vert_ratio_pct"], "- Verticale ratio: {:.1f}%"),
    ) if t]
    if not regels:
        return None

    ratio_goed = vertical_ratio_is_good(dyn["vert_ratio_pct"])
    if ratio_goed:
        regels.append(
            f"  (verticale ratio onder {VERTICAL_RATIO_GOOD_PCT:.0f}% is prima — "
            "hier valt niets te verbeteren)"
        )

    trend = dynamics_trend(activities)
    if not trend.empty:
        cad = trend.dropna(subset=["cadans_spm"]).tail(6)
        if len(cad) >= 2:
            waarden = " → ".join(f"{v:.0f}" for v in cad["cadans_spm"])
            regels.append(f"- Cadans-trend laatste loopsessies (spm): {waarden}")
        gct = trend.dropna(subset=["gct_ms"]).tail(6)
        if len(gct) >= 2:
            waarden = " → ".join(f"{v:.0f}" for v in gct["gct_ms"])
            regels.append(f"- Grondcontacttijd-trend (ms): {waarden}")

    regels.append(
        "Leesinstructie: dit is LANGETERMIJN-context (project voor ná de "
        f"eerste race). Richtlijn cadans rustig duurlooptempo ~{CADENCE_GUIDE_SPM[0]}–"
        f"{CADENCE_GUIDE_SPM[1]} spm, maar dat is een richting, geen norm — "
        "beoordeel een losse sessie hier NIET op en presenteer een lage cadans "
        "niet als fout. Hooguit één rustige observatie over de trend over "
        "weken/maanden."
    )
    return "\n".join(regels)
