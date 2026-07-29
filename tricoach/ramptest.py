"""Herkenning van FTP-tests en het afleiden van een FTP-voorstel.

Sinds de Wahoo Kickr Core 2 is een echte FTP-test haalbaar geworden. Twee
protocollen komen in de praktijk voor:

- **Ramptest** (Zwift/Kickr, indoor). Het vermogen loopt in vaste blokken van
  ~1 minuut trapsgewijs op tot je afhaakt. De gangbare schatting is
  **75% van het hoogste 1-minuut-gemiddelde** — de laatste volledige trap.
- **Veldtest / 20-minutentest**. Twintig minuten zo hard mogelijk, waarna
  **95% van het beste 20-minutenvermogen** als FTP geldt (zie
  :data:`tricoach.power.FTP_EST_FACTOR`).

Deze module doet twee dingen en niet meer: een sessie *herkennen* als
FTP-test, en er een *voorstel* uit rekenen. Het voorstel wordt nooit
automatisch opgeslagen — de atleet bevestigt het op de instellingenpagina.
Dat is bewust: een afgebroken test, een mislukte koppeling of een gewone
intervaltraining die toevallig op een ramp lijkt mag de FTP nooit stilletjes
verzetten.

De vaststelling zelf wordt vastgelegd in ``memory/inzichten.md`` (datum,
methode, waarde), zodat de FTP-ontwikkeling over de tijd te volgen is naast
de drempelgeschiedenis in ``memory/lthr_geschiedenis.md``.
"""

from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from tricoach.power import FTP_EST_FACTOR, FTP_EST_WINDOW_S

# Ramptest: venster van één trap en de factor op het beste 1-minuutvermogen.
RAMP_WINDOW_S = 60
RAMP_FTP_FACTOR = 0.75

# Herkenningsdrempels voor een ramptest. Een ramp is te herkennen aan een
# vermogensprofiel dat over de sessie duidelijk en vrijwel monotoon oploopt
# naar een piek aan het eind — heel anders dan een duurrit (vlak) of een
# klassieke intervaltraining (op en neer rond hetzelfde niveau).
RAMP_MIN_DURATION_S = 8 * 60      # korter dan ~8 min is geen bruikbare test
RAMP_MAX_DURATION_S = 45 * 60     # langer dan ~45 min is geen ramptest meer
RAMP_MIN_RISE = 1.6               # piek moet ≥1,6× het startniveau zijn
RAMP_MIN_INCREASING = 0.75        # ≥75% van de trappen moet oplopen
RAMP_PEAK_TAIL = 0.75             # de piek ligt in het laatste kwart

# Veldtest: een 20-minutentest is alleen zinvol als er ook echt 20 aaneengesloten
# minuten hard is gereden. We stellen hem alleen voor bij een duidelijk
# inspanningsniveau ten opzichte van de rest van de rit.
FIELD_MIN_INTENSITY = 1.15        # beste 20 min ≥1,15× het ritgemiddelde

# Methodenamen, ook gebruikt in de logregel in inzichten.md.
METHOD_RAMP = "ramptest (75% van het beste 1-minuutvermogen)"
METHOD_FIELD = "veldtest (95% van het beste 20-minutenvermogen)"


@dataclass(frozen=True)
class FTPProposal:
    """Een FTP-voorstel uit één sessie, ter bevestiging door de atleet.

    ``basis_watt`` is het gemeten vermogen waarop het voorstel rust (het beste
    1-minuut- respectievelijk 20-minutengemiddelde), ``ftp_watt`` de daaruit
    afgeleide FTP. ``confidence`` is "hoog" bij een herkende ramptest en
    "indicatie" bij een 20-minutenafleiding uit een gewone rit.

    ``lthr_bpm`` is de bonus van een 20-minutentest: de gemiddelde hartslag
    over precies dat blok is de klassieke schatting van de **fiets-LTHR**. Uit
    één inspanning komen zo twee drempels. Bij een ramptest blijft dit None —
    daar is de hartslag aan het eind maximaal, niet drempelniveau, en die
    waarde zou de fiets-LTHR fors overschatten.
    """

    method: str
    ftp_watt: float
    basis_watt: float
    window_s: int
    factor: float
    confidence: str
    explanation: str
    lthr_bpm: int | None = None

    def as_text(self) -> str:
        """Het voorstel als één leesbare regel (UI, log en prompt)."""
        regel = (f"FTP {self.ftp_watt:.0f} W — {self.method}: "
                 f"{self.basis_watt:.0f} W × {self.factor:.0%}")
        if self.lthr_bpm:
            regel += f"; fiets-LTHR {self.lthr_bpm} bpm (gem. HR over het blok)"
        return regel


# ---------------------------------------------------------- vermogensprofiel --

def _power_series(records: pd.DataFrame) -> pd.Series:
    """Vermogen per meetpunt op de tijdas; leeg zonder bruikbare powerdata."""
    if records is None or records.empty or "power" not in records:
        return pd.Series(dtype=float)
    df = records.dropna(subset=["timestamp", "power"]).sort_values("timestamp")
    if df.empty:
        return pd.Series(dtype=float)
    return pd.Series(df["power"].astype(float).values,
                     index=pd.DatetimeIndex(df["timestamp"]))


def best_1min_power(records: pd.DataFrame) -> float | None:
    """Het hoogste 1-minuut-gemiddelde vermogen binnen één sessie, of None.

    Dit is de laatste volledige trap van een ramptest — de waarde waar de
    75%-regel op rekent.
    """
    serie = _power_series(records)
    if serie.empty:
        return None
    if (serie.index[-1] - serie.index[0]) < pd.Timedelta(seconds=RAMP_WINDOW_S):
        return None
    rollend = serie.rolling(f"{RAMP_WINDOW_S}s").mean()
    vol = rollend[serie.index >= serie.index[0] + pd.Timedelta(seconds=RAMP_WINDOW_S)]
    return float(vol.max()) if not vol.empty else None


def step_profile(records: pd.DataFrame, step_s: int = RAMP_WINDOW_S) -> list[float]:
    """Het gemiddelde vermogen per blok van ``step_s`` seconden.

    Het blokkige profiel van een ramptest is hierin direct zichtbaar: een
    reeks die trede voor trede oploopt. Lege lijst zonder powerdata.
    """
    serie = _power_series(records)
    if serie.empty:
        return []
    blokken = serie.resample(f"{step_s}s").mean().dropna()
    return [float(v) for v in blokken.to_numpy()]


def best_20min_window(records: pd.DataFrame) -> dict | None:
    """Het beste 20-minutenblok, mét de gemiddelde hartslag over datzelfde blok.

    Waar :func:`tricoach.power.best_20min_power` alleen het vermogen geeft,
    zoekt deze functie ook *wáár* dat blok lag, zodat de hartslag erover
    gemiddeld kan worden. Dat is precies wat een 20-minutentest zo nuttig
    maakt: uit één inspanning komen zowel de FTP (95% van het vermogen) als de
    fiets-LTHR (de gemiddelde hartslag over het blok).

    Geeft ``{"avg_watt", "avg_hr", "start", "end"}`` of None als de rit korter
    is dan 20 minuten of geen powerdata heeft. ``avg_hr`` is None zonder
    hartslagdata.
    """
    serie = _power_series(records)
    if serie.empty:
        return None
    venster = pd.Timedelta(seconds=FTP_EST_WINDOW_S)
    if (serie.index[-1] - serie.index[0]) < venster:
        return None

    rollend = serie.rolling(f"{FTP_EST_WINDOW_S}s").mean()
    vol = rollend[serie.index >= serie.index[0] + venster]
    if vol.empty:
        return None
    einde = vol.idxmax()          # het rolling-venster eindigt hier
    start = einde - venster

    resultaat = {"avg_watt": float(vol.max()), "avg_hr": None,
                 "start": start, "end": einde}
    if "heart_rate" not in records:
        return resultaat
    hr = records.dropna(subset=["timestamp", "heart_rate"])
    if hr.empty:
        return resultaat
    hr = hr.set_index(pd.DatetimeIndex(hr["timestamp"]))["heart_rate"]
    blok = hr[(hr.index >= start) & (hr.index <= einde)]
    if not blok.empty:
        resultaat["avg_hr"] = float(blok.mean())
    return resultaat


def is_ramp_test(records: pd.DataFrame, indoor: bool) -> bool:
    """Ziet deze sessie eruit als een ramptest (indoor, oplopend blokkig)?

    Een ramptest wordt alleen binnen gereden (Zwift/Kickr), duurt ~8–45
    minuten en heeft een profiel dat vrijwel monotoon oploopt naar een piek
    in het laatste kwart, met een duidelijke stijging ten opzichte van de
    starttrappen. Bewust streng: liever een test missen dan een gewone
    intervalrit als test aanmerken en de atleet een verkeerde FTP voorstellen.
    """
    if not indoor:
        return False
    serie = _power_series(records)
    if serie.empty:
        return False
    duur = (serie.index[-1] - serie.index[0]).total_seconds()
    if not RAMP_MIN_DURATION_S <= duur <= RAMP_MAX_DURATION_S:
        return False

    trappen = step_profile(records)
    if len(trappen) < 6:
        return False
    # Warming-up (het eerste blok) telt niet mee voor het startniveau: die is
    # bij Zwift vaak al hoger dan de eerste echte trap.
    start = float(np.median(trappen[:3]))
    piek = max(trappen)
    if start <= 0 or piek / start < RAMP_MIN_RISE:
        return False

    # De piek hoort aan het eind te liggen: bij een ramp haak je bovenaan af.
    piek_index = int(np.argmax(trappen))
    if piek_index < RAMP_PEAK_TAIL * (len(trappen) - 1):
        return False

    # En het profiel moet overwegend stijgen tot aan die piek.
    tot_piek = trappen[:piek_index + 1]
    stijgend = sum(1 for a, b in zip(tot_piek, tot_piek[1:]) if b >= a)
    overgangen = max(len(tot_piek) - 1, 1)
    return stijgend / overgangen >= RAMP_MIN_INCREASING


# ------------------------------------------------------------- het voorstel --

def ftp_proposal(records: pd.DataFrame, indoor: bool) -> FTPProposal | None:
    """Leid een FTP-voorstel af uit één sessie, of None als dat niet kan.

    Voorkeursvolgorde:

    1. Een herkende **ramptest** → 75% van het beste 1-minuutvermogen.
    2. Anders een **20-minuten-inspanning** die er duidelijk uitspringt ten
       opzichte van de rest van de rit → 95% van dat vermogen, als indicatie,
       plus de fiets-LTHR uit de gemiddelde hartslag over datzelfde blok.

    Een gewone duurrit levert niets op: daar zit geen maximale inspanning in,
    dus elke afleiding zou de FTP onderschatten.
    """
    if is_ramp_test(records, indoor):
        beste = best_1min_power(records)
        if beste:
            return FTPProposal(
                method=METHOD_RAMP,
                ftp_watt=beste * RAMP_FTP_FACTOR,
                basis_watt=beste,
                window_s=RAMP_WINDOW_S,
                factor=RAMP_FTP_FACTOR,
                confidence="hoog",
                explanation=(
                    "Deze sessie heeft het profiel van een ramptest: indoor, "
                    "met een blokkig vermogen dat trede voor trede oploopt naar "
                    "een piek aan het eind. De gangbare afleiding is 75% van de "
                    "laatste volledige minuut. Let op: een ramptest levert "
                    "alleen vermogen op — de hartslag is aan het eind maximaal, "
                    "niet drempelniveau, dus hier komt géén fiets-LTHR uit. "
                    "Een 20-minutentest geeft je beide drempels in één keer."
                ),
            )

    venster = best_20min_window(records)
    if not venster:
        return None
    serie = _power_series(records)
    gemiddeld = float(serie.mean()) if not serie.empty else 0.0
    if gemiddeld <= 0 or venster["avg_watt"] / gemiddeld < FIELD_MIN_INTENSITY:
        return None
    lthr = round(venster["avg_hr"]) if venster["avg_hr"] else None
    uitleg = (
        "Geen ramptest herkend, maar er zit een aaneengesloten blok van 20 "
        "minuten in dat duidelijk boven het ritgemiddelde ligt. 95% daarvan "
        "is de klassieke veldtest-schatting — een ondergrens zolang het "
        "geen echte, volledig uitgereden test was."
    )
    if lthr:
        uitleg += (
            f" Bonus: de gemiddelde hartslag over precies dat blok is "
            f"{lthr} bpm — de klassieke schatting van de **fiets-LTHR**. Uit "
            "één inspanning komen zo beide drempels, waar een ramptest alleen "
            "het vermogen geeft."
        )
    return FTPProposal(
        method=METHOD_FIELD,
        ftp_watt=venster["avg_watt"] * FTP_EST_FACTOR,
        basis_watt=venster["avg_watt"],
        window_s=FTP_EST_WINDOW_S,
        factor=FTP_EST_FACTOR,
        confidence="indicatie",
        explanation=uitleg,
        lthr_bpm=lthr,
    )


# -------------------------------------------------------------- vastleggen --

INZICHTEN_HEADER = """# Inzichten

Langetermijnpatronen die de tool (of de coach) in de data ontdekt.
"""


def log_ftp_determination(memory_dir: Path, ftp_watt: float, method: str,
                          basis_watt: float | None = None,
                          session_date: "date | datetime | None" = None,
                          note: str = "") -> None:
    """Leg een FTP-vaststelling vast in ``memory/inzichten.md``.

    Eén sectie per vaststelling, met de datum, de gebruikte methode en het
    gemeten vermogen waarop hij rust. Zo is de FTP-ontwikkeling over de tijd
    terug te lezen naast de drempelgeschiedenis — inclusief het protocol, want
    een ramptest-FTP en een 20-minuten-FTP zijn niet zomaar vergelijkbaar.
    """
    path = memory_dir / "inzichten.md"
    if not path.exists():
        path.write_text(INZICHTEN_HEADER, encoding="utf-8")

    wanneer = session_date or date.today()
    if isinstance(wanneer, datetime):
        wanneer = wanneer.date()
    entry = (
        f"\n## {wanneer:%Y-%m-%d} — FTP vastgesteld: {ftp_watt:.0f} W\n\n"
        f"- **Methode:** {method}\n"
    )
    if basis_watt:
        entry += f"- **Gemeten basis:** {basis_watt:.0f} W\n"
    entry += f"- **Vastgelegd op:** {date.today():%Y-%m-%d}\n"
    if note:
        entry += f"- **Opmerking:** {note}\n"
    with open(path, "a", encoding="utf-8") as f:
        f.write(entry)
