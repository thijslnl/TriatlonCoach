"""Duurschatting uit de eigen sessies — niet uit een standaardtabel.

Een voedingsplan valt of staat bij de duur: 60 g/uur over drie uur is iets
heel anders dan over vijf. Die duur komt hier uit de eigen database, met een
expliciete onderbouwing ("waar komt dit getal vandaan") en altijd als
**bandbreedte**, want een puntschatting suggereert een precisie die er niet is.

De werkwijze per onderdeel:

1. **Vergelijkbare sessies zoeken.** Recente sessies van dezelfde sport op de
   gekozen intensiteit — bij fietsen op het vermogensvenster (%FTP), bij lopen
   op het hartslagvenster van de zone. Transport-ritjes en indoorsessies
   (virtuele snelheid) vallen af.
2. **Temperatuur eruit rekenen.** Elke sessie wordt teruggerekend naar een
   neutrale temperatuur en daarna vooruit naar de verwachte temperatuur van de
   geplande sessie. Hoeveel tempo warmte kost, wordt uit de eigen sessies
   geschat (regressie van snelheid op temperatuur); is daar te weinig data
   voor, dan geldt :data:`HEAT_DEFAULT_PCT_PER_C`.
3. **Naar de doelafstand schalen.** Met Riegel (dezelfde exponent als de
   racevoorspelling): een sessie van 37 km zegt iets over 90 km, maar niet dat
   je dat tempo volhoudt. Wordt de duur direct opgegeven, dan slaat deze stap
   over.
4. **Bandbreedte.** De p25-p75 van de geschaalde sessietijden, verbreed met een
   sportafhankelijke marge. Bij fietsen is die marge het grootst: wind is daar
   de dominante onzekerheid en die zit niet in de historie.

Bij een brick of triatlon worden de onderdelen opgeteld plus de wisseltijden;
die komen uit de eigen bevestigde combo's (zie :mod:`tricoach.combos`) met een
terugval op :data:`DEFAULT_TRANSITION_S`.
"""

import sqlite3
from dataclasses import dataclass, field

import pandas as pd

from tricoach.formatting import fmt_duration
from tricoach.sportzones import ftp as athlete_ftp, hr_zone_bounds
from tricoach.storage import training_activities

# Riegel-exponent: dezelfde als in tricoach.progress, zodat de duurschatting en
# de racevoorspelling niet uit elkaar lopen.
RIEGEL = 1.06

# Alleen sessies uit deze periode tellen mee; ouder materiaal beschrijft een
# andere atleet.
LOOKBACK_DAYS = 180

# Minimaal aantal sessies voor een schatting op de gekozen intensiteit; daaronder
# valt de schatting terug op zone 2 met een intensiteitsopslag.
MIN_SESSIONS = 3

# Sessies korter dan dit zeggen te weinig over een duurinspanning.
MIN_SESSION_S = {"running": 900.0, "cycling": 1200.0, "swimming": 600.0}

# Vermogensvensters per intensiteit als fractie van de FTP (Coggan-achtig):
# rustig = duurzone, racetempo = tempo/sweetspot, hard = drempel.
POWER_WINDOWS = {
    "rustig": (0.56, 0.75),
    "racetempo": (0.76, 0.90),
    "hard": (0.91, 1.05),
}

# Hartslagvensters per intensiteit, uitgedrukt in zone-index: rustig = Z2,
# racetempo = Z3, hard = Z4 en hoger.
HR_ZONE_INDEX = {"rustig": 0, "racetempo": 1, "hard": 2}

# Terugval als er te weinig sessies op de gevraagde intensiteit zijn: opslag op
# de zone 2-snelheid. Bewust bescheiden — het is een schatting, geen belofte.
INTENSITY_SPEED_FACTOR = {"rustig": 1.00, "racetempo": 1.08, "hard": 1.15}

# Extra verbreding van de bandbreedte per sport (fractie van de mediaan). Bij
# fietsen het grootst: wind bepaalt daar de dag meer dan de vorm.
BAND_WIDEN = {"cycling": 0.02, "running": 0.01, "swimming": 0.03}

# Ondergrens voor de halve bandbreedte als er weinig sessies zijn (fractie).
MIN_HALF_BAND = {"cycling": 0.08, "running": 0.05, "swimming": 0.08}

# Temperatuurcorrectie. Boven deze temperatuur begint warmte tempo te kosten.
HEAT_NEUTRAL_C = 18.0
# Terugval als de eigen data geen bruikbaar verband geeft: % snelheidsverlies
# per graad boven de neutrale temperatuur.
HEAT_DEFAULT_PCT_PER_C = 0.4
# Grenzen waarbinnen een uit eigen data geschat verband geloofwaardig is.
HEAT_FIT_MIN_SESSIONS = 6
HEAT_FIT_MAX_PCT_PER_C = 1.5
# Maximale correctie op de snelheid (nooit meer dan 15% trager door hitte).
HEAT_MAX_PENALTY = 0.15
# Onder dit percentage wordt de temperatuurcorrectie niet in de onderbouwing
# genoemd: een half procent is ruis en zou de tekst alleen maar drukker maken.
HEAT_NOTE_MIN_PCT = 0.5

# Open water is trager dan het bad: geen muur om af te zetten, golven, navigeren.
# Alleen gebruikt als terugval wanneer er te weinig open water-sessies zijn.
OPEN_WATER_FACTOR = 0.92

# Wisseltijden (s) als er nog geen bevestigde combo's zijn.
DEFAULT_TRANSITION_S = {"T1": 300.0, "T2": 180.0}

SPORT_LABEL = {"swimming": "Zwemmen", "cycling": "Fietsen", "running": "Hardlopen"}


@dataclass
class LegRequest:
    """Wat de atleet voor één onderdeel invult: een afstand óf een duur."""

    sport: str
    distance_m: float | None = None
    duration_s: float | None = None

    @property
    def is_duration_based(self) -> bool:
        return self.duration_s is not None and self.duration_s > 0


@dataclass
class LegEstimate:
    """De duurschatting van één onderdeel, met de onderbouwing erbij."""

    sport: str
    low_s: float
    mid_s: float
    high_s: float
    distance_m: float | None = None
    n_sessions: int = 0
    basis: str = ""
    exact_intensity: bool = True
    speed_ms: float | None = None

    @property
    def label(self) -> str:
        return SPORT_LABEL.get(self.sport, self.sport)

    def range_text(self) -> str:
        """De schatting als bereik, bijv. '3:05-3:25'."""
        if round(self.low_s) == round(self.high_s):
            return fmt_duration(self.mid_s)
        return f"{fmt_duration(self.low_s)}-{fmt_duration(self.high_s)}"


@dataclass
class DurationEstimate:
    """De totale duurschatting: de onderdelen, de wissels en het totaal."""

    legs: list[LegEstimate] = field(default_factory=list)
    transitions: list[tuple[str, float]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def transition_s(self) -> float:
        return sum(s for _, s in self.transitions)

    @property
    def low_s(self) -> float:
        return sum(l.low_s for l in self.legs) + self.transition_s

    @property
    def mid_s(self) -> float:
        return sum(l.mid_s for l in self.legs) + self.transition_s

    @property
    def high_s(self) -> float:
        return sum(l.high_s for l in self.legs) + self.transition_s

    def range_text(self) -> str:
        if round(self.low_s) == round(self.high_s):
            return fmt_duration(self.mid_s)
        return f"{fmt_duration(self.low_s)}-{fmt_duration(self.high_s)}"


# ------------------------------------------------------------ sessieselectie --

def _session_speeds(conn: sqlite3.Connection, acts: pd.DataFrame,
                    sport: str) -> pd.DataFrame:
    """Bruikbare sessies van één sport met afstand, tijd, snelheid en temperatuur.

    Alleen echte trainingen (geen transport), buiten (geen Zwift/trainer: die
    snelheid is virtueel) en lang genoeg om iets over een duurinspanning te
    zeggen. De tijdsbasis is de actieve tijd (zie :mod:`tricoach.timebasis`),
    met terugval op de totale duur.
    """
    if acts.empty:
        return pd.DataFrame()
    df = training_activities(acts)
    df = df[df["sport"] == sport].copy()
    if df.empty:
        return pd.DataFrame()

    grens = pd.Timestamp.now(tz=df["start_time"].dt.tz) - pd.Timedelta(days=LOOKBACK_DAYS)
    df = df[df["start_time"] >= grens]
    if "is_indoor" in df:
        df = df[df["is_indoor"].fillna(0) != 1]
    df["seconds"] = df["active_s"].fillna(df["duration_s"]) if "active_s" in df \
        else df["duration_s"]
    df = df.dropna(subset=["distance_m", "seconds"])
    df = df[(df["distance_m"] > 0) & (df["seconds"] >= MIN_SESSION_S.get(sport, 600.0))]
    if df.empty:
        return pd.DataFrame()
    df["speed_ms"] = df["distance_m"] / df["seconds"]
    return df


def _filter_intensity(df: pd.DataFrame, sport: str, intensity: str,
                      athlete: dict) -> tuple[pd.DataFrame, str]:
    """Filter de sessies op de gevraagde intensiteit.

    Bij fietsen op het genormaliseerde vermogen als de FTP bekend is (vermogen
    is daar de betere maat), anders op hartslag. Bij lopen altijd op hartslag.
    Geeft (sessies, beschrijving van het gebruikte venster) terug; een lege
    beschrijving betekent "geen intensiteitsfilter toegepast".
    """
    if df.empty:
        return df, ""

    if sport == "cycling":
        ftp = athlete_ftp(athlete)
        lo_pct, hi_pct = POWER_WINDOWS.get(intensity, POWER_WINDOWS["rustig"])
        if ftp and "np_power" in df and df["np_power"].notna().any():
            lo, hi = ftp * lo_pct, ftp * hi_pct
            sel = df[df["np_power"].between(lo, hi)]
            return sel, (f"genormaliseerd vermogen {lo:.0f}-{hi:.0f} W "
                         f"({lo_pct:.0%}-{hi_pct:.0%} van FTP {ftp:.0f} W)")

    bounds = hr_zone_bounds(athlete, sport)
    if bounds is None or "avg_hr" not in df:
        return df, ""
    idx = HR_ZONE_INDEX.get(intensity, 0)
    lo = bounds[idx]
    hi = bounds[idx + 1] - 1 if idx + 1 < len(bounds) else 250
    sel = df[df["avg_hr"].between(lo, hi)]
    return sel, f"gemiddelde hartslag {lo}-{hi} bpm"


# --------------------------------------------------------- temperatuureffect --

def heat_pct_per_degree(df: pd.DataFrame) -> tuple[float, bool]:
    """Hoeveel procent snelheid kost één graad boven :data:`HEAT_NEUTRAL_C`?

    Geschat uit de eigen sessies: een rechte lijn door (temperatuur,
    afstand-genormaliseerde snelheid). Om te voorkomen dat een toevallige
    correlatie het plan stuurt, wordt de uitkomst alleen aangenomen bij genoeg
    sessies én een plausibele helling; anders geldt
    :data:`HEAT_DEFAULT_PCT_PER_C`. Geeft (percentage per graad, uit_eigen_data).
    """
    if df.empty or "temperature_c" not in df:
        return HEAT_DEFAULT_PCT_PER_C, False
    sel = df.dropna(subset=["temperature_c", "speed_ms", "distance_m", "seconds"])
    if len(sel) < HEAT_FIT_MIN_SESSIONS:
        return HEAT_DEFAULT_PCT_PER_C, False

    # Snelheid hangt ook van de afstand af; normaliseer eerst met Riegel naar de
    # mediane afstand, zodat alleen de temperatuur overblijft.
    ref = float(sel["distance_m"].median())
    genormaliseerd = ref / (sel["seconds"] * (ref / sel["distance_m"]) ** RIEGEL)
    try:
        helling, _ = _linear_fit(sel["temperature_c"].to_numpy(dtype=float),
                                 genormaliseerd.to_numpy(dtype=float))
    except (ValueError, ZeroDivisionError):
        return HEAT_DEFAULT_PCT_PER_C, False

    gemiddelde = float(genormaliseerd.mean())
    if gemiddelde <= 0:
        return HEAT_DEFAULT_PCT_PER_C, False
    pct = -helling / gemiddelde * 100
    if not 0 < pct <= HEAT_FIT_MAX_PCT_PER_C:
        return HEAT_DEFAULT_PCT_PER_C, False
    return pct, True


def _linear_fit(x, y) -> tuple[float, float]:
    """Kleinste-kwadratenlijn y = a*x + b; geeft (a, b)."""
    n = len(x)
    if n < 2:
        raise ValueError("te weinig punten")
    gem_x, gem_y = x.mean(), y.mean()
    noemer = ((x - gem_x) ** 2).sum()
    if noemer == 0:
        raise ZeroDivisionError("geen spreiding in x")
    a = ((x - gem_x) * (y - gem_y)).sum() / noemer
    return float(a), float(gem_y - a * gem_x)


def heat_speed_factor(temp_c: float | None, pct_per_degree: float) -> float:
    """Snelheidsfactor bij deze temperatuur (1,0 = geen effect, 0,95 = 5% trager)."""
    if temp_c is None:
        return 1.0
    verlies = max(temp_c - HEAT_NEUTRAL_C, 0.0) * pct_per_degree / 100
    return 1.0 - min(verlies, HEAT_MAX_PENALTY)


# ---------------------------------------------------------------- schatting --

def _riegel_seconds(seconds: float, from_m: float, to_m: float) -> float:
    """Schaal een sessietijd naar een andere afstand volgens Riegel."""
    return seconds * (to_m / from_m) ** RIEGEL


def estimate_leg(conn: sqlite3.Connection, acts: pd.DataFrame, athlete: dict,
                 leg: LegRequest, intensity: str,
                 temp_c: float | None) -> LegEstimate:
    """Schat de duur van één onderdeel uit de eigen sessies.

    Is er een duur opgegeven, dan wordt die één op één overgenomen (de atleet
    weet wat hij van plan is). Bij een afstand worden vergelijkbare sessies
    gezocht, op temperatuur gecorrigeerd en met Riegel naar de doelafstand
    geschaald; de bandbreedte is de p25-p75 van die geschaalde tijden.
    """
    if leg.is_duration_based:
        return LegEstimate(
            sport=leg.sport, low_s=leg.duration_s, mid_s=leg.duration_s,
            high_s=leg.duration_s, distance_m=leg.distance_m,
            basis="duur handmatig opgegeven",
        )

    doel_m = float(leg.distance_m or 0)
    if doel_m <= 0:
        return LegEstimate(sport=leg.sport, low_s=0, mid_s=0, high_s=0,
                           basis="geen afstand of duur opgegeven")

    alle = _session_speeds(conn, acts, leg.sport)
    if leg.sport == "swimming":
        return _estimate_swim(alle, doel_m)

    sel, venster = _filter_intensity(alle, leg.sport, intensity, athlete)
    exact = len(sel) >= MIN_SESSIONS
    opslag, terugval = 1.0, ""
    if not exact:
        # Te weinig sessies op deze intensiteit: eerst terugvallen op zone 2 met
        # een opslag, en anders op alles wat er is. Beide worden gemeld — een
        # schatting die niet zegt waarop hij rust, is een gok met decimalen.
        sel, venster = _filter_intensity(alle, leg.sport, "rustig", athlete)
        opslag = INTENSITY_SPEED_FACTOR.get(intensity, 1.0)
        terugval = (f"te weinig sessies op {intensity} — geschat vanaf zone 2 "
                    f"met {(opslag - 1) * 100:.0f}% opslag")
        if len(sel) < MIN_SESSIONS:
            sel, venster, opslag = alle, "alle recente sessies", 1.0
            terugval = (f"te weinig sessies op {intensity} én in zone 2 — "
                        f"geschat op alle recente sessies samen")

    if sel.empty:
        return LegEstimate(sport=leg.sport, low_s=0, mid_s=0, high_s=0,
                           distance_m=doel_m,
                           basis="geen vergelijkbare sessies in de database — "
                                 "vul de duur handmatig in")

    pct, uit_data = heat_pct_per_degree(alle)
    doel_factor = heat_speed_factor(temp_c, pct)

    tijden = []
    for _, r in sel.iterrows():
        # Twee correcties op de gemeten sessietijd, in deze volgorde:
        # 1. temperatuur — terug naar neutraal (delen door de eigen factor) en
        #    vooruit naar de verwachte temperatuur; plus de intensiteitsopslag;
        # 2. afstand — met Riegel van de sessieafstand naar de doelafstand.
        eigen_factor = heat_speed_factor(r.get("temperature_c"), pct)
        gecorrigeerd = r["seconds"] * eigen_factor / (doel_factor * opslag)
        tijden.append(_riegel_seconds(gecorrigeerd, r["distance_m"], doel_m))

    reeks = pd.Series(tijden, dtype=float)
    mid = float(reeks.median())
    low = float(reeks.quantile(0.25))
    high = float(reeks.quantile(0.75))

    # Verbreden: de spreiding van eigen sessies onderschat de onzekerheid van
    # één specifieke dag (wind, gevoel, parcours).
    verbreed = BAND_WIDEN.get(leg.sport, 0.02)
    low, high = low * (1 - verbreed), high * (1 + verbreed)
    minimum = MIN_HALF_BAND.get(leg.sport, 0.05) * mid
    low, high = min(low, mid - minimum), max(high, mid + minimum)

    onderbouwing = f"{len(sel)} sessie(s), {venster}" if venster else f"{len(sel)} sessie(s)"
    if terugval:
        onderbouwing += f"; {terugval}"
    if (1 - doel_factor) * 100 >= HEAT_NOTE_MIN_PCT:
        bron = "eigen sessies" if uit_data else "richtlijn"
        onderbouwing += (f"; {(1 - doel_factor) * 100:.1f}% trager door "
                         f"{temp_c:.0f} °C ({pct:.2f}%/°C uit {bron})")

    return LegEstimate(
        sport=leg.sport, low_s=low, mid_s=mid, high_s=high, distance_m=doel_m,
        n_sessions=len(sel), basis=onderbouwing, exact_intensity=exact,
        speed_ms=doel_m / mid if mid > 0 else None,
    )


def _estimate_swim(alle: pd.DataFrame, doel_m: float) -> LegEstimate:
    """Zwemschatting: bij voorkeur uit open water-sessies (trager dan het bad).

    Zwemmen kent bewust geen zone-oordeel, dus er is geen intensiteitsfilter;
    de schatting draait op het tempo per 100 m van recente sessies. Zijn er te
    weinig open water-sessies, dan wordt het badtempo met
    :data:`OPEN_WATER_FACTOR` afgewaardeerd.
    """
    if alle.empty:
        return LegEstimate(sport="swimming", low_s=0, mid_s=0, high_s=0,
                           distance_m=doel_m,
                           basis="geen zwemsessies in de database — "
                                 "vul de duur handmatig in")
    open_water = alle[alle.get("sub_sport", pd.Series(dtype=str)) == "open_water"] \
        if "sub_sport" in alle else pd.DataFrame()
    if len(open_water) >= 2:
        sel, factor, venster = open_water, 1.0, f"{len(open_water)} open water-sessie(s)"
    else:
        sel, factor = alle, OPEN_WATER_FACTOR
        venster = (f"{len(alle)} zwemsessie(s), overwegend bad — "
                   f"{(1 - OPEN_WATER_FACTOR) * 100:.0f}% trager gerekend voor open water")

    tijden = [_riegel_seconds(r["seconds"] / factor, r["distance_m"], doel_m)
              for _, r in sel.iterrows()]
    reeks = pd.Series(tijden, dtype=float)
    mid = float(reeks.median())
    low = float(reeks.quantile(0.25)) * (1 - BAND_WIDEN["swimming"])
    high = float(reeks.quantile(0.75)) * (1 + BAND_WIDEN["swimming"])
    minimum = MIN_HALF_BAND["swimming"] * mid
    low, high = min(low, mid - minimum), max(high, mid + minimum)
    return LegEstimate(sport="swimming", low_s=low, mid_s=mid, high_s=high,
                       distance_m=doel_m, n_sessions=len(sel), basis=venster,
                       speed_ms=doel_m / mid if mid > 0 else None)


def transition_seconds(conn: sqlite3.Connection, acts: pd.DataFrame) -> dict[str, float]:
    """De eigen wisseltijden (T1/T2) uit bevestigde combo's, met terugval.

    Faalt zacht: zonder combo-tabel of zonder bevestigde combo's komen de
    standaardwaarden uit :data:`DEFAULT_TRANSITION_S` terug.
    """
    uit = dict(DEFAULT_TRANSITION_S)
    try:
        from tricoach.combos import combo_history
        from tricoach.storage import load_records

        hist = combo_history(conn, acts, load_records)
    except Exception:
        return uit
    if hist is None or hist.empty:
        return uit
    for kolom, sleutel in (("t1_s", "T1"), ("t2_s", "T2")):
        if kolom in hist and hist[kolom].notna().any():
            uit[sleutel] = float(hist[kolom].dropna().median())
    return uit


def estimate_duration(conn: sqlite3.Connection, acts: pd.DataFrame, athlete: dict,
                      legs: list[LegRequest], intensity: str,
                      temp_c: float | None) -> DurationEstimate:
    """De complete duurschatting voor een sessie van één of meer onderdelen.

    De onderdelen worden opgeteld; bij twee of meer onderdelen komen de
    wisseltijden erbij (T1 zwem→fiets, T2 fiets→loop, zie
    :func:`transition_seconds`).
    """
    schatting = DurationEstimate()
    for leg in legs:
        schatting.legs.append(
            estimate_leg(conn, acts, athlete, leg, intensity, temp_c))

    if len(legs) >= 2:
        wissels = transition_seconds(conn, acts)
        sporten = [l.sport for l in legs]
        for van, naar, label in (("swimming", "cycling", "T1"),
                                 ("cycling", "running", "T2")):
            if van in sporten and naar in sporten:
                schatting.transitions.append((label, wissels[label]))
        if not schatting.transitions:
            schatting.transitions.append(
                ("wissel", DEFAULT_TRANSITION_S["T2"]))
        schatting.notes.append(
            "Wisseltijden uit je eigen bevestigde combo's "
            f"({', '.join(f'{l} {fmt_duration(s)}' for l, s in schatting.transitions)})."
        )
    return schatting
