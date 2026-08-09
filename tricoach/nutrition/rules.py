"""De sportvoedingsrichtlijnen als expliciete, testbare regels.

Dit is het *normatieve* hart van de voedingsplanner: alle drempelwaarden uit de
algemene sportvoedingsrichtlijnen staan hier als benoemde constante, met een
pure functie eromheen. Geen data, geen LLM, geen UI — zo is elke uitkomst van
de planner terug te voeren op één regel die je hier kunt nalezen en aanpassen.

De twee harde constraints die de planner moet bewaken:

**A. De single-source limiet.** Producten met alleen glucose/maltodextrine
gaan via één darmtransporter (SGLT1) naar binnen en lopen tegen een plafond
van ~60 g/uur; alles daarboven blijft in de darm. Een dual-source product
(glucose + fructose, ratio ~1:0,8) gebruikt daarnaast de fructosetransporter
(GLUT5) en haalt het plafond naar ~90 g/uur, bij een getrainde darm tot ~120.

**B. De concentratielimiet van drank.** Sportdrank is isotoon rond 6-8%
koolhydraten (~30-40 g per 500 ml). Meer koolhydraten in hetzelfde vocht maakt
de drank hypertoon: de maag leegt trager en de opname kan juist afnemen. De
planner rekent daarom eerst uit hoeveel koolhydraten er *passen* in het
geplande vochtvolume en verschuift de rest naar gels.

Zie ``memory/beslissingen.md`` voor de onderbouwing en de bronnen.
"""

from dataclasses import dataclass

# --------------------------------------------------------- koolhydraatbehoefte --

# Koolhydraten per uur op basis van de TOTALE duur van de sessie. Per rij:
# (bovengrens in minuten of None, lo g/uur, hi g/uur, label). De rijen zijn
# oplopend en de eerste rij waarvan de duur onder de bovengrens valt, wint.
CARB_TIERS: list[tuple[int | None, float, float, str]] = [
    (75, 0.0, 30.0, "< 75 min"),
    (150, 30.0, 60.0, "75-150 min"),
    (180, 60.0, 60.0, "2,5-3 uur"),
    (None, 60.0, 90.0, "> 3 uur"),
]

# Binnen de tier "> 3 uur" schuift het startadvies mee met de duur: een race
# van drie uur zit rond het midden van de band, een race van zes uur of langer
# aan de bovenkant. Reden: hoe langer de inspanning, hoe meer de *totale*
# substraatbehoefte gaat domineren over het risico van maagklachten.
LONG_ADVICE_START_H, LONG_ADVICE_END_H = 3.0, 6.0
LONG_ADVICE_LO_G_H, LONG_ADVICE_HI_G_H = 75.0, 90.0

# Onder deze duur is bijvoeden meestal niet nodig: de glycogeenvoorraad dekt
# de inspanning ruim.
NO_FUEL_BELOW_MIN = 75


@dataclass(frozen=True)
class CarbTarget:
    """De koolhydraatbehoefte voor één sessie.

    ``lo``/``hi`` zijn de grenzen van de richtlijn (g/uur) en ``advice`` is het
    concrete startpunt dat de planner gebruikt zolang de atleet niets anders
    invult. ``tier`` is het label van de gebruikte rij uit :data:`CARB_TIERS`.
    """

    lo_g_h: float
    hi_g_h: float
    advice_g_h: float
    tier: str

    @property
    def needed(self) -> bool:
        """Is bijvoeden voor deze duur überhaupt zinvol?"""
        return self.advice_g_h > 0

    def as_text(self) -> str:
        """Leesbare richtlijnregel, bijv. '> 3 uur → 60-90 g/uur'."""
        if self.lo_g_h == self.hi_g_h:
            band = f"~{self.hi_g_h:.0f} g/uur"
        else:
            band = f"{self.lo_g_h:.0f}-{self.hi_g_h:.0f} g/uur"
        return f"{self.tier} → {band}"


def carb_target(duration_s: float) -> CarbTarget:
    """De koolhydraatbehoefte (g/uur) bij een totale duur in seconden.

    Volgt :data:`CARB_TIERS`. Het startadvies is het midden van de band, met
    twee uitzonderingen: onder de 75 minuten is het advies 0 ("meestal niet
    nodig") en boven de drie uur schuift het advies met de duur mee van 75
    naar 90 g/uur (zie :data:`LONG_ADVICE_START_H`).
    """
    minutes = max(duration_s, 0) / 60
    for upper, lo, hi, label in CARB_TIERS:
        if upper is not None and minutes >= upper:
            continue
        if lo == 0.0:
            advice = 0.0
        elif upper is None:
            hours = minutes / 60
            fractie = (hours - LONG_ADVICE_START_H) / (LONG_ADVICE_END_H - LONG_ADVICE_START_H)
            fractie = min(max(fractie, 0.0), 1.0)
            ruw = LONG_ADVICE_LO_G_H + fractie * (LONG_ADVICE_HI_G_H - LONG_ADVICE_LO_G_H)
            advice = round(ruw / 5) * 5.0
        else:
            advice = round((lo + hi) / 2 / 5) * 5.0
        return CarbTarget(lo_g_h=lo, hi_g_h=hi, advice_g_h=advice, tier=label)
    raise AssertionError("CARB_TIERS moet met een open bovengrens eindigen")


# ------------------------------------------------------ constraint A: opname --

# Plafond (g/uur) van producten die alleen glucose/maltodextrine bevatten: één
# transporter, één plafond. Alles daarboven wordt niet opgenomen.
SINGLE_SOURCE_CAP_G_H = 60.0

# Met dual-source (glucose + fructose) komt de tweede transporter erbij.
DUAL_SOURCE_CAP_G_H = 90.0

# Met een bewust getrainde darm haalbaar; alleen na het in training te hebben
# getest, nooit als startpunt.
DUAL_SOURCE_TRAINED_CAP_G_H = 120.0

# De ratio waarop dual-source producten zijn gebouwd.
DUAL_SOURCE_RATIO = "~1:0,8 (glucose:fructose)"


def absorption_cap_g_h(has_dual_source: bool, trained_gut: bool = False) -> float:
    """Het opnameplafond (g/uur) dat bij deze productselectie hoort.

    ``has_dual_source`` is True zodra er minstens één dual-source product in
    de selectie zit. ``trained_gut`` tilt het dual-source plafond naar
    :data:`DUAL_SOURCE_TRAINED_CAP_G_H` — alleen aanzetten als hogere innames
    in training aantoonbaar goed vielen.
    """
    if not has_dual_source:
        return SINGLE_SOURCE_CAP_G_H
    return DUAL_SOURCE_TRAINED_CAP_G_H if trained_gut else DUAL_SOURCE_CAP_G_H


# ------------------------------------------------ constraint B: concentratie --

# Isotoon bereik van sportdrank in procenten koolhydraten (g per 100 ml).
ISOTONIC_PCT_LO, ISOTONIC_PCT_HI = 6.0, 8.0


def max_carbs_from_fluid_g(fluid_ml: float) -> float:
    """Hoeveel koolhydraten er maximaal in dit vochtvolume passen (isotoon).

    Boven deze grens wordt de drank hypertoon; de planner verschuift het
    overschot dan naar gels in plaats van de bidon zwaarder te maken.
    """
    return max(fluid_ml, 0.0) * ISOTONIC_PCT_HI / 100


def concentration_pct(carbs_g: float, fluid_ml: float) -> float | None:
    """De koolhydraatconcentratie (g per 100 ml) van een mengsel, of None
    zonder vocht."""
    if not fluid_ml or fluid_ml <= 0:
        return None
    return carbs_g / fluid_ml * 100


# ------------------------------------------------------------ vocht & natrium --

# Vochtinname per uur, lineair oplopend met de temperatuur tussen deze punten
# (°C → ml/uur). Buiten de punten wordt afgekapt.
FLUID_TEMP_POINTS: list[tuple[float, float]] = [(10.0, 400.0), (30.0, 800.0)]

# Temperatuur die wordt aangenomen als er geen weersverwachting is.
DEFAULT_TEMP_C = 18.0

# Natrium per liter vocht, lineair tussen deze punten (°C → mg/l).
SODIUM_TEMP_POINTS: list[tuple[float, float]] = [
    (15.0, 300.0), (25.0, 600.0), (32.0, 1000.0)]

# Boven de 600 mg/l gaan we alleen bij hitte én lange duur; korter dan dit
# blijft het advies op 600 mg/l staan.
SODIUM_LONG_DURATION_MIN = 180

# Onder dit percentage van het natriumdoel meldt de planner een tekort.
SODIUM_SHORTFALL_FRACTION = 0.7


def _interpolate(points: list[tuple[float, float]], x: float) -> float:
    """Lineaire interpolatie tussen (x, y)-punten, afgekapt op de randen."""
    if x <= points[0][0]:
        return points[0][1]
    if x >= points[-1][0]:
        return points[-1][1]
    for (x1, y1), (x2, y2) in zip(points, points[1:]):
        if x1 <= x <= x2:
            return y1 + (y2 - y1) * (x - x1) / (x2 - x1)
    return points[-1][1]


def fluid_ml_per_hour(temp_c: float | None) -> float:
    """Vochtadvies (ml/uur) bij deze temperatuur, afgerond op 50 ml.

    400 ml/uur bij koud weer oplopend tot 800 ml/uur vanaf 30 °C. Zonder
    temperatuur wordt :data:`DEFAULT_TEMP_C` aangenomen.
    """
    temp = DEFAULT_TEMP_C if temp_c is None else temp_c
    return round(_interpolate(FLUID_TEMP_POINTS, temp) / 50) * 50.0


def sodium_mg_per_liter(temp_c: float | None, duration_s: float) -> float:
    """Natriumadvies (mg per liter vocht) bij deze temperatuur en duur.

    300 mg/l bij gematigd weer, oplopend tot 600 mg/l rond 25 °C. Het bereik
    boven 600 mg/l (tot ~1000) geldt alleen bij hitte én een sessie langer dan
    :data:`SODIUM_LONG_DURATION_MIN` minuten.
    """
    temp = DEFAULT_TEMP_C if temp_c is None else temp_c
    mg = _interpolate(SODIUM_TEMP_POINTS, temp)
    if duration_s / 60 <= SODIUM_LONG_DURATION_MIN:
        mg = min(mg, 600.0)
    return round(mg / 25) * 25.0


# ------------------------------------------------------------------ cafeïne --

# Maximum cafeïne per race, per kilo lichaamsgewicht.
CAFFEINE_MAX_MG_PER_KG = 3.0

# Cafeïne hoort in de tweede helft: daar levert hij het meeste op (de
# waargenomen inspanning loopt dan op) en zit hij de maag het minst in de weg.
CAFFEINE_FROM_FRACTION = 0.5


def caffeine_cap_mg(weight_kg: float | None) -> float | None:
    """Het cafeïneplafond (mg) voor deze atleet, of None zonder gewicht."""
    if not weight_kg or weight_kg <= 0:
        return None
    return weight_kg * CAFFEINE_MAX_MG_PER_KG


# -------------------------------------------------------------------- timing --

# Eerste inname na deze tijd (minuten): niet meteen bij de start — de maag moet
# eerst wennen aan de inspanning. In T1 van een triatlon mag het wél direct:
# dat is het rustigste eetmoment van de hele race.
FIRST_INTAKE_RANGE_MIN = (20, 45)
FIRST_INTAKE_MIN = 30

# Daarna elke 20-30 minuten.
INTAKE_INTERVAL_RANGE_MIN = (20, 30)
INTAKE_INTERVAL_MIN = 25

# Vlak voor de finish heeft innemen geen zin meer.
LAST_INTAKE_BEFORE_END_MIN = 10

# Standaard bidoninhoud (ml) voor de meeneemlijst.
BOTTLE_ML = 750.0


# -------------------------------------------------------------- voorbehoud --

DISCLAIMER = (
    "Dit plan is een startpunt op basis van algemene sportvoedingsrichtlijnen, "
    "bedoeld om in training te testen — geen voedingsadvies. Bij twijfel of "
    "maagklachten: raadpleeg een sportdiëtist."
)
