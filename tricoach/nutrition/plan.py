"""De planner: van duurschatting + productselectie naar een concreet innameplan.

Alles in deze module is **deterministisch**. Dezelfde invoer geeft altijd
hetzelfde plan; er komt geen taalmodel aan te pas. De regels staan in
:mod:`tricoach.nutrition.rules`, de producten in
:mod:`tricoach.nutrition.products` en de duur in
:mod:`tricoach.nutrition.duration`.

De volgorde waarin de planner beslist — die volgorde is de kern, want de
constraints kunnen elkaar tegenspreken:

1. **Duur → behoefte.** De totale duur bepaalt de band g/uur en het startadvies.
2. **Opnameplafond.** Bevat de selectie alleen single-source producten, dan
   wordt het advies afgetopt op 60 g/uur — met melding, want dit is de meest
   voorkomende reden dat een plan op papier klopt en in de praktijk niet.
3. **Eetbare tijd.** Tijdens het zwemmen kun je niets innemen; de behoefte
   wordt over de fiets, de loop en de wissels verdeeld.
4. **Vocht eerst vullen.** Drank levert koolhydraten, vocht én natrium in één
   keer, dus die gaat voor — maar niet verder dan de isotone concentratie in
   het geplande vochtvolume toelaat.
5. **Rest naar gels.** Wat niet in de bidon past, wordt vast/gel. Kan dat niet
   (geen gel geselecteerd), dan haalt het plan het doel niet en zegt dat.
6. **Timing.** Eerste inname na ~30 min, daarna elke 20-30 min, wissels als
   vaste eetmomenten, cafeïne alleen in de tweede helft en binnen 3 mg/kg.
7. **Controle achteraf.** Natriumtekort, cafeïneplafond en innamedichtheid
   worden op het gebouwde plan nagerekend, niet vooraf aangenomen.
"""

import math
from dataclasses import dataclass, field
from datetime import date

from tricoach.formatting import fmt_duration
from tricoach.nutrition import rules
from tricoach.nutrition.duration import DurationEstimate, LegRequest
from tricoach.nutrition.products import (
    KIND_DRINK,
    SOURCE_DUAL,
    Product,
    has_dual_source,
)

# Sporten waarbij innemen tijdens het onderdeel niet gaat.
NON_FEEDABLE_SPORTS = ("swimming",)

INTENSITIES = ("rustig", "racetempo", "hard")
INTENSITY_LABEL = {
    "rustig": "rustig (zone 2)",
    "racetempo": "racetempo",
    "hard": "hard",
}

# Samengestelde sessietypes en de onderdelen waaruit ze bestaan.
SESSION_TYPES = {
    "running": ("Hardlopen", ["running"]),
    "cycling": ("Fietsen", ["cycling"]),
    "brick": ("Brick (fiets + loop)", ["cycling", "running"]),
    "triathlon": ("Triatlon (zwem + fiets + loop)", ["swimming", "cycling", "running"]),
}

SEVERITY_WARNING, SEVERITY_INFO = "waarschuwing", "info"


@dataclass
class AidStation:
    """Een verzorgingspost: op welk onderdeel en op welke kilometer."""

    leg_index: int
    km: float


@dataclass
class PlanRequest:
    """Alles wat de atleet invult voordat de planner gaat rekenen."""

    session_type: str = "cycling"
    legs: list[LegRequest] = field(default_factory=list)
    intensity: str = "rustig"
    temp_c: float | None = None
    product_names: list[str] = field(default_factory=list)
    aid_stations: list[AidStation] = field(default_factory=list)
    weight_kg: float | None = None
    target_g_h: float | None = None
    trained_gut: bool = False
    bottle_ml: float = rules.BOTTLE_ML
    override_duration_s: float | None = None
    planned_date: date | None = None
    name: str = ""

    @property
    def type_label(self) -> str:
        return SESSION_TYPES.get(self.session_type, (self.session_type, []))[0]


@dataclass
class Segment:
    """Eén blok van de sessie: een onderdeel of een wissel."""

    label: str
    sport: str | None
    start_s: float
    end_s: float
    feedable: bool
    priority: bool = False
    distance_m: float | None = None

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s

    @property
    def speed_ms(self) -> float | None:
        if not self.distance_m or self.duration_s <= 0:
            return None
        return self.distance_m / self.duration_s


@dataclass
class IntakeEvent:
    """Eén inname op de tijdlijn."""

    t_s: float
    segment: str
    product: str
    amount: str
    carbs_g: float
    sodium_mg: float = 0.0
    caffeine_mg: float = 0.0
    km: float | None = None
    note: str = ""
    cumulative_carbs_g: float = 0.0

    @property
    def time_label(self) -> str:
        return fmt_duration(self.t_s)


@dataclass
class DrinkPlan:
    """De drankcomponent: hoeveel, hoe geconcentreerd en hoe verdeeld."""

    product: Product | None = None
    servings: float = 0.0
    carbs_g: float = 0.0
    sodium_mg: float = 0.0
    fluid_ml: float = 0.0
    ml_per_hour: float = 0.0

    @property
    def concentration_pct(self) -> float | None:
        return rules.concentration_pct(self.carbs_g, self.fluid_ml)


@dataclass
class CarryItem:
    """Eén regel van de meeneemlijst."""

    label: str
    amount: str
    detail: str = ""


@dataclass
class PlanWarning:
    """Een melding waar een constraint knelt."""

    code: str
    text: str
    severity: str = SEVERITY_WARNING

    @property
    def icon(self) -> str:
        return "⚠️" if self.severity == SEVERITY_WARNING else "ℹ️"


@dataclass
class NutritionPlan:
    """Het complete plan: tijdlijn, meeneemlijst, totalen en waarschuwingen."""

    request: PlanRequest
    duration: DurationEstimate
    total_s: float
    feedable_s: float
    target: rules.CarbTarget
    requested_g_h: float
    planned_g_h: float
    cap_g_h: float
    segments: list[Segment] = field(default_factory=list)
    events: list[IntakeEvent] = field(default_factory=list)
    drink: DrinkPlan = field(default_factory=DrinkPlan)
    carry: list[CarryItem] = field(default_factory=list)
    warnings: list[PlanWarning] = field(default_factory=list)
    totals: dict = field(default_factory=dict)

    @property
    def disclaimer(self) -> str:
        return rules.DISCLAIMER

    def warning_texts(self) -> list[str]:
        return [f"{w.icon} {w.text}" for w in self.warnings]

    def summary_text(self) -> str:
        """Compacte samenvatting, voor het logboek en de LLM-toelichting."""
        t = self.totals
        intensiteit = INTENSITY_LABEL.get(self.request.intensity, self.request.intensity)
        kop = f"Sessie: {self.request.type_label}, {intensiteit}"
        if self.request.temp_c is not None:
            kop += f", {self.request.temp_c:.0f} °C"
        regels = [
            kop,
            f"Duur: {self.duration.range_text()} (gerekend met {fmt_duration(self.total_s)})",
            f"Richtlijn: {self.target.as_text()}; plan: {self.planned_g_h:.0f} g/uur "
            f"(plafond {self.cap_g_h:.0f} g/uur)",
            f"Totaal: {t.get('carbs_g', 0):.0f} g koolhydraten, "
            f"{t.get('fluid_ml', 0):.0f} ml vocht, "
            f"{t.get('sodium_mg', 0):.0f} mg natrium, "
            f"{t.get('caffeine_mg', 0):.0f} mg cafeïne",
        ]
        regels += [f"Let op: {w.text}" for w in self.warnings
                   if w.severity == SEVERITY_WARNING]
        return "\n".join(regels)


# ------------------------------------------------------------------ segmenten --

def build_segments(duration: DurationEstimate, total_s: float) -> list[Segment]:
    """Zet de duurschatting om in opeenvolgende blokken op één tijdlijn.

    De onderdelen staan in racevolgorde met de wissels ertussen. ``total_s``
    kan afwijken van de som van de schattingen (handmatige overschrijving); de
    blokken worden dan evenredig geschaald, zodat de tijdlijn op het gekozen
    totaal uitkomt.
    """
    ruw: list[tuple[str, str | None, float, bool, bool, float | None]] = []
    wissels = list(duration.transitions)
    for i, leg in enumerate(duration.legs):
        ruw.append((leg.label, leg.sport, leg.mid_s, leg.sport not in NON_FEEDABLE_SPORTS,
                    False, leg.distance_m))
        if i < len(duration.legs) - 1 and wissels:
            label, seconden = wissels.pop(0)
            ruw.append((label, None, seconden, True, True, None))

    som = sum(r[2] for r in ruw)
    schaal = (total_s / som) if som > 0 else 1.0
    segmenten, t = [], 0.0
    for label, sport, seconden, feedable, priority, afstand in ruw:
        duur = seconden * schaal
        segmenten.append(Segment(label=label, sport=sport, start_s=t, end_s=t + duur,
                                 feedable=feedable, priority=priority,
                                 distance_m=afstand))
        t += duur
    return segmenten


def _segment_at(segments: list[Segment], t_s: float) -> Segment | None:
    """Het blok waarin dit tijdstip valt."""
    for seg in segments:
        if seg.start_s <= t_s < seg.end_s:
            return seg
    return segments[-1] if segments else None


def _km_at(segments: list[Segment], t_s: float) -> float | None:
    """De kilometerstand binnen het eigen onderdeel op dit tijdstip."""
    seg = _segment_at(segments, t_s)
    if seg is None or seg.speed_ms is None:
        return None
    return seg.speed_ms * (t_s - seg.start_s) / 1000


# -------------------------------------------------------------------- slots --

def candidate_slots(segments: list[Segment], total_s: float,
                    interval_min: float, first_intake_min: float) -> list[float]:
    """De momenten waarop innemen kan, op volgorde.

    Wissels zijn altijd een moment (de makkelijkste eetmomenten van een race);
    daarbuiten een raster van ``interval_min`` binnen de eetbare blokken, met
    de eerste inname niet vóór ``first_intake_min`` en de laatste niet in de
    slotminuten (zie :data:`rules.LAST_INTAKE_BEFORE_END_MIN`).
    """
    einde = total_s - rules.LAST_INTAKE_BEFORE_END_MIN * 60
    slots: list[float] = []

    for seg in segments:
        if not seg.feedable:
            continue
        if seg.priority:
            # Midden in de wissel: rustig moment, sta of loop je toch.
            slots.append(seg.start_s + seg.duration_s / 2)
            continue
        t = max(seg.start_s, first_intake_min * 60)
        # Raster binnen dit blok, uitgelijnd op de start van het blok.
        while t <= min(seg.end_s, einde):
            slots.append(t)
            t += interval_min * 60

    return _enforce_spacing(sorted(set(float(round(s)) for s in slots)),
                            {float(round(s.start_s + s.duration_s / 2))
                             for s in segments if s.priority})


def _enforce_spacing(slots: list[float], priority: set[float]) -> list[float]:
    """Haal momenten weg die te dicht op het vorige zitten.

    Nodig omdat een wissel en het raster van het volgende blok vlak na elkaar
    kunnen vallen: zonder deze stap staat er een gel in T1 en twee minuten
    later nog een op de fiets. Een wisselmoment wint altijd — dat is het beste
    eetmoment dat er is — en duwt een te dicht ervóór liggend gewoon moment weg.
    """
    minimaal = rules.INTAKE_INTERVAL_RANGE_MIN[0] * 60
    gehouden: list[float] = []
    for t in slots:
        if gehouden and t - gehouden[-1] < minimaal:
            if t in priority and gehouden[-1] not in priority:
                gehouden[-1] = t  # de wissel verdringt het gewone moment
            continue
        gehouden.append(t)
    return gehouden


def _spread(slots: list[float], n: int, priority: set[float]) -> list[float]:
    """Kies ``n`` momenten uit de beschikbare slots, zo gelijkmatig mogelijk.

    Wisselmomenten (``priority``) worden altijd meegenomen: die zijn te goed om
    over te slaan. De rest wordt niet om-en-om uit de lijst geplukt maar op
    *tijd* verdeeld — er worden gelijkmatig verdeelde streeftijden over de
    sessie gelegd en per streeftijd het dichtstbijzijnde vrije moment gekozen.
    Zo ontstaan er geen gaten van meer dan een uur naast twee momenten kort na
    elkaar, wat er bij verdelen over de lijst wél gebeurt (de momenten liggen
    niet gelijkmatig over de tijd: een wissel levert er één midden in het veld).
    """
    if n <= 0 or not slots:
        return []
    if n >= len(slots):
        return list(slots)

    gekozen = [s for s in slots if s in priority][:n]
    vrij = [s for s in slots if s not in gekozen]
    te_kiezen = n - len(gekozen)
    if te_kiezen <= 0 or not vrij:
        return sorted(gekozen)

    # Streeftijden over het hele venster; de momenten die al vastliggen (de
    # wissels) claimen elk de streeftijd waar ze het dichtst bij zitten.
    start, eind = slots[0], slots[-1]
    doelen = [start + (eind - start) * i / max(n - 1, 1) for i in range(n)]
    for vast in gekozen:
        doelen.remove(min(doelen, key=lambda d: abs(d - vast)))

    for doel in doelen[:te_kiezen]:
        if not vrij:
            break
        beste = min(vrij, key=lambda s: abs(s - doel))
        gekozen.append(beste)
        vrij.remove(beste)
    return sorted(gekozen)


# ------------------------------------------------------------------ planner --

def _pick_drink(selection: list[Product]) -> Product | None:
    """De drank waarmee het plan rekent: dual-source gaat voor, dan de meeste
    koolhydraten per portie (minder sachets mee)."""
    dranken = [p for p in selection if p.kind == KIND_DRINK and p.carbs_g > 0]
    if not dranken:
        return None
    return sorted(dranken,
                  key=lambda p: (p.effective_source != SOURCE_DUAL, -p.carbs_g))[0]


def _pick_fuel(selection: list[Product]) -> Product | None:
    """Het hoofd-vast/gelproduct: het meeste koolhydraten per eenheid zonder
    cafeïne — dan hoeven er de minste mee en blijft cafeïne stuurbaar."""
    vast = [p for p in selection if p.kind != KIND_DRINK and p.carbs_g > 0]
    if not vast:
        return None
    zonder_caf = [p for p in vast if not p.caffeine_mg]
    kandidaten = zonder_caf or vast
    return sorted(kandidaten, key=lambda p: -p.carbs_g)[0]


def _pick_caffeine(selection: list[Product]) -> Product | None:
    """Het cafeïneproduct met de meeste koolhydraten per eenheid, of None."""
    met_caf = [p for p in selection
               if p.kind != KIND_DRINK and p.caffeine_mg > 0 and p.carbs_g > 0]
    if not met_caf:
        return None
    return sorted(met_caf, key=lambda p: -p.carbs_g)[0]


def build_plan(request: PlanRequest, selection: list[Product],
               duration: DurationEstimate) -> NutritionPlan:
    """Bouw het volledige voedingsplan. Puur rekenwerk, geen taalmodel.

    ``selection`` zijn de producten die de atleet heeft aangevinkt;
    ``duration`` is de schatting uit :mod:`tricoach.nutrition.duration`. Een
    handmatige duur (``request.override_duration_s``) wint van de schatting.
    """
    total_s = request.override_duration_s or duration.mid_s
    total_s = max(float(total_s or 0), 0.0)
    # Zonder onderdelen (alleen een handmatige duur) is de hele sessie één
    # eetbaar blok; anders kan er geen enkel innamemoment worden gevonden.
    segments = (build_segments(duration, total_s) if duration.legs else
                [Segment(label="Sessie", sport=None, start_s=0.0, end_s=total_s,
                         feedable=True)])
    feedable_s = sum(s.duration_s for s in segments if s.feedable) or total_s

    target = rules.carb_target(total_s)
    requested = request.target_g_h if request.target_g_h is not None else target.advice_g_h
    cap = rules.absorption_cap_g_h(has_dual_source(selection), request.trained_gut)

    plan = NutritionPlan(
        request=request, duration=duration, total_s=total_s,
        feedable_s=feedable_s, target=target, requested_g_h=float(requested),
        planned_g_h=float(requested), cap_g_h=cap, segments=segments,
    )

    if total_s <= 0:
        plan.warnings.append(PlanWarning(
            "geen_duur", "Zonder duur of afstand valt er niets te plannen."))
        plan.totals = _empty_totals()
        return plan

    _apply_absorption_cap(plan, selection)
    _plan_fluid_and_drink(plan, selection)
    _plan_solids(plan, selection)
    _check_sodium(plan)
    _build_carry_list(plan, selection)
    _finalize_totals(plan, selection)
    _add_context_notes(plan, duration)
    return plan


def _empty_totals() -> dict:
    return {"carbs_g": 0.0, "fluid_ml": 0.0, "sodium_mg": 0.0, "caffeine_mg": 0.0,
            "carbs_per_hour": 0.0, "carbs_per_feedable_hour": 0.0}


# ------------------------------------------------------ stap 2: opnameplafond --

def _apply_absorption_cap(plan: NutritionPlan, selection: list[Product]) -> None:
    """Constraint A: top het advies af op het opnameplafond van de selectie."""
    if not selection:
        if plan.target.needed:
            plan.warnings.append(PlanWarning(
                "geen_producten",
                "Er is geen enkel product geselecteerd, dus er valt niets in te "
                "plannen. Vink hierboven aan wat je bij je hebt."))
        return

    if plan.requested_g_h <= plan.cap_g_h:
        return

    plan.planned_g_h = plan.cap_g_h
    if plan.cap_g_h == rules.SINGLE_SOURCE_CAP_G_H:
        plan.warnings.append(PlanWarning(
            "single_source_plafond",
            f"Je selectie bevat alleen single-source producten "
            f"(glucose/maltodextrine). Die gaan via één darmtransporter naar "
            f"binnen en lopen vast op ~{rules.SINGLE_SOURCE_CAP_G_H:.0f} g/uur — "
            f"de gevraagde {plan.requested_g_h:.0f} g/uur wordt niet opgenomen. "
            f"Het plan is afgetopt op {plan.cap_g_h:.0f} g/uur. Wil je hoger: "
            f"voeg een dual-source product toe (glucose + fructose, ratio "
            f"{rules.DUAL_SOURCE_RATIO}); daarmee kan het plan naar "
            f"{rules.DUAL_SOURCE_CAP_G_H:.0f} g/uur."))
    else:
        plan.warnings.append(PlanWarning(
            "dual_source_plafond",
            f"{plan.requested_g_h:.0f} g/uur ligt boven het dual-source plafond "
            f"van {plan.cap_g_h:.0f} g/uur; het plan is daarop afgetopt. Meer dan "
            f"{rules.DUAL_SOURCE_CAP_G_H:.0f} g/uur vraagt een getrainde darm en "
            f"hoort eerst in training getest te zijn."))


# --------------------------------------------- stap 4: vocht en drankvulling --

def _plan_fluid_and_drink(plan: NutritionPlan, selection: list[Product]) -> None:
    """Constraint B: vul zoveel mogelijk uit drank, binnen de concentratiegrens."""
    uren = plan.feedable_s / 3600
    ml_per_uur = rules.fluid_ml_per_hour(plan.request.temp_c)
    totaal_vocht = ml_per_uur * uren
    plan.drink.ml_per_hour = ml_per_uur
    plan.drink.fluid_ml = totaal_vocht

    nodig = plan.planned_g_h * uren
    drank = _pick_drink(selection)
    if drank is None or nodig <= 0:
        return

    # Een drank met een eigen fabrikantopgave (koolhydraten per aanbevolen
    # mengvolume) gebruikt die eigen, eventueel hypertone, verhouding; alleen
    # zonder die opgave geldt de algemene isotone grens.
    max_pct = rules.concentration_cap_pct(drank.carbs_g, drank.serving_ml)
    eigen_opgave = bool(drank.carbs_g and drank.serving_ml)
    maximaal_uit_vocht = rules.max_carbs_from_fluid_g(totaal_vocht, max_pct)
    gewenst = min(nodig, maximaal_uit_vocht)
    porties = math.floor(gewenst / drank.carbs_g) if drank.carbs_g else 0
    if porties == 0 and drank.carbs_g <= maximaal_uit_vocht and nodig > 0:
        porties = 1
    if porties <= 0:
        return

    plan.drink.product = drank
    plan.drink.servings = float(porties)
    plan.drink.carbs_g = porties * drank.carbs_g
    plan.drink.sodium_mg = porties * drank.sodium_mg

    if nodig > maximaal_uit_vocht + 1e-6:
        tekort = nodig - plan.drink.carbs_g
        heeft_gel = any(p.kind != KIND_DRINK and p.carbs_g > 0 for p in selection)
        concentratie = rules.concentration_pct(nodig, totaal_vocht) or 0
        grens_tekst = (f"de aanbevolen verhouding van {drank.name} ({max_pct:.0f}%)"
                        if eigen_opgave else
                        f"de isotone {rules.ISOTONIC_PCT_LO:.0f}-"
                        f"{rules.ISOTONIC_PCT_HI:.0f}%")
        basis = (
            f"{nodig:.0f} g koolhydraten past niet in de geplande {totaal_vocht:.0f} ml "
            f"vocht: dat zou {concentratie:.1f}% worden, boven {grens_tekst}. Te "
            f"geconcentreerde drank verlaat de maag trager, dus meer poeder in "
            f"dezelfde bidon levert juist mínder opname."
        )
        if heeft_gel:
            plan.warnings.append(PlanWarning(
                "concentratie",
                basis + f" De drank is daarom begrensd op {plan.drink.carbs_g:.0f} g "
                        f"({plan.drink.concentration_pct:.1f}%) en de resterende "
                        f"{tekort:.0f} g is naar gels verschoven."))
        else:
            plan.warnings.append(PlanWarning(
                "concentratie_geen_gel",
                basis + f" De drank is begrensd op {plan.drink.carbs_g:.0f} g "
                        f"({plan.drink.concentration_pct:.1f}%), maar er is geen gel of "
                        f"vast product geselecteerd om de resterende {tekort:.0f} g "
                        f"naar te verschuiven. Voeg een gel toe of accepteer een "
                        f"lagere inname."))


# --------------------------------------------------- stap 5 + 6: gels en timing --

def _plan_solids(plan: NutritionPlan, selection: list[Product]) -> None:
    """Verdeel de resterende koolhydraten over gels/vast voedsel en zet ze op tijd."""
    uren = plan.feedable_s / 3600
    nodig = plan.planned_g_h * uren - plan.drink.carbs_g
    brandstof = _pick_fuel(selection)
    if nodig <= 0 or brandstof is None:
        return

    # Naar boven afronden op hele eenheden — je neemt geen halve gel — maar
    # nooit tot voorbij het opnameplafond: dat is een harde grens, geen doel.
    # Wat er niet meer bij past, laten we liever liggen dan dat het onopgenomen
    # in de darm blijft.
    eenheden = max(1, math.ceil(nodig / brandstof.carbs_g))
    plafond_g = plan.cap_g_h * uren - plan.drink.carbs_g
    if eenheden * brandstof.carbs_g > plafond_g:
        eenheden = int(plafond_g // brandstof.carbs_g)
    if eenheden <= 0:
        plan.warnings.append(PlanWarning(
            "gel_past_niet",
            f"Eén {brandstof.name} ({brandstof.carbs_g:.0f} g) past er naast de "
            f"drank niet meer bij binnen het plafond van {plan.cap_g_h:.0f} g/uur.",
            SEVERITY_INFO))
        return

    # Slots zoeken; wordt het te krap, dan naar het kortste toegestane interval.
    interval = rules.INTAKE_INTERVAL_MIN
    eerste = rules.FIRST_INTAKE_MIN
    slots = candidate_slots(plan.segments, plan.total_s, interval, eerste)
    if len(slots) < eenheden:
        interval = rules.INTAKE_INTERVAL_RANGE_MIN[0]
        slots = candidate_slots(plan.segments, plan.total_s, interval, eerste)

    prioriteit = {s.start_s + s.duration_s / 2 for s in plan.segments if s.priority}
    prioriteit = {float(round(p)) for p in prioriteit}

    if not slots:
        plan.warnings.append(PlanWarning(
            "geen_momenten",
            "De sessie is te kort voor een innamemoment na de opstartfase; neem "
            "eventueel vlak voor de start iets."))
        return

    if eenheden > len(slots):
        plan.warnings.append(PlanWarning(
            "innamedichtheid",
            f"Er zijn {eenheden} eenheden {brandstof.name} nodig maar slechts "
            f"{len(slots)} momenten van minstens "
            f"{rules.INTAKE_INTERVAL_RANGE_MIN[0]} minuten uit elkaar. Het plan "
            f"legt er meerdere op één moment; overweeg een product met meer "
            f"koolhydraten per eenheid of een lager doel."))

    momenten = _spread(slots, min(eenheden, len(slots)), prioriteit)
    per_moment = [1] * len(momenten)
    for i in range(eenheden - len(momenten)):
        per_moment[i % len(per_moment)] += 1

    cafeine = _pick_caffeine(selection)
    plafond = rules.caffeine_cap_mg(plan.request.weight_kg)
    caf_gepland = 0.0
    tweede_helft_vanaf = plan.total_s * rules.CAFFEINE_FROM_FRACTION

    # Cafeïne toewijzen: zo laat mogelijk, zolang het cumulatieve totaal onder
    # het plafond blijft. Gekozen uit het volledige rooster van momenten
    # (``slots``), niet alleen uit de momenten die de koolhydraatplanning
    # toevallig al koos — anders sneuvelt een gel onterecht zodra er maar één
    # moment nodig is en dat toevallig in de eerste helft valt, ook al past
    # de cafeïne zelf ruim binnen het plafond.
    caf_momenten: dict[float, int] = {}
    if cafeine is not None and cafeine.caffeine_mg > 0:
        if plafond is None:
            plan.warnings.append(PlanWarning(
                "cafeine_geen_gewicht",
                "Zonder lichaamsgewicht kan het cafeïneplafond (3 mg/kg) niet "
                "worden berekend; cafeïnegels zijn daarom niet ingepland.",
                SEVERITY_INFO))
        else:
            max_caf_eenheden = int(plafond // cafeine.caffeine_mg)
            tweede_helft_slots = sorted(
                (s for s in slots if s >= tweede_helft_vanaf), reverse=True)
            te_plannen = min(max_caf_eenheden, len(tweede_helft_slots), eenheden)
            for moment in tweede_helft_slots[:te_plannen]:
                if moment not in momenten:
                    # Geen bestaand innamemoment op deze tijd: verplaats het
                    # dichtstbijzijnde moment uit de eerste helft hierheen, zodat
                    # het totaal aantal eenheden (en dus de koolhydraten) gelijk
                    # blijft.
                    eerste_helft = [m for m in momenten if m < tweede_helft_vanaf]
                    if not eerste_helft:
                        continue
                    te_verplaatsen = min(eerste_helft, key=lambda m: abs(m - moment))
                    momenten[momenten.index(te_verplaatsen)] = moment
                caf_momenten[moment] = caf_momenten.get(moment, 0) + 1
                caf_gepland += cafeine.caffeine_mg
            if caf_gepland == 0:
                if not tweede_helft_slots:
                    plan.warnings.append(PlanWarning(
                        "cafeine_geen_moment",
                        "Er is geen innamemoment in de tweede helft van de sessie; "
                        "cafeïne is daarom niet ingepland.",
                        SEVERITY_INFO))
                else:
                    plan.warnings.append(PlanWarning(
                        "cafeine_past_niet",
                        f"Eén {cafeine.name} levert {cafeine.caffeine_mg:.0f} mg cafeïne "
                        f"en dat past niet binnen je plafond van {plafond:.0f} mg "
                        f"({rules.CAFFEINE_MAX_MG_PER_KG:.0f} mg/kg). Niet ingepland.",
                        SEVERITY_INFO))

    for moment, aantal in zip(momenten, per_moment):
        n_caf = caf_momenten.get(moment, 0)
        for product, n in ((cafeine, n_caf), (brandstof, aantal - n_caf)):
            if n <= 0 or product is None:
                continue
            seg = _segment_at(plan.segments, moment)
            notities = []
            if seg is not None and seg.priority:
                notities.append("wissel — rustigste eetmoment")
            if product.caffeine_mg:
                notities.append("cafeïne: bewust in de tweede helft")
            plan.events.append(IntakeEvent(
                t_s=moment,
                segment=seg.label if seg else "",
                product=product.name,
                amount=f"{n}× {product.unit_label}",
                carbs_g=n * product.carbs_g,
                sodium_mg=n * product.sodium_mg,
                caffeine_mg=n * product.caffeine_mg,
                km=_km_at(plan.segments, moment),
                note="; ".join(notities),
            ))

    plan.events.sort(key=lambda e: (e.t_s, e.product))


# ------------------------------------------------------ stap 7: natriumcheck --

def _check_sodium(plan: NutritionPlan) -> None:
    """Levert de selectie genoeg natrium voor deze temperatuur en duur?"""
    liter = plan.drink.fluid_ml / 1000
    if liter <= 0:
        return
    doel_per_l = rules.sodium_mg_per_liter(plan.request.temp_c, plan.total_s)
    doel = doel_per_l * liter
    geleverd = plan.drink.sodium_mg + sum(e.sodium_mg for e in plan.events)
    plan.totals["sodium_target_mg"] = doel
    plan.totals["sodium_mg_per_l_target"] = doel_per_l
    if doel <= 0:
        return
    if geleverd < doel * rules.SODIUM_SHORTFALL_FRACTION:
        tekort = doel - geleverd
        plan.warnings.append(PlanWarning(
            "natrium_laag",
            f"Je producten leveren {geleverd:.0f} mg natrium, terwijl bij "
            f"{plan.request.temp_c if plan.request.temp_c is not None else rules.DEFAULT_TEMP_C:.0f} °C "
            f"en deze duur ~{doel:.0f} mg past ({doel_per_l:.0f} mg/l over "
            f"{liter:.1f} l). Tekort ~{tekort:.0f} mg: overweeg een zouttablet, "
            f"een natriumrijkere drank of een snufje zout in de bidon."))


# ------------------------------------------------------------ meeneemlijst --

def _build_carry_list(plan: NutritionPlan, selection: list[Product]) -> None:
    """Wat gaat er mee: gels per soort, bidons, poeder — en wat je onderweg vult."""
    per_product: dict[str, float] = {}
    for e in plan.events:
        aantal = float(e.amount.split("×")[0])
        per_product[e.product] = per_product.get(e.product, 0) + aantal
    namen = {p.name: p for p in selection}
    for naam, aantal in per_product.items():
        product = namen.get(naam)
        eenheid = product.unit_label if product else "stuks"
        detail = product.as_text() if product else ""
        plan.carry.append(CarryItem(
            label=naam, amount=f"{aantal:.0f}× {eenheid}", detail=detail))

    if plan.drink.fluid_ml <= 0:
        return

    bottle_ml = plan.request.bottle_ml or rules.BOTTLE_ML
    bidons = max(1, math.ceil(plan.drink.fluid_ml / bottle_ml))
    per_bidon_g = plan.drink.carbs_g / bidons if bidons else 0
    per_bidon_ml = plan.drink.fluid_ml / bidons
    concentratie = rules.concentration_pct(per_bidon_g, per_bidon_ml)

    if plan.drink.product is not None and plan.drink.servings > 0:
        eigen_opgave = bool(plan.drink.product.carbs_g and plan.drink.product.serving_ml)
        grens_tekst = (f"aanbevolen verhouding is "
                        f"{rules.concentration_cap_pct(plan.drink.product.carbs_g, plan.drink.product.serving_ml):.0f}%"
                        if eigen_opgave else
                        f"isotoon is {rules.ISOTONIC_PCT_LO:.0f}-{rules.ISOTONIC_PCT_HI:.0f}%")
        plan.carry.append(CarryItem(
            label=plan.drink.product.name,
            amount=f"{plan.drink.servings:.0f}× {plan.drink.product.unit_label}",
            detail=(f"totaal {plan.drink.carbs_g:.0f} g koolhydraten, opgelost in "
                    f"{plan.drink.fluid_ml:.0f} ml "
                    f"({plan.drink.concentration_pct:.1f}% — {grens_tekst})"),
        ))

    posten = _stations_on_route(plan)
    mee = bidons if not posten else max(1, math.ceil(bidons / (len(posten) + 1)))
    detail = (f"{per_bidon_ml:.0f} ml per bidon"
              + (f" met ~{per_bidon_g:.0f} g koolhydraten "
                 f"({concentratie:.1f}%)" if per_bidon_g else " water"))
    if posten:
        detail += (f"; {bidons - mee} keer bijvullen onderweg "
                   f"({', '.join(posten)})")
    elif bidons > 2:
        detail += ("; dat is meer dan er op de fiets past — plan bijvullen of "
                   "geef verzorgingsposten op")
    plan.carry.append(CarryItem(
        label="Bidons", amount=f"{mee}× {bottle_ml:.0f} ml mee", detail=detail))


def _stations_on_route(plan: NutritionPlan) -> list[str]:
    """Leesbare beschrijving van de verzorgingsposten, op volgorde."""
    uit = []
    onderdelen = [s for s in plan.segments if s.sport]
    for post in sorted(plan.request.aid_stations, key=lambda a: (a.leg_index, a.km)):
        if 0 <= post.leg_index < len(onderdelen):
            uit.append(f"{onderdelen[post.leg_index].label.lower()} km {post.km:g}")
        else:
            uit.append(f"km {post.km:g}")
    return uit


# ------------------------------------------------------------------ totalen --

def _finalize_totals(plan: NutritionPlan, selection: list[Product]) -> None:
    """Tel het plan op en zet het lopende koolhydraattotaal op de tijdlijn.

    Het lopende totaal telt de gels op het moment zelf mee en de drank naar rato
    van de verstreken eetbare tijd — je drinkt immers continu, niet in porties.
    """
    vast_kh = sum(e.carbs_g for e in plan.events)
    totaal_kh = vast_kh + plan.drink.carbs_g
    drank_per_s = plan.drink.carbs_g / plan.feedable_s if plan.feedable_s else 0.0

    lopend = 0.0
    for e in plan.events:
        lopend += e.carbs_g
        e.cumulative_carbs_g = lopend + drank_per_s * _feedable_elapsed(plan, e.t_s)

    uren = plan.total_s / 3600
    eet_uren = plan.feedable_s / 3600
    plan.totals.update({
        "carbs_g": totaal_kh,
        "carbs_from_drink_g": plan.drink.carbs_g,
        "carbs_from_solids_g": vast_kh,
        "fluid_ml": plan.drink.fluid_ml,
        "sodium_mg": plan.drink.sodium_mg + sum(e.sodium_mg for e in plan.events),
        "caffeine_mg": sum(e.caffeine_mg for e in plan.events),
        "carbs_per_hour": totaal_kh / uren if uren else 0.0,
        "carbs_per_feedable_hour": totaal_kh / eet_uren if eet_uren else 0.0,
        "fluid_ml_per_hour": plan.drink.ml_per_hour,
    })

    plafond = rules.caffeine_cap_mg(plan.request.weight_kg)
    if plafond is not None and plan.totals["caffeine_mg"] > plafond:
        plan.warnings.append(PlanWarning(
            "cafeine_plafond",
            f"Het plan komt op {plan.totals['caffeine_mg']:.0f} mg cafeïne, boven "
            f"je plafond van {plafond:.0f} mg "
            f"({rules.CAFFEINE_MAX_MG_PER_KG:.0f} mg/kg bij "
            f"{plan.request.weight_kg:.0f} kg). Laat een cafeïnegel weg."))

    # Zonder producten is "het doel niet gehaald" geen extra informatie: dat
    # zegt de melding over de lege selectie al.
    haalbaar = plan.planned_g_h * eet_uren
    if selection and haalbaar > 0 and totaal_kh < haalbaar * 0.9:
        plan.warnings.append(PlanWarning(
            "doel_niet_gehaald",
            f"Het plan levert {totaal_kh:.0f} g terwijl {haalbaar:.0f} g het doel "
            f"was ({plan.planned_g_h:.0f} g/uur over {eet_uren:.1f} eetbare uur). "
            f"Met de geselecteerde producten komt het niet hoger."))


def _feedable_elapsed(plan: NutritionPlan, t_s: float) -> float:
    """Hoeveel eetbare tijd is er verstreken op tijdstip ``t_s``?"""
    totaal = 0.0
    for seg in plan.segments:
        if not seg.feedable:
            continue
        totaal += max(0.0, min(seg.end_s, t_s) - seg.start_s)
    return totaal


# ------------------------------------------------------------- context-notes --

def _add_context_notes(plan: NutritionPlan, duration: DurationEstimate) -> None:
    """Informatieve regels die geen constraint zijn maar wel het plan duiden."""
    if not plan.target.needed and plan.planned_g_h <= 0:
        plan.warnings.insert(0, PlanWarning(
            "niet_nodig",
            f"Bij {fmt_duration(plan.total_s)} is bijvoeden meestal niet nodig — je "
            f"glycogeen dekt dit ruim. Water volstaat; wil je toch iets, houd het "
            f"dan onder de {rules.CARB_TIERS[0][2]:.0f} g. Start je nuchter of ligt er "
            f"een intensief blok in, dan is een kleine gel prima.",
            SEVERITY_INFO))

    niet_eetbaar = [s for s in plan.segments if not s.feedable]
    if niet_eetbaar:
        plan.warnings.append(PlanWarning(
            "zwemmen_niet_eetbaar",
            "Tijdens het zwemmen kun je niets innemen; de behoefte van dat blok "
            f"({fmt_duration(sum(s.duration_s for s in niet_eetbaar))}) is over de "
            f"fiets, de loop en de wissels verdeeld. In T1 mag je meteen wat nemen — "
            f"dat is het rustigste eetmoment van de race.",
            SEVERITY_INFO))

    if plan.request.temp_c is None:
        plan.warnings.append(PlanWarning(
            "geen_temperatuur",
            f"Geen temperatuur bekend; er is met {rules.DEFAULT_TEMP_C:.0f} °C "
            f"gerekend. Vul de verwachting in als het warmer wordt: vocht en "
            f"natrium schuiven daar flink mee.",
            SEVERITY_INFO))

    for note in duration.notes:
        plan.warnings.append(PlanWarning("duurschatting", note, SEVERITY_INFO))


# ------------------------------------------------------------- hulpfuncties --

def legs_for(session_type: str) -> list[str]:
    """De sporten waaruit dit sessietype bestaat, in racevolgorde."""
    return list(SESSION_TYPES.get(session_type, ("", []))[1])


def build_legs(session_type: str, per_sport: dict[str, tuple[float | None, float | None]]
               ) -> list[LegRequest]:
    """Bouw de onderdeel-invoer uit een dict ``{sport: (afstand_m, duur_s)}``."""
    return [LegRequest(sport=s, distance_m=per_sport.get(s, (None, None))[0],
                       duration_s=per_sport.get(s, (None, None))[1])
            for s in legs_for(session_type)]
