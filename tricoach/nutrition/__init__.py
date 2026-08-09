"""Voedingsplanner voor trainingen en races.

Vooraf invoeren wat je gaat doen, terugkrijgen hoeveel koolhydraten je wanneer
uit welk product haalt en wat er mee moet. De opzet in vier lagen, elk in een
eigen module:

- :mod:`~tricoach.nutrition.rules` — de richtlijnen en drempelwaarden als pure
  functies (koolhydraten per uur, opnameplafonds, concentratie, vocht, natrium,
  cafeïne, timing).
- :mod:`~tricoach.nutrition.products` — de bewerkbare productdatabase.
- :mod:`~tricoach.nutrition.duration` — de duurschatting uit de eigen sessies,
  als bandbreedte en met onderbouwing.
- :mod:`~tricoach.nutrition.plan` — de planner die dat alles tot een tijdlijn,
  meeneemlijst en waarschuwingen combineert.
- :mod:`~tricoach.nutrition.store` — opslaan van plannen en de terugkoppeling
  achteraf (wat ging erin, hoe viel het).

**Het algoritme rekent; een taalmodel komt er niet aan te pas.** Elk getal in
een plan is terug te voeren op een benoemde constante in ``rules.py``, en
dezelfde invoer geeft altijd hetzelfde plan.
"""

from tricoach.nutrition.duration import (
    DurationEstimate,
    LegEstimate,
    LegRequest,
    estimate_duration,
)
from tricoach.nutrition.plan import (
    INTENSITIES,
    INTENSITY_LABEL,
    SESSION_TYPES,
    AidStation,
    NutritionPlan,
    PlanRequest,
    build_legs,
    build_plan,
    legs_for,
)
from tricoach.nutrition.products import (
    DEFAULT_PRODUCTS,
    KINDS,
    SOURCES,
    Product,
    load_products,
    products_dataframe,
    products_from_dataframe,
    reset_products,
    save_products,
)
from tricoach.nutrition.rules import DISCLAIMER, CarbTarget, carb_target

__all__ = [
    "AidStation", "CarbTarget", "DEFAULT_PRODUCTS", "DISCLAIMER",
    "DurationEstimate", "INTENSITIES", "INTENSITY_LABEL", "KINDS", "LegEstimate",
    "LegRequest", "NutritionPlan", "PlanRequest", "Product", "SESSION_TYPES",
    "SOURCES", "build_legs", "build_plan", "carb_target", "estimate_duration",
    "legs_for", "load_products", "products_dataframe", "products_from_dataframe",
    "reset_products", "save_products",
]
