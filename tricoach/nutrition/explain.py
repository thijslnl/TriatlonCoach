"""De optionele toelichting van 2-3 zinnen bij een plan — en niets meer.

Het taalmodel rekent hier **niets** uit. Het krijgt het al berekende plan als
platte tekst en mag daar hooguit een korte toelichting bij schrijven: waar zit
de spanning, waar moet je op letten. Alle getallen staan vast voordat deze
module wordt aangeroepen, en de systeemprompt verbiedt expliciet om er nieuwe
bij te verzinnen of bestaande te herberekenen.

Waarom die scheiding zo strikt is: een voedingsplan moet reproduceerbaar en
controleerbaar zijn. Twee keer dezelfde invoer hoort twee keer hetzelfde plan
te geven, en elk getal moet terug te voeren zijn op een regel in
:mod:`tricoach.nutrition.rules`. Een taalmodel geeft die garantie niet — dus
krijgt het de rol van tekstschrijver, niet van rekenmachine.

De taak heet ``nutrition_note`` en staat in ``config.yaml`` onder
``llm.routing``; standaard gaat hij naar het lokale Ollama-model, want het is
lichte tekst en er hoeft geen API-geld naartoe.
"""

from tricoach.llm.router import LLMRouter
from tricoach.nutrition.plan import NutritionPlan

TASK = "nutrition_note"

SYSTEM = (
    "Je bent de assistent van een triatlon-trainingsdashboard en schrijft in het "
    "Nederlands. Je krijgt een VOLLEDIG UITGEREKEND voedingsplan. Je schrijft "
    "hooguit 3 korte zinnen toelichting: wat is hier de kern, en waar moet de "
    "atleet op letten. HARDE REGELS: (1) verzin geen getallen en herbereken "
    "niets — noem alleen cijfers die letterlijk in het plan staan, en liever nog "
    "minder; (2) spreek de meldingen in het plan niet tegen; (3) geef geen "
    "medisch of diëtistisch advies en geen calorie-, dieet- of afvaldoelen; "
    "(4) geen opsomming, geen kopjes en geen herhaling van de tijdlijn — die "
    "ziet de atleet er al bij staan. Schrijf nuchter en concreet."
)


def explain_plan(router: LLMRouter, plan: NutritionPlan) -> str:
    """Vraag een korte toelichting bij een al berekend plan.

    Het plan gaat als samenvatting, tijdlijn en meldingenlijst mee — precies
    wat de atleet zelf ook ziet, zodat het model niets kan toevoegen wat niet
    op het scherm staat. Een ontbrekende routing of een onbereikbaar model
    komt als exception naar boven; de UI toont dat als melding en het plan
    zelf blijft gewoon staan, want de toelichting is sierwerk.
    """
    tijdlijn = "\n".join(
        f"- {e.time_label} ({e.segment}): {e.amount} {e.product}"
        + (f" — {e.note}" if e.note else "")
        for e in plan.events
    ) or "- geen losse innamemomenten"
    meldingen = "\n".join(f"- {w.text}" for w in plan.warnings) or "- geen"

    context = (
        f"# Het berekende plan\n\n{plan.summary_text()}\n\n"
        f"# Tijdlijn\n\n{tijdlijn}\n\n"
        f"# Meldingen\n\n{meldingen}\n\n"
        "Schrijf nu de toelichting van hooguit 3 zinnen."
    )
    return router.ask(TASK, context, system=SYSTEM).strip()
