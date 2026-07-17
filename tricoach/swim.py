"""Zwem-specifieke basisberekeningen: baanlengte per baan en de SWOLF.

SWOLF (tijd + slagen per baan) is alleen vergelijkbaar bij dezelfde slag en
dezelfde baanlengte. Daarom rekenen alle trends, records en feedbackregels
met één en dezelfde selectie: crawlbanen in het 25m-bad (:func:`crawl_swolf`).
Overal waar zo'n SWOLF wordt getoond hoort het filter erbij vermeld te staan
(:data:`SWOLF_FILTER_LABEL`), zodat duidelijk is waarom banen kunnen ontbreken.

Dit module staat onderaan de importketen (alleen pandas) zodat zowel
``analysis`` als ``progress``, ``trainingslog`` en ``feedback_context`` er
zonder cyclische imports uit kunnen putten.
"""

import pandas as pd

# Zo benoemen we het filter overal in de weergave (grafiektitels, records,
# feedback- en logregels).
SWOLF_FILTER_LABEL = "alleen crawl, 25m-bad"


def lane_meters(lengths: pd.DataFrame, pool_length: float | None,
                distance_m: float | None) -> pd.Series:
    """Afstand per baan (m), robuust voor banen van wisselende lengte.

    Normaal is elke baan simpelweg ``pool_length``. Maar als de totale
    sessieafstand duidelijk afwijkt van banen × baanlengte (bijv. het zwembad
    ging halverwege van 25 m- naar 15 m-banen, terwijl het horloge op 25 m
    bleef staan), dan is dát niet meer waar. In dat geval verdelen we de
    sessieafstand naar rato van het aantal slagen per baan: een kortere baan
    kost evenredig minder slagen, en het sessietotaal klopt weer. Dit is een
    schatting per baan, maar veel eerlijker dan elke baan even lang rekenen.
    """
    n = len(lengths)
    pool = float(pool_length) if pool_length and not pd.isna(pool_length) else 25.0
    uniform = pd.Series([pool] * n, index=lengths.index)
    if not distance_m or pd.isna(distance_m) or n == 0:
        return uniform
    if abs(n * pool - distance_m) <= 0.02 * distance_m:
        return uniform
    slagen = pd.to_numeric(lengths.get("total_strokes"), errors="coerce")
    if slagen.isna().any() or slagen.sum() <= 0:
        # Geen bruikbare slagdata: gemiddelde afstand per baan als terugval.
        return pd.Series([distance_m / n] * n, index=lengths.index)
    return slagen / slagen.sum() * distance_m


def crawl_swolf(lengths: pd.DataFrame, pool_length: float | None,
                distance_m: float | None) -> float | None:
    """Gemiddelde SWOLF over crawlbanen van ~25 m; None zonder zulke banen.

    Andere slagen en afwijkende baanlengtes (ook bínnen een sessie, via
    :func:`lane_meters`) tellen niet mee — die geven wezenlijk andere
    waarden, waardoor sessies onderling niet vergelijkbaar zouden zijn.
    """
    if lengths.empty or "swim_stroke" not in lengths or "total_strokes" not in lengths:
        return None
    meters = lane_meters(lengths, pool_length, distance_m)
    sel = lengths[(lengths["swim_stroke"] == "freestyle")
                  & ((meters - 25.0).abs() <= 1.0)]
    if sel.empty:
        return None
    swolf = (sel["total_timer_time"] + sel["total_strokes"]).mean()
    return float(swolf) if pd.notna(swolf) else None
