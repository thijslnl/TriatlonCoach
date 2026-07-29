"""Hartslagzone-berekeningen op basis van %LTHR.

De zonegrenzen zijn een lijst met de ondergrens van Z2 t/m Z5: alles onder de
eerste grens is Z1, alles vanaf de laatste grens is Z5.

Deze module rekent alleen; wélke drempel erin gaat is sport-afhankelijk en
wordt door :mod:`tricoach.sportzones` bepaald (loop-LTHR voor hardlopen,
fiets-LTHR voor fietsen, geen zones voor zwemmen). Vraag de grenzen daar op
met ``hr_zone_bounds(athlete, sport)`` in plaats van ze zelf af te leiden.
"""

import pandas as pd

ZONE_NAMES = ["Z1", "Z2", "Z3", "Z4", "Z5"]

# Ondergrens van Z2 t/m Z5 als fractie van de LTHR (standaard %LTHR-indeling).
# Met de loop-LTHR van 164 geeft dit 131 / 146 / 156 / 164.
DEFAULT_ZONE_PCT = [0.80, 0.89, 0.95, 1.00]


def bounds_from_lthr(lthr: int, pcts: list[float] | None = None) -> list[int]:
    """Reken de zonegrenzen uit op basis van de LTHR."""
    return [round(lthr * p) for p in (pcts or DEFAULT_ZONE_PCT)]

# Records liggen normaal ~1 seconde uit elkaar. Bij grotere gaten (pauze,
# signaalverlies) tellen we maximaal dit aantal seconden mee, zodat een
# koffiestop niet als trainingstijd in een zone belandt.
MAX_GAP_S = 10


def zone_for_hr(hr: int, bounds: list[int]) -> str:
    """Bepaal de zonenaam (Z1..Z5) voor één hartslagwaarde."""
    zone = 0  # start in Z1
    for bound in bounds:
        if hr >= bound:
            zone += 1
    return ZONE_NAMES[min(zone, len(ZONE_NAMES) - 1)]


def time_in_zones(records: pd.DataFrame, bounds: list[int] | None) -> dict[str, int]:
    """Bereken seconden per hartslagzone uit de record-data van een sessie.

    Verwacht een DataFrame met kolommen 'timestamp' en 'heart_rate'.
    Geeft een dict terug zoals {"Z1": 120, "Z2": 1800, ...}.

    ``bounds=None`` betekent "deze sport krijgt geen zone-oordeel" (zwemmen):
    alle zones blijven dan op 0, zodat er nergens per ongeluk toch een
    zoneverdeling opduikt.
    """
    result = dict.fromkeys(ZONE_NAMES, 0)
    if bounds is None or records.empty or "heart_rate" not in records:
        return result
    df = records.dropna(subset=["heart_rate"]).sort_values("timestamp")
    if len(df) < 2:
        return result

    # Tijd tussen opeenvolgende records, afgekapt op MAX_GAP_S.
    seconds = df["timestamp"].diff().dt.total_seconds().clip(upper=MAX_GAP_S)
    for hr, sec in zip(df["heart_rate"].iloc[1:], seconds.iloc[1:]):
        result[zone_for_hr(int(hr), bounds)] += int(sec)
    return result


# Vanaf dit aandeel rustige tijd (Z1+Z2) noemen we een sessie "overwegend
# rustig". De aerobe-efficiëntie-trend vergelijkt bij voorkeur alleen sessies
# binnen dezelfde categorie, zodat een rustige Z2-rit niet tegen een harde
# Z3-rit wordt afgezet.
EASY_SHARE_THRESHOLD = 60.0


def pct_in_zone2(z1: int, z2: int, z3: int, z4: int, z5: int) -> float | None:
    """Aandeel (%) van de gemeten hartslagtijd dat in zone 2 viel.

    Werkt op de al berekende seconden-per-zone (z1..z5), zodat dit zowel bij
    import als bij een herberekening uit dezelfde bron komt. Geeft None terug
    als er geen hartslagdata is (som = 0).
    """
    total = z1 + z2 + z3 + z4 + z5
    if total <= 0:
        return None
    return 100.0 * z2 / total


def easy_share(z1: int, z2: int, z3: int, z4: int, z5: int) -> float | None:
    """Aandeel (%) rustige tijd (Z1 + Z2). None zonder hartslagdata."""
    total = z1 + z2 + z3 + z4 + z5
    if total <= 0:
        return None
    return 100.0 * (z1 + z2) / total


def intensity_category(z1: int, z2: int, z3: int, z4: int, z5: int) -> str | None:
    """Grove intensiteitscategorie voor like-for-like vergelijken.

    "rustig" als minstens ``EASY_SHARE_THRESHOLD`` % in Z1+Z2 zat, anders
    "intensief". None zonder hartslagdata.
    """
    share = easy_share(z1, z2, z3, z4, z5)
    if share is None:
        return None
    return "rustig" if share >= EASY_SHARE_THRESHOLD else "intensief"
