"""Pure businesslogica voor krachttraining: geen ``sqlite3``-import.

Vier verantwoordelijkheden, in deze module gebundeld omdat ze allemaal zuiver
zijn (invoer -> uitvoer, geen database, geen Streamlit):

- **Progressie**: Epley-1RM, volume, beste set, persoonlijke records per
  ``load_type`` (weight/bodyweight/time/distance).
- **Rotatie**: welk sjabloon is aan de beurt.
- **Signalen**: een deload-suggestie na 7 weken, en een conflictwaarschuwing
  als een krachtsessie te dicht op een kwaliteitssessie (interval, tempo,
  lange rit) ligt — de reden dat deze module in tricoach zit en niet in een
  losse krachtapp: ze leest de bestaande trainingsdata en zones.
- **Fase-instelling**: welke doelsets/reps gelden (1/2/3), uit config.yaml.

``tricoach/strength/catalog.py`` en ``store.py`` roepen deze functies aan met
data die ze uit SQLite hebben geladen; hier komt geen ``sqlite3`` aan te pas,
zodat deze module zonder database te testen is.
"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from tricoach.formatting import local_time
from tricoach.sportzones import bike_lthr, ftp, run_lthr

# --------------------------------------------------------------------- fase --


def current_phase(config: dict) -> int:
    """De actuele krachtfase (1, 2 of 3) uit config.yaml, geklemd, default 1."""
    try:
        fase = int((config.get("training") or {}).get("phase", 1))
    except (TypeError, ValueError):
        return 1
    return min(max(fase, 1), 3)


def phase_target(item: dict, phase: int) -> tuple[int | None, str | None]:
    """``(target_sets, target_reps)`` voor deze fase uit een sjabloon-item.

    Valt terug op fase 1 als de kolommen voor de gevraagde fase leeg zijn
    (bijv. een net toegevoegd item waarvan nog niet alle fases zijn ingevuld).
    """
    sets = item.get(f"target_sets_p{phase}")
    reps = item.get(f"target_reps_p{phase}")
    if sets is None and reps is None and phase != 1:
        return item.get("target_sets_p1"), item.get("target_reps_p1")
    return sets, reps


def parse_target_reps(text: str | None) -> int | None:
    """Het numerieke deel van een doelnotatie (``"8"``->8, ``"40m"``->40,
    ``"30s"``->30). ``"max"`` (bijv. bij pull-ups) heeft geen doelaantal om
    een beste set tegen af te zetten en geeft ``None``.
    """
    if not text:
        return None
    tekst = str(text).strip().lower()
    if tekst == "max":
        return None
    cijfers = "".join(c for c in tekst if c.isdigit())
    return int(cijfers) if cijfers else None


# ----------------------------------------------------------------- rotatie --


def next_template_id(active_templates: list[dict], last_template_id: int | None) -> int | None:
    """Het sjabloon dat nu aan de beurt is.

    ``active_templates`` moet al gesorteerd zijn op ``sort_order``. Geeft de
    opvolger van ``last_template_id`` met wrap-around naar de eerste; geeft
    de eerste als er nog geen historie is óf als het laatst gebruikte
    sjabloon niet meer in de actieve lijst zit (inmiddels gedeactiveerd).
    ``None`` bij een lege lijst.
    """
    if not active_templates:
        return None
    ids = [t["id"] for t in active_templates]
    if last_template_id not in ids:
        return ids[0]
    idx = ids.index(last_template_id)
    return ids[(idx + 1) % len(ids)]


# ----------------------------------------------------------- opwarmfilter --


def working_sets(sets: pd.DataFrame) -> pd.DataFrame:
    """De sets die meetellen voor volume, 1RM en records: geen opwarmsets.

    Dit is HET filterpunt — elke aggregatiefunctie hieronder roept dit als
    eerste regel aan, zoals :func:`tricoach.storage.training_activities` dat
    is voor transport-ritten. Nergens anders in deze module staat een
    ``is_warmup``-vergelijking.
    """
    if sets.empty or "is_warmup" not in sets:
        return sets
    return sets[sets["is_warmup"] == 0]


# --------------------------------------------------------------- progressie --


def epley_1rm(weight_kg: float | None, reps: int | None) -> float | None:
    """Geschat 1RM via Epley: gewicht x (1 + reps/30)."""
    if not weight_kg or not reps or reps <= 0:
        return None
    return float(weight_kg) * (1 + float(reps) / 30)


def workout_1rm(sets: pd.DataFrame) -> float | None:
    """Hoogste geschatte 1RM over de werksets van één workout."""
    werk = working_sets(sets)
    if werk.empty:
        return None
    waarden = [epley_1rm(r.get("weight_kg"), r.get("reps")) for _, r in werk.iterrows()]
    waarden = [w for w in waarden if w is not None]
    return max(waarden) if waarden else None


def workout_volume(sets: pd.DataFrame) -> float:
    """Totaal volume (kg) van de werksets: som(reps x gewicht)."""
    werk = working_sets(sets)
    if werk.empty or "reps" not in werk or "weight_kg" not in werk:
        return 0.0
    reps = werk["reps"].fillna(0)
    gewicht = werk["weight_kg"].fillna(0)
    return float((reps * gewicht).sum())


def workout_best_set(sets: pd.DataFrame, target_reps: int | None = None) -> dict | None:
    """De zwaarste werkset die minstens het doelaantal reps haalde.

    Zonder ``target_reps`` gewoon de zwaarste werkset.
    """
    werk = working_sets(sets)
    if werk.empty or "weight_kg" not in werk:
        return None
    kandidaten = werk.dropna(subset=["weight_kg"])
    if target_reps:
        kandidaten = kandidaten[kandidaten["reps"].fillna(0) >= target_reps]
    if kandidaten.empty:
        return None
    return kandidaten.loc[kandidaten["weight_kg"].idxmax()].to_dict()


def workout_totals(sets: pd.DataFrame, load_type: str, target_reps: int | None = None) -> dict:
    """Dispatcher: de relevante totalen voor dit ``load_type``, over de
    werksets (opwarmsets zijn al uitgefilterd via :func:`working_sets`)."""
    werk = working_sets(sets)
    if load_type == "weight":
        return {
            "e1rm": workout_1rm(sets),
            "volume_kg": workout_volume(sets),
            "best_set": workout_best_set(sets, target_reps),
        }
    if load_type == "bodyweight":
        totaal = int(werk["reps"].fillna(0).sum()) if not werk.empty and "reps" in werk else 0
        return {"total_reps": totaal}
    if load_type == "time":
        totaal = float(werk["seconds"].fillna(0).sum()) if not werk.empty and "seconds" in werk else 0.0
        return {"total_seconds": totaal}
    if load_type == "distance":
        totaal = float(werk["meters"].fillna(0).sum()) if not werk.empty and "meters" in werk else 0.0
        return {"total_meters": totaal}
    return {}


def progression_series(sets: pd.DataFrame, workouts: pd.DataFrame, load_type: str,
                       target_reps: int | None = None) -> pd.DataFrame:
    """Eén rij per workout met de relevante totalen, voor de voortgangsgrafiek.

    ``is_deload`` gaat mee zodat de UI die punten apart kan markeren.
    """
    if workouts.empty:
        return pd.DataFrame()
    rijen = []
    for _, w in workouts.iterrows():
        # workouts komt uit store.load_workouts(): de primary key heet daar
        # "id" (zoals bij elke andere tabel), niet "workout_id" — dat laatste
        # is alleen de kolomnaam in strength_set die ernaar verwijst.
        workout_id = w["id"]
        eigen = sets[sets["workout_id"] == workout_id] if not sets.empty else sets
        totals = workout_totals(eigen, load_type, target_reps)
        rijen.append({
            "workout_id": workout_id, "started_at": w["started_at"],
            "is_deload": bool(w.get("is_deload", 0)), **totals,
        })
    return pd.DataFrame(rijen).sort_values("started_at").reset_index(drop=True)


def personal_records(series: pd.DataFrame, load_type: str) -> dict:
    """Persoonlijke records per relevante metriek, met deloads uitgesloten.

    Een deload kan een PR dus nooit verhogen (het is geen serieuze poging)
    of verlagen (het record blijft gewoon op de niet-deload-waarde staan).
    """
    if series.empty:
        return {}
    niet_deload = series[~series["is_deload"]]
    if niet_deload.empty:
        return {}
    metriek_per_type = {
        "weight": ["e1rm", "volume_kg"],
        "bodyweight": ["total_reps"],
        "time": ["total_seconds"],
        "distance": ["total_meters"],
    }
    uit = {}
    for kolom in metriek_per_type.get(load_type, []):
        if kolom not in niet_deload or niet_deload[kolom].dropna().empty:
            continue
        idx = niet_deload[kolom].idxmax()
        beste = niet_deload.loc[idx]
        uit[kolom] = {"waarde": beste[kolom], "datum": beste["started_at"],
                     "workout_id": beste["workout_id"]}
    return uit


# ------------------------------------------------------------------ deload --

DELOAD_MELDING = ("Zeven weken zonder deload. Overweeg deze week dezelfde "
                  "oefeningen op 60% van het gewicht.")


def suggest_deload(workouts: pd.DataFrame, today: date | None = None,
                   weeks: int = 7, min_workouts: int = 10) -> bool:
    """Is het tijd voor een deload-suggestie?

    ``True`` als er minstens ``min_workouts`` voltooide workouts zijn gelogd
    én er in de afgelopen ``weeks`` weken geen enkele deload-workout was.
    Zwijgt vanzelf weer voor ``weeks`` weken zodra er een deload gelogd
    wordt, want die valt dan zelf binnen het venster.
    """
    if workouts.empty or "completed_at" not in workouts:
        return False
    voltooid = workouts[workouts["completed_at"].notna()]
    if len(voltooid) < min_workouts:
        return False
    vandaag = today or date.today()
    grens = pd.Timestamp(vandaag) - pd.Timedelta(weeks=weeks)
    gestart = pd.to_datetime(voltooid["started_at"])
    recent_deload = voltooid[voltooid["is_deload"].astype(bool) & (gestart >= grens)]
    return recent_deload.empty


# ------------------------------------------------ koppeling met trainingsdata --

QUALITY_MIN_DURATION_S = 7200
QUALITY_HR_FRACTION = 0.90
QUALITY_POWER_FRACTION = 0.85


def is_quality_session(row, athlete: dict) -> bool:
    """Is dit een kwaliteitssessie (interval, tempo, lange duur) die niet
    vlak naast een zware krachtsessie moet liggen?

    Vorm naar :func:`tricoach.transport.suggest_transport`: vroege
    ``return``s, één criterium per regel.

    1. Elke sport: duur > 2 uur.
    2. Hardlopen: gem. HR boven 90% van de loop-LTHR.
    3. Fietsen: vermogen (NP, val terug op gem. vermogen) boven 85% van de
       FTP als die bekend is; anders (FTP onbekend of geen vermogensdata)
       terugval op gem. HR boven 90% van de fiets-LTHR (die heeft altijd een
       waarde, in tegenstelling tot FTP).
    """
    duration_s = row.get("duration_s")
    if duration_s and duration_s > QUALITY_MIN_DURATION_S:
        return True

    sport = row.get("sport")
    avg_hr = row.get("avg_hr")

    if sport == "running":
        return bool(avg_hr and avg_hr > run_lthr(athlete) * QUALITY_HR_FRACTION)

    if sport == "cycling":
        power = row.get("np_power") or row.get("avg_power")
        drempel_ftp = ftp(athlete)
        if power and drempel_ftp:
            return power > drempel_ftp * QUALITY_POWER_FRACTION
        return bool(avg_hr and avg_hr > bike_lthr(athlete) * QUALITY_HR_FRACTION)

    return False


def check_session_conflict(activities: pd.DataFrame, on_day: date, athlete: dict) -> list[dict]:
    """Kwaliteitssessies op de kalenderdag vóór, op, of ná ``on_day``.

    Blokkeert niets — geeft alleen de treffers terug zodat de UI kan
    signaleren. ``activities`` moet al ``training_activities()``-gefilterd
    zijn: een transportrit of wisselsegment is geen kwaliteitssessie.

    Kijkt bewust naar hele kalenderdagen (niet een strikt 24-uursvenster rond
    middernacht) — dat laatste zou een intervalloop van gisterochtend 07:00
    missen. Dit is de letterlijke programmaregel: "geen zware benen op de
    dag vóór of ná" een kwaliteitssessie.
    """
    if activities.empty:
        return []
    dagen = {on_day - timedelta(days=1): "gisteren", on_day: "vandaag",
             on_day + timedelta(days=1): "morgen"}
    uit = []
    for _, row in activities.iterrows():
        start = row.get("start_time")
        if pd.isna(start):
            continue
        dag = local_time(start).date()
        if dag not in dagen:
            continue
        if not is_quality_session(row, athlete):
            continue
        uit.append({
            "activity_key": row.get("activity_key"), "sport": row.get("sport"),
            "start_time": start, "when": dagen[dag],
            "label": f"{row.get('sport')} ({dagen[dag]})",
        })
    return uit


# ------------------------------------------------------------- weekaggregatie --


def iso_week_label(ts: pd.Series) -> pd.Series:
    """ISO-weeklabel (``'2026-W23'``), zelfde formaat als ``analysis.add_week()``.

    Eigen functie omdat krachtworkouts hun tijdstempel ``started_at`` heten,
    niet ``start_time`` — ``add_week()`` verwacht die laatste kolomnaam.
    Levert identieke labels op dezelfde datum, zodat de merge in de
    herstelgrafiek gegarandeerd aansluit.
    """
    tijden = pd.to_datetime(ts)
    iso = tijden.dt.isocalendar()
    return iso["year"].astype(str) + "-W" + iso["week"].astype(str).str.zfill(2)
