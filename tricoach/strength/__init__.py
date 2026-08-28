"""Krachttraining: rotatie, progressie en de koppeling met trainingsdata.

Drie lagen, elk in een eigen module:

- :mod:`~tricoach.strength.rules` — pure businesslogica (geen ``sqlite3``):
  Epley-1RM/volume/beste-set, sjabloonrotatie, deload-signaal,
  kwaliteitssessie-classificatie, opwarmfilter, fase-instelling.
- :mod:`~tricoach.strength.catalog` — de bewerkbare oefeningen- en
  sjabloonbibliotheek (``strength_exercise``/``strength_template``/
  ``strength_template_item``).
- :mod:`~tricoach.strength.store` — het loggen van sessies en sets
  (``strength_workout``/``strength_set``) en de leesqueries voor voortgang,
  gewichtsuggestie en weekvolume.

**Niets wordt ooit hard verwijderd.** Oefeningen en sjablonen krijgen
``is_active``; loghistorie blijft na een hernoeming of deactivering leesbaar
via een live join (val terug op een naam-snapshot alleen als het echt niet
anders kan). Zie ``memory/beslissingen.md`` voor de volledige onderbouwing.
"""

from tricoach.strength.catalog import (
    LOAD_TYPES,
    add_exercise,
    add_template,
    add_template_item,
    load_exercises,
    load_template_items,
    load_templates,
    set_exercise_active,
    set_template_active,
    set_template_item_active,
    update_exercise,
    update_template,
    update_template_item,
)
from tricoach.strength.rules import (
    DELOAD_MELDING,
    check_session_conflict,
    current_phase,
    epley_1rm,
    is_quality_session,
    next_template_id,
    parse_target_reps,
    personal_records,
    phase_target,
    progression_series,
    suggest_deload,
    working_sets,
    workout_1rm,
    workout_best_set,
    workout_totals,
    workout_volume,
)
from tricoach.strength.store import (
    complete_workout,
    last_values_for_exercise,
    load_sets,
    load_workouts,
    log_set,
    next_template,
    open_workout,
    restore_workout,
    soft_delete_workout,
    start_workout,
    weekly_strength_volume,
)

__all__ = [
    "DELOAD_MELDING", "LOAD_TYPES",
    "add_exercise", "add_template", "add_template_item",
    "check_session_conflict", "complete_workout", "current_phase",
    "epley_1rm", "is_quality_session", "last_values_for_exercise",
    "load_exercises", "load_sets", "load_template_items", "load_templates",
    "load_workouts", "log_set", "next_template", "next_template_id",
    "open_workout", "parse_target_reps", "personal_records", "phase_target",
    "progression_series", "restore_workout", "set_exercise_active",
    "set_template_active", "set_template_item_active", "soft_delete_workout",
    "start_workout", "suggest_deload", "update_exercise", "update_template",
    "update_template_item", "weekly_strength_volume", "working_sets",
    "workout_1rm", "workout_best_set", "workout_totals", "workout_volume",
]
