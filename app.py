"""Triatlon Training Dashboard — Streamlit-app.

Starten met:  streamlit run app.py

De app leest de geïmporteerde sessies uit SQLite (data/training.db) en de
memory-bestanden uit memory/. Uploads gaan via de zijbalk en doorlopen
dezelfde import-pipeline als de commandline (tricoach.importer).
"""

import copy
import json
from datetime import date, datetime, timedelta

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pydeck as pdk
import streamlit as st
import dotenv

from tricoach import body
from tricoach import heatmap as heatmap_mod
from tricoach import garmin_sync
from tricoach import wellness
from tricoach import profile as profile_mod
from tricoach.advice import generate_advice, generate_insights, last_advice, last_insights
from tricoach.analysis import (
    aerobic_efficiency_trend,
    pace_at_hr,
    lane_meters,
    run_power,
    stroke_distribution,
    swim_length_matrix,
    swim_per_session,
    weekly_intensity_share,
    weekly_totals,
    weekly_volume,
    weekly_zone_time,
)
from tricoach.chat import answer_question
from tricoach.combos import (
    combo_history,
    combo_membership,
    detect_and_store_proposals,
    load_combos,
    max_gap_min,
    race_similarity,
    run_transition_analysis,
    set_combo_status,
)
from tricoach.config import PROJECT_ROOT, load_config, resolve_path
from tricoach.feedback import generate_feedback, load_session_feedback
from tricoach.formatting import (
    GEEN_WAARDE,
    effective_speed_ms,
    fmt_duration,
    fmt_aantal,
    fmt_hours_hhmm,
    local_time,
    sessie_tempo,
    sport_label,
    stroke_label,
)
from tricoach.memory_review import MAX_LEEFTIJD_WEKEN, review_dataframe
from tricoach.nutrition import products as voeding_producten
from tricoach.nutrition import rules as voeding_regels
from tricoach.nutrition import store as voeding_opslag
from tricoach.nutrition.duration import LegRequest, estimate_duration
from tricoach.nutrition.explain import explain_plan
from tricoach.nutrition.plan import (
    INTENSITY_LABEL,
    SESSION_TYPES,
    SEVERITY_WARNING,
    AidStation,
    PlanRequest,
    build_plan,
    legs_for,
)
from tricoach.progress import (
    acwr_status,
    best_efforts,
    css_estimate,
    decoupling,
    efficiency_factor,
    load_curves,
    personal_records,
    progress_summary_text,
    race_distances,
    race_prediction,
    race_prediction_history,
    readiness,
    swim_progression,
)
from tricoach.importer import import_zip
from tricoach.power import (
    FTP_EST_FACTOR,
    POWER_ZONE_NAMES,
    estimate_ftp,
    power_decoupling_trend,
    power_trend,
    power_zone_bounds,
    time_in_power_zones,
)
from tricoach.rundynamics import (
    CADENCE_GUIDE_SPM,
    GCT_TRAINED_MS,
    VERTICAL_RATIO_GOOD_PCT,
    dynamics_from_summary,
    dynamics_trend,
    vertical_ratio_is_good,
)
from tricoach.llm import LLMRouter
from tricoach.llm.log import usage_summary
from tricoach.llm.observations import session_observation
from tricoach.lthr import (
    BIKE_FTP as LTHR_BIKE_FTP,
    BIKE_LTHR as LTHR_BIKE,
    RUN_LTHR as LTHR_RUN,
    append_entry as lthr_append,
    load_history as lthr_history,
)
from tricoach.palette import get_palette, with_alpha
from tricoach.viz import (
    PLOTLY_CONFIG,
    add_race_marker,
    date_xaxis,
    pace_as_time,
    pace_axis,
    pad_single_point,
    style_fig,
)
from tricoach.schedule import add_note_row, load_schedule, save_schedule
from tricoach.settings import save_config
from tricoach.feedback_context import hr_drift_values, run_splits_df
from tricoach.archive import migrate_originals, uploads_root, verify_originals
from tricoach.removal import purge_session, remove_session, restore_session
from tricoach.storage import (
    connect,
    load_activities,
    load_deleted_activities,
    load_lengths,
    load_records,
    recompute_power_zones,
    recompute_zones,
    set_training_label,
    swim_active_seconds,
    training_activities,
)
from tricoach.transport import is_transport, mark_transport, unmark_transport
from tricoach.weather import forecast_temperature, wind_for_activity
from tricoach.zones import bounds_from_lthr
from tricoach.sportzones import (
    CYCLING,
    RUNNING,
    SWIMMING,
    bike_lthr,
    bike_lthr_is_estimated,
    ftp as athlete_ftp,
    hr_zone_bounds,
    run_lthr,
    run_lthr_source,
    set_thresholds,
    threshold_notes,
    zone2_range,
    zone_model,
    zone_overview,
)
from tricoach.ramptest import (
    RAMP_FTP_FACTOR,
    ftp_proposal,
    is_ramp_test,
    log_ftp_determination,
)

dotenv.load_dotenv()

st.set_page_config(page_title="Triatlon Coach", page_icon="🏊", layout="wide")

config = load_config()
ATHLETE = config["athlete"]
# Drempels en zonegrenzen zijn sport-afhankelijk (zie tricoach.sportzones):
# hardlopen op de loop-LTHR, fietsen op de fiets-LTHR (of %FTP zodra die
# bekend is), zwemmen zonder zones.
RUN_BOUNDS = hr_zone_bounds(ATHLETE, RUNNING)
BIKE_BOUNDS = hr_zone_bounds(ATHLETE, CYCLING)
RUN_Z2 = zone2_range(ATHLETE, RUNNING)    # bijv. (131, 145) bij LTHR 164
BIKE_Z2 = zone2_range(ATHLETE, CYCLING)
# De hartslagzonegrenzen per sport, voor grafieken die één sessie tekenen.
SPORT_BOUNDS = {RUNNING: RUN_BOUNDS, CYCLING: BIKE_BOUNDS, SWIMMING: None}
# FTP (W) voor de vermogenszones; None zolang er geen (geteste of geschatte)
# waarde op de instellingen-tab staat — fietsen valt dan terug op de
# fiets-LTHR-hartslagzones.
FTP = athlete_ftp(ATHLETE)
MEMORY_DIR = resolve_path(config, "memory_dir")
# Origineel-archief van alle geüploade FIT-bestanden (zie tricoach.archive).
UPLOADS_DIR = uploads_root(config)
TZ = "Europe/Amsterdam"  # Garmin slaat tijden op in UTC; tonen in lokale tijd

# Vaste kleuren zodat sporten en zones in elke grafiek hetzelfde ogen.
# Het palet komt uit tricoach/palette.py (gevalideerd, licht + donker) en
# volgt het actieve Streamlit-thema.
PAL = get_palette(getattr(getattr(st.context, "theme", None), "type", "light"))
SPORT_COLORS = PAL["sport"]
# Zonelabels zonder hartslaggrenzen: de grenzen verschillen per sport, dus in
# een grafiek die sporten optelt (weekoverzicht) zou één vast bereik liegen.
# Voor één sport tegelijk geeft zone_labels_for() de labels mét grenzen.
ZONE_LABELS = {
    "Z1": "Zone 1 (herstel)",
    "Z2": "Zone 2 (rustig duurtempo)",
    "Z3": "Zone 3 (grijze zone)",
    "Z4": "Zone 4 (drempel)",
    "Z5": "Zone 5 (maximaal)",
}
ZONE_COLORS = dict(zip(ZONE_LABELS.values(), PAL["zones"]))


def zone_labels_for(bounds: list[int]) -> dict[str, str]:
    """Zonelabels mét hartslaggrenzen, voor een grafiek over één sport."""
    return {
        "Z1": f"Zone 1 (< {bounds[0]})",
        "Z2": f"Zone 2 ({bounds[0]}–{bounds[1] - 1})",
        "Z3": f"Zone 3 ({bounds[1]}–{bounds[2] - 1})",
        "Z4": f"Zone 4 ({bounds[2]}–{bounds[3] - 1})",
        "Z5": f"Zone 5 (> {bounds[3]})",
    }


def zone_methode_caption() -> str:
    """Eén regel die per sport toont waarop beoordeeld wordt (UI-onderschrift)."""
    return " · ".join(
        f"**{sport_label(m.sport)}**: "
        + (m.bounds_text() if m.has_zones else "geen zones")
        for m in zone_overview(ATHLETE)
    )
# Vaste kleur per zwemslag (Nederlandse labels), zodat elke slag in elke
# grafiek dezelfde kleur houdt — ook als een sessie maar twee slagen bevat.
STROKE_COLORS = dict(zip(
    [stroke_label(s) for s in
     ("freestyle", "breaststroke", "backstroke", "butterfly", "mixed", "drill", "im")],
    PAL["cats"],
))
def chart(fig, show_legend: bool = True, key: str | None = None):
    """Render een figuur in huisstijl, met de gedeelde modebar-config.

    ``key`` is nodig wanneer dezelfde soort grafiek meerdere keren op een
    pagina staat (zoals per combinatietraining): twee figuren met identieke
    parameters krijgen anders hetzelfde auto-ID en dat laat Streamlit vallen.
    """
    st.plotly_chart(style_fig(fig, show_legend), width="stretch",
                    config=PLOTLY_CONFIG, key=key)


# Mini zoneverdeling-balk: 10 gekleurde blokjes per sessie, Z1..Z5 in oplopende
# kleur. In één blik zie je of een sessie echt rustig (veel groen) was of stiekem
# veel Z3/Z4 (geel/oranje) bevatte.
ZONE_SQUARES = ["🟦", "🟩", "🟨", "🟧", "🟥"]  # Z1, Z2, Z3, Z4, Z5


def zone_bar(z1: int, z2: int, z3: int, z4: int, z5: int, n: int = 10) -> str:
    """Tijd-in-zones als balkje van ``n`` gekleurde blokjes (grootste-rest-afronding)."""
    secs = [z1, z2, z3, z4, z5]
    total = sum(secs)
    if total <= 0:
        return "—"
    raw = [s / total * n for s in secs]
    blokken = [int(x) for x in raw]
    rest = n - sum(blokken)
    # De resterende blokjes naar de zones met de grootste afgekapte fractie.
    volgorde = sorted(range(5), key=lambda i: raw[i] - blokken[i], reverse=True)
    for i in range(rest):
        blokken[volgorde[i]] += 1
    return "".join(ZONE_SQUARES[i] * blokken[i] for i in range(5))


def z2_kleur(pct: float | None, z1: int, z2: int, z3: int, z4: int, z5: int) -> str:
    """Subtiel kleuraccent voor de %-Zone-2-cel: groen bij veel rustige tijd,
    oranje als er juist veel in Z3+ zat, anders neutraal."""
    total = z1 + z2 + z3 + z4 + z5
    if total <= 0 or pct is None or pd.isna(pct):
        return ""
    z3plus = 100.0 * (z3 + z4 + z5) / total
    if pct >= 60:
        return "background-color: rgba(84, 162, 75, 0.30)"   # subtiel groen
    if z3plus >= 40:
        return "background-color: rgba(245, 133, 24, 0.28)"  # subtiel oranje
    return ""


def trend_cell(info: dict | None) -> str:
    """Trendpijl + percentageverschil als compacte celtekst (⚠ bij terugval-vergelijking)."""
    if not info or info.get("delta_pct") is None:
        return "—"
    teken = "+" if info["delta_pct"] >= 0 else "−"
    tekst = f"{info['symbol']} {teken}{abs(info['delta_pct']):.0f}%"
    return tekst + " ⚠" if not info["exact"] else tekst


def veilig_cel(func, *args, fallback=GEEN_WAARDE):
    """Voer een cel-formatter uit; bij een fout de fallback i.p.v. een crash.

    Algemeen vangnet voor de 'Recente sessies'-tabel: een probleem op één
    sessie (een ontbrekend of NaN-veld) levert hooguit een placeholder in die
    cel op, nooit een ValueError die de hele pagina onderuit haalt. Ook bruikbaar
    voor niet-tekstuele berekeningen door een eigen ``fallback`` mee te geven
    (bijv. ``None`` als "kon niet worden bepaald").
    """
    try:
        return func(*args)
    except Exception:
        return fallback


def get_conn():
    """Open een verse databaseverbinding (goedkoop; vermijdt thread-gedoe)."""
    return connect(resolve_path(config, "database"))


def run_garmin_sync(client) -> None:
    """Volledige Garmin-sync: wellness + nieuwe activiteiten + feedback.

    Nieuwe activiteiten doorlopen exact dezelfde afhandeling als een
    handmatige upload: transport-vermoedens gaan naar de bevestigingsvraag,
    de rest krijgt coach-feedback die bovenaan de pagina verschijnt (en in de
    database bewaard blijft). Fouten per onderdeel worden meldingen, geen
    crashes — het dashboard draait altijd door op de bestaande data.
    """
    g = config.get("garmin", {})
    c = get_conn()
    router_sync = LLMRouter(config, MEMORY_DIR)
    meldingen = []
    try:
        w = garmin_sync.sync_wellness(
            client, c, days=int(g.get("wellness_days", 30)))
        meldingen.append(f"wellness {w.fetched} dag(en) bijgewerkt")

        a = garmin_sync.sync_activities(
            client, c, config, MEMORY_DIR,
            days=int(g.get("activities_days", 14)),
            observation_fn=lambda act, tiz: session_observation(
                router_sync, act, tiz),
            weather_fn=lambda act: wind_for_activity(act, MEMORY_DIR),
            uploads_dir=UPLOADS_DIR,
        )
        if a.new:
            meldingen.append(f"{len(a.new)} nieuwe activiteit(en)")
        bekend = a.skipped_known + a.duplicates
        if bekend:
            meldingen.append(f"{bekend} al bekend")
        if a.deleted_kept:
            meldingen.append(f"{a.deleted_kept} verwijderd gebleven")
        if a.errors:
            meldingen.append(f"{len(a.errors)} activiteit(en) mislukt")

        for r in a.new:
            if r.transport_suggested:
                st.session_state.setdefault("transport_suggesties", []).append(r)
                continue
            try:
                fb = generate_feedback(
                    router_sync, c, MEMORY_DIR, config,
                    r.activity, r.tiz, r.observation,
                    user_note=r.user_note, wind=r.wind,
                    training_label=r.training_label,
                )
                st.session_state.setdefault("upload_feedback", []).append(fb)
            except Exception as e:
                meldingen.append(f"feedback overgeslagen: {e}")

        garmin_sync.mark_synced(c, " · ".join(meldingen))
        st.session_state["sync_flash"] = "✅ Garmin-sync: " + " · ".join(meldingen)
    finally:
        c.close()


# ---------------------------------------------------------------- zijbalk --
with st.sidebar:
    st.title("🏊🚴🏃 Triatlon Coach")

    for race in config.get("races", []):
        race_date = race["date"] if isinstance(race["date"], date) else date.fromisoformat(str(race["date"]))
        days = (race_date - date.today()).days
        st.metric(race["name"], f"{days} dagen", help=race.get("distances", ""))

    st.divider()
    st.subheader("📤 Upload training")
    # Bewust gescheiden stappen: eerst bestand kiezen + eventueel een opmerking
    # typen, en pas op de knop wordt er ingelezen, opgeslagen en geanalyseerd.
    uploads = st.file_uploader(
        "Zip met FIT-bestanden (Garmin Connect → Origineel exporteren)",
        type="zip", accept_multiple_files=True,
    )
    user_note = st.text_area(
        "Opmerking (optioneel)",
        placeholder="bijv. voelde me moe · meewind heen, tegenwind terug · "
                    "nieuwe schoenen · intervaltraining bedoeld",
        help="Vrije context bij deze upload. De coach weegt dit mee bij de "
             "feedback en het wordt bij de sessie bewaard.",
    )
    # Sessielabel: een techniek-/cadanssessie hoort een hogere hartslag te
    # mogen hebben zonder dat de coach "te hard getraind" oordeelt.
    UPLOAD_LABELS = {
        "Geen label (normale sessie)": None,
        "Techniek/cadans — bewust op hogere cadans gelopen": "techniek/cadans",
    }
    label_keuze = st.selectbox(
        "Sessielabel (optioneel)", list(UPLOAD_LABELS),
        help="Bij 'techniek/cadans' weet de coach dat een hogere hartslag "
             "bij het oefenen hoort en beoordeelt hij de sessie niet als "
             "te hard getraind. Geldt voor alle sessies in deze upload; "
             "achteraf aan te passen op de 🔍 Sessie-tab.",
    )
    upload_label = UPLOAD_LABELS[label_keuze]
    start_upload = st.button(
        "🚀 Uploaden en analyseren", type="primary", disabled=not uploads,
        use_container_width=True,
    )

    if start_upload and uploads:
        router_upload = LLMRouter(config, MEMORY_DIR)
        conn = get_conn()
        verse_feedback = []
        with st.spinner("Importeren en analyseren..."):
            for up in uploads:
                results = import_zip(
                    up, conn, config, MEMORY_DIR,
                    observation_fn=lambda act, tiz: session_observation(router_upload, act, tiz),
                    weather_fn=lambda act: wind_for_activity(act, MEMORY_DIR),
                    user_note=user_note,
                    training_label=upload_label,
                    uploads_dir=UPLOADS_DIR,
                )
                for r in results:
                    icon = {"nieuw": "✅", "duplicaat": "↩️"}.get(r.status, "🗑️")
                    st.write(f"{icon} {local_time(r.activity.start_time):%d-%m %H:%M} "
                             f"{sport_label(r.activity.sport)} — {r.status}")
                    if r.status == "verwijderd":
                        st.caption(
                            "Deze sessie is eerder verwijderd en blijft verwijderd. "
                            "Herstellen kan via ⚙️ Instellingen → Verwijderde sessies.")
                    if r.enriched:
                        st.caption(
                            "📈 Aangevuld met velden die bij de eerdere import "
                            "nog niet werden opgeslagen (o.a. loopdynamiek).")
                    if r.status == "nieuw" and r.wind is not None:
                        st.caption(f"🌬️ Wind: {r.wind.as_text()}")
                    # Alleen nieuwe sessies krijgen coaching-feedback (Sonnet);
                    # duplicaten niet, dat zou onnodig een API-call kosten.
                    # Bij een transport-vermoeden (korte, rustige fietsrit)
                    # wordt de feedback uitgesteld tot de gebruiker de
                    # suggestie bovenaan de pagina bevestigt of afwijst —
                    # transport-ritjes horen geen coach-feedback te krijgen.
                    if r.status == "nieuw" and r.transport_suggested:
                        st.session_state.setdefault("transport_suggesties", []).append(r)
                        st.caption(
                            "🛒 Lijkt een transport-ritje (kort en rustig) — "
                            "bevestig of wijs af bovenaan de pagina.")
                    elif r.status == "nieuw":
                        try:
                            fb = generate_feedback(
                                router_upload, conn, MEMORY_DIR, config,
                                r.activity, r.tiz, r.observation,
                                user_note=r.user_note, wind=r.wind,
                                training_label=r.training_label,
                            )
                            verse_feedback.append(fb)
                        except Exception as e:
                            st.warning(f"Feedback overgeslagen: {e}")
        conn.close()
        if verse_feedback:
            # Bovenaan de hoofdpagina tonen (buiten de zijbalk); blijft staan tot
            # de volgende upload of tot 'sluiten'.
            st.session_state["upload_feedback"] = verse_feedback

    st.divider()
    st.subheader("🔄 Garmin-sync")
    _c = get_conn()
    laatste_sync = garmin_sync.last_sync(_c)
    _c.close()
    if laatste_sync:
        st.caption(f"Laatst gesynct: **{laatste_sync:%d-%m-%Y %H:%M}**")
    else:
        st.caption(
            "Nog nooit gesynct. Zet `GARMIN_EMAIL` en `GARMIN_PASSWORD` in "
            "`.env` (naast de API-key) en klik hieronder; eenmalig inloggen "
            "is genoeg, daarna werken de bewaarde tokens."
        )

    mfa_pending = st.session_state.get("garmin_mfa")
    if mfa_pending:
        # Garmin vraagt een MFA-code (mail/app); de login wacht in
        # session_state tot de code er is.
        mfa_code = st.text_input(
            "MFA-code van Garmin", key="garmin_mfa_code",
            help="Garmin stuurde een code per e-mail of app; vul die hier in "
                 "om de koppeling af te ronden.",
        )
        col_ok, col_stop = st.columns(2)
        if col_ok.button("✅ Code bevestigen", disabled=not mfa_code,
                         use_container_width=True):
            client, mfa_state = mfa_pending
            try:
                client = garmin_sync.complete_mfa(client, mfa_state, mfa_code)
                del st.session_state["garmin_mfa"]
                with st.spinner("Synchroniseren met Garmin..."):
                    run_garmin_sync(client)
                st.rerun()
            except garmin_sync.GarminSyncError as e:
                st.error(str(e))
        if col_stop.button("Annuleren", use_container_width=True):
            del st.session_state["garmin_mfa"]
            st.rerun()
    elif st.button("🔄 Nu synchroniseren", use_container_width=True):
        # Zacht falen: elke fout wordt een melding in de zijbalk, nooit een
        # crash — het dashboard blijft op de bestaande data draaien.
        try:
            with st.spinner("Verbinden met Garmin..."):
                status, payload = garmin_sync.connect_client()
            if status == "mfa":
                st.session_state["garmin_mfa"] = payload
                st.rerun()
            else:
                with st.spinner("Synchroniseren met Garmin..."):
                    run_garmin_sync(payload)
                st.rerun()
        except garmin_sync.GarminSyncError as e:
            st.error(str(e))
        except Exception as e:
            st.error(f"Garmin-sync mislukt: {e}")

    st.caption("Weather data by [Open-Meteo.com](https://open-meteo.com) (CC BY 4.0)")

# ------------------------------------------------------------------- data --
conn = get_conn()


@st.cache_resource(show_spinner=False)
def archief_inhaalslag() -> int:
    """Eenmalig per serverstart: sluimerende originelen (oude importmap)
    alsnog in het uploads-archief zetten; zie tricoach.archive."""
    c = get_conn()
    try:
        return migrate_originals(c, UPLOADS_DIR,
                                 [resolve_path(config, "import_dir")])
    finally:
        c.close()


archief_inhaalslag()


def auto_sync_indien_nodig() -> None:
    """Stille periodieke Garmin-sync bij het laden van de pagina.

    Draait hooguit één keer per browsersessie, alleen als hij in de config
    aanstaat, er bewaarde tokens zijn (nooit een interactieve login of MFA
    afdwingen) en de vorige sync oud genoeg is. Elke fout is een stille
    waarschuwing — het dashboard werkt gewoon door op de bestaande data.
    """
    g = config.get("garmin", {})
    if not g.get("auto_sync", False) or st.session_state.get("auto_sync_gedaan"):
        return
    if not garmin_sync.has_tokens():
        # Nog nooit ingelogd: stil overslaan — de zijbalk legt de setup uit.
        # De waarschuwing hieronder is voor échte problemen (verlopen tokens,
        # netwerk), niet voor een koppeling die simpelweg nog niet bestaat.
        return
    st.session_state["auto_sync_gedaan"] = True
    laatste = garmin_sync.last_sync(conn)
    uren = float(g.get("auto_sync_hours", 6))
    if laatste and datetime.now() - laatste < timedelta(hours=uren):
        return
    try:
        with st.spinner("Automatische Garmin-sync..."):
            _, client = garmin_sync.connect_client(tokens_only=True)
            run_garmin_sync(client)
    except Exception as e:
        st.warning(f"Automatische Garmin-sync mislukt ({e}) — het dashboard "
                   "toont de laatst bekende data. Handmatig syncen kan via "
                   "de zijbalk.")


auto_sync_indien_nodig()

acts = load_activities(conn)

if acts.empty:
    st.info("Nog geen trainingen geïmporteerd. Upload een zip via de zijbalk.")
    st.stop()

acts["start_time"] = acts["start_time"].dt.tz_convert(TZ)
acts["Sport"] = acts["sport"].map(sport_label)
# De analysesubset: alles behalve als-transport-gemarkeerde sessies. Trends,
# records, zone-statistieken en brick-detectie rekenen hierop; de volledige
# `acts` blijft voor weektotalen, belasting en de sessielijst (daar staan
# transport-ritjes gedimd bij).
trainingen = training_activities(acts)
# Eén keer per rerun: de zuivere zwemtijd per sessie (som van actieve banen),
# gebruikt door de sessietabel, de tempografieken en het sessie-detail.
zwem_actief = swim_active_seconds(conn)

# Combinatietrainingen: detecteer nieuwe brick-/triatlonvoorstellen (goedkoop,
# alleen sessie-metadata) en haal het lidmaatschap op voor de sessietabel.
# Voorstellen worden nooit stilzwijgend samengevoegd: bevestigen of losmaken
# gebeurt op de 🧱 Bricks-tab. Transport-ritjes doen niet mee: terugfietsen
# van het zwembad is geen brick.
detect_and_store_proposals(conn, trainingen, max_gap_min(config))
combo_leden = combo_membership(conn)

router = LLMRouter(config, MEMORY_DIR)

# Cache rond de berekeningen die alle seconde-records doorlopen: die worden
# anders bij elke rerun (elke klik) opnieuw gedaan en groeien mee met de
# database. De datastand (aantal sessies + nieuwste sessie + aantal
# transport-markeringen) is de cachesleutel: na een upload of een (de)markering
# verandert die en wordt alles vers berekend.
DATA_VERSIE = (len(acts), acts["start_time"].max().isoformat(),
               int(acts["excluded_reason"].notna().sum()))


@st.cache_data(show_spinner=False)
def cache_pace_at_hr(sport: str, bereik: tuple, versie: tuple) -> pd.DataFrame:
    return pace_at_hr(conn, trainingen, sport, bereik)


@st.cache_data(show_spinner=False)
def cache_decoupling(versie: tuple) -> pd.DataFrame:
    return decoupling(conn, trainingen)


@st.cache_data(show_spinner=False)
def cache_personal_records(versie: tuple) -> pd.DataFrame:
    return personal_records(conn, trainingen)


@st.cache_data(show_spinner=False)
def cache_best_efforts(versie: tuple) -> pd.DataFrame:
    return best_efforts(conn, trainingen)


@st.cache_data(show_spinner=False)
def cache_prediction_history(race_in: dict, versie: tuple) -> pd.DataFrame:
    return race_prediction_history(conn, trainingen, race_in)


@st.cache_data(show_spinner=False)
def cache_css(versie: tuple) -> dict | None:
    return css_estimate(conn, trainingen)


@st.cache_data(show_spinner=False)
def cache_power_decoupling(versie: tuple) -> pd.DataFrame:
    return power_decoupling_trend(conn, trainingen)


@st.cache_data(show_spinner=False)
def cache_ftp_estimate(versie: tuple) -> dict | None:
    return estimate_ftp(conn, trainingen)


# De heatmap heeft een eigen cachesleutel: niet de sessiestand maar de stand
# van de trackcache (zie tricoach.heatmap.cache_stats). Die verandert alleen
# als er GPS is bijgekomen, dus filteren en inzoomen kost geen herberekening.
@st.cache_data(show_spinner=False)
def cache_track_points(hm_versie: tuple) -> pd.DataFrame:
    """Alle gecachede trackpunten met hun sessiecontext (sport, datum, markeringen)."""
    c = get_conn()
    try:
        return heatmap_mod.load_track_points(c, tz=TZ)
    finally:
        c.close()


@st.cache_data(show_spinner=False)
def cache_heatmap_cells(hm_versie: tuple, filters: tuple, cell_m: float,
                        schaal: str, zone: tuple) -> pd.DataFrame:
    """Gekleurde rastercellen voor de huidige filters (zie tricoach.heatmap)."""
    punten = cache_track_points(hm_versie)
    categorieen, start, eind, met_transport, met_verwijderd = filters
    punten = heatmap_mod.filter_points(
        punten, categories=categorieen, start=start, end=eind,
        include_transport=met_transport, include_deleted=met_verwijderd)
    if zone[0] is not None:
        punten = heatmap_mod.apply_privacy_zone(punten, zone[0], zone[1], zone[2])
    return heatmap_mod.heatmap_cells(punten, cell_m=cell_m, method=schaal)


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def cache_verwachte_temperatuur(lat: float, lon: float, dag: date) -> float | None:
    """De verwachte temperatuur voor een geplande sessie (Open-Meteo).

    Gecachet met een TTL van zes uur: de voedingsplanner wil de verwachting
    standaard invullen, maar een weerbericht verandert niet per klik en de app
    hoort niet bij elke rerun een externe dienst te bevragen. Per dag en
    locatie gaat er dus hooguit een paar keer per dag één request uit; falen
    is zacht (None → zelf invullen).
    """
    try:
        return forecast_temperature(lat, lon, dag, memory_dir=MEMORY_DIR)
    except Exception:
        return None


def render_upload_feedback():
    """Toon de feedback van de zojuist geüploade sessies prominent bovenaan.

    Per sessie: de coaching-feedback, de kerncijfers + zoneverdeling, en — als
    de coach iets voorstelt — een opvallend aanpassingsblok met een knop om die
    aanpassing in de planning over te nemen.
    """
    fbs = st.session_state.get("upload_feedback")
    if not fbs:
        return
    with st.container(border=True):
        kop, sluit = st.columns([6, 1])
        kop.subheader("🆕 Feedback op je upload")
        if sluit.button("Sluiten", key="dismiss_feedback"):
            del st.session_state["upload_feedback"]
            st.rerun()

        for i, fb in enumerate(fbs):
            st.markdown(f"#### {fb.sport} — {fb.start_time}")
            st.success(fb.feedback)
            st.caption(f"**Kerncijfers:** {fb.kerncijfers}  \n**Tijd in zones:** {fb.zoneverdeling}")
            if fb.aanpassing:
                st.warning(f"**Voorgestelde aanpassing volgende sessie:** {fb.aanpassing}")
                if st.button("➡️ Aanpassing overnemen in planning", key=f"adopt_{i}"):
                    add_note_row(MEMORY_DIR, fb.aanpassing)
                    st.toast("Aanpassing toegevoegd aan het weekschema (Coach-tab).")
            else:
                st.info("Volgende sessie zoals gepland — geen aanpassing nodig.")
            if i < len(fbs) - 1:
                st.divider()


def render_transport_suggesties():
    """Vraag per vermoedelijk transport-ritje om bevestiging, bovenaan de pagina.

    De import markeert nooit zelf (zie tricoach.transport): hier kiest de
    gebruiker met één klik. "Ja" markeert de sessie als transport (geen
    feedback, telt niet mee in trends); "Nee" laat de uitgestelde
    coach-feedback alsnog genereren.
    """
    suggesties = st.session_state.get("transport_suggesties")
    if not suggesties:
        return
    with st.container(border=True):
        st.subheader("🛒 Transport-ritje?")
        st.caption(
            "Deze uploads lijken verplaatsingen (korte fietsrit, hartslag "
            "onder trainingsintensiteit). Als transport tellen ze wél mee in "
            "je weektotalen en belasting, maar niet in trends, records en "
            "de feedback-vergelijking. Achteraf aanpassen kan op de 🔍 Sessie-tab."
        )
        for r in list(suggesties):
            key = r.activity.activity_key
            info, ja, nee = st.columns([4, 1, 2])
            info.write(
                f"**{local_time(r.activity.start_time):%d-%m %H:%M}** · "
                f"{sport_label(r.activity.sport)} · "
                f"{r.activity.distance_m / 1000:.1f} km · "
                f"HR gem {r.activity.summary.get('avg_heart_rate', '—')}"
            )
            if ja.button("✅ Ja, transport", key=f"transport_ja_{key}"):
                c = get_conn()
                mark_transport(c, MEMORY_DIR, key)
                c.close()
                suggesties.remove(r)
                st.toast("Gemarkeerd als transport — geen coach-feedback nodig.")
                st.rerun()
            if nee.button("➖ Nee, gewone training", key=f"transport_nee_{key}"):
                suggesties.remove(r)
                # De bij de import uitgestelde feedback alsnog genereren.
                try:
                    with st.spinner("Coach-feedback genereren..."):
                        c = get_conn()
                        fb = generate_feedback(
                            router, c, MEMORY_DIR, config,
                            r.activity, r.tiz, r.observation,
                            user_note=r.user_note, wind=r.wind,
                            training_label=r.training_label,
                        )
                        c.close()
                    st.session_state.setdefault("upload_feedback", []).append(fb)
                except Exception as e:
                    st.warning(f"Feedback overgeslagen: {e}")
                st.rerun()


if sync_flash := st.session_state.pop("sync_flash", None):
    st.success(sync_flash)
render_transport_suggesties()
render_upload_feedback()

(tab_overzicht, tab_trends, tab_voortgang, tab_sessie, tab_lopen, tab_fietsen,
 tab_zwemmen, tab_bricks, tab_lichaam, tab_herstel, tab_voeding, tab_coach,
 tab_chat, tab_heatmap, tab_log, tab_settings) = st.tabs(
    ["📋 Overzicht", "📈 Trends", "🚀 Voortgang", "🔍 Sessie", "🏃 Lopen", "🚴 Fietsen",
     "🏊 Zwemmen", "🧱 Bricks", "🧍 Lichaam", "🌙 Herstel", "🥤 Voeding", "🧠 Coach",
     "💬 Chat", "🗺️ Heatmap", "📖 Logboek", "⚙️ Instellingen"]
)

# --------------------------------------------------------------- overzicht --
with tab_overzicht:
    week_ago = pd.Timestamp.now(tz=TZ) - pd.Timedelta(days=7)
    # Volume-metrics op álle sessies (compleet belastingsbeeld, incl.
    # transport); de zone-aandelen alleen op echte trainingen.
    recent = acts[acts["start_time"] >= week_ago]
    recent_training = trainingen[trainingen["start_time"] >= week_ago]

    c1, c2, c3, c4 = st.columns(4)
    n_transport = len(recent) - len(recent_training)
    c1.metric("Sessies (7 dagen)", len(recent),
              help=f"Waarvan {n_transport} transport/verplaatsing."
              if n_transport else None)
    c2.metric("Trainingsuren (7 dagen)", f"{recent['duration_s'].sum() / 3600:.1f}")
    z2_share = (recent_training["z2_s"].sum()
                / max(recent_training["duration_s"].sum(), 1) * 100)
    c3.metric("Aandeel zone 2 (7 dagen)", f"{z2_share:.0f}%",
              help="Doel: dit omhoog krijgen — je traint structureel te hard. "
                   "Transport-ritjes tellen hier niet mee.")
    hard = ((recent_training["z4_s"].sum() + recent_training["z5_s"].sum())
            / max(recent_training["duration_s"].sum(), 1) * 100)
    c4.metric("Aandeel zone 4+5 (7 dagen)", f"{hard:.0f}%")

    col_links, col_rechts = st.columns(2)
    with col_links:
        st.subheader("Weekvolume per sport")
        vol = weekly_volume(acts)
        vol["sport"] = vol["sport"].map(sport_label)
        fig = px.bar(
            vol, x="week", y="uren", color="sport",
            color_discrete_map=SPORT_COLORS,
            category_orders={"week": sorted(vol["week"].unique())},
            labels={"week": "Week", "uren": "Uren", "sport": "Sport"},
        )
        fig.update_traces(
            marker_line=dict(width=1, color=PAL["surface"]),
            hovertemplate="%{x} · %{fullData.name}: %{y:.1f} uur<extra></extra>")
        chart(fig)

    with col_rechts:
        st.subheader("Tijd in hartslagzones per week")
        st.caption(
            "De grenzen verschillen per sport, dus de balken tellen zones op "
            "die elk tegen hun eigen drempel zijn bepaald: "
            + zone_methode_caption()
            + ". Zwemmen telt hier niet mee (geen zones)."
        )
        tz_df = weekly_zone_time(trainingen)
        tz_df["zone"] = tz_df["zone"].map(ZONE_LABELS)
        als_pct = st.toggle(
            "Toon als percentage", key="zonetijd_pct",
            help="Percentages maken weken met verschillend volume vergelijkbaar.")
        if als_pct:
            tz_df["waarde"] = (tz_df["minuten"] / tz_df.groupby("week")["minuten"]
                               .transform("sum") * 100)
            y_col, y_label, hover_fmt = "waarde", "Aandeel (%)", "%{y:.0f}%"
        else:
            y_col, y_label, hover_fmt = "minuten", "Minuten", "%{y:.0f} min"
        fig = px.bar(
            tz_df, x="week", y=y_col, color="zone",
            color_discrete_map=ZONE_COLORS,
            category_orders={
                "week": sorted(tz_df["week"].unique()),
                "zone": list(ZONE_COLORS),
            },
            labels={"week": "Week", y_col: y_label, "zone": "Zone"},
        )
        fig.update_traces(
            marker_line=dict(width=1, color=PAL["surface"]),
            hovertemplate=f"%{{x}} · %{{fullData.name}}: {hover_fmt}<extra></extra>")
        chart(fig)

    st.subheader("Mijn drempels & zones per sport")
    st.caption(
        "Elke sport heeft een eigen drempel; ze zijn niet uitwisselbaar. "
        + zone_methode_caption()
        + ". Pas ze aan op de ⚙️ Instellingen-tab; de zonetijden van alle "
        "sessies worden dan herrekend."
    )
    m_run, m_bike, m_swim = zone_overview(ATHLETE)
    k1, k2, k3 = st.columns(3)
    k1.metric("Loop-LTHR", f"{run_lthr(ATHLETE)} bpm",
              help=run_lthr_source(ATHLETE) or "Handmatig ingesteld.")
    k2.metric("FTP (fietsen)", f"{FTP:.0f} W" if FTP else "onbekend",
              help="Zodra de FTP bekend is, worden fietssessies op "
                   "%FTP-vermogenszones beoordeeld in plaats van op hartslag.")
    k3.metric("Fiets-LTHR", f"{bike_lthr(ATHLETE)} bpm",
              help="Geschat uit de loop-LTHR." if bike_lthr_is_estimated(ATHLETE)
                   else "Handmatig ingesteld.")
    if m_bike.provisional:
        st.info(
            "🔧 **Tussenoplossing voor fietsen:** er is nog geen FTP, dus "
            f"fietssessies worden voorlopig op hartslagzones rond de "
            f"fiets-LTHR ({bike_lthr(ATHLETE)} bpm) beoordeeld. Doe een "
            "**20-minutentest** op de Kickr: daar haal je in één inspanning "
            "zowel je FTP (95% van het gemiddelde vermogen) als je fiets-LTHR "
            "(gemiddelde hartslag over dat blok) uit. Een ramptest geeft je "
            "alleen het vermogen."
        )
    # Drempels die nog op een schatting rusten, expliciet als zodanig tonen.
    for notitie in threshold_notes(ATHLETE):
        st.caption(f"❓ {notitie}")

    st.markdown("**Loop-LTHR door de tijd** (de zones voor hardlopen schuiven mee)")
    st.caption(
        "Bij een wijziging worden alle zonetijden herrekend, dus de trends "
        "blijven onderling vergelijkbaar — zie memory/beslissingen.md. Let op: "
        "Garmin's automatische schatting zakt tijdelijk ná een brick-training "
        "(de fietsbelasting vóór het lopen drukt de loopdrempel). Zo'n dip is "
        "een meetartefact; neem hem niet over zonder bevestiging uit lósse "
        "loopsessies."
    )
    hist = lthr_history(MEMORY_DIR, run_lthr(ATHLETE), kind=LTHR_RUN)
    if hist.empty:
        st.info("Nog geen loop-LTHR-geschiedenis vastgelegd.")
    else:
        dates = [pd.Timestamp(d) for d in hist["datum"]]
        end = pd.Timestamp(date.today())
        if end <= dates[-1]:
            end = dates[-1] + pd.Timedelta(days=30)
        dates.append(end)
        lthrs = list(hist["lthr"]) + [int(hist["lthr"].iloc[-1])]
        pcts = ATHLETE.get("zone_pct_lthr")
        per_date = [bounds_from_lthr(l, pcts) for l in lthrs]

        # Labels mét grenzen: deze grafiek gaat over één sport (hardlopen).
        run_labels = zone_labels_for(RUN_BOUNDS)
        fig = go.Figure()
        floor = min(b[0] for b in per_date) - 25
        fig.add_trace(go.Scatter(  # onzichtbare onderkant van de zone-1-band
            x=dates, y=[floor] * len(dates), line=dict(width=0),
            line_shape="hv", hoverinfo="skip", showlegend=False,
        ))
        tops = [
            [b[0] for b in per_date], [b[1] for b in per_date],
            [b[2] for b in per_date], [b[3] for b in per_date],
            [ATHLETE["max_hr"]] * len(dates),
        ]
        for (zone, kleur), top in zip(zip(ZONE_LABELS, PAL["zones"]), tops):
            label = run_labels[zone]
            fig.add_trace(go.Scatter(
                x=dates, y=top, name=label, fill="tonexty",
                line=dict(width=0), fillcolor=with_alpha(kleur, 0.55),
                line_shape="hv",
                hovertemplate=f"{label}: tot %{{y}} bpm<extra></extra>",
            ))
        fig.add_trace(go.Scatter(
            x=dates, y=lthrs, name="Loop-LTHR",
            line=dict(color=PAL["ink"], dash="dash", width=2),
            line_shape="hv", hovertemplate="Loop-LTHR: %{y} bpm<extra></extra>",
        ))
        fig.update_layout(yaxis_title="Hartslag (bpm)", xaxis_title="Datum")
        chart(fig)

    st.subheader("Recente sessies")
    trend = aerobic_efficiency_trend(trainingen)
    tabel = acts.head(15).copy()
    # Transport-ritjes blijven zichtbaar (compleet logboek) maar gedimd en
    # met label; markeren/demarkeren gebeurt op de 🔍 Sessie-tab.
    tabel["is_transport"] = tabel.apply(is_transport, axis=1)
    tabel.loc[tabel["is_transport"], "Sport"] += " · 🛒 transport"
    tabel["Duur"] = pd.to_datetime(tabel["duration_s"], unit="s").dt.time
    tabel["Afstand"] = tabel["distance_m"] / 1000
    # Tempo/snelheid op de actieve tijd (zie tricoach.timebasis): rust aan de
    # kant en stilstand tellen niet mee; terugval op de timer-duur als de
    # actieve tijd ontbreekt. Elke cel is afgeschermd zodat één rij met een
    # gat in de data de tabel niet laat crashen.
    tabel["Tempo / snelheid"] = tabel.apply(
        lambda r: veilig_cel(sessie_tempo, r["sport"], r["distance_m"],
                             r["duration_s"], r.get("active_s")),
        axis=1)
    # Vermogen en cadans, alleen zinvol bij fietssessies met powerdata (sinds
    # de Rally/Kickr); alle andere rijen tonen een streepje.
    def vermogen_cel(r) -> str:
        if r["sport"] != "cycling" or pd.isna(r.get("avg_power")):
            return GEEN_WAARDE
        tekst = f"{r['avg_power']:.0f} W"
        if not pd.isna(r.get("np_power")):
            tekst += f" · NP {r['np_power']:.0f}"
        if r.get("is_indoor"):
            tekst += " 🏠"
        return tekst

    def cadans_cel(r) -> str:
        if r["sport"] != "cycling":
            return GEEN_WAARDE
        cadans = r.get("avg_cadence_excl0")
        if cadans is None or pd.isna(cadans):
            cadans = r.get("avg_cadence")
        if cadans is None or pd.isna(cadans):
            return GEEN_WAARDE
        return f"{cadans:.0f} rpm"

    tabel["Vermogen"] = tabel.apply(lambda r: veilig_cel(vermogen_cel, r), axis=1)
    tabel["Cadans"] = tabel.apply(lambda r: veilig_cel(cadans_cel, r), axis=1)
    tabel["% Z2"] = tabel["pct_in_zone2"].map(
        lambda v: GEEN_WAARDE if pd.isna(v) else f"{v:.0f}%")
    tabel["Zones"] = tabel.apply(
        lambda r: veilig_cel(zone_bar, r["z1_s"], r["z2_s"], r["z3_s"], r["z4_s"], r["z5_s"]),
        axis=1)
    tabel["Trend"] = tabel["activity_key"].map(
        lambda k: veilig_cel(trend_cell, trend.get(k)))
    # Onderdelen van een (voorgestelde of bevestigde) combinatietraining krijgen
    # een marker; het samengestelde blok zelf staat op de 🧱 Bricks-tab.
    tabel["Combo"] = tabel["activity_key"].map(
        lambda k: ("🧱 " + combo_leden[k]["kind"]
                   + ("?" if combo_leden[k]["status"] == "voorgesteld" else ""))
        if k in combo_leden else "")

    vis = tabel[["start_time", "Sport", "Duur", "Afstand", "avg_hr",
                 "Tempo / snelheid", "Vermogen", "Cadans", "% Z2", "Zones",
                 "Trend", "Combo"]]
    # Kleuraccent voor de %-Z2-cel, per rij vooraf bepaald (index-gekoppeld).
    css = {idx: ("" if r["is_transport"] else
                 z2_kleur(r["pct_in_zone2"], r["z1_s"], r["z2_s"],
                          r["z3_s"], r["z4_s"], r["z5_s"]))
           for idx, r in tabel.iterrows()}
    styler = vis.style.apply(
        lambda col: [css[i] for i in col.index], subset=["% Z2"])
    # Transport-rijen gedimd: wel zichtbaar in het logboek, duidelijk geen
    # training.
    dim = "color: #898781; font-style: italic; opacity: 0.75"
    styler = styler.apply(
        lambda row: [dim if tabel.loc[row.name, "is_transport"] else ""] * len(row),
        axis=1)
    st.dataframe(
        styler,
        column_config={
            "start_time": st.column_config.DatetimeColumn("Datum", format="DD-MM-YYYY HH:mm"),
            "Duur": st.column_config.TimeColumn("Duur", format="H:mm:ss"),
            "Afstand": st.column_config.NumberColumn("Afstand", format="%.2f km"),
            "Tempo / snelheid": st.column_config.TextColumn(
                "Actief tempo / snelheid",
                help="Op actieve/bewegende tijd: rust aan de badrand, "
                     "stilstand en pauzes tellen niet mee. Kan daardoor "
                     "sneller zijn dan de weergave in Garmin Connect."),
            "avg_hr": st.column_config.NumberColumn("Gem. HR"),
            "Vermogen": st.column_config.TextColumn(
                "Vermogen",
                help="Gemiddeld vermogen · NP (normalized power) van "
                     "fietssessies met vermogensmeter (Rally buiten, Kickr "
                     "binnen — 🏠 = indoor). NP weegt pieken zwaarder en is "
                     "voor wisselend buitenrijden de betere intensiteitsmaat."),
            "Cadans": st.column_config.TextColumn(
                "Cadans",
                help="Gemiddelde trapfrequentie (rpm, excl. freewheelen — "
                     "zoals Garmin). Alleen voor fietssessies."),
            "% Z2": st.column_config.TextColumn(
                "% Zone 2",
                help="Aandeel van de gemeten hartslagtijd in zone 2. De grens "
                     f"verschilt per sport: hardlopen {RUN_Z2[0]}–{RUN_Z2[1]}, "
                     f"fietsen {BIKE_Z2[0]}–{BIKE_Z2[1]}. Groen = veel rustige "
                     "tijd; oranje = juist veel in Z3+. Zwemmen krijgt geen "
                     "zone-oordeel."),
            "Zones": st.column_config.TextColumn(
                "Zoneverdeling",
                help="Tijd per zone in 10 blokjes: 🟦 Z1 · 🟩 Z2 · 🟨 Z3 · 🟧 Z4 · 🟥 Z5."),
            "Trend": st.column_config.TextColumn(
                "Aerobe trend",
                help="Snelheid per hartslag t.o.v. de vorige vergelijkbare sessie "
                     "(zelfde sport én intensiteit). ▲ efficiënter · ▼ minder · "
                     "▬ gelijk (±2%). ⚠ = vergeleken met de dichtstbijzijnde i.p.v. een "
                     "gelijke-intensiteit sessie. Zwemmen krijgt geen pijl."),
            "Combo": st.column_config.TextColumn(
                "Combo",
                help="Onderdeel van een combinatietraining (brick of "
                     "triatlon-training). Een ? betekent: nog niet bevestigd — "
                     "zie de 🧱 Bricks-tab."),
        },
        hide_index=True, width="stretch",
    )
    st.caption(
        "**Aerobe trend** — snelheid bij gelijke hartslag t.o.v. je vorige vergelijkbare "
        "sessie (zelfde sport én intensiteit): ▲ sneller · ▼ langzamer · ▬ gelijk (±2%) · "
        "— geen vergelijkbare sessie. ⚠ markeert een vergelijking met de dichtstbijzijnde "
        "sessie omdat er geen eerdere sessie van gelijke intensiteit was. Zwemmen krijgt "
        "geen pijl: de pols-hartslag onder water is onbetrouwbaar."
    )

    st.subheader("Weektotalen")
    totalen = weekly_totals(acts)
    # De transport-kolom alleen tonen als er ooit iets te tonen valt.
    toon_transport = totalen["uren_transport"].sum() > 0
    # Uren als u:mm — '4:18' leest makkelijker dan '4.3 uur'.
    totalen["uren"] = totalen["uren"].map(fmt_hours_hhmm)
    totalen["delta_uren"] = totalen["delta_uren"].map(
        lambda u: fmt_hours_hhmm(u, signed=True))
    for kolom in ("uren_swimming", "uren_cycling", "uren_running",
                  "uren_transport"):
        totalen[kolom] = totalen[kolom].map(fmt_hours_hhmm)
    week_kolommen = ["week", "sessies", "uren", "delta_uren", "uren_swimming",
                     "uren_cycling", "uren_running"]
    if toon_transport:
        week_kolommen.append("uren_transport")
    st.dataframe(
        totalen[week_kolommen + ["km", "trimp"]],
        column_config={
            "week": st.column_config.TextColumn("Week"),
            "sessies": st.column_config.NumberColumn("Sessies"),
            "uren": st.column_config.TextColumn("Uren"),
            "delta_uren": st.column_config.TextColumn(
                "Δ uren", help="Verschil in trainingsuren met de week eronder."),
            "uren_swimming": st.column_config.TextColumn("🏊 Zwemmen"),
            "uren_cycling": st.column_config.TextColumn("🚴 Fietsen"),
            "uren_running": st.column_config.TextColumn("🏃 Lopen"),
            "uren_transport": st.column_config.TextColumn(
                "🛒 Transport",
                help="Verplaatsingen (naar het zwembad, boodschappen): tellen "
                     "mee in het weektotaal en de belasting, niet in trends "
                     "en trainingsstatistieken."),
            "km": st.column_config.NumberColumn("Km totaal", format="%.0f"),
            "trimp": st.column_config.NumberColumn(
                "TRIMP", format="%.0f",
                help="Totale trainingsbelasting van de week (tijd × zonegewicht; "
                     "zwemmen op basis van de gemiddelde hartslag)."),
        },
        hide_index=True, width="stretch",
    )

# ------------------------------------------------------------------ trends --
with tab_trends:
    st.subheader("Tempo bij gelijke hartslag (zone 2)")
    st.caption(
        "De belangrijkste grafiek: gemiddeld tempo van alle meetpunten binnen zone 2, "
        "per sessie. Sneller worden bij dezelfde hartslag = grotere aerobe basis. "
        "Sessies met minder dan 5 minuten in zone 2 worden weggelaten. Zone 2 "
        f"loopt bij hardlopen van {RUN_Z2[0]} tot {RUN_Z2[1]} (loop-LTHR "
        f"{run_lthr(ATHLETE)}) en bij fietsen van {BIKE_Z2[0]} tot {BIKE_Z2[1]} "
        f"(fiets-LTHR {bike_lthr(ATHLETE)}) — elke sport tegen zijn eigen drempel."
    )

    col_run, col_bike = st.columns(2)
    with col_run:
        run_trend = cache_pace_at_hr("running", RUN_Z2, DATA_VERSIE)
        n_runs = (trainingen["sport"] == "running").sum()
        if run_trend.empty:
            st.info("Nog geen loopsessies met ≥5 min in zone 2 — dat zegt op zich al iets 😉")
        else:
            run_trend["tempo"] = pace_as_time(1000 / run_trend["speed_ms"])
            fig = px.line(
                run_trend, x="start_time", y="tempo", markers=True,
                labels={"start_time": "Datum", "tempo": "Tempo (min/km)"},
            )
            pace_axis(fig)
            fig.update_traces(
                marker=dict(size=11),
                hovertemplate="%{x|%d-%m-%Y} · %{y|%M:%S} min/km<extra></extra>",
            )
            fig.update_layout(
                title=f"Hardlopen — tempo in zone 2 ({RUN_Z2[0]}–{RUN_Z2[1]}, "
                      "sneller = hoger)")
            toon_legenda = len(run_trend) >= 4
            if toon_legenda:  # voortschrijdend gemiddelde dempt dagvorm en weer
                fig.update_traces(name="Per sessie", showlegend=True)
                gemiddeld = (1000 / run_trend["speed_ms"]).rolling(3).mean()
                fig.add_scatter(
                    x=run_trend["start_time"], y=pace_as_time(gemiddeld),
                    mode="lines", name="Gemiddelde (3 sessies)",
                    line=dict(color=PAL["muted"], dash="dot", width=2),
                    hoverinfo="skip")
            pad_single_point(fig, run_trend["start_time"],
                             y_center=run_trend["tempo"].iloc[0],
                             y_pad=pd.Timedelta(seconds=30), reversed_y=True)
            chart(fig, show_legend=toon_legenda)
        if 0 < len(run_trend) < n_runs:
            st.caption(
                f"{n_runs - len(run_trend)} van je {n_runs} loopsessies is weggelaten: "
                "minder dan 5 minuten in zone 2 (die sessie was vrijwel volledig Z3+)."
            )

    with col_bike:
        bike_trend = cache_pace_at_hr("cycling", BIKE_Z2, DATA_VERSIE)
        n_rides = (trainingen["sport"] == "cycling").sum()
        if bike_trend.empty:
            st.info("Nog geen fietssessies met ≥5 min in zone 2.")
        else:
            fig = px.line(
                bike_trend, x="start_time", y="snelheid_kmh", markers=True,
                labels={"start_time": "Datum", "snelheid_kmh": "Snelheid (km/h)"},
            )
            fig.update_traces(
                marker=dict(size=11),
                hovertemplate="%{x|%d-%m-%Y} · %{y:.1f} km/h<extra></extra>",
            )
            fig.update_layout(
                title=f"Fietsen — snelheid in zone 2 ({BIKE_Z2[0]}–{BIKE_Z2[1]}, "
                      "fiets-LTHR)")
            toon_legenda = len(bike_trend) >= 4
            if toon_legenda:
                fig.update_traces(name="Per sessie", showlegend=True)
                fig.add_scatter(
                    x=bike_trend["start_time"],
                    y=bike_trend["snelheid_kmh"].rolling(3).mean(),
                    mode="lines", name="Gemiddelde (3 sessies)",
                    line=dict(color=PAL["muted"], dash="dot", width=2),
                    hoverinfo="skip")
            pad_single_point(fig, bike_trend["start_time"],
                             y_center=bike_trend["snelheid_kmh"].iloc[0], y_pad=3)
            chart(fig, show_legend=toon_legenda)
        if 0 < len(bike_trend) < n_rides:
            st.caption(
                f"{n_rides - len(bike_trend)} van je {n_rides} fietssessies is weggelaten: "
                "minder dan 5 minuten in zone 2."
            )

    st.divider()
    st.subheader("Intensiteitsverdeling per week (80/20-check)")
    st.caption(
        "Duursporters trainen idealiter ± 80% rustig (zone 1–2) en maar een klein "
        "deel echt hard. Deze balken tonen per week hoe je hartslagtijd verdeeld "
        "was; de stippellijn is het 80%-doel voor het rustige aandeel."
    )
    intens = weekly_intensity_share(trainingen)
    if intens.empty:
        st.info("Nog geen sessies met zonetijd.")
    else:
        cat_kleuren = {
            "Rustig (Z1–Z2)": PAL["zones"][1],
            "Grijze zone (Z3)": PAL["muted"],
            "Hard (Z4–Z5)": PAL["zones"][4],
        }
        fig = px.bar(
            intens, x="week", y="pct", color="categorie",
            color_discrete_map=cat_kleuren,
            category_orders={"week": sorted(intens["week"].unique()),
                             "categorie": list(cat_kleuren)},
            labels={"week": "Week", "pct": "Aandeel (%)", "categorie": ""},
        )
        fig.add_hline(y=80, line_dash="dash", line_color=PAL["ref_line"],
                      annotation_text="doel: ≥80% rustig",
                      annotation_font_color=PAL["muted"])
        fig.update_traces(
            marker_line=dict(width=1, color=PAL["surface"]),
            hovertemplate="%{x} · %{fullData.name}: %{y:.0f}%<extra></extra>")
        chart(fig)

    st.subheader("Snelheid/tempo tegenover hartslag — per sport")
    st.caption(
        "Per sport een eigen grafiek met de juiste eenheid en schaal: zwemmen, "
        "lopen en fietsen liggen te ver uiteen voor één gedeelde as. Elke stip is "
        "één sessie; grotere stippen duurden langer."
    )
    # Effectieve snelheid: avg_speed_ms waar aanwezig, anders afgeleid uit
    # afstand en duur (voor zwemmen de zuivere zwemtijd). Zo valt een sessie
    # zonder avg_speed_ms — zoals een samengevoegde zwemsessie — niet meer uit
    # de grafiek.
    sc = trainingen.copy()
    sc["eff_speed_ms"] = sc.apply(lambda r: effective_speed_ms(r, zwem_actief), axis=1)
    sc = sc.dropna(subset=["eff_speed_ms", "avg_hr"]).copy()
    sc["Datum"] = sc["start_time"].dt.strftime("%d-%m-%Y")

    # Per sport de natuurlijke prestatiemaat: lopen min/km, fietsen km/h,
    # zwemmen min/100m. Tempo's worden als tijd geplot met omgekeerde as
    # (sneller = hoger), snelheid gewoon oplopend.
    specs = [
        ("running", "Hardlopen", "tempo", 1000, "%{customdata[0]} · HR %{x} · %{y|%M:%S} min/km<extra></extra>"),
        ("cycling", "Fietsen", "snelheid", None, "%{customdata[0]} · HR %{x} · %{y:.1f} km/h<extra></extra>"),
        ("swimming", "Zwemmen", "tempo", 100, "%{customdata[0]} · HR %{x} · %{y|%M:%S} /100m<extra></extra>"),
    ]
    sport_cols = st.columns(3)
    for col, (sport_key, titel, soort, afstand, hover) in zip(sport_cols, specs):
        with col:
            deel = sc[sc["sport"] == sport_key]
            if deel.empty:
                st.info(f"Nog geen {titel.lower()}-sessies.")
                continue
            deel = deel.copy()
            if soort == "tempo":
                deel["y"] = pace_as_time(afstand / deel["eff_speed_ms"])
                y_label = ("Actief tempo (min/km)" if sport_key == "running"
                           else "Actief tempo (min/100m)")
            else:
                deel["y"] = deel["eff_speed_ms"] * 3.6
                y_label = "Snelheid (km/h, actief)"
            fig = px.scatter(
                deel, x="avg_hr", y="y", size="duration_s", custom_data=["Datum"],
                color_discrete_sequence=[SPORT_COLORS[titel]],
                labels={"avg_hr": "Gem. hartslag", "y": y_label},
            )
            fig.update_traces(hovertemplate=hover)
            if soort == "tempo":
                pace_axis(fig)
            # Zone-2-band van déze sport: zo zie je meteen welke sessies echt
            # rustig waren. Zwemmen krijgt geen band — daar gelden geen zones.
            sport_z2 = zone2_range(ATHLETE, sport_key)
            if sport_z2:
                fig.add_vrect(
                    x0=sport_z2[0], x1=sport_z2[1], line_width=0,
                    fillcolor=with_alpha(PAL["zones"][1], 0.15),
                    annotation_text=f"zone 2 ({sport_z2[0]}–{sport_z2[1]})",
                    annotation_position="top left",
                    annotation_font_color=PAL["muted"])
            fig.update_layout(title=titel)
            chart(fig, show_legend=False)

# --------------------------------------------------------------- voortgang --
with tab_voortgang:
    # -- Belasting & fitheid ------------------------------------------------
    st.subheader("🔋 Belasting & fitheid")
    curves = load_curves(acts)
    vandaag = curves.iloc[-1]
    # Minder dan 4 weken data: de fitheidslijn is nog niet 'ingelopen' en de
    # verhouding dan kunstmatig hoog — toon dan geen (vals) alarm.
    acwr_val = vandaag["acwr"] if len(curves) >= 28 and vandaag["ctl"] >= 10 else None
    status, kleur = acwr_status(acwr_val)
    c1, c2, c3 = st.columns(3)
    c1.metric("Fitheid (CTL)", f"{vandaag['ctl']:.0f}",
              help="Traag voortschrijdend gemiddelde (42 dagen) van je trainingsbelasting. Hoger = fitter.")
    c2.metric("Vermoeidheid (ATL)", f"{vandaag['atl']:.0f}",
              help="Snel voortschrijdend gemiddelde (7 dagen): de belasting van de laatste week.")
    c3.metric("Opbouw", f"{kleur} {status}",
              help="Verhouding vermoeidheid/fitheid (ACWR). 0,8–1,3 is een gezonde opbouw; "
                   "daarboven stijgt de belasting sneller dan je fitheid aankan.")

    # Drie panelen met gedeelde datum-as: één zware sessie (TRIMP in de
    # honderden) drukt anders de CTL/ATL-lijnen (tientallen) plat tegen de
    # nullijn. Het middelste paneel toont de ACWR-verhouding met de gezonde
    # bandbreedte, zodat je niet alleen de status van vandaag ziet maar ook
    # hoe je opbouw zich ontwikkelt.
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        row_heights=[0.45, 0.25, 0.3], vertical_spacing=0.05)
    fig.add_trace(go.Scatter(
        x=curves["datum"], y=curves["ctl"], name="Fitheid (CTL)",
        line=dict(color=PAL["cats"][3], width=3),
        hovertemplate="%{x|%d-%m-%Y}: fitheid %{y:.1f}<extra></extra>"),
        row=1, col=1)
    fig.add_trace(go.Scatter(
        x=curves["datum"], y=curves["atl"], name="Vermoeidheid (ATL)",
        line=dict(color=PAL["cats"][7], width=2, dash="dot"),
        hovertemplate="%{x|%d-%m-%Y}: vermoeidheid %{y:.1f}<extra></extra>"),
        row=1, col=1)
    fig.add_trace(go.Scatter(
        x=curves["datum"], y=curves["tsb"], name="Vorm (TSB)",
        line=dict(color=PAL["cats"][4], width=2, dash="dash"),
        hovertemplate="%{x|%d-%m-%Y}: vorm %{y:.1f}<extra></extra>"),
        row=1, col=1)
    fig.add_hrect(y0=0.8, y1=1.3, line_width=0,
                  fillcolor=with_alpha(PAL["zones"][1], 0.15),
                  row=2, col=1)
    fig.add_trace(go.Scatter(
        x=curves["datum"], y=curves["acwr"], name="Opbouw (ACWR)",
        line=dict(color=PAL["ink"], width=2),
        hovertemplate="%{x|%d-%m-%Y}: ACWR %{y:.2f}<extra></extra>"),
        row=2, col=1)
    fig.add_trace(go.Bar(
        x=curves["datum"], y=curves["trimp"], name="Dagbelasting (TRIMP)",
        marker_color=with_alpha(PAL["muted"], 0.45),
        hovertemplate="%{x|%d-%m-%Y}: %{y:.0f} TRIMP<extra></extra>"),
        row=3, col=1)
    fig.update_yaxes(title_text="Fitheid, vermoeidheid & vorm", row=1, col=1)
    fig.update_yaxes(title_text="ACWR", row=2, col=1)
    fig.update_yaxes(title_text="Dag-TRIMP", row=3, col=1)
    fig.update_xaxes(title_text="Datum", row=3, col=1)
    fig.update_layout(height=640, hovermode="x unified")
    # Racedatum alleen markeren als hij (bijna) in beeld is; anders rekt de
    # lijn de hele datum-as op tot maanden zonder data.
    race = (config.get("races") or [{}])[0]
    if race.get("date"):
        race_dt = pd.Timestamp(str(race["date"]))
        if race_dt <= pd.Timestamp(curves["datum"].max()) + pd.Timedelta(days=45):
            add_race_marker(fig, race_dt, race.get("name", "Race"), PAL["ref_line"])
    chart(fig)
    st.caption(
        "Elke sessie krijgt een TRIMP-score uit de tijd per hartslagzone "
        "(zone 1 telt 1×, zone 5 telt 5× per minuut); zwemmen — bewust "
        "zonder zones — telt mee via de gemiddelde hartslag (Banister). "
        "De fitheidslijn moet "
        "gestaag stijgen, met de vermoeidheidslijn er niet te ver bovenuit. "
        "**Vorm (TSB)** = fitheid − vermoeidheid: rond de race wil je dit "
        "positief hebben. **ACWR** hoort in de groene band (0,8–1,3): "
        "daarboven stijgt de belasting sneller dan je fitheid aankan. "
        "De eerste weken is dit beeld nog onbetrouwbaar — de lijnen moeten 'inlopen'."
    )

    st.divider()
    # -- Efficiëntie ---------------------------------------------------------
    st.subheader("📐 Aerobe efficiëntie")
    st.caption(
        "**Efficiency factor**: afgelegde meters per minuut, per hartslag. Stijgt deze bij "
        "vergelijkbare sessies, dan groeit je aerobe basis. **Decoupling**: hoeveel je "
        "efficiëntie wegzakt in de tweede helft van een sessie — onder de 5% bij een "
        "duurtraining duidt op een goede basis."
    )
    ef = efficiency_factor(trainingen)
    # Fietsen met vermogensdata krijgt de zuivere EF (NP per hartslag,
    # windonafhankelijk); de oude snelheid/HR-maat blijft voor ritten zonder
    # power, zodat de historie van vóór de vermogensmeter zichtbaar blijft.
    pw_voortgang = power_trend(trainingen)
    power_start_times = set(pw_voortgang["start_time"]) \
        if not pw_voortgang.empty else set()
    c1, c2 = st.columns(2)
    for col, sport_key, titel in ((c1, "running", "Hardlopen"), (c2, "cycling", "Fietsen")):
        with col:
            ef_sport = ef[ef["sport"] == sport_key]
            if sport_key == "cycling" and not pw_voortgang.empty:
                ef_pw = pw_voortgang.dropna(subset=["ef_watt"]).copy()
                if not ef_pw.empty:
                    ef_pw["Waar"] = ef_pw["indoor"].map(
                        {True: "Indoor (trainer)", False: "Buiten"})
                    fig = px.line(
                        ef_pw, x="start_time", y="ef_watt", color="Waar",
                        markers=True,
                        color_discrete_map={"Buiten": SPORT_COLORS["Fietsen"],
                                            "Indoor (trainer)": PAL["cats"][5]},
                        labels={"start_time": "Datum",
                                "ef_watt": "EF (W per hartslag)", "Waar": ""},
                    )
                    fig.update_traces(
                        marker=dict(size=11),
                        hovertemplate="%{x|%d-%m-%Y} · EF %{y:.2f} · "
                                      "%{fullData.name}<extra></extra>")
                    fig.update_layout(
                        title="Fietsen — EF op vermogen (NP/HR, windonafhankelijk)")
                    date_xaxis(fig, ef_pw["start_time"])
                    pad_single_point(fig, ef_pw["start_time"])
                    chart(fig)
                # De oude maat alleen nog voor ritten zonder vermogensdata.
                ef_sport = ef_sport[~ef_sport["start_time"].isin(power_start_times)]
                if ef_sport.empty:
                    continue
            if ef_sport.empty:
                st.info(f"Nog geen {titel.lower()}-sessies.")
                continue
            titel_suffix = " (ritten zonder vermogensdata)" \
                if sport_key == "cycling" and power_start_times else ""
            fig = px.line(
                ef_sport, x="start_time", y="ef", markers=True,
                labels={"start_time": "Datum", "ef": "EF (m/min per hartslag)"},
            )
            fig.update_traces(
                marker=dict(size=11), line=dict(color=SPORT_COLORS[titel]),
                hovertemplate="%{x|%d-%m-%Y} · EF %{y:.2f}<extra></extra>")
            fig.update_layout(
                title=f"{titel} — efficiency factor (hoger = beter){titel_suffix}")
            pad_single_point(fig, ef_sport["start_time"])
            chart(fig, show_legend=False)

    dec = cache_decoupling(DATA_VERSIE)
    if not dec.empty:
        dec = dec.copy().sort_values("start_time")
        dec["Sport"] = dec["sport"].map(sport_label)
        dec["label"] = dec["start_time"].dt.strftime("%d-%m")
        fig = px.bar(
            dec, x="label", y="decoupling_pct", color="Sport",
            color_discrete_map=SPORT_COLORS, barmode="group",
            labels={"label": "Datum", "decoupling_pct": "Decoupling (%)"},
            category_orders={"label": dec["label"].tolist()},
        )
        # Forceer een categorie-as: op een datum-as maakt Plotly elke staaf maar
        # één dag breed over een reeks van weken, waardoor de balken tot
        # onzichtbare haarlijntjes verschrompelen. Als categorie blijven ze
        # gewoon dag-maand en chronologisch.
        fig.update_xaxes(type="category")
        fig.add_hline(y=5, line_dash="dash", line_color=PAL["ref_line"],
                      annotation_text="richtwaarde 5%",
                      annotation_font_color=PAL["muted"])
        fig.update_traces(
            marker_line=dict(width=1, color=PAL["surface"]),
            hovertemplate="%{x} · %{fullData.name}: %{y:.1f}%<extra></extra>")
        fig.update_layout(title="HR-decoupling per sessie (lager = betere aerobe basis)")
        chart(fig)

    st.divider()
    # -- Racevoorspelling -----------------------------------------------------
    afst = race_distances(race)
    st.subheader(
        f"🏁 Racevoorspelling — {race.get('name') or 'standaard'} "
        f"({afst['swim_m'] / 1000:g} km / {afst['bike_m'] / 1000:g} km / "
        f"{afst['run_m'] / 1000:g} km)"
    )
    pred = race_prediction(conn, trainingen, race)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric(f"Zwemmen {afst['swim_m'] / 1000:g} km", fmt_duration(pred["zwem"]))
    c2.metric(f"Fietsen {afst['bike_m'] / 1000:g} km", fmt_duration(pred["fiets"]))
    c3.metric(f"Lopen {afst['run_m'] / 1000:g} km", fmt_duration(pred["loop"]))
    c4.metric("Wissels (T1+T2)", fmt_duration(pred["wissels"]))
    c5.metric("Totaal (schatting)", fmt_duration(pred["totaal"]))
    st.caption(
        "Ruwe schatting: lopen via Riegel-schaling vanaf je beste recente loop, fietsen via je "
        "snelste rit (≥15 km), zwemmen via het tempo van je laatste zwemsessie — dat verandert "
        "nu het snelst, dus deze voorspelling wordt elke zwemsessie beter. Racedag-effecten "
        "(wetsuit, drafting, spanning) zitten er niet in. De afstanden volgen de eerste race "
        "op de instellingen-tab."
    )
    for emoji, tekst in readiness(trainingen, race):
        st.markdown(f"{emoji} {tekst}")

    hist = cache_prediction_history(race, DATA_VERSIE)
    if len(hist) >= 2:
        naam = {"zwem": "Zwemmen", "fiets": "Fietsen", "loop": "Lopen", "totaal": "Totaal"}
        kleuren = {"Zwemmen": SPORT_COLORS["Zwemmen"], "Fietsen": SPORT_COLORS["Fietsen"],
                   "Lopen": SPORT_COLORS["Hardlopen"], "Totaal": PAL["ink"]}
        lang = hist.melt(id_vars="datum", var_name="onderdeel",
                         value_name="seconden").dropna(subset=["seconden"])
        lang["onderdeel"] = lang["onderdeel"].map(naam)
        lang["tijd"] = pace_as_time(lang["seconden"])
        fig = px.line(
            lang, x="datum", y="tijd", color="onderdeel", markers=True,
            color_discrete_map=kleuren,
            labels={"datum": "Datum", "tijd": "Geschatte tijd", "onderdeel": ""},
        )
        fig.update_yaxes(tickformat="%H:%M")
        fig.update_traces(
            marker=dict(size=9),
            hovertemplate="%{x|%d-%m-%Y} · %{fullData.name}: %{y|%H:%M:%S}<extra></extra>")
        fig.update_layout(title="Voorspelde racetijd over tijd (lager = beter)")
        date_xaxis(fig, lang["datum"])
        chart(fig)
        st.caption(
            "Elke punt is de voorspelling zoals die er die week uitzag, berekend "
            "met alleen de trainingen tot dat moment. Een dalende lijn betekent: "
            "je wordt sneller op de raceafstanden."
        )

    st.divider()
    # -- Records & zwemprogressie ---------------------------------------------
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("🏆 Persoonlijke records")
        prs = cache_personal_records(DATA_VERSIE)
        if prs.empty:
            st.info("Nog geen records — importeer trainingen.")
        else:
            st.dataframe(prs, hide_index=True, width="stretch")
    with c2:
        st.subheader("🏊 Zwemprogressie")
        aandeel, swolf_sessie = swim_progression(conn, trainingen)
        if aandeel.empty:
            st.info("Nog geen zwemsessies met baandata.")
        else:
            fig = px.line(
                aandeel, x="start_time", y="crawl_pct", markers=True,
                labels={"start_time": "Datum", "crawl_pct": "Aandeel borstcrawl (%)"},
            )
            fig.update_traces(
                marker=dict(size=11), line=dict(color=SPORT_COLORS["Zwemmen"]),
                hovertemplate="%{x|%d-%m-%Y} · %{y:.0f}% crawl<extra></extra>")
            fig.update_yaxes(range=[0, 100])
            fig.update_layout(title="Aandeel borstcrawl per sessie (doel: richting 100%)")
            pad_single_point(fig, aandeel["start_time"])
            chart(fig, show_legend=False)
        if not swolf_sessie.empty:
            fig = px.line(
                swolf_sessie, x="start_time", y="swolf", markers=True,
                labels={"start_time": "Datum", "swolf": "SWOLF"},
            )
            fig.update_traces(
                marker=dict(size=11), line=dict(color=SPORT_COLORS["Zwemmen"]),
                hovertemplate="%{x|%d-%m-%Y} · SWOLF %{y:.0f}<extra></extra>")
            fig.update_layout(
                title="SWOLF per sessie — alleen crawl, 25m-bad "
                      "(lager = efficiënter)")
            pad_single_point(fig, swolf_sessie["start_time"])
            chart(fig, show_legend=False)
        elif not aandeel.empty:
            st.info("Nog geen crawlbanen in het 25m-bad voor de SWOLF-trend "
                    "(andere slagen en baanlengtes tellen niet mee).")

# ------------------------------------------------------------------ sessie --
with tab_sessie:
    st.subheader("🔍 Sessie-detail")
    st.caption(
        "Kies een sessie voor het verloop bínnen de training: hartslag met "
        "zonebanden, tempo/snelheid en hoogte, plus kilometersplits of banen."
    )
    keuze_df = acts.sort_values("start_time", ascending=False)
    sessie_labels = {
        r["activity_key"]: (
            f"{r['start_time']:%d-%m-%Y %H:%M} · {r['Sport']}"
            + (f" · {r['distance_m'] / 1000:.1f} km" if pd.notna(r["distance_m"]) else "")
            + (" · 🛒 transport" if is_transport(r) else "")
        )
        for _, r in keuze_df.iterrows()
    }
    gekozen = st.selectbox(
        "Sessie", keuze_df["activity_key"].tolist(),
        format_func=sessie_labels.get, label_visibility="collapsed",
    )
    sessie = keuze_df.loc[keuze_df["activity_key"] == gekozen].iloc[0]
    is_zwem = sessie["sport"] == "swimming"
    rec = pd.DataFrame() if is_zwem else load_records(conn, gekozen).sort_values("timestamp")
    banen = load_lengths(conn, gekozen) if is_zwem else pd.DataFrame()

    drift = hr_drift_values(rec) if not rec.empty else None
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Duur", fmt_duration(sessie["duration_s"]))
    c2.metric("Afstand", f"{sessie['distance_m'] / 1000:.2f} km"
              if pd.notna(sessie["distance_m"]) else GEEN_WAARDE)
    c3.metric("Actief tempo", veilig_cel(
        sessie_tempo, sessie["sport"], sessie["distance_m"],
        sessie["duration_s"], sessie.get("active_s")),
        help="Op actieve/bewegende tijd (excl. rust en stilstand); "
             "kan afwijken van Garmin Connect.")
    c4.metric("Gem. / max HR",
              f"{sessie['avg_hr']:.0f} / {sessie['max_hr']:.0f}"
              if pd.notna(sessie["avg_hr"]) and pd.notna(sessie["max_hr"]) else GEEN_WAARDE)
    c5.metric("HR-drift", f"{drift[1] - drift[0]:+.0f} bpm" if drift else GEEN_WAARDE,
              help="Gemiddelde hartslag van de tweede helft minus de eerste helft. "
                   "Duidelijk oplopend = vermoeidheid of een te hoog begintempo.")

    # Waarop wordt déze sessie beoordeeld? Expliciet in beeld, zodat nooit
    # onduidelijk is welke drempel en welke zonegrenzen gelden.
    sessie_heeft_power = (not rec.empty and "power" in rec
                          and rec["power"].notna().any())
    sessie_model = zone_model(ATHLETE, sessie["sport"], has_power=sessie_heeft_power)
    if not sessie_model.has_zones:
        st.caption(f"⚖️ **Beoordeling:** {sessie_model.reason} — geen zone-oordeel.")
    elif sessie_model.provisional:
        st.warning(
            f"⚖️ **Beoordeling:** {sessie_model.method_label} op "
            f"{sessie_model.threshold_label} → {sessie_model.bounds_text()}. "
            "Dit is een tussenoplossing: zodra je de FTP invult (ramptest op de "
            "Kickr) stapt deze rit over op %FTP-vermogenszones."
        )
    else:
        st.caption(
            f"⚖️ **Beoordeling:** {sessie_model.method_label} op "
            f"{sessie_model.threshold_label} → {sessie_model.bounds_text()}"
            + (f" · {sessie_model.reason}" if sessie_model.reason else "")
        )

    # Bewaarde coach-feedback bij deze sessie (database, sinds de sync-
    # uitbreiding): zo is de coaching altijd terug te lezen, ook van
    # automatisch gesyncte sessies waarvan niemand de upload-flits zag.
    bewaarde_fb = load_session_feedback(conn, gekozen)
    if bewaarde_fb:
        with st.expander(f"🗣️ Coach-feedback bij deze sessie ({len(bewaarde_fb)})"):
            for f in bewaarde_fb:
                st.caption(f["created_at"][:16].replace("T", " "))
                st.success(f["feedback"])
                if f["aanpassing"]:
                    st.warning(f"**Voorgestelde aanpassing:** {f['aanpassing']}")

    if sessie["sport"] == "cycling" and pd.notna(sessie.get("avg_power")):
        # Vermogenskerncijfers (sinds de Rally/Kickr). EF = NP per hartslag:
        # windonafhankelijk, dus de zuiverste aerobe maat voor het fietsen.
        p1, p2, p3, p4, p5 = st.columns(5)
        p1.metric("Gem. vermogen", f"{sessie['avg_power']:.0f} W",
                  help="Gemiddelde over de hele rit, inclusief freewheelen (0 W).")
        p2.metric("NP", f"{sessie['np_power']:.0f} W"
                  if pd.notna(sessie.get("np_power")) else GEEN_WAARDE,
                  help="Normalized power: 30s-gemiddelde → 4e macht → gemiddelde "
                       "→ 4e-machtswortel. Weegt pieken zwaarder; voor wisselend "
                       "buitenrijden de betere intensiteitsmaat.")
        cadans_detail = sessie.get("avg_cadence_excl0")
        if pd.isna(cadans_detail):
            cadans_detail = sessie.get("avg_cadence")
        p3.metric("Cadans", f"{cadans_detail:.0f} rpm"
                  if pd.notna(cadans_detail) else GEEN_WAARDE,
                  help="Gemiddelde trapfrequentie exclusief freewheelen (zoals "
                       "Garmin); het maximum staat in de grafiek hieronder.")
        p4.metric("EF", f"{sessie['ef_watt']:.2f} W/slag"
                  if pd.notna(sessie.get("ef_watt")) else GEEN_WAARDE,
                  help="Efficiency factor: NP gedeeld door je gemiddelde "
                       "hartslag. Stijgend over weken bij vergelijkbare ritten "
                       "= aerobe vooruitgang, onafhankelijk van de wind.")
        bron = sessie.get("power_source")
        p5.metric("Bron", ("🏠 " if sessie.get("is_indoor") else "🌍 ")
                  + (str(bron) if bron and pd.notna(bron) else "onbekend"),
                  help="Pedalen (buiten) en trainer (binnen) meten net iets "
                       "anders; vergelijk trends binnen dezelfde bron.")

        # FTP-test herkennen (ramptest op de Kickr of een 20-minuten-veldtest)
        # en er een voorstel uit afleiden. Het voorstel wordt nooit automatisch
        # opgeslagen: de atleet bevestigt hem hieronder.
        voorstel = veilig_cel(ftp_proposal, rec, bool(sessie.get("is_indoor")),
                              fallback=None)
        if voorstel:
            ramp = veilig_cel(is_ramp_test, rec, bool(sessie.get("is_indoor")),
                              fallback=False)
            kop = ("🧗 **Dit ziet eruit als een ramptest.**" if ramp
                   else "📈 **Er zit een 20-minuten-inspanning in deze rit.**")
            titel = f"{kop.strip('*# ')} — FTP-voorstel: {voorstel.ftp_watt:.0f} W"
            if voorstel.lthr_bpm:
                titel += f" + fiets-LTHR {voorstel.lthr_bpm} bpm"
            with st.expander(titel, expanded=bool(ramp)):
                regels = (
                    f"{kop}\n\n{voorstel.explanation}\n\n"
                    f"- **Gemeten:** {voorstel.basis_watt:.0f} W over "
                    f"{voorstel.window_s // 60} min\n"
                    f"- **Afleiding FTP:** {voorstel.factor:.0%} → "
                    f"**{voorstel.ftp_watt:.0f} W**\n"
                )
                if voorstel.lthr_bpm:
                    regels += (
                        f"- **Afleiding fiets-LTHR:** gemiddelde hartslag over "
                        f"datzelfde blok → **{voorstel.lthr_bpm} bpm**\n")
                regels += f"- **Betrouwbaarheid:** {voorstel.confidence}\n"
                st.markdown(regels)

                huidige_ftp = f"{FTP:.0f} W" if FTP else "nog niet ingesteld"
                st.caption(
                    f"Huidige FTP-instelling: {huidige_ftp}; fiets-LTHR "
                    f"{bike_lthr(ATHLETE)} bpm. Bevestigen zet de waarde(n) in "
                    "config.yaml, herrekent de zones van alle ritten en legt de "
                    "vaststelling vast in memory/inzichten.md en de "
                    "drempelgeschiedenis."
                )
                # Een 20-minutentest levert beide drempels; de atleet kiest of
                # hij de fiets-LTHR meteen meeneemt.
                ook_lthr = False
                if voorstel.lthr_bpm:
                    ook_lthr = st.checkbox(
                        f"Ook de fiets-LTHR bijwerken naar "
                        f"{voorstel.lthr_bpm} bpm (nu {bike_lthr(ATHLETE)})",
                        value=True, key=f"ook_lthr_{gekozen}",
                        help="De gemiddelde hartslag over het 20-minutenblok is "
                             "de klassieke schatting van je fiets-drempel. "
                             "Hiermee kloppen ook de ritten zónder vermogen.")
                knop = f"✅ FTP op {voorstel.ftp_watt:.0f} W zetten"
                if ook_lthr:
                    knop += f" + fiets-LTHR {voorstel.lthr_bpm}"
                if st.button(knop, key=f"ftp_bevestig_{gekozen}"):
                    nieuwe_ftp = float(round(voorstel.ftp_watt))
                    nieuwe_bike_lthr = (voorstel.lthr_bpm if ook_lthr
                                        else bike_lthr(ATHLETE))
                    nieuw = copy.deepcopy(config)
                    set_thresholds(nieuw["athlete"], run_lthr(ATHLETE),
                                   nieuwe_bike_lthr, nieuwe_ftp)
                    save_config(nieuw)
                    profile_mod.update_doelen(MEMORY_DIR, config, nieuw,
                                              note=f"FTP uit {voorstel.method}")
                    n_pw = recompute_power_zones(conn, nieuwe_ftp)
                    lthr_append(MEMORY_DIR, int(nieuwe_ftp),
                                f"Vastgesteld met {voorstel.method} "
                                f"({voorstel.basis_watt:.0f} W)",
                                kind=LTHR_BIKE_FTP)
                    log_ftp_determination(
                        MEMORY_DIR, nieuwe_ftp, voorstel.method,
                        basis_watt=voorstel.basis_watt,
                        session_date=sessie["start_time"],
                        note="Bevestigd vanuit het sessie-detail in het dashboard.")
                    # Fiets-LTHR uit dezelfde test: ook de hartslagzones van
                    # alle sessies moeten dan mee herrekend worden.
                    if ook_lthr and nieuwe_bike_lthr != bike_lthr(ATHLETE):
                        lthr_append(MEMORY_DIR, nieuwe_bike_lthr,
                                    f"Gem. HR over het 20-minutenblok van "
                                    f"{voorstel.method} (was {bike_lthr(ATHLETE)})",
                                    kind=LTHR_BIKE)
                        recompute_zones(conn, nieuw["athlete"])
                    melding = (
                        f"FTP op {nieuwe_ftp:.0f} W gezet ({voorstel.method}); "
                        f"vermogenszones van {n_pw} ritten herrekend."
                    )
                    if ook_lthr:
                        melding += (f" Fiets-LTHR op {nieuwe_bike_lthr} bpm "
                                    "gezet; hartslagzones herrekend.")
                    st.session_state["settings_flash"] = melding
                    st.success(melding)
                    st.rerun()

    if sessie["sport"] == "running":
        # Loopdynamiek van deze sessie (zelfde getallen als Garmin Connect;
        # cadans als totale stappen per minuut). Oudere imports missen de
        # meeste velden tot de zip opnieuw is geüpload.
        try:
            dyn = dynamics_from_summary(json.loads(sessie.get("summary_json") or "{}"))
        except (TypeError, ValueError):
            dyn = {}
        if any(v is not None for v in dyn.values()):
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Cadans",
                      f"{dyn['cadans_spm']:.0f} spm" if dyn.get("cadans_spm") else GEEN_WAARDE,
                      help="Totale stappen per minuut (beide benen), zoals "
                           "Garmin Connect. Je eigen trend telt; de richtlijn "
                           f"voor rustig duurlooptempo ({CADENCE_GUIDE_SPM[0]}–"
                           f"{CADENCE_GUIDE_SPM[1]}) is geen norm.")
            k2.metric("Staplengte",
                      f"{dyn['staplengte_m']:.2f} m" if dyn.get("staplengte_m") else GEEN_WAARDE,
                      help="Hangt samen met de cadans: hogere cadans bij gelijk "
                           "tempo = kortere pas.")
            k3.metric("Grondcontacttijd",
                      f"{dyn['gct_ms']:.0f} ms" if dyn.get("gct_ms") else GEEN_WAARDE,
                      help="Korter = efficiënter; daalt vanzelf mee als de "
                           "cadans over de maanden stijgt.")
            ratio = dyn.get("vert_ratio_pct")
            k4.metric("Verticale ratio",
                      f"{ratio:.1f}%" + (" ✓" if vertical_ratio_is_good(ratio) else "")
                      if ratio is not None else GEEN_WAARDE,
                      help=f"Onder {VERTICAL_RATIO_GOOD_PCT:.0f}% is prima — "
                           "het vinkje betekent: hier valt niets te verbeteren.")

    # Sessielabel (achteraf) instellen of weghalen: een techniek-/cadanssessie
    # mag een hogere hartslag hebben zonder als te hard getraind te gelden.
    SESSIE_LABELS = ["(geen)", "techniek/cadans"]
    huidig_label = sessie.get("training_label")
    if huidig_label is None or (isinstance(huidig_label, float) and pd.isna(huidig_label)):
        huidig_label = "(geen)"
    nieuw_label = st.selectbox(
        "🏷️ Sessielabel", SESSIE_LABELS,
        index=SESSIE_LABELS.index(huidig_label) if huidig_label in SESSIE_LABELS else 0,
        key=f"label_{gekozen}",
        help="'techniek/cadans' = bewuste cadans-oefensessie; de coach weet "
             "dan dat een hogere hartslag bij het oefenen hoort. Wijzigingen "
             "gelden voor toekomstige feedback en analyses.",
    )
    if nieuw_label != huidig_label:
        set_training_label(conn, gekozen, None if nieuw_label == "(geen)" else nieuw_label)
        st.toast(f"Sessielabel bijgewerkt: {nieuw_label}")
        st.rerun()

    # Transport-markering (aan/uit): een verplaatsing — naar het zwembad
    # fietsen, boodschappen — blijft in het logboek en de weektotalen, maar
    # telt niet mee in trends, records, zone-statistieken, brick-detectie en
    # de feedback-vergelijking.
    was_transport = is_transport(sessie)
    wordt_transport = st.toggle(
        "🛒 Transport/verplaatsing (geen training)",
        value=was_transport, key=f"transport_{gekozen}",
        help="Telt mee in weektotalen en belasting (compleet beeld), maar "
             "niet in trends, records en de feedback-vergelijking. De sessie "
             "staat gedimd in de sessielijst.",
    )
    if wordt_transport != was_transport:
        if wordt_transport:
            mark_transport(conn, MEMORY_DIR, gekozen)
            st.toast("Gemarkeerd als transport — telt niet meer mee in trends.")
        else:
            unmark_transport(conn, MEMORY_DIR, gekozen)
            st.toast("Transport-markering verwijderd — telt weer mee als training.")
        st.rerun()

    # Verwijderen is een soft delete met expliciete bevestiging: één misklik
    # kost geen data, en herstellen kan altijd via de instellingen-tab.
    with st.expander("🗑️ Deze sessie verwijderen"):
        st.caption(
            "De sessie verdwijnt uit alle tabellen, grafieken, trends en de "
            "feedback-context, maar blijft bewaard in de database. Herstellen "
            "of definitief wissen kan via ⚙️ Instellingen → Verwijderde sessies. "
            "Bij een herimport van dezelfde zip blijft de sessie verwijderd."
        )
        del_reden = st.text_input(
            "Reden (optioneel, komt in het trainingslog)",
            placeholder="bijv. verkeerd bestand · dubbele upload",
            key=f"del_reden_{gekozen}",
        )
        del_zeker = st.checkbox("Weet je het zeker?", key=f"del_zeker_{gekozen}")
        if st.button("🗑️ Verwijder deze sessie", type="secondary",
                     disabled=not del_zeker, key=f"del_knop_{gekozen}"):
            remove_session(conn, MEMORY_DIR, gekozen, reason=del_reden)
            st.toast(f"Sessie {sessie_labels[gekozen]} verwijderd.")
            st.rerun()

    if is_zwem:
        if banen.empty:
            st.info("Geen baandata voor deze sessie (bijv. open water).")
        else:
            banen = banen.reset_index(drop=True)
            banen["baan"] = banen.index + 1
            banen["Slag"] = banen["swim_stroke"].map(stroke_label)
            # Tempo per 100 m i.p.v. tijd per baan: vergelijkbaar, ook als de
            # baanlengte binnen de sessie wisselde (zie lane_meters).
            banen["meters"] = lane_meters(
                banen, sessie.get("pool_length"), sessie.get("distance_m"))
            banen["tempo_100m"] = (banen["total_timer_time"] / banen["meters"]
                                   * 100).where(banen["meters"] > 0)
            fig = px.bar(
                banen, x="baan", y="tempo_100m", color="Slag",
                color_discrete_map=STROKE_COLORS,
                custom_data=["total_strokes", "meters"],
                labels={"baan": "Baan", "tempo_100m": "Tempo (s/100m)"},
            )
            fig.update_traces(
                marker_line=dict(width=1, color=PAL["surface"]),
                hovertemplate="baan %{x} · %{y:.0f} s/100m · ~%{customdata[1]:.0f} m · "
                              "%{customdata[0]:.0f} slagen · %{fullData.name}<extra></extra>")
            fig.update_layout(title="Baan voor baan als tempo (kleur = slagtype)")
            chart(fig)
            st.caption(
                "Gelijkmatige balken = constant tempo. Worden de crawlbanen naar "
                "het einde toe langzamer, dan zakt de techniek onder vermoeidheid in."
            )
    elif rec.empty or "heart_rate" not in rec or rec["heart_rate"].dropna().empty:
        st.info("Geen seconde-data voor deze sessie.")
    else:
        rec = rec.dropna(subset=["timestamp"]).reset_index(drop=True)
        rec["minuut"] = (rec["timestamp"] - rec["timestamp"].iloc[0]).dt.total_seconds() / 60
        # Rollend gemiddelde van ~10 s tegen sensorruis; de records zijn ~1/s.
        hr_glad = rec["heart_rate"].rolling(10, min_periods=1).mean()
        snel_glad = rec["speed_ms"].rolling(10, min_periods=1).mean() \
            if "speed_ms" in rec else pd.Series(dtype=float)
        alt = rec["altitude_m"].rolling(10, min_periods=1).mean() \
            if "altitude_m" in rec and rec["altitude_m"].notna().any() else pd.Series(dtype=float)
        # Vermogen en cadans (fietsen met vermogensmeter): eigen panelen onder
        # de hartslag, zodat vermogen vs hartslag in één blik (en één unified
        # hover) te vergelijken is. Nullen zijn echte waarden (freewheelen).
        power_glad = rec["power"].rolling(10, min_periods=1).mean() \
            if sessie["sport"] == "cycling" and "power" in rec \
            and rec["power"].notna().any() else pd.Series(dtype=float)
        cad_glad = rec["cadence"].rolling(10, min_periods=1).mean() \
            if sessie["sport"] == "cycling" and "cadence" in rec \
            and rec["cadence"].notna().any() else pd.Series(dtype=float)
        rijen = 2 + (0 if power_glad.empty else 1) + (0 if cad_glad.empty else 1) \
            + (0 if alt.empty else 1)
        # Hartslag en tempo/snelheid wat ruimer, de rest gelijk verdeeld.
        gewichten = [0.3, 0.25] + [0.45 / (rijen - 2)] * (rijen - 2) \
            if rijen > 2 else [0.55, 0.45]

        fig = make_subplots(rows=rijen, cols=1, shared_xaxes=True,
                            row_heights=gewichten,
                            vertical_spacing=0.05)
        # Zonebanden achter de hartslaglijn, met de grenzen van déze sport: in
        # één blik zie je de zone. Zwemmen krijgt geen banden — daar is bewust
        # geen zone-oordeel (onbetrouwbare pols-HR, techniek in opbouw).
        sessie_bounds = SPORT_BOUNDS.get(sessie["sport"])
        if sessie_bounds:
            onder = float(min(hr_glad.min(), sessie_bounds[0] - 10))
            boven = float(max(hr_glad.max() + 5, sessie_bounds[3] + 5))
            grenzen = [onder, *sessie_bounds, boven]
            for i in range(5):
                fig.add_hrect(y0=grenzen[i], y1=grenzen[i + 1], line_width=0,
                              fillcolor=with_alpha(PAL["zones"][i], 0.14),
                              row=1, col=1)
        else:
            onder = float(hr_glad.min() - 5)
            boven = float(hr_glad.max() + 5)
        fig.add_trace(go.Scatter(
            x=rec["minuut"], y=hr_glad, name="Hartslag",
            line=dict(color=PAL["ink"], width=2),
            hovertemplate="min %{x:.0f}: HR %{y:.0f}<extra></extra>"),
            row=1, col=1)
        fig.update_yaxes(title_text="HR (bpm)", range=[onder, boven], row=1, col=1)

        if sessie["sport"] == "running":
            geldig = snel_glad.where(snel_glad > 0.5)  # stilstand geeft oneindig tempo
            fig.add_trace(go.Scatter(
                x=rec["minuut"], y=pace_as_time(1000 / geldig), name="Tempo",
                line=dict(color=SPORT_COLORS["Hardlopen"], width=2),
                hovertemplate="min %{x:.0f}: %{y|%M:%S}/km<extra></extra>"),
                row=2, col=1)
            fig.update_yaxes(title_text="min/km", tickformat="%M:%S",
                             autorange="reversed", row=2, col=1)
        else:
            fig.add_trace(go.Scatter(
                x=rec["minuut"], y=snel_glad * 3.6, name="Snelheid",
                line=dict(color=SPORT_COLORS["Fietsen"], width=2),
                hovertemplate="min %{x:.0f}: %{y:.1f} km/h<extra></extra>"),
                row=2, col=1)
            fig.update_yaxes(title_text="km/h", row=2, col=1)

        rij = 3
        if not power_glad.empty:
            fig.add_trace(go.Scatter(
                x=rec["minuut"], y=power_glad, name="Vermogen",
                line=dict(color=PAL["cats"][3], width=2),
                hovertemplate="min %{x:.0f}: %{y:.0f} W<extra></extra>"),
                row=rij, col=1)
            if FTP:
                # Bovengrens zone 2 als referentie: daaronder is echt rustig.
                fig.add_hline(y=power_zone_bounds(FTP)[1], line_dash="dash",
                              line_color=PAL["ref_line"],
                              annotation_text="bovengrens P2 (rustig)",
                              annotation_font_color=PAL["muted"],
                              row=rij, col=1)
            fig.update_yaxes(title_text="Vermogen (W)", rangemode="tozero",
                             row=rij, col=1)
            rij += 1
        if not cad_glad.empty:
            fig.add_trace(go.Scatter(
                x=rec["minuut"], y=cad_glad, name="Cadans",
                line=dict(color=PAL["cats"][7], width=2),
                hovertemplate="min %{x:.0f}: %{y:.0f} rpm<extra></extra>"),
                row=rij, col=1)
            fig.update_yaxes(title_text="Cadans (rpm)", rangemode="tozero",
                             row=rij, col=1)
            rij += 1
        if not alt.empty:
            fig.add_trace(go.Scatter(
                x=rec["minuut"], y=alt, name="Hoogte",
                line=dict(color=PAL["muted"], width=2),
                hovertemplate="min %{x:.0f}: %{y:.0f} m<extra></extra>"),
                row=rij, col=1)
            fig.update_yaxes(title_text="Hoogte (m)", row=rij, col=1)

        fig.update_xaxes(title_text="Minuten", row=rijen, col=1)
        fig.update_layout(height=150 * rijen + 120, hovermode="x unified")
        chart(fig)

        # Tijd in vermogenszones: vers berekend uit de seconde-data met de
        # actuele FTP, naast de HR-zonebanden hierboven. Zonder FTP geen
        # zone-oordeel — alleen een hint dat de instelling bestaat.
        if sessie["sport"] == "cycling" and not power_glad.empty:
            if FTP:
                tip = time_in_power_zones(rec, FTP)
                if sum(tip.values()) > 0:
                    st.subheader("Tijd in vermogenszones")
                    grenzen_w = [round(w) for w in power_zone_bounds(FTP)]
                    zone_labels = {
                        "P1": f"P1 herstel (<{grenzen_w[0]} W)",
                        "P2": f"P2 duur ({grenzen_w[0]}–{grenzen_w[1]} W)",
                        "P3": f"P3 tempo ({grenzen_w[1]}–{grenzen_w[2]} W)",
                        "P4": f"P4 drempel ({grenzen_w[2]}–{grenzen_w[3]} W)",
                        "P5": f"P5 VO2max ({grenzen_w[3]}–{grenzen_w[4]} W)",
                        "P6": f"P6 anaeroob (>{grenzen_w[4]} W)",
                    }
                    tip_df = pd.DataFrame({
                        "Zone": [zone_labels[z] for z in POWER_ZONE_NAMES],
                        "Minuten": [tip[z] / 60 for z in POWER_ZONE_NAMES],
                    })
                    fig = px.bar(
                        tip_df, x="Minuten", y="Zone", orientation="h",
                        labels={"Minuten": "Minuten", "Zone": ""},
                    )
                    fig.update_traces(
                        marker_color=[PAL["zones"][min(i, 4)] for i in range(6)],
                        marker_line=dict(width=1, color=PAL["surface"]),
                        hovertemplate="%{y}: %{x:.0f} min<extra></extra>")
                    fig.update_yaxes(autorange="reversed")
                    fig.update_layout(
                        title=f"Vermogenszones (Coggan, FTP {FTP:.0f} W) — naast "
                              "de hartslagzones, niet in de plaats ervan",
                        height=280)
                    chart(fig, show_legend=False)
            else:
                st.caption(
                    "💡 Deze rit heeft vermogensdata, maar er is nog geen FTP "
                    "ingesteld — zet hem op de ⚙️ Instellingen-tab en hier "
                    "verschijnt de tijd per vermogenszone."
                )

        if sessie["sport"] == "running":
            splits = run_splits_df(rec)
            if not splits.empty:
                st.subheader("Kilometersplits")
                splits = splits.copy()
                splits["Tempo"] = splits["duur_s"].map(fmt_duration)
                splits["Gem. HR"] = splits["gem_hr"].map(
                    lambda v: GEEN_WAARDE if v is None or pd.isna(v) else f"{v:.0f}")
                st.dataframe(
                    splits[["km", "Tempo", "Gem. HR"]].rename(columns={"km": "Km"}),
                    hide_index=True, width="stretch")

# ------------------------------------------------------- discipline-tabs --
with tab_lopen:
    runs = trainingen[trainingen["sport"] == "running"].sort_values("start_time")
    if runs.empty:
        st.info("Nog geen loopsessies.")
    else:
        runs = runs.copy()
        runs["tempo"] = pace_as_time(1000 / runs["avg_speed_ms"])
        # Loopdynamiek per sessie uit summary_json: cadans (spm, incl. de
        # fractie zodat het getal exact met Garmin Connect overeenkomt),
        # staplengte, grondcontacttijd en verticale ratio.
        dynamiek = dynamics_trend(runs)
        c1, c2 = st.columns(2)
        with c1:
            fig = px.scatter(
                runs, x="start_time", y="tempo", color="avg_hr",
                color_continuous_scale=PAL["seq"],
                labels={"start_time": "Datum", "tempo": "Actief tempo (min/km)",
                        "avg_hr": "Gem. HR"},
            )
            pace_axis(fig)
            fig.update_traces(
                mode="lines+markers",
                marker=dict(size=14, line=dict(width=1, color=PAL["surface"])),
                line=dict(color=with_alpha(PAL["muted"], 0.4)),
                hovertemplate="%{x|%d-%m-%Y} · %{y|%M:%S} min/km · HR %{marker.color}<extra></extra>",
            )
            fig.update_layout(
                title="Actief tempo per sessie (kleur = gemiddelde hartslag)")
            chart(fig, show_legend=False)
        with c2:
            cad = dynamiek.dropna(subset=["cadans_spm"])
            fig = px.line(
                cad, x="start_time", y="cadans_spm", markers=True,
                labels={"start_time": "Datum", "cadans_spm": "Cadans (stappen/min)"},
            )
            fig.update_traces(
                marker=dict(size=11), line=dict(color=SPORT_COLORS["Hardlopen"]),
                hovertemplate="%{x|%d-%m-%Y} · %{y:.0f} stappen/min<extra></extra>",
            )
            # Zachte richtlijn voor rustig duurlooptempo, bewust in neutraal
            # grijs: de eigen trend is de maat, niet dit gebied. De oude,
            # harde 170–180-band kwam van de elite-"180-regel" en was voor
            # rustig zone 2-werk verkeerd gekalibreerd.
            fig.add_hrect(y0=CADENCE_GUIDE_SPM[0], y1=CADENCE_GUIDE_SPM[1],
                          line_width=0,
                          fillcolor=with_alpha(PAL["muted"], 0.15),
                          annotation_text=f"richtlijn duurloop "
                                          f"{CADENCE_GUIDE_SPM[0]}–{CADENCE_GUIDE_SPM[1]} "
                                          f"(geen norm)",
                          annotation_position="top left",
                          annotation_font_color=PAL["muted"])
            fig.update_layout(title="Cadans per sessie (totale stappen/min, als Garmin Connect)")
            chart(fig, show_legend=False)
            st.caption(
                "Je eigen trend is hier de maat. Het grijze gebied is een "
                "zachte richtlijn voor rústig duurlooptempo — geen norm, en "
                "bewust niet de bekende 170–180-\"regel\", die van "
                "elite-lopers op wedstrijdtempo komt. Cadans is een middel "
                "(loopeconomie, blessurebestendigheid), geen doel: geleidelijk "
                "(~+5% per paar weken) is genoeg, en dit is iets voor ná "
                "Ouderkerk (22 augustus) — vlak voor een race geen nieuw "
                "looppatroon inslijpen."
            )

        st.subheader("Loopdynamiek")
        st.caption(
            "Trendcijfers van het horloge om over máánden te volgen, niet om "
            "nu op te sturen. Let op: bewust op hogere cadans lopen maakt de "
            "hartslag tijdelijk hoger (onwennig patroon) — dat is normaal en "
            "geen te hard trainen. Label zo'n sessie bij de upload als "
            "'techniek/cadans', dan weegt de coach dat mee."
        )
        heeft_dynamiek = (not dynamiek.empty and dynamiek[
            ["staplengte_m", "gct_ms", "vert_ratio_pct"]].notna().any().any())
        if not heeft_dynamiek:
            st.info(
                "Nog geen loopdynamiek-velden in de database. Die zitten wel "
                "in je FIT-bestanden, maar werden bij eerdere imports niet "
                "opgeslagen: upload je Garmin-zips gewoon opnieuw — duplicaten "
                "worden niet dubbel geteld, alleen aangevuld."
            )
        else:
            d1, d2, d3 = st.columns(3)
            with d1:
                stap = dynamiek.dropna(subset=["staplengte_m"])
                if not stap.empty:
                    fig = px.line(
                        stap, x="start_time", y="staplengte_m", markers=True,
                        labels={"start_time": "Datum", "staplengte_m": "Staplengte (m)"},
                    )
                    fig.update_traces(
                        marker=dict(size=10), line=dict(color=SPORT_COLORS["Hardlopen"]),
                        hovertemplate="%{x|%d-%m-%Y} · %{y:.2f} m<extra></extra>")
                    fig.update_layout(title="Staplengte per sessie")
                    date_xaxis(fig, stap["start_time"])
                    pad_single_point(fig, stap["start_time"])
                    chart(fig, show_legend=False)
                    st.caption(
                        "Context bij de cadans: bij gelijk tempo en een hogere "
                        "cadans wordt de pas vanzelf korter — een dalende lijn "
                        "is hier dus geen achteruitgang."
                    )
            with d2:
                gct = dynamiek.dropna(subset=["gct_ms"])
                if not gct.empty:
                    fig = px.line(
                        gct, x="start_time", y="gct_ms", markers=True,
                        labels={"start_time": "Datum", "gct_ms": "Grondcontacttijd (ms)"},
                    )
                    fig.update_traces(
                        marker=dict(size=10), line=dict(color=SPORT_COLORS["Hardlopen"]),
                        hovertemplate="%{x|%d-%m-%Y} · %{y:.0f} ms<extra></extra>")
                    fig.add_hrect(y0=GCT_TRAINED_MS[0], y1=GCT_TRAINED_MS[1],
                                  line_width=0,
                                  fillcolor=with_alpha(PAL["muted"], 0.15),
                                  annotation_text=f"geoefend: {GCT_TRAINED_MS[0]}–"
                                                  f"{GCT_TRAINED_MS[1]} ms (referentie)",
                                  annotation_position="bottom left",
                                  annotation_font_color=PAL["muted"])
                    fig.update_layout(title="Grondcontacttijd per sessie")
                    date_xaxis(fig, gct["start_time"])
                    pad_single_point(fig, gct["start_time"])
                    chart(fig, show_legend=False)
                    st.caption(
                        "Korter = efficiënter. Hangt samen met de cadans: gaat "
                        "die over de maanden omhoog, dan zakt dit vanzelf mee. "
                        "Volg de trend, forceer niets."
                    )
            with d3:
                ratio = dynamiek.dropna(subset=["vert_ratio_pct"])
                if not ratio.empty:
                    fig = px.line(
                        ratio, x="start_time", y="vert_ratio_pct", markers=True,
                        labels={"start_time": "Datum",
                                "vert_ratio_pct": "Verticale ratio (%)"},
                    )
                    fig.update_traces(
                        marker=dict(size=10), line=dict(color=SPORT_COLORS["Hardlopen"]),
                        hovertemplate="%{x|%d-%m-%Y} · %{y:.1f}%<extra></extra>")
                    # Groene band: onder ~8% is dit gewoon goed — expliciet,
                    # zodat de grafiek niet iets suggereert dat beter moet.
                    fig.add_hrect(y0=0, y1=VERTICAL_RATIO_GOOD_PCT, line_width=0,
                                  fillcolor=with_alpha(PAL["zones"][1], 0.15),
                                  annotation_text=f"prima (< {VERTICAL_RATIO_GOOD_PCT:.0f}%)",
                                  annotation_position="bottom left",
                                  annotation_font_color=PAL["muted"])
                    fig.update_yaxes(rangemode="tozero")
                    fig.update_layout(title="Verticale ratio per sessie")
                    date_xaxis(fig, ratio["start_time"])
                    pad_single_point(fig, ratio["start_time"])
                    chart(fig, show_legend=False)
                    st.caption(
                        "Verticale beweging t.o.v. de staplengte — in het "
                        "groene gebied zit dit gewoon goed en valt er niets "
                        "te verbeteren."
                    )

        inspanningen = cache_best_efforts(DATA_VERSIE)
        if not inspanningen.empty:
            st.subheader("Beste inspanningen per sessie")
            st.caption(
                "Het snelste aaneengesloten stuk van 1 km en 5 km bínnen elke "
                "loopsessie, als tempo. Dalende lijnen = sneller worden, ook als "
                "de sessie als geheel rustig was."
            )
            inspanningen = inspanningen.copy()
            inspanningen["Afstand"] = (inspanningen["afstand_m"] // 1000).map(
                lambda k: f"Snelste {k} km")
            inspanningen["tempo"] = pace_as_time(
                inspanningen["seconden"] / (inspanningen["afstand_m"] / 1000))
            kleuren = {"Snelste 1 km": PAL["cats"][0], "Snelste 5 km": PAL["cats"][1]}
            fig = px.line(
                inspanningen, x="start_time", y="tempo", color="Afstand",
                markers=True, color_discrete_map=kleuren,
                custom_data=["seconden"],
                labels={"start_time": "Datum", "tempo": "Tempo (min/km)", "Afstand": ""},
            )
            pace_axis(fig)
            fig.update_traces(
                marker=dict(size=9),
                hovertemplate="%{x|%d-%m-%Y} · %{fullData.name}: %{y|%M:%S}/km "
                              "(totaal %{customdata[0]:.0f} s)<extra></extra>")
            date_xaxis(fig, inspanningen["start_time"])
            pad_single_point(fig, inspanningen["start_time"])
            chart(fig)

        vermogen = run_power(trainingen)
        if not vermogen.empty:
            st.subheader("Loopvermogen per sessie")
            st.caption(
                "Genormaliseerd vermogen (watt), door het horloge aan de pols "
                "geschat — indicatief, maar consistent tussen sessies. Stijgend "
                "vermogen bij gelijke hartslag wijst op groeiende fitheid. "
                "Kanttekening: loopvermogen is een onrijpe, merk-afhankelijke "
                "maat — aardig om te volgen, geen stuurmiddel. Sturen doe je "
                "op hartslag."
            )
            fig = px.line(
                vermogen, x="start_time", y="watt", markers=True,
                labels={"start_time": "Datum", "watt": "Vermogen (W)"},
            )
            fig.update_traces(
                marker=dict(size=10), line=dict(color=SPORT_COLORS["Hardlopen"]),
                hovertemplate="%{x|%d-%m-%Y} · %{y:.0f} W<extra></extra>")
            date_xaxis(fig, vermogen["start_time"])
            pad_single_point(fig, vermogen["start_time"])
            chart(fig, show_legend=False)

with tab_fietsen:
    rides = trainingen[trainingen["sport"] == "cycling"].sort_values("start_time")
    if rides.empty:
        st.info("Nog geen fietssessies.")
    else:
        rides = rides.copy()
        rides["snelheid_kmh"] = rides["avg_speed_ms"] * 3.6
        c1, c2 = st.columns(2)
        with c1:
            fig = px.scatter(
                rides, x="start_time", y="snelheid_kmh", color="avg_hr",
                color_continuous_scale=PAL["seq"],
                labels={"start_time": "Datum", "snelheid_kmh": "Snelheid (km/h, actief)",
                        "avg_hr": "Gem. HR"},
            )
            fig.update_traces(
                mode="lines+markers",
                marker=dict(size=14, line=dict(width=1, color=PAL["surface"])),
                line=dict(color=with_alpha(PAL["muted"], 0.4)),
                hovertemplate="%{x|%d-%m-%Y} · %{y:.1f} km/h · HR %{marker.color}<extra></extra>",
            )
            fig.update_layout(
                title="Snelheid per rit — op rijtijd (kleur = gemiddelde hartslag)")
            chart(fig, show_legend=False)
        with c2:
            fig = px.bar(
                rides, x="start_time", y="total_ascent",
                labels={"start_time": "Datum", "total_ascent": "Hoogtemeters"},
            )
            fig.update_traces(
                marker_color=SPORT_COLORS["Fietsen"],
                hovertemplate="%{x|%d-%m-%Y} · %{y:.0f} m<extra></extra>")
            fig.update_layout(title="Hoogtemeters per rit")
            chart(fig, show_legend=False)

        st.divider()
        st.subheader("⚡ Vermogen")
        pw = power_trend(rides)
        if pw.empty:
            st.info(
                "Nog geen ritten met vermogensdata. Sinds de Rally-pedalen "
                "(buiten) en de Kickr-trainer (binnen) komt die vanzelf mee "
                "met nieuwe FIT-bestanden; eerder geïmporteerde ritten van "
                "ná de aanschaf krijgen hem alsnog door de Garmin-zip "
                "opnieuw te uploaden."
            )
        else:
            # FTP-status: ingesteld → vermogen is leidend; onbekend → de
            # fiets-hartslagzones zijn de tussenoplossing, plus een schatting.
            if FTP:
                grenzen_w = [round(w) for w in power_zone_bounds(FTP)]
                st.caption(
                    f"FTP: **{FTP:.0f} W** (instellingen-tab) → fietssessies "
                    "worden op deze **vermogenszones** beoordeeld (Coggan): "
                    f"P1 <{grenzen_w[0]} · P2 {grenzen_w[0]}–{grenzen_w[1]} · "
                    f"P3 {grenzen_w[1]}–{grenzen_w[2]} · P4 {grenzen_w[2]}–{grenzen_w[3]} · "
                    f"P5 {grenzen_w[3]}–{grenzen_w[4]} · P6 >{grenzen_w[4]} W. "
                    f"Ritten zonder vermogen vallen terug op de fiets-LTHR "
                    f"({bike_lthr(ATHLETE)} bpm)."
                )
            else:
                st.warning(
                    "🔧 **Tussenoplossing:** er is nog geen FTP, dus "
                    "fietssessies worden voorlopig beoordeeld op "
                    f"hartslagzones rond de fiets-LTHR ({bike_lthr(ATHLETE)} "
                    f"bpm; zone 2 = {BIKE_Z2[0]}–{BIKE_Z2[1]}). Vermogen wordt "
                    "wel getoond, maar zonder zone-oordeel."
                )
                schatting = cache_ftp_estimate(DATA_VERSIE)
                if schatting:
                    st.info(
                        f"💡 **FTP-schatting uit je data: ~{schatting['ftp_watt']:.0f} W** "
                        f"({FTP_EST_FACTOR:.0%} van je beste 20-minutenvermogen, "
                        f"{schatting['best20_watt']:.0f} W op "
                        f"{schatting['datum']:%d-%m-%Y}). Zet hem op de ⚙️ "
                        "Instellingen-tab om vermogenszones te krijgen — een "
                        "echte ramptest op de Kickr blijft nauwkeuriger dan "
                        "deze schatting uit gewone trainingen. Rijd je er een, "
                        "dan herkent de 🔍 Sessie-tab hem en doet daar meteen "
                        "een FTP-voorstel."
                    )
                else:
                    st.caption(
                        "Nog geen rit met een vol 20-minutenblok voor een "
                        "schatting. Een ramptest op de Kickr wordt op de "
                        "🔍 Sessie-tab automatisch herkend, met een "
                        "FTP-voorstel ter bevestiging."
                    )

            # Bron als kleur: pedalen (buiten) en trainer (binnen) meten net
            # iets anders, dus trends horen per bron gelezen te worden.
            pw = pw.copy()
            pw["Bron"] = pw["power_source"].fillna("onbekend")
            bron_kleuren = dict(zip(sorted(pw["Bron"].unique()), PAL["cats"]))

            e1, e2 = st.columns(2)
            with e1:
                ef_pw = pw.dropna(subset=["ef_watt"])
                if ef_pw.empty:
                    st.info("Nog geen ritten met vermogen én hartslag.")
                else:
                    fig = px.line(
                        ef_pw, x="start_time", y="ef_watt", color="Bron",
                        markers=True, color_discrete_map=bron_kleuren,
                        labels={"start_time": "Datum",
                                "ef_watt": "EF (W per hartslag)", "Bron": ""},
                    )
                    fig.update_traces(
                        marker=dict(size=11),
                        hovertemplate="%{x|%d-%m-%Y} · EF %{y:.2f} · "
                                      "%{fullData.name}<extra></extra>")
                    fig.update_layout(
                        title="Efficiency factor — NP per hartslag (hoger = beter)")
                    date_xaxis(fig, ef_pw["start_time"])
                    pad_single_point(fig, ef_pw["start_time"])
                    chart(fig)
                    st.caption(
                        "Dé aerobe fietstrend, eindelijk windonafhankelijk: "
                        "meer watt bij dezelfde hartslag = grotere motor. "
                        "Vergelijk binnen één kleur (bron) — pedalen en "
                        "trainer meten net iets anders."
                    )
            with e2:
                np_pw = pw.copy()
                np_pw["watt"] = np_pw["np_power"].where(
                    np_pw["np_power"].notna(), np_pw["avg_power"])
                fig = px.line(
                    np_pw, x="start_time", y="watt", color="Bron",
                    markers=True, color_discrete_map=bron_kleuren,
                    custom_data=["avg_power"],
                    labels={"start_time": "Datum", "watt": "NP (W)", "Bron": ""},
                )
                fig.update_traces(
                    marker=dict(size=11),
                    hovertemplate="%{x|%d-%m-%Y} · NP %{y:.0f} W (gem. "
                                  "%{customdata[0]:.0f} W) · %{fullData.name}"
                                  "<extra></extra>")
                fig.update_layout(title="Normalized power per rit")
                date_xaxis(fig, np_pw["start_time"])
                pad_single_point(fig, np_pw["start_time"])
                chart(fig)

            f1, f2 = st.columns(2)
            with f1:
                cad_pw = pw.dropna(subset=["cadans"])
                if not cad_pw.empty:
                    fig = px.line(
                        cad_pw, x="start_time", y="cadans", color="Bron",
                        markers=True, color_discrete_map=bron_kleuren,
                        labels={"start_time": "Datum",
                                "cadans": "Cadans (rpm)", "Bron": ""},
                    )
                    fig.update_traces(
                        marker=dict(size=11),
                        hovertemplate="%{x|%d-%m-%Y} · %{y:.0f} rpm · "
                                      "%{fullData.name}<extra></extra>")
                    fig.update_layout(
                        title="Cadans per rit (excl. freewheelen, als Garmin)")
                    date_xaxis(fig, cad_pw["start_time"])
                    pad_single_point(fig, cad_pw["start_time"])
                    chart(fig)
                    st.caption(
                        "Observatie, geen doel: je eigen patroon is de maat. "
                        "Iets hogere cadans spaart de benen richting het "
                        "lopen erna — relevant voor triatlon, niets om te "
                        "forceren."
                    )
            with f2:
                pw_dec = cache_power_decoupling(DATA_VERSIE)
                if not pw_dec.empty:
                    pw_dec = pw_dec.copy().sort_values("start_time")
                    pw_dec["Waar"] = pw_dec["indoor"].map(
                        {True: "Indoor (trainer)", False: "Buiten"})
                    pw_dec["label"] = pw_dec["start_time"].dt.strftime("%d-%m")
                    fig = px.bar(
                        pw_dec, x="label", y="decoupling_pct", color="Waar",
                        barmode="group",
                        color_discrete_map={"Buiten": SPORT_COLORS["Fietsen"],
                                            "Indoor (trainer)": PAL["cats"][5]},
                        labels={"label": "Datum",
                                "decoupling_pct": "Pw:Hr decoupling (%)",
                                "Waar": ""},
                        category_orders={"label": pw_dec["label"].tolist()},
                    )
                    # Categorie-as: op een datum-as verschrompelen staven over
                    # weken tot onzichtbare haarlijntjes (zie HR-decoupling).
                    fig.update_xaxes(type="category")
                    fig.add_hline(y=5, line_dash="dash", line_color=PAL["ref_line"],
                                  annotation_text="richtwaarde 5%",
                                  annotation_font_color=PAL["muted"])
                    fig.update_traces(
                        marker_line=dict(width=1, color=PAL["surface"]),
                        hovertemplate="%{x} · %{y:.1f}%<extra></extra>")
                    fig.update_layout(
                        title="Aerobic decoupling (Pw:Hr) — lager = betere basis")
                    chart(fig)
                    st.caption(
                        "Vermogen/hartslag van de eerste vs de tweede helft "
                        "van ritten ≥30 min. Onder de 5% bij een duurrit "
                        "duidt op een goed ontwikkelde aerobe basis — de "
                        "opvolger van de HR-drift-analyse."
                    )

with tab_zwemmen:
    swim = swim_per_session(conn, trainingen)
    if swim.empty:
        st.info("Nog geen zwemsessies.")
    else:
        css = cache_css(DATA_VERSIE)
        if css:
            m1, m2, m3 = st.columns(3)
            css_s = css["css_per_100m"]
            m1.metric("CSS (kritieke zwemsnelheid)",
                      f"{int(css_s // 60)}:{int(css_s % 60):02d}/100m",
                      help="Geschat uit je snelste aaneengesloten 400 m en 200 m "
                           "crawl: (t400 − t200) / 2. Dit tempo kun je in theorie "
                           "lang volhouden — rustige banen zwem je erboven (langzamer), "
                           "intervallen eromheen.")
            m2.metric("Snelste 400 m crawl", fmt_duration(css["t400"]))
            m3.metric("Snelste 200 m crawl", fmt_duration(css["t200"]))
        else:
            st.caption(
                "💡 Vanaf je volgende zwem-upload verschijnt hier je **CSS** "
                "(kritieke zwemsnelheid): daarvoor zijn per-baan-starttijden "
                "nodig die pas sinds deze versie worden opgeslagen."
            )
        c1, c2 = st.columns(2)
        with c1:
            swolf_df = swim.dropna(subset=["swolf"])
            if swolf_df.empty:
                st.info("Nog geen crawlbanen in het 25m-bad voor de SWOLF-trend.")
            else:
                fig = px.line(
                    swolf_df, x="start_time", y="swolf", markers=True,
                    labels={"start_time": "Datum", "swolf": "SWOLF"},
                )
                fig.update_traces(
                    marker=dict(size=11), line=dict(color=SPORT_COLORS["Zwemmen"]),
                    hovertemplate="%{x|%d-%m-%Y} · SWOLF %{y:.0f}<extra></extra>",
                )
                fig.update_layout(
                    title="Gemiddelde SWOLF per sessie — alleen crawl, 25m-bad "
                          "(lager = efficiënter)")
                chart(fig, show_legend=False)
        with c2:
            swim = swim.copy()
            swim["tempo"] = pace_as_time(swim["tempo_s_per_100m"])
            fig = px.line(
                swim, x="start_time", y="tempo", markers=True,
                labels={"start_time": "Datum", "tempo": "Actief tempo (min/100m)"},
            )
            pace_axis(fig)
            fig.update_traces(
                marker=dict(size=11), line=dict(color=SPORT_COLORS["Zwemmen"]),
                hovertemplate="%{x|%d-%m-%Y} · %{y|%M:%S} /100m<extra></extra>",
            )
            if css:
                css_y = pace_as_time(pd.Series([css["css_per_100m"]])).iloc[0]
                fig.update_traces(name="Per sessie", showlegend=True)
                fig.add_scatter(
                    x=[swim["start_time"].min(), swim["start_time"].max()],
                    y=[css_y, css_y], mode="lines", name="CSS",
                    line=dict(color=PAL["ref_line"], dash="dash", width=2),
                    hoverinfo="skip")
            fig.update_layout(
                title="Actief tempo per 100 meter — excl. rust (sneller = hoger)")
            chart(fig, show_legend=css is not None)

        st.subheader("Tempo per baan")
        st.caption(
            "Elke rij is een sessie, elke cel een baan, als tempo per 100 m: "
            "donkerder = langzamer. Zo zie je in één blik waar het tempo in een "
            "sessie wegzakte — en of dat punt per sessie later komt te liggen. "
            "Bij een sessie met wisselende baanlengtes wordt de afstand per baan "
            "geschat via het aantal slagen."
        )
        matrix = swim_length_matrix(conn, trainingen)
        if matrix.empty:
            st.info("Nog geen baandata.")
        else:
            matrix.index = [f"{d:%d-%m-%Y}" for d in matrix.index]
            fig = px.imshow(
                matrix, aspect="auto", color_continuous_scale=PAL["seq"],
                labels=dict(x="Baan", y="Sessie", color="s/100m"),
            )
            fig.update_traces(
                hovertemplate="%{y} · baan %{x}: %{z:.0f} s/100m<extra></extra>")
            chart(fig, show_legend=False)

        st.subheader("Slagverdeling per sessie")
        verdeling = stroke_distribution(conn, trainingen)
        if not verdeling.empty:
            verdeling["Slag"] = verdeling["slag"].map(stroke_label)
            verdeling["pct"] = (verdeling["banen"] / verdeling.groupby("start_time")
                                ["banen"].transform("sum") * 100)
            fig = px.bar(
                verdeling, x="start_time", y="pct", color="Slag",
                color_discrete_map=STROKE_COLORS, custom_data=["banen"],
                labels={"start_time": "Datum", "pct": "Aandeel banen (%)", "Slag": ""},
            )
            fig.update_traces(
                marker_line=dict(width=1, color=PAL["surface"]),
                hovertemplate="%{x|%d-%m-%Y} · %{fullData.name}: %{y:.0f}% "
                              "(%{customdata[0]} banen)<extra></extra>")
            fig.update_layout(title="Aandeel per slagtype (doel: steeds meer borstcrawl)")
            date_xaxis(fig, verdeling["start_time"])
            chart(fig)

# ------------------------------------------------------------------ bricks --
with tab_bricks:
    st.subheader("🧱 Combinatietrainingen (bricks & triatlon-trainingen)")
    st.caption(
        f"Losse sessies van dezelfde dag die binnen **{max_gap_min(config)} "
        "minuten** na elkaar starten, in racevolgorde (zwemmen → fietsen → "
        "hardlopen), worden hier als één training voorgesteld. Twee onderdelen "
        "= brick, drie = triatlon-training. Jij bevestigt of maakt los — er "
        "wordt nooit stilzwijgend samengevoegd. De drempel is instelbaar in "
        "config.yaml (combo.max_gap_min)."
    )

    SPORT_ICOON = {"swimming": "🏊", "cycling": "🚴", "running": "🏃"}

    def combo_titel(combo) -> str:
        """Kopregel van een combo-blok: soort, datum en de onderdelen."""
        soort = ("🏊🚴🏃 Triatlon-training" if combo["kind"] == "triatlon"
                 else "🧱 Brick")
        delen = " → ".join(
            f"{SPORT_ICOON[m['sport']]} {(m['distance_m'] or 0) / 1000:.1f} km"
            for m in combo["members"])
        return f"{soort} — {combo['start_time']:%a %d-%m-%Y}: {delen}"

    def render_combo(combo):
        """Eén combinatietraining als samengesteld blok.

        Per onderdeel de kerncijfers, de wisseltijden ertussen, een totaal
        inclusief wissels, de race-simulatievergelijking en — bij een loop na
        het fietsen — de bakstenen-benen-analyse met tempoverloopgrafiekje.
        """
        members = combo["members"]
        kolommen = st.columns(len(members) * 2 - 1)
        for i, m in enumerate(members):
            open_water = "open_water" in str(m.get("sub_sport") or "")
            naam = sport_label(m["sport"]) + (" (open water)" if open_water else "")
            hr = (f"HR gem {m['avg_hr']:.0f}"
                  if pd.notna(m.get("avg_hr")) else "HR —")
            kolommen[i * 2].markdown(
                f"**{SPORT_ICOON[m['sport']]} {naam}**  \n"
                f"{(m['distance_m'] or 0) / 1000:.2f} km · "
                f"{fmt_duration(m['duration_s'])}  \n"
                f"{veilig_cel(sessie_tempo, m['sport'], m['distance_m'], m['duration_s'], m.get('active_s'))}"
                f" · {hr}")
            if i < len(members) - 1:
                t = combo["transitions"][i]
                kolommen[i * 2 + 1].metric(
                    f"⏱️ {t['label']}", fmt_duration(t["seconds"]),
                    help="Tijd tussen het einde van het vorige onderdeel en de "
                         "start van het volgende — oefenbare racetijd.")
        st.markdown(
            f"**Totaal: {fmt_duration(combo['totaal_s'])}** (incl. "
            f"{fmt_duration(combo['wissel_s'])} wisseltijd)")

        afstanden = {m["sport"]: m.get("distance_m") for m in members}
        sim = race_similarity(afstanden, config)
        if sim:
            st.info(f"🏁 {sim}")

        # De kern van brick-training: de loop na het fietsen. Eerste stuk vs
        # de rest, plus het tempoverloop van de eerste kilometers.
        for i, t in enumerate(combo["transitions"]):
            if not (t["van"] == "cycling" and t["naar"] == "running"):
                continue
            run = members[i + 1]
            rec = load_records(conn, run["activity_key"])
            analyse = run_transition_analysis(rec)
            if analyse:
                e, r = analyse["eerste"], analyse["rest"]
                c1, c2, c3 = st.columns(3)
                c1.metric(
                    f"Tempo {analyse['basis']} na het fietsen",
                    fmt_duration(e["tempo_s_per_km"]) + "/km"
                    if e["tempo_s_per_km"] else GEEN_WAARDE)
                c2.metric(
                    "Tempo rest van de loop",
                    fmt_duration(r["tempo_s_per_km"]) + "/km"
                    if r["tempo_s_per_km"] else GEEN_WAARDE,
                    delta=(f"{e['tempo_s_per_km'] - r['tempo_s_per_km']:+.0f} "
                           "s/km t.o.v. eerste stuk"
                           if e["tempo_s_per_km"] and r["tempo_s_per_km"]
                           else None),
                    delta_color="off")
                c3.metric(
                    "HR eerste stuk → rest",
                    f"{e['gem_hr']:.0f} → {r['gem_hr']:.0f}"
                    if e["gem_hr"] and r["gem_hr"] else GEEN_WAARDE,
                    help="Trekt het tempo na het eerste stuk aan bij gelijke "
                         "HR, dan kwamen de 'bakstenen benen' los.")
            splits = run_splits_df(rec, max_km=6)
            if not splits.empty:
                splits = splits.copy()
                splits["Tempo"] = pace_as_time(splits["duur_s"])
                fig = px.line(
                    splits, x="km", y="Tempo", markers=True,
                    custom_data=["gem_hr"],
                    labels={"km": "Kilometer van de loop", "Tempo": "Tempo (min/km)"},
                )
                pace_axis(fig)
                fig.update_traces(
                    marker=dict(size=11),
                    line=dict(color=SPORT_COLORS["Hardlopen"]),
                    hovertemplate="km %{x} · %{y|%M:%S}/km · HR %{customdata[0]:.0f}"
                                  "<extra></extra>")
                fig.update_xaxes(dtick=1)
                fig.update_layout(
                    title="Tempoverloop van de loop na het fietsen "
                          "(eerste kilometers = inlopen van de bakstenen benen)")
                chart(fig, show_legend=False,
                      key=f"combo_run_{combo['combo_id']}_{i}")

    alle_combos = load_combos(conn, acts)
    voorstellen = [c for c in alle_combos if c["status"] == "voorgesteld"]
    bevestigde = [c for c in alle_combos if c["status"] == "bevestigd"]

    if not alle_combos:
        st.info(
            "Nog geen combinatietrainingen gevonden. Upload de losse "
            "FIT-bestanden van een brick (bijv. fietsen en direct daarna "
            "hardlopen) en het voorstel verschijnt hier."
        )

    for combo in voorstellen:
        with st.container(border=True):
            st.markdown(f"#### 💡 Voorstel: {combo_titel(combo)}")
            st.caption(
                "Deze sessies volgen kort op elkaar in racevolgorde. Horen ze "
                "bij elkaar als één combinatietraining?")
            render_combo(combo)
            knop_ja, knop_nee, _ = st.columns([1, 1, 3])
            if knop_ja.button("✅ Ja, samen één training",
                              key=f"combo_ok_{combo['combo_id']}", type="primary"):
                set_combo_status(conn, combo["combo_id"], "bevestigd")
                st.toast("Combinatietraining bevestigd — telt nu mee in de trends.")
                st.rerun()
            if knop_nee.button("✖️ Nee, losse trainingen",
                               key=f"combo_nee_{combo['combo_id']}"):
                set_combo_status(conn, combo["combo_id"], "losgemaakt")
                st.toast("Losgemaakt — dit voorstel komt niet terug.")
                st.rerun()

    if bevestigde:
        st.subheader("Bevestigde combinatietrainingen")
        for combo in bevestigde:
            with st.expander(combo_titel(combo), expanded=(combo is bevestigde[0])):
                render_combo(combo)
                if st.button("🔓 Losmaken (toch losse trainingen)",
                             key=f"combo_los_{combo['combo_id']}"):
                    set_combo_status(conn, combo["combo_id"], "losgemaakt")
                    st.toast("Losgemaakt — deze groep komt niet opnieuw als voorstel.")
                    st.rerun()

    # ---- Trends over de bevestigde combos: worden de wissels korter en de
    # ---- overgang soepeler? Dat is de kernmaat van brick-training.
    historie = combo_history(conn, acts, load_records)
    if len(historie) >= 2:
        st.subheader("Trend over je combinatietrainingen")
        c1, c2 = st.columns(2)
        with c1:
            wissels = historie.melt(
                id_vars=["start_time"], value_vars=["t1_s", "t2_s"],
                var_name="wissel", value_name="seconden").dropna(subset=["seconden"])
            if not wissels.empty:
                wissels["Wissel"] = wissels["wissel"].map(
                    {"t1_s": "T1 (zwem → fiets)", "t2_s": "T2 (fiets → loop)"})
                wissels["Tijd"] = pace_as_time(wissels["seconden"])
                fig = px.line(
                    wissels, x="start_time", y="Tijd", color="Wissel", markers=True,
                    color_discrete_map={"T1 (zwem → fiets)": PAL["cats"][0],
                                        "T2 (fiets → loop)": PAL["cats"][2]},
                    labels={"start_time": "Datum", "Tijd": "Wisseltijd (min)"},
                )
                fig.update_yaxes(tickformat="%M:%S")
                fig.update_traces(
                    marker=dict(size=11),
                    hovertemplate="%{x|%d-%m-%Y} · %{fullData.name}: "
                                  "%{y|%M:%S}<extra></extra>")
                fig.update_layout(title="Wisseltijden per training (korter = beter)")
                date_xaxis(fig, wissels["start_time"])
                chart(fig)
        with c2:
            delta = historie.dropna(subset=["delta_tempo_s_per_km"]).copy()
            if not delta.empty:
                # De +/- vooraf in Python opmaken: Plotly's hover-format snapt de
                # d3-tekenvlag (%{y:+.0f}) niet en toont dan de rauwe float met
                # alle decimalen. Via customdata houden we het teken én ronden af.
                delta["delta_txt"] = delta["delta_tempo_s_per_km"].map(
                    lambda v: f"{v:+.0f}")
                fig = px.line(
                    delta, x="start_time", y="delta_tempo_s_per_km", markers=True,
                    custom_data=["delta_txt"],
                    labels={"start_time": "Datum",
                            "delta_tempo_s_per_km": "Eerste stuk t.o.v. rest (s/km)"},
                )
                fig.update_traces(
                    marker=dict(size=11), line=dict(color=SPORT_COLORS["Hardlopen"]),
                    hovertemplate="%{x|%d-%m-%Y} · eerste stuk %{customdata[0]} s/km "
                                  "t.o.v. de rest<extra></extra>")
                fig.add_hline(y=0, line_dash="dash", line_color=PAL["ref_line"])
                fig.update_layout(
                    title="Hoe zwaar was de start van de loop na het fietsen? "
                          "(dichter bij 0 = soepelere overgang)")
                date_xaxis(fig, delta["start_time"])
                chart(fig, show_legend=False)
        st.caption(
            "Beide grafieken tellen alleen **bevestigde** combinatietrainingen. "
            "Worden de wisseltijden korter en zakt het tempoverschil van het "
            "eerste stuk richting 0, dan werpt de brick-training vruchten af."
        )

# ----------------------------------------------------------------- lichaam --
with tab_lichaam:
    st.subheader("🧍 Lichaamssamenstelling")
    st.caption(
        "Neutrale data voor je sportprestatie — geen afval- of dieetcoach. "
        "De nadruk ligt op de **trend over weken**, vooral vet% en spiermassa. "
        "BMI zegt weinig bij veel spiermassa en krijgt daarom weinig gewicht."
    )

    body.ensure_table(conn)
    metingen = body.load_measurements(conn)

    # Flits van de vorige rerun: de st.rerun() na het opslaan waaide de
    # succesmelding én de verse trendduiding meteen weg, waardoor de duiding
    # alleen nog onderin het logboek terechtkwam. Nu blijft hij in beeld.
    if body_flash := st.session_state.pop("body_flash", None):
        datum_flash, duiding_flash = body_flash
        st.success(f"Meting van {datum_flash:%d-%m-%Y} opgeslagen.")
        if duiding_flash:
            st.info(f"**Trendduiding:** {duiding_flash}")

    # -- Invoer (voorgevuld met de laatste meting of een screenshot) --------
    with st.expander("➕ Nieuwe meting invoeren", expanded=False):
        st.caption(
            "De velden staan voorgevuld met je laatste meting — pas alleen aan "
            "wat veranderd is. Vul je liever automatisch: upload eerst een "
            "screenshot van de Fitdays-app (lokaal gemma-model, gratis). "
            "Velden op 0 worden niet opgeslagen."
        )
        shot = st.file_uploader(
            "Screenshot Fitdays (optioneel)", type=["png", "jpg", "jpeg"],
            key="body_shot",
        )
        if shot is not None and st.button("📷 Velden uit screenshot lezen"):
            with st.spinner("Gemma leest de screenshot..."):
                try:
                    st.session_state["body_prefill"] = body.extract_from_screenshot(
                        router, shot.getvalue())
                    st.success("Velden voorgevuld — controleer ze hieronder.")
                except Exception as e:
                    st.error(f"Uitlezen mislukt, vul handmatig in: {e}")

        prefill = st.session_state.get("body_prefill", {})
        laatste_meting = metingen.iloc[-1] if not metingen.empty else None
        with st.form("body_form"):
            meetdatum = st.date_input("Datum", value=date.today())
            cols = st.columns(3)
            ingevuld = {}
            for idx, (col, label, eenheid, stap) in enumerate(body.FIELDS):
                with cols[idx % 3]:
                    label_txt = f"{label}{f' ({eenheid})' if eenheid else ''}"
                    # Voorvullen: screenshot-waarde > laatste meting > leeg.
                    # Een nieuwe meting wijkt meestal weinig af van de vorige,
                    # dus zo hoef je alleen de verschillen aan te passen.
                    basis = prefill.get(col) or 0.0
                    if not basis and laatste_meting is not None \
                            and pd.notna(laatste_meting.get(col)):
                        basis = float(laatste_meting[col])
                    val = st.number_input(
                        label_txt, value=float(basis),
                        step=stap, format="%.1f", key=f"body_{col}",
                    )
                    ingevuld[col] = val
            opslaan = st.form_submit_button("💾 Meting opslaan")

        if opslaan:
            # 0.0 betekent 'niet ingevuld' (de weegschaal geeft geen nullen).
            waarden = {k: (v if v else None) for k, v in ingevuld.items()}
            if not any(v is not None for v in waarden.values()):
                st.warning("Niets ingevuld — voer minstens één waarde in.")
            else:
                body.save_measurement(conn, meetdatum, waarden)
                body.log_measurement(MEMORY_DIR, meetdatum, waarden)
                with st.spinner("Korte trendduiding..."):
                    duiding = body.summarize_trend(router, conn, MEMORY_DIR)
                st.session_state.pop("body_prefill", None)
                # Via een flash over de rerun heen, anders is de melding (en
                # de duiding) meteen weer weg — zie bovenaan deze tab.
                st.session_state["body_flash"] = (meetdatum, duiding)
                st.rerun()

    # -- Meting verwijderen -------------------------------------------------
    if not metingen.empty:
        with st.expander("🗑️ Meting verwijderen", expanded=False):
            datums = sorted(metingen["measured_on"].dt.date.tolist(), reverse=True)
            te_wissen = st.selectbox(
                "Kies de datum die je wilt verwijderen", datums,
                format_func=lambda d: f"{d:%d-%m-%Y}",
            )
            if st.button("🗑️ Verwijder deze meting", type="secondary"):
                body.delete_measurement(conn, te_wissen)
                st.success(f"Meting van {te_wissen:%d-%m-%Y} verwijderd.")
                st.rerun()

    if metingen.empty:
        st.info("Nog geen metingen. Voer er een in via 'Nieuwe meting invoeren'.")
    else:
        # -- Laatste meting + datumbereik -----------------------------------
        laatste = metingen.iloc[-1]
        kerncijfers = [
            ("Gewicht", "weight_kg", "kg"), ("Lichaamsvet", "fat_pct", "%"),
            ("Spiermassa", "muscle_mass_kg", "kg"), ("Visceraal vet", "visceral_fat", ""),
        ]
        kcols = st.columns(len(kerncijfers))
        for kc, (titel, col, eenheid) in zip(kcols, kerncijfers):
            waarde = laatste[col]
            delta = None
            if len(metingen) > 1:
                eerdere = metingen[col].dropna()
                if len(eerdere) > 1 and pd.notna(waarde):
                    delta = f"{waarde - eerdere.iloc[-2]:+.1f}"
            kc.metric(titel, f"{waarde:g} {eenheid}".strip() if pd.notna(waarde) else "—",
                      delta=delta, delta_color="off")

        dmin = metingen["measured_on"].min().date()
        dmax = metingen["measured_on"].max().date()
        if dmin == dmax:
            start_d, end_d = dmin, dmax
            st.caption(f"Eén meting op {dmin:%d-%m-%Y} — trends verschijnen vanaf de tweede meting.")
        else:
            start_d, end_d = st.slider(
                "Datumbereik", min_value=dmin, max_value=dmax,
                value=(dmin, dmax), format="DD-MM-YYYY",
            )
        sel = body.in_range(metingen, start_d, end_d)

        # -- Trendrooster: alle maten met data --------------------------------
        st.subheader("Trends per maat")
        st.caption("Alle maten van de weegschaal, klein naast elkaar; "
                   "maten zonder metingen worden overgeslagen.")
        grid = st.columns(3)
        paneel = 0
        for col, label, eenheid, _stap in body.FIELDS:
            serie = sel[["measured_on", col]].dropna()
            if serie.empty:
                continue
            titel = f"{label}{f' ({eenheid})' if eenheid else ''}"
            with grid[paneel % 3]:
                fig = px.line(serie, x="measured_on", y=col, markers=True,
                              labels={"measured_on": "", col: ""})
                fig.update_traces(
                    line=dict(color=PAL["cats"][0]), marker=dict(size=8),
                    hovertemplate="%{x|%d-%m-%Y} · %{y:.1f}<extra></extra>")
                fig.update_layout(title=titel, height=240,
                                  margin=dict(t=40, b=10, l=10, r=10))
                date_xaxis(fig, serie["measured_on"])
                pad_single_point(fig, serie["measured_on"], days=7)
                st.plotly_chart(style_fig(fig, show_legend=False), width="stretch",
                                config=PLOTLY_CONFIG, key=f"body_trend_{col}")
            paneel += 1

        # -- Gecombineerd (genormaliseerd) ----------------------------------
        combo = body.normalized_trends(sel, body.TREND_FIELDS)
        if not combo.empty and combo["measured_on"].nunique() > 1:
            st.subheader("Gecombineerd (relatieve verandering)")
            st.caption("Elke reeks geïndexeerd op de eerste meting in het bereik (=100), "
                       "zodat maten met verschillende eenheden samen vergelijkbaar zijn.")
            fig = px.line(combo, x="measured_on", y="index", color="reeks", markers=True,
                          labels={"measured_on": "Datum", "index": "Index (eerste = 100)",
                                  "reeks": "Maat"})
            fig.add_hline(y=100, line_dash="dot", line_color=with_alpha(PAL["muted"], 0.6))
            fig.update_traces(hovertemplate="%{fullData.name}: %{y:.1f}<extra></extra>")
            fig.update_layout(hovermode="x unified")
            fig.update_xaxes(hoverformat="%d-%m-%Y")
            date_xaxis(fig, combo["measured_on"])
            chart(fig)

        # -- Kruising met trainingsdata -------------------------------------
        kruising = body.weight_vs_cycling(conn, trainingen)
        if not kruising.empty and len(kruising) > 1:
            st.subheader("Gewicht naast fietsvolume (richting power-to-weight)")
            st.caption(
                "Vermogen wordt niet opgeslagen, dus fietskilometers per week dienen "
                "als prestatieproxy. Daalt het gewicht terwijl het fietsvolume op peil "
                "blijft, dan beweegt je power-to-weight de goede kant op."
            )
            # Twee panelen met gedeelde week-as in plaats van een dubbele
            # y-as: twee schalen over elkaar suggereren een verband tussen
            # lijnhoogtes dat er niet is en lezen lastig.
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                row_heights=[0.55, 0.45], vertical_spacing=0.08)
            fig.add_trace(go.Scatter(
                x=kruising["week"], y=kruising["weight_kg"], name="Gewicht (kg)",
                line=dict(color=PAL["cats"][0], width=3), mode="lines+markers",
                hovertemplate="%{x}: %{y:.1f} kg<extra></extra>"),
                row=1, col=1)
            fig.add_trace(go.Bar(
                x=kruising["week"], y=kruising["fiets_km"], name="Fietskilometers/week",
                marker_color=with_alpha(SPORT_COLORS["Fietsen"], 0.6),
                hovertemplate="%{x}: %{y:.0f} km<extra></extra>"),
                row=2, col=1)
            fig.update_yaxes(title_text="Gewicht (kg)", row=1, col=1)
            fig.update_yaxes(title_text="Fiets-km/week", row=2, col=1)
            fig.update_xaxes(title_text="Week", row=2, col=1)
            fig.update_layout(hovermode="x unified")
            chart(fig)

    st.divider()
    log_path = MEMORY_DIR / "lichaamssamenstelling.md"
    if log_path.exists():
        with st.expander("📖 Logboek & trendduiding"):
            st.markdown(log_path.read_text(encoding="utf-8"))

# ----------------------------------------------------------------- herstel --
with tab_herstel:
    st.subheader("🌙 Herstel — rustpols, HRV & slaap")
    st.caption(
        "Dagelijkse wellness-data uit Garmin Connect. Dag-tot-dag-ruis is "
        "groot: de **dikke lijn (7-daags gemiddelde)** is de maat, niet de "
        "losse dag. Doel: objectief zien of de vroegere bedtijd (22:00) het "
        "herstel verbetert en hoe rustpols en HRV meebewegen met zware "
        "trainingsweken."
    )
    wdf = wellness.load_wellness(conn)
    sync_moment = garmin_sync.last_sync(conn)
    if sync_moment:
        st.caption(f"Laatst gesynct: **{sync_moment:%d-%m-%Y %H:%M}** — "
                   "opnieuw syncen kan via de zijbalk.")

    if wdf.empty:
        st.info(
            "Nog geen wellness-data. Zet `GARMIN_EMAIL` en `GARMIN_PASSWORD` "
            "in `.env` en klik in de zijbalk op **🔄 Nu synchroniseren** — "
            "rustpols, HRV, slaap, body battery en training readiness komen "
            "dan automatisch binnen (dagelijks bij te werken)."
        )
    else:
        # -- Kerncijfers: laatste dag t.o.v. het 7-daags gemiddelde ---------
        def _laatste_waarde(col: str):
            """(dag, waarde, 7-daags gemiddelde ervóór) van de laatste dag met data."""
            serie = wdf[["day", col]].dropna()
            if serie.empty:
                return None
            dag, waarde = serie.iloc[-1]["day"], serie.iloc[-1][col]
            venster = serie[(serie["day"] >= dag - pd.Timedelta(days=7))
                            & (serie["day"] < dag)][col]
            gem = venster.mean() if len(venster) >= 3 else None
            return dag, waarde, gem

        k1, k2, k3, k4, k5 = st.columns(5)
        rp = _laatste_waarde("resting_hr")
        k1.metric("Rustpols", f"{rp[1]:.0f} bpm" if rp else GEEN_WAARDE,
                  delta=(f"{rp[1] - rp[2]:+.0f} vs 7d" if rp and rp[2] else None),
                  delta_color="inverse",
                  help="Lager is doorgaans beter herstel; vergelijk met het "
                       "7-daags gemiddelde, niet met gisteren.")
        hv = _laatste_waarde("hrv_last_night")
        k2.metric("HRV (nacht)", f"{hv[1]:.0f} ms" if hv else GEEN_WAARDE,
                  delta=(f"{hv[1] - hv[2]:+.0f} vs 7d" if hv and hv[2] else None),
                  help="Nachtelijk gemiddelde. Hoger = doorgaans beter "
                       "hersteld; binnen je persoonlijke baseline is alles "
                       "goed.")
        sl = _laatste_waarde("sleep_s")
        k3.metric("Slaap", f"{int(sl[1] // 3600)}u{int(sl[1] % 3600 // 60):02d}"
                  if sl else GEEN_WAARDE,
                  delta=(f"{(sl[1] - sl[2]) / 3600:+.1f}u vs 7d"
                         if sl and sl[2] else None),
                  help="Totale slaapduur van de laatste nacht met data.")
        bb = _laatste_waarde("body_battery_high")
        k4.metric("Body battery (max)", f"{bb[1]:.0f}" if bb else GEEN_WAARDE,
                  help="Hoogste stand van de dag — hoe ver je 's nachts "
                       "hebt opgeladen.")
        tr = _laatste_waarde("training_readiness")
        k5.metric("Readiness", f"{tr[1]:.0f}" if tr else GEEN_WAARDE,
                  help="Garmin's training readiness (0-100), o.a. uit slaap, "
                       "HRV en recente belasting.")

        # -- Periode + voortschrijdend gemiddelde ---------------------------
        periodes = {"Laatste 30 dagen": 30, "Laatste 90 dagen": 90,
                    "Laatste 180 dagen": 180, "Alles": None}
        periode_keuze = st.selectbox("Periode", list(periodes), index=1)
        dagen_terug = periodes[periode_keuze]
        start_p = (date.today() - timedelta(days=dagen_terug)
                   if dagen_terug else None)
        sel = wellness.with_rolling(wellness.in_range(wdf, start_p, None))

        # -- Hoofdgrafieken: rustpols en HRV --------------------------------
        RUST_KLEUR, HRV_KLEUR = PAL["cats"][0], PAL["cats"][1]
        for col, titel, eenheid, kleur, extra_band in (
            ("resting_hr", "Rustpols", "bpm", RUST_KLEUR, False),
            ("hrv_last_night", "HRV — nachtelijk gemiddelde", "ms", HRV_KLEUR, True),
        ):
            serie = sel.dropna(subset=[col])
            if serie.empty:
                continue
            fig = go.Figure()
            if extra_band:
                # Persoonlijke Garmin-baseline als rustige band op de
                # achtergrond: binnen de band = in balans.
                band = sel.dropna(subset=["hrv_baseline_low", "hrv_baseline_high"])
                if not band.empty:
                    fig.add_trace(go.Scatter(
                        x=band["day"], y=band["hrv_baseline_high"],
                        line=dict(width=0), showlegend=False, hoverinfo="skip"))
                    fig.add_trace(go.Scatter(
                        x=band["day"], y=band["hrv_baseline_low"],
                        line=dict(width=0), fill="tonexty",
                        fillcolor=with_alpha(kleur, 0.10),
                        name="Persoonlijke baseline", hoverinfo="skip"))
            fig.add_trace(go.Scatter(
                x=serie["day"], y=serie[col], mode="markers",
                marker=dict(size=5, color=with_alpha(kleur, 0.35)),
                name="Per dag",
                hovertemplate="%{x|%d-%m-%Y}: %{y:.0f} " + eenheid
                              + "<extra></extra>"))
            ma = sel.dropna(subset=[f"{col}_ma"]) if f"{col}_ma" in sel else pd.DataFrame()
            if not ma.empty:
                fig.add_trace(go.Scatter(
                    x=ma["day"], y=ma[f"{col}_ma"], mode="lines",
                    line=dict(color=kleur, width=3), name="7-daags gemiddelde",
                    hovertemplate="%{x|%d-%m-%Y}: gem. %{y:.1f} " + eenheid
                                  + "<extra></extra>"))
            fig.update_layout(title=f"{titel} ({eenheid})")
            date_xaxis(fig, serie["day"])
            pad_single_point(fig, serie["day"], days=7)
            chart(fig, key=f"herstel_{col}")

        # -- Klein rooster: slaap, stress, body battery, readiness, VO2 max --
        st.subheader("Overige maten")
        klein = [
            ("sleep_s", "Slaap (uren)", lambda s: s / 3600, "%{y:.1f} u"),
            ("sleep_score", "Slaapscore", None, "%{y:.0f}"),
            ("stress_avg", "Stress (daggemiddelde)", None, "%{y:.0f}"),
            ("body_battery_high", "Body battery (max)", None, "%{y:.0f}"),
            ("training_readiness", "Training readiness", None, "%{y:.0f}"),
            ("vo2max_run", "VO2 max (lopen)", None, "%{y:.1f}"),
            ("vo2max_bike", "VO2 max (fietsen)", None, "%{y:.1f}"),
        ]
        grid = st.columns(3)
        paneel = 0
        for col, titel, transform, yfmt in klein:
            serie = sel[["day", col]].dropna()
            if serie.empty:
                continue
            serie = serie.copy()
            if transform:
                serie[col] = transform(serie[col])
            with grid[paneel % 3]:
                fig = px.line(serie, x="day", y=col, markers=True,
                              labels={"day": "", col: ""})
                fig.update_traces(
                    line=dict(color=PAL["cats"][paneel % len(PAL["cats"])]),
                    marker=dict(size=6),
                    hovertemplate="%{x|%d-%m-%Y} · " + yfmt + "<extra></extra>")
                fig.update_layout(title=titel, height=240,
                                  margin=dict(t=40, b=10, l=10, r=10))
                date_xaxis(fig, serie["day"])
                pad_single_point(fig, serie["day"], days=7)
                st.plotly_chart(style_fig(fig, show_legend=False),
                                width="stretch", config=PLOTLY_CONFIG,
                                key=f"herstel_klein_{col}")
            paneel += 1

        # -- Kruising met de trainingsbelasting -----------------------------
        herstel_pw = wellness.weekly_recovery(conn)
        if not herstel_pw.empty and not acts.empty:
            uren_pw = (weekly_volume(acts).groupby("week", as_index=False)
                       ["uren"].sum())
            kruis = herstel_pw.merge(uren_pw, on="week", how="inner")
            if len(kruis) > 1:
                st.subheader("Herstel naast trainingsbelasting")
                st.caption(
                    "Weekgemiddelden van rustpols en HRV boven de "
                    "trainingsuren per week: zo zie je of je herstel "
                    "meebeweegt met zware weken — en of hij op tijd "
                    "terugveert. Structureel oplopende rustpols of dalende "
                    "HRV bij hoog volume is het signaal om te remmen."
                )
                fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                                    row_heights=[0.3, 0.3, 0.4],
                                    vertical_spacing=0.07)
                fig.add_trace(go.Scatter(
                    x=kruis["week"], y=kruis["rustpols"], name="Rustpols (weekgem.)",
                    line=dict(color=RUST_KLEUR, width=3), mode="lines+markers",
                    hovertemplate="%{x}: %{y:.1f} bpm<extra></extra>"),
                    row=1, col=1)
                fig.add_trace(go.Scatter(
                    x=kruis["week"], y=kruis["hrv"], name="HRV (weekgem.)",
                    line=dict(color=HRV_KLEUR, width=3), mode="lines+markers",
                    hovertemplate="%{x}: %{y:.1f} ms<extra></extra>"),
                    row=2, col=1)
                fig.add_trace(go.Bar(
                    x=kruis["week"], y=kruis["uren"], name="Trainingsuren/week",
                    marker_color=with_alpha(PAL["cats"][2], 0.6),
                    hovertemplate="%{x}: %{y:.1f} u<extra></extra>"),
                    row=3, col=1)
                fig.update_yaxes(title_text="bpm", row=1, col=1)
                fig.update_yaxes(title_text="ms", row=2, col=1)
                fig.update_yaxes(title_text="uren", row=3, col=1)
                fig.update_xaxes(title_text="Week", row=3, col=1)
                fig.update_layout(hovermode="x unified")
                chart(fig, key="herstel_kruising")

# ----------------------------------------------------------------- voeding --
# Voedingsplanner: invoeren wat je gaat doen, terugkrijgen wat je wanneer neemt.
# Het rekenwerk zit volledig in tricoach.nutrition (deterministisch, testbaar);
# hier staat alleen de UI. Het taalmodel mag hooguit een korte toelichting
# schrijven bij een al berekend plan — nooit de cijfers zelf.
with tab_voeding:
    st.subheader("🥤 Voedingsplan voor training of race")
    st.caption(voeding_regels.DISCLAIMER)

    laatste_gewicht = None
    metingen = body.load_measurements(conn)
    if not metingen.empty and metingen["weight_kg"].notna().any():
        laatste_gewicht = float(metingen["weight_kg"].dropna().iloc[-1])

    alle_producten = voeding_producten.load_products(conn)

    plan_tab, product_tab, historie_tab = st.tabs(
        ["📝 Plan maken", "📦 Producten", "📚 Opgeslagen plannen"])

    # ------------------------------------------------------------ plan maken --
    with plan_tab:
        c1, c2, c3 = st.columns([2, 2, 2])
        sessietype = c1.selectbox(
            "Sport", list(SESSION_TYPES),
            format_func=lambda k: SESSION_TYPES[k][0],
            index=1, key="voeding_type",
        )
        intensiteit = c2.selectbox(
            "Intensiteit", list(INTENSITY_LABEL),
            format_func=lambda k: INTENSITY_LABEL[k], key="voeding_intensiteit",
        )
        plandatum = c3.date_input(
            "Datum van de sessie", value=date.today(), key="voeding_datum",
            help="Bepaalt welke weersverwachting wordt opgehaald.",
        )

        st.markdown("**Onderdelen** — vul per onderdeel een afstand óf een duur in.")
        per_sport: dict[str, tuple[float | None, float | None]] = {}
        posten: list[AidStation] = []
        for i, sport in enumerate(legs_for(sessietype)):
            k1, k2, k3 = st.columns([1, 2, 2])
            k1.markdown(f"<div style='padding-top:2rem'>{sport_label(sport)}</div>",
                        unsafe_allow_html=True)
            basis = k2.radio(
                "Invoer", ["Afstand", "Duur"], horizontal=True,
                key=f"voeding_basis_{sport}", label_visibility="collapsed",
            )
            if basis == "Afstand":
                eenheid = "m" if sport == "swimming" else "km"
                standaard = {"swimming": 1900.0, "cycling": 90.0, "running": 21.1}[sport]
                waarde = k3.number_input(
                    f"Afstand ({eenheid})", min_value=0.0, value=standaard,
                    step=1.0 if sport == "swimming" else 0.1,
                    key=f"voeding_afstand_{sport}",
                )
                meters = waarde if sport == "swimming" else waarde * 1000
                per_sport[sport] = (meters, None)
            else:
                minuten = k3.number_input(
                    "Duur (minuten)", min_value=0.0, value=180.0, step=5.0,
                    key=f"voeding_duur_{sport}",
                )
                per_sport[sport] = (None, minuten * 60)

            if sport != "swimming":
                ruw = st.text_input(
                    f"Verzorgingsposten op de {sport_label(sport).lower()} "
                    f"(km, komma-gescheiden)",
                    key=f"voeding_posten_{sport}", placeholder="bijv. 30, 60",
                    help="Waar je kunt bijvullen; dan hoeft niet alles mee.",
                )
                for stuk in ruw.replace(";", ",").split(","):
                    try:
                        posten.append(AidStation(leg_index=i, km=float(stuk.strip())))
                    except ValueError:
                        continue

        # Temperatuur: standaard uit de verwachting van Open-Meteo voor de
        # geplande dag (thuislocatie uit de privacyzone), handmatig aanpasbaar.
        t1, t2 = st.columns([2, 3])
        thuis = heatmap_mod.privacy_settings(config)
        verwacht = (cache_verwachte_temperatuur(thuis["lat"], thuis["lon"], plandatum)
                    if thuis.get("lat") and thuis.get("lon") else None)
        temp = t1.number_input(
            "Verwachte temperatuur (°C)",
            value=float(verwacht if verwacht is not None
                        else voeding_regels.DEFAULT_TEMP_C),
            step=0.5, key=f"voeding_temp_{plandatum.isoformat()}",
        )
        t2.caption(
            f"Verwachting Open-Meteo voor {plandatum:%d-%m-%Y}: **{verwacht:.1f} °C** "
            f"— aanpasbaar." if verwacht is not None else
            "Geen verwachting beschikbaar (te ver vooruit, geen thuislocatie of "
            "geen internet) — vul de temperatuur zelf in."
        )

        st.markdown("**Beschikbare producten** — vink aan wat je bij je hebt.")
        gekozen_namen = []
        kolommen = st.columns(2)
        for i, product in enumerate([p for p in alle_producten if p.active]):
            label = (f"{product.name} — {product.carbs_g:.0f} g"
                     + (" · dual-source" if product.effective_source ==
                        voeding_producten.SOURCE_DUAL else " · single-source")
                     + (f" · {product.caffeine_mg:.0f} mg cafeïne"
                        if product.caffeine_mg else ""))
            if kolommen[i % 2].checkbox(label, key=f"voeding_p_{product.name}"):
                gekozen_namen.append(product.name)
        selectie = [p for p in alle_producten if p.name in gekozen_namen]

        with st.expander("Fijnafstelling (doel, gewicht, duur overschrijven)"):
            f1, f2 = st.columns(2)
            eigen_doel = f1.checkbox("Zelf een doel in g/uur kiezen",
                                     key="voeding_eigen_doel")
            doel_g_h = f1.number_input(
                "Koolhydraten (g/uur)", min_value=0.0, max_value=150.0,
                value=60.0, step=5.0, disabled=not eigen_doel,
                key="voeding_doel_waarde",
            )
            getrainde_darm = f1.checkbox(
                "Getrainde darm (tot 120 g/uur toestaan)",
                key="voeding_darm",
                help="Alleen aanzetten als hogere innames in training "
                     "aantoonbaar goed vielen — zie de tolerantiegeschiedenis.",
            )
            gewicht = f2.number_input(
                "Lichaamsgewicht (kg, voor het cafeïneplafond)",
                min_value=0.0, value=float(laatste_gewicht or 0.0), step=0.5,
                key="voeding_gewicht",
                help="Standaard je laatste meting op de Lichaam-tab.",
            )
            eigen_duur = f2.checkbox("Duurschatting overschrijven",
                                     key="voeding_eigen_duur")
            duur_min = f2.number_input(
                "Totale duur (minuten)", min_value=0.0, value=180.0, step=5.0,
                disabled=not eigen_duur, key="voeding_duur_waarde",
            )

        if st.button("🧮 Bereken plan", type="primary", key="voeding_bereken"):
            legs = [LegRequest(sport=s, distance_m=per_sport[s][0],
                               duration_s=per_sport[s][1])
                    for s in legs_for(sessietype)]
            with st.spinner("Duur schatten uit je eigen sessies..."):
                schatting = estimate_duration(
                    conn, trainingen, config["athlete"], legs, intensiteit, temp)
            verzoek = PlanRequest(
                session_type=sessietype, legs=legs, intensity=intensiteit,
                temp_c=temp, product_names=gekozen_namen, aid_stations=posten,
                weight_kg=gewicht or None,
                target_g_h=doel_g_h if eigen_doel else None,
                trained_gut=getrainde_darm,
                override_duration_s=duur_min * 60 if eigen_duur else None,
                planned_date=plandatum,
                name=f"{SESSION_TYPES[sessietype][0]} {plandatum:%d-%m-%Y}",
            )
            st.session_state["voeding_plan"] = build_plan(verzoek, selectie, schatting)
            st.session_state.pop("voeding_toelichting", None)

        plan = st.session_state.get("voeding_plan")
        if plan is None:
            st.info("Vul hierboven je sessie in en klik op **Bereken plan**.")
        else:
            st.divider()
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Geschatte duur", plan.duration.range_text(),
                      help="Uit je eigen sessies; zie de onderbouwing hieronder.")
            m2.metric("Koolhydraten", f"{plan.totals['carbs_g']:.0f} g",
                      f"{plan.totals['carbs_per_hour']:.0f} g/uur")
            m3.metric("Vocht", f"{plan.totals['fluid_ml']:.0f} ml",
                      f"{plan.totals['fluid_ml_per_hour']:.0f} ml/uur")
            m4.metric("Natrium", f"{plan.totals['sodium_mg']:.0f} mg",
                      f"{plan.totals['caffeine_mg']:.0f} mg cafeïne"
                      if plan.totals["caffeine_mg"] else None)

            # Het plafond hangt aan de productselectie; zonder selectie is er
            # niets om een plafond van af te leiden en zou het getal suggereren
            # dat er al iets gekozen is.
            plafond = (f" · opnameplafond van je selectie: "
                       f"**{plan.cap_g_h:.0f} g/uur**"
                       if plan.request.product_names else
                       " · nog geen producten aangevinkt")
            st.caption(
                f"Richtlijn: **{plan.target.as_text()}** · plan: "
                f"**{plan.planned_g_h:.0f} g/uur** over de eetbare tijd "
                f"({fmt_duration(plan.feedable_s)}){plafond}."
            )
            for been in plan.duration.legs:
                st.caption(f"↳ {been.label}: **{been.range_text()}** — {been.basis}")

            for melding in plan.warnings:
                (st.warning if melding.severity == SEVERITY_WARNING else st.info)(
                    melding.text)

            if plan.events:
                st.markdown("#### ⏱️ Tijdlijn")
                tijdlijn = pd.DataFrame([{
                    "Moment": e.time_label,
                    "Onderdeel": e.segment,
                    "Km": f"{e.km:.1f}" if e.km is not None else GEEN_WAARDE,
                    "Wat": f"{e.amount} {e.product}",
                    "Koolhydraten": f"{e.carbs_g:.0f} g",
                    "Totaal tot hier": f"{e.cumulative_carbs_g:.0f} g",
                    "Opmerking": e.note,
                } for e in plan.events])
                st.dataframe(tijdlijn, hide_index=True, width="stretch")
            if plan.drink.carbs_g or plan.drink.fluid_ml:
                st.caption(
                    f"Drinken loopt continu door: ~"
                    f"{plan.drink.ml_per_hour / 4:.0f} ml elke 15 minuten "
                    f"({plan.drink.ml_per_hour:.0f} ml/uur). Het lopende totaal "
                    f"hierboven telt de drank naar rato mee."
                )

            if plan.carry:
                st.markdown("#### 🎒 Meenemen")
                st.dataframe(pd.DataFrame([{
                    "Wat": c.label, "Hoeveel": c.amount, "Toelichting": c.detail,
                } for c in plan.carry]), hide_index=True, width="stretch")

            st.markdown("#### 💾 Bewaren bij een geplande sessie")
            b1, b2 = st.columns([3, 1])
            plannaam = b1.text_input("Naam", value=plan.request.name,
                                     key="voeding_plannaam")
            if b2.button("💾 Plan opslaan", key="voeding_opslaan"):
                voeding_opslag.save_plan(conn, plan, plannaam, plandatum)
                st.success(f"Plan '{plannaam}' opgeslagen — vul na afloop op de "
                           f"tab **Opgeslagen plannen** in hoe het viel.")

            st.caption(
                "De toelichting is het enige wat een taalmodel hier schrijft; "
                "alle cijfers hierboven zijn door het algoritme berekend."
            )
            if st.button("💬 Korte toelichting vragen", key="voeding_toelicht"):
                try:
                    with st.spinner("Toelichting schrijven..."):
                        st.session_state["voeding_toelichting"] = explain_plan(
                            router, plan)
                except Exception as e:
                    st.error(f"Toelichting mislukt: {e}")
            if toelichting := st.session_state.get("voeding_toelichting"):
                st.info(toelichting)

    # -------------------------------------------------------------- producten --
    with product_tab:
        st.caption(
            "Waarden per eenheid (één gel, één schepje/sachet, één portie), van "
            "de verpakking — controleer ze bij een nieuwe batch, recepturen "
            "veranderen. **Bron** bepaalt het opnameplafond: `single` = "
            "glucose/maltodextrine (~60 g/uur), `dual` = glucose + fructose "
            "(~90 g/uur), `onbekend` = meerdere suikers zonder ratio en telt "
            "conservatief als single."
        )
        bewerkt = st.data_editor(
            voeding_producten.products_dataframe(alle_producten),
            num_rows="dynamic", hide_index=True, width="stretch",
            key="voeding_producteditor",
            column_config={
                "name": st.column_config.TextColumn("Product", required=True),
                "kind": st.column_config.SelectboxColumn(
                    "Type", options=list(voeding_producten.KINDS)),
                "carbs_g": st.column_config.NumberColumn(
                    "Koolhydraten (g)", min_value=0.0, step=1.0),
                "source": st.column_config.SelectboxColumn(
                    "Bron", options=list(voeding_producten.SOURCES)),
                "ratio": st.column_config.TextColumn("Ratio / samenstelling"),
                "sodium_mg": st.column_config.NumberColumn(
                    "Natrium (mg)", min_value=0.0, step=1.0),
                "caffeine_mg": st.column_config.NumberColumn(
                    "Cafeïne (mg)", min_value=0.0, step=5.0),
                "serving_ml": st.column_config.NumberColumn("Volume (ml)"),
                "serving_g": st.column_config.NumberColumn("Portie (g)"),
                "note": st.column_config.TextColumn("Opmerking"),
                "active": st.column_config.CheckboxColumn("In voorraad"),
            },
        )
        p1, p2 = st.columns([1, 4])
        if p1.button("💾 Producten opslaan", key="voeding_prod_opslaan"):
            n = voeding_producten.save_products(
                conn, voeding_producten.products_from_dataframe(bewerkt))
            st.success(f"{n} producten opgeslagen.")
            st.rerun()
        if p2.button("↩️ Terug naar de standaardlijst", key="voeding_prod_reset"):
            voeding_producten.reset_products(conn)
            st.rerun()

    # ------------------------------------------------------ opgeslagen plannen --
    with historie_tab:
        tolerantie = voeding_opslag.tolerance_summary(conn)
        if tolerantie:
            st.markdown("#### 🧪 Wat mijn maag aankan")
            for regel in tolerantie:
                st.markdown(f"- {regel}")
            st.caption(
                "Dit zijn je eigen cijfers en die wegen zwaarder dan een "
                "algemene richtlijn — ze gaan over jouw darm."
            )
            geschiedenis = voeding_opslag.tolerance_history(conn)
            if len(geschiedenis) >= 2 and geschiedenis["g_per_uur"].notna().any():
                fig = px.scatter(
                    geschiedenis.dropna(subset=["g_per_uur"]),
                    x="datum", y="g_per_uur", color="gut",
                    labels={"datum": "", "g_per_uur": "g/uur", "gut": "Maag"},
                    # Groen/oranje/rood uit de zonereeks: goed → klachten.
                    color_discrete_map={
                        voeding_opslag.GUT_GOOD: PAL["zones"][1],
                        voeding_opslag.GUT_MILD: PAL["zones"][2],
                        voeding_opslag.GUT_BAD: PAL["zones"][4],
                    },
                )
                chart(style_fig(fig), key="voeding_tolerantie")
            st.divider()

        plannen = voeding_opslag.load_plans(conn)
        if plannen.empty:
            st.info("Nog geen plannen opgeslagen.")
        else:
            for _, rij in plannen.iterrows():
                ingevuld = pd.notna(rij["gut"])
                kop = (f"{'✅' if ingevuld else '📝'} {rij['name']}"
                       + (f" — {rij['planned_date']}" if rij["planned_date"] else ""))
                with st.expander(kop, expanded=not ingevuld):
                    st.code(rij["summary"] or "", language=None)
                    if ingevuld:
                        st.markdown(
                            f"**Achteraf:** {voeding_opslag.GUT_ICON.get(rij['gut'], '')} "
                            f"{rij['gut']} — {rij['actual_carbs_g']:.0f} g in "
                            f"{fmt_duration(rij['actual_duration_s'])}"
                            + (f" · _{rij['note']}_" if rij["note"] else "")
                        )
                    with st.form(f"voeding_fb_{rij['plan_id']}"):
                        st.markdown("**Hoe ging het?**")
                        v1, v2, v3 = st.columns(3)
                        werkelijk = v1.number_input(
                            "Werkelijk ingenomen (g koolhydraten)", min_value=0.0,
                            value=float(rij["actual_carbs_g"] or 0), step=5.0)
                        werkelijke_duur = v2.number_input(
                            "Werkelijke duur (minuten)", min_value=0.0,
                            value=float((rij["actual_duration_s"] or 0) / 60),
                            step=5.0)
                        maag = v3.selectbox(
                            "Maag", list(voeding_opslag.GUT_OPTIONS),
                            index=(list(voeding_opslag.GUT_OPTIONS).index(rij["gut"])
                                   if ingevuld else 0))
                        notitie = st.text_input("Notitie", value=rij["note"] or "")
                        opslaan, verwijderen = st.columns([1, 1])
                        if opslaan.form_submit_button("💾 Opslaan"):
                            voeding_opslag.save_feedback(
                                conn, int(rij["plan_id"]), werkelijk,
                                werkelijke_duur * 60, maag, notitie)
                            st.rerun()
                        if verwijderen.form_submit_button("🗑️ Plan verwijderen"):
                            voeding_opslag.delete_plan(conn, int(rij["plan_id"]))
                            st.rerun()


# ------------------------------------------------------------------- coach --
with tab_coach:
    st.subheader("📅 Weekschema")
    st.caption("Pas het schema aan en klik op Opslaan; het advies volgt dit schema.")
    schema = st.data_editor(
        load_schedule(MEMORY_DIR), num_rows="dynamic",
        hide_index=True, width="stretch", key="schema_editor",
    )
    if st.button("💾 Schema opslaan"):
        save_schedule(schema, MEMORY_DIR)
        st.success("Weekschema opgeslagen in memory/weekschema.md")

    st.divider()
    st.subheader("🎯 Trainingsadvies")

    advies = last_advice(MEMORY_DIR)
    if advies:
        st.markdown(advies)
    else:
        st.info("Nog geen advies gegenereerd.")

    st.caption(
        "Een nieuw advies kost een API-call naar Anthropic; het laatste advies "
        "blijft bewaard in memory/adviezen.md en wordt hierboven getoond."
    )
    if st.button("✨ Genereer nieuw advies (Anthropic API)"):
        try:
            with st.spinner("De coach kijkt naar je data..."):
                generate_advice(router, conn, MEMORY_DIR, config)
            st.rerun()
        except Exception as e:
            st.error(str(e))

    st.divider()
    st.subheader("🔍 Inzichten (trendanalyse)")

    inzichten = last_insights(MEMORY_DIR)
    if inzichten:
        st.markdown(inzichten)
    else:
        st.info("Nog geen trendanalyse uitgevoerd.")

    st.caption(
        "De cloud-coach analyseert al je data (logboek, belasting, efficiëntie, "
        "records, zwemprogressie) op langetermijnpatronen en legt de bevindingen "
        "vast in memory/inzichten.md. Die inzichten voeden ook het trainingsadvies."
    )
    if st.button("🔍 Analyseer trends (Anthropic API)"):
        try:
            with st.spinner("De coach zoekt naar patronen..."):
                generate_insights(
                    router, conn, MEMORY_DIR,
                    progress_text=progress_summary_text(conn, trainingen),
                    config=config)
            st.rerun()
        except Exception as e:
            st.error(str(e))

# -------------------------------------------------------------------- chat --
with tab_chat:
    st.subheader("💬 Vragen over je data")
    escaleer = st.toggle(
        "Vraag de cloud-coach (Anthropic API)",
        help="Uit = lokaal Ollama-model (gratis, onbeperkt). "
             "Aan = Anthropic API, voor vragen die echt redeneerwerk vragen.",
    )

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    for vraag, antwoord, bron in st.session_state.chat_history:
        st.chat_message("user").write(vraag)
        st.chat_message("assistant").write(f"{antwoord}\n\n_— {bron}_")

    if vraag := st.chat_input("Bijv.: hoeveel uur heb ik deze week getraind?"):
        st.chat_message("user").write(vraag)
        bron = "Anthropic (cloud)" if escaleer else "Ollama (lokaal)"
        try:
            with st.spinner(f"Antwoord van {bron}..."):
                antwoord = answer_question(
                    router, conn, MEMORY_DIR, vraag,
                    escalate=escaleer, config=config)
        except Exception as e:
            antwoord = f"Er ging iets mis: {e}"
        st.chat_message("assistant").write(f"{antwoord}\n\n_— {bron}_")
        st.session_state.chat_history.append((vraag, antwoord, bron))

# ----------------------------------------------------------------- heatmap --
# Voor de lol, geen trainingsanalyse: alle GPS-tracks op één donkere kaart,
# feller waar vaker gereden/gelopen. De rekenketen zit in tricoach.heatmap;
# hier staat alleen de UI en de caching eromheen.
with tab_heatmap:
    st.subheader("🗺️ Mijn heatmap")
    st.caption(
        "Alle routes die je ooit hebt gefietst, gelopen en open water gezwommen. "
        "Feller = vaker langsgekomen. De dichtheid wordt geteld over punten op "
        "**vaste afstand** (elke 10 m), niet per seconde: waar je stilstaat "
        "voor een stoplicht licht de kaart dus níet onterecht op."
    )

    # -- trackcache bijwerken: alleen nieuwe activiteiten worden geparst ------
    conn_hm = get_conn()
    try:
        heatmap_mod.prune_track_cache(conn_hm)
        nog_te_doen = len(heatmap_mod.pending_activities(conn_hm))
        if nog_te_doen:
            balk = st.progress(0.0, text="GPS uit de FIT-bestanden lezen...")

            def _voortgang(gedaan: int, totaal: int, label: str) -> None:
                balk.progress(gedaan / max(totaal, 1),
                              text=f"GPS inlezen ({gedaan}/{totaal}) — {label}")

            telling = heatmap_mod.refresh_track_cache(conn_hm, progress=_voortgang)
            balk.empty()
            st.success(
                f"{telling[heatmap_mod.STATUS_OK]} nieuwe tracks ingelezen "
                f"({fmt_aantal(telling['punten'])} punten)."
                + (f" {telling[heatmap_mod.STATUS_NO_GPS]} sessies zonder GPS "
                   "overgeslagen (banenzwemmen/indoor)."
                   if telling[heatmap_mod.STATUS_NO_GPS] else "")
                + (f" {telling[heatmap_mod.STATUS_ERROR]} onleesbaar."
                   if telling[heatmap_mod.STATUS_ERROR] else "")
            )
            cache_track_points.clear()
            cache_heatmap_cells.clear()
        hm_stats = heatmap_mod.cache_stats(conn_hm)
    finally:
        conn_hm.close()

    HM_VERSIE = (hm_stats.get(heatmap_mod.STATUS_OK, 0), hm_stats["punten"],
                 hm_stats.get("laatst"))
    alle_punten = cache_track_points(HM_VERSIE)

    if alle_punten.empty:
        st.info(
            "Nog geen GPS-tracks in de cache. Die komen uit de originele "
            "FIT-bestanden in het archief (uploads/); sessies van vóór het "
            "archief en zwembadsessies hebben geen bruikbare GPS."
        )
    else:
        # -- filters ---------------------------------------------------------
        aanwezig = [label for label in heatmap_mod.SPORT_CATEGORIES
                    if label in set(alle_punten["categorie"])]
        f1, f2, f3 = st.columns([2, 2, 1.4])
        gekozen_sporten = f1.multiselect(
            "Sport", aanwezig, default=aanwezig, key="hm_sporten",
            help="Banenzwemmen staat er niet bij: dat heeft geen GPS.")
        datum_min, datum_max = alle_punten["datum"].min(), alle_punten["datum"].max()
        periode = f2.date_input(
            "Periode", value=(datum_min, datum_max),
            min_value=datum_min, max_value=datum_max, key="hm_periode",
            help="Zo zie je hoe je actieradius zich over de maanden ontwikkelt.")
        schaal = f3.selectbox(
            "Kleurschaal", [heatmap_mod.SCALE_LOG, heatmap_mod.SCALE_PERCENTILE],
            key="hm_schaal",
            help="Logaritmisch houdt de verhoudingen herkenbaar; percentiel "
                 "licht álles wat je meer dan één keer deed sterk uit. "
                 "Lineair zou de woon-werkroute al het andere wegdrukken.")

        o1, o2, o3 = st.columns([1.4, 1.4, 1.4])
        met_transport = o1.toggle(
            "Transport meetellen", value=True, key="hm_transport",
            help="Voor een heatmap gaat het om waar je komt, niet om de "
                 "trainingsprikkel — daarom standaard aan.")
        met_verwijderd = o2.toggle(
            "Verwijderde sessies meetellen", value=False, key="hm_verwijderd",
            help="Soft-deleted sessies blijven standaard van de kaart.")
        cel_m = o3.select_slider(
            "Rasterfijnheid (m)", options=[10, 15, 20, 30, 50], value=20,
            key="hm_cel",
            help="Kleiner = fijnere lijnen maar meer gevoelig voor GPS-ruis "
                 "(dezelfde route valt dan in verschillende cellen).")

        # -- privacyzone -----------------------------------------------------
        zone_cfg = heatmap_mod.privacy_settings(config)
        voorstel = heatmap_mod.suggest_home(alle_punten)
        # De kop moet de stand tonen die nú op de kaart geldt, niet de laatst
        # opgeslagen: de toggle werkt meteen en het middelpunt valt terug op de
        # schatting zolang er niets is opgeslagen. De actuele widgetwaarden
        # staan in session_state zodra ze één keer getekend zijn; bij de eerste
        # render gelden de opgeslagen waarden.
        zone_aan_nu = st.session_state.get("hm_zone_aan", zone_cfg["enabled"])
        zone_radius_nu = st.session_state.get("hm_zone_radius",
                                             int(zone_cfg["radius_m"]))
        heeft_midden = zone_cfg["lat"] is not None or voorstel is not None
        with st.expander(
                f"🔒 Privacyzone — aan ({zone_radius_nu} m)"
                if zone_aan_nu and heeft_midden else "🔒 Privacyzone — uit",
                expanded=False):
            st.caption(
                "Je tracks beginnen en eindigen bij de voordeur. Punten binnen "
                "deze zone worden uit de heatmap weggelaten — vooral van belang "
                "als je ooit een screenshot deelt. De routes lopen dan gewoon "
                "door tot de rand van de zone. Het middelpunt wordt opgeslagen "
                f"in `{heatmap_mod.privacy_path(config).relative_to(PROJECT_ROOT)}` "
                "en niet in config.yaml: dat laatste staat in versiebeheer, en "
                "het middelpunt *is* je huisadres."
            )
            z1, z2, z3 = st.columns(3)
            zone_lat = z1.number_input(
                "Breedtegraad", value=float(zone_cfg["lat"] if zone_cfg["lat"]
                                            else (voorstel[0] if voorstel else 52.0)),
                format="%.6f", step=0.0001, key="hm_zone_lat")
            zone_lon = z2.number_input(
                "Lengtegraad", value=float(zone_cfg["lon"] if zone_cfg["lon"]
                                           else (voorstel[1] if voorstel else 5.0)),
                format="%.6f", step=0.0001, key="hm_zone_lon")
            zone_radius = z3.number_input(
                "Straal (m)", min_value=0, max_value=5000,
                value=int(zone_cfg["radius_m"]), step=50, key="hm_zone_radius")
            zone_aan = st.toggle(
                "Privacyzone toepassen", value=zone_cfg["enabled"],
                key="hm_zone_aan",
                help="Standaard aan; uitzetten mag voor persoonlijk gebruik.")
            if zone_cfg["lat"] is None and voorstel:
                st.info(
                    f"Nog geen middelpunt opgeslagen — de zone gebruikt nu de "
                    f"schatting uit de startpunten van al je tracks (mediaan): "
                    f"{voorstel[0]:.6f}, {voorstel[1]:.6f}. Sla die op (of "
                    "corrigeer hem) om hem vast te zetten."
                )
            elif voorstel:
                st.caption(
                    f"Schatting uit de startpunten van al je tracks (mediaan): "
                    f"{voorstel[0]:.6f}, {voorstel[1]:.6f}"
                )
            if st.button("Privacyzone opslaan", key="hm_zone_opslaan"):
                pad = heatmap_mod.store_privacy_settings(
                    config, zone_aan, zone_lat, zone_lon, zone_radius)
                st.toast(f"Privacyzone opgeslagen in "
                         f"{pad.relative_to(PROJECT_ROOT)}.")
                st.rerun()

        # -- kaart -----------------------------------------------------------
        start_datum, eind_datum = (periode if isinstance(periode, (tuple, list))
                                   and len(periode) == 2 else (datum_min, datum_max))
        zone = ((zone_lat, zone_lon, float(zone_radius)) if zone_aan
                else (None, None, 0.0))
        cellen = cache_heatmap_cells(
            HM_VERSIE,
            (tuple(gekozen_sporten), start_datum, eind_datum,
             met_transport, met_verwijderd),
            float(cel_m), schaal, zone)

        if cellen.empty:
            st.warning("Geen tracks binnen deze selectie.")
        else:
            view = heatmap_mod.fit_view(cellen)
            laag = pdk.Layer(
                "ScatterplotLayer",
                data=cellen[["lat", "lon", "count", "sessies", "r", "g", "b", "a"]],
                get_position="[lon, lat]",
                get_fill_color="[r, g, b, a]",
                # Radius in meters, dus de lijnen houden hun echte breedte bij
                # inzoomen; de min/max in pixels houdt ze bij ver uitzoomen
                # zichtbaar en bij ver inzoomen slank.
                get_radius=float(cel_m) * 0.75,
                radius_min_pixels=1,
                radius_max_pixels=7,
                stroked=False,
                filled=True,
                pickable=True,
            )
            st.pydeck_chart(pdk.Deck(
                layers=[laag],
                initial_view_state=pdk.ViewState(**view),
                map_provider="carto",
                map_style=pdk.map_styles.CARTO_DARK,
                tooltip={"text": "{count}× langsgekomen in {sessies} sessies"},
            ), height=640)

            # Legenda: welke kleur hoort bij hoeveel passages.
            stappen = heatmap_mod.legend_stops(cellen["count"], schaal)
            blokjes = " ".join(
                f"<span style='display:inline-block;width:2.1rem;height:0.85rem;"
                f"background:{kleur};border-radius:2px;vertical-align:middle'></span>"
                f"<span style='margin:0 0.9rem 0 0.35rem'>{aantal}×</span>"
                for aantal, kleur in stappen)
            st.markdown(
                "<div style='margin-top:0.4rem'>"
                "<span style='opacity:0.7;margin-right:0.6rem'>Aantal keer "
                f"langsgekomen:</span>{blokjes}</div>",
                unsafe_allow_html=True)

            # Buiten het startbeeld gevallen cellen: die staan er wél, maar een
            # losse rit in het buitenland mag de kaart niet naar landniveau
            # uitzoomen. Zeggen dat ze bestaan is genoeg — uitzoomen doet de rest.
            la0, la1, lo0, lo1 = heatmap_mod.view_bounds(cellen)
            buiten = int((~(cellen["lat"].between(la0, la1)
                            & cellen["lon"].between(lo0, lo1))).sum())
            if buiten:
                st.caption(
                    f"De kaart opent op het gebied waar "
                    f"{100 - 100 * buiten / len(cellen):.0f}% van je routes ligt. "
                    f"{fmt_aantal(buiten)} rastercellen liggen daarbuiten (bijv. een rit of "
                    "loopje elders) — zoom uit om ze te zien."
                )

            m1, m2, m3, m4 = st.columns(4)
            n_sessies = (heatmap_mod.filter_points(
                alle_punten, categories=tuple(gekozen_sporten),
                start=start_datum, end=eind_datum,
                include_transport=met_transport,
                include_deleted=met_verwijderd)["activity_key"].nunique())
            m1.metric("Sessies op de kaart", n_sessies)
            m2.metric("Unieke km weg/pad",
                      f"{heatmap_mod.covered_km(cellen, float(cel_m)):.0f}",
                      help="Elke bezochte rastercel staat voor ongeveer één "
                           "celbreedte route; dubbel gereden stukken tellen één "
                           "keer. Een maat voor actieradius, geen kilometerstand.")
            m3.metric("Vaakst gereden plek", f"{int(cellen['count'].max())}×")
            m4.metric("Rastercellen", fmt_aantal(len(cellen)))

            st.caption(
                "Kaartachtergrond: © [OpenStreetMap](https://www.openstreetmap.org/copyright)"
                "-contributors, © [CARTO](https://carto.com/attributions) "
                "(Dark Matter). Tracks uit je eigen FIT-archief."
            )

    with st.expander("Hoe deze kaart gemaakt wordt", expanded=False):
        st.markdown(
            f"""
- **Bron**: de originele FIT-bestanden in `uploads/`. FIT slaat posities op in
  *semicircles*; die worden omgerekend met `graden = semicircles × 180 / 2³¹`.
- **Herbemonstering op afstand** ({heatmap_mod.RESAMPLE_M:.0f} m): tussen de
  gelogde punten wordt geïnterpoleerd, zodat er elke
  {heatmap_mod.RESAMPLE_M:.0f} meter precies één punt ligt. Zonder deze stap
  telt een heatmap *seconden* in plaats van *meters* en lichten stoplichten en
  klimmetjes onterecht op. Sprongen groter dan
  {heatmap_mod.MAX_GAP_M:.0f} m (pauze, autorit) worden niet overbrugd.
- **Passages per cel**: opeenvolgende punten in dezelfde rastercel gelden als
  één passage; een latere terugkomst telt opnieuw. Zo maakt de hoek waaronder
  je een cel kruist niet uit.
- **Cache**: `track_points` + `track_extract` in dezelfde SQLite-database.
  Elk FIT-bestand wordt één keer geparst; bij het openen van deze tab worden
  alleen nieuwe activiteiten bijgewerkt.

Stand van de cache: **{hm_stats.get(heatmap_mod.STATUS_OK, 0)}** sessies met
GPS ({fmt_aantal(hm_stats['punten'])} punten),
**{hm_stats.get(heatmap_mod.STATUS_NO_GPS, 0)}** zonder GPS (banenzwemmen,
indoor), **{hm_stats.get(heatmap_mod.STATUS_NO_FILE, 0)}** zonder gearchiveerd
origineel, **{hm_stats.get(heatmap_mod.STATUS_ERROR, 0)}** onleesbaar.
"""
        )
        if st.button("Trackcache opnieuw opbouwen", key="hm_herbouw",
                     help="Wist de cache en parst alle FIT-bestanden opnieuw. "
                          "Alleen nodig na een wijziging in de extractie."):
            c = get_conn()
            try:
                c.execute("DELETE FROM track_points")
                c.execute("DELETE FROM track_extract")
                c.commit()
            finally:
                c.close()
            cache_track_points.clear()
            cache_heatmap_cells.clear()
            st.rerun()

# ----------------------------------------------------------------- logboek --
with tab_log:
    log_path = MEMORY_DIR / "trainingslog.md"
    if log_path.exists():
        st.markdown(log_path.read_text(encoding="utf-8"))
    else:
        st.info("Nog geen trainingslog — importeer eerst een training.")

# ------------------------------------------------------------- instellingen --
with tab_settings:
    if flash := st.session_state.pop("settings_flash", None):
        st.success(flash)

    st.caption(
        "Je profielwaarden hieronder dienen als context voor ál het advies en de "
        "feedback. Bij opslaan worden ze (leesbaar) gespiegeld naar "
        "memory/doelen.md met een changelog, zodat eerdere zones reproduceerbaar blijven."
    )

    st.subheader("🏁 Races & streeftijden")
    races_df = pd.DataFrame(config.get("races", []))
    for col in ["name", "date", "distances", "goal", "target_time"]:
        if col not in races_df:
            races_df[col] = ""
    for col in ["swim_m", "bike_m", "run_m"]:
        if col not in races_df:
            races_df[col] = None
    races_df["date"] = pd.to_datetime(races_df["date"])
    edited_races = st.data_editor(
        races_df[["name", "date", "swim_m", "bike_m", "run_m",
                  "distances", "goal", "target_time"]],
        column_config={
            "name": st.column_config.TextColumn("Race"),
            "date": st.column_config.DateColumn("Datum", format="DD-MM-YYYY"),
            "swim_m": st.column_config.NumberColumn(
                "Zwem (m)", min_value=0, step=100,
                help="Zwemafstand in meters; leeg = standaard 1500 m."),
            "bike_m": st.column_config.NumberColumn(
                "Fiets (m)", min_value=0, step=1000,
                help="Fietsafstand in meters; leeg = standaard 40.000 m."),
            "run_m": st.column_config.NumberColumn(
                "Loop (m)", min_value=0, step=500,
                help="Loopafstand in meters; leeg = standaard 10.000 m."),
            "distances": st.column_config.TextColumn(
                "Afstanden (tekst)", help="Vrije omschrijving, alleen voor weergave."),
            "goal": st.column_config.TextColumn("Doel"),
            "target_time": st.column_config.TextColumn("Streeftijd"),
        },
        num_rows="dynamic", hide_index=True, width="stretch", key="races_editor",
    )
    st.caption(
        "De racevoorspelling en gereedheid op de voortgang-tab rekenen met de "
        "meterafstanden van de eerste race; lege velden vallen terug op de "
        "standaardafstand (1,5 / 40 / 10 km)."
    )

    st.subheader("🗓️ Trainingsdagen & beschikbare tijd")
    c1, c2 = st.columns(2)
    new_training_days = c1.text_input(
        "Geoormerkte trainingsdagen", str(config["athlete"].get("training_days", "")),
        help="Bijv.: zwemmen ma/vr-ochtend, lange duurtraining zondag.")
    new_session_time = c2.text_input(
        "Beschikbare tijd per sessie", str(config["athlete"].get("session_time", "")),
        help="Bijv.: 30-45 min doordeweeks, 1,5-2 uur zondag.")

    st.subheader("🎯 Drempels per sport")
    st.caption(
        "Elke sport heeft een eigen drempel — ze zijn niet uitwisselbaar. "
        "Hardlopen stuurt op hartslag (%LTHR), fietsen op vermogen (%FTP) "
        "zodra dat kan, en zwemmen kent bewust geen zones. De maximale "
        "hartslag blijft een los profielveld: de %max-methode is alleen een "
        "terugval, %drempel is leidend."
    )
    for notitie in threshold_notes(config["athlete"]):
        st.caption(f"❓ **Voorlopige drempel** — {notitie}")
    new_max_hr = st.number_input(
        "Maximale hartslag", 120, 230, int(config["athlete"]["max_hr"]),
        help="Alleen terugval en referentie; de zones worden van de "
             "sport-drempels afgeleid, niet hiervan.")

    st.markdown("**🏃 Hardlopen — LTHR (drempelhartslag)**")
    huidige_run_lthr = run_lthr(config["athlete"])
    c1, c2 = st.columns([1, 2])
    new_run_lthr = c1.number_input(
        "Loop-LTHR (bpm)", 100, 220, huidige_run_lthr, key="run_lthr_input",
        help="Garmin schat deze waarde automatisch tijdens hardlopen. Wat je "
             "hier invult is de waarde die de app gebruikt — bevestig de "
             "Garmin-schatting of overschrijf hem.")
    new_run_source = c2.text_input(
        "Herkomst van deze waarde", run_lthr_source(config["athlete"]) or "",
        key="run_lthr_source_input",
        help="Bijv. 'Garmin-schatting (bevestigd)' of 'veldtest 30 min'. "
             "Puur documentatie; gaat mee in het profiel en de changelog.")
    run_preview = bounds_from_lthr(new_run_lthr, config["athlete"].get("zone_pct_lthr"))
    st.caption(
        f"Loopzones bij LTHR {new_run_lthr} (%LTHR): "
        f"zone 1 < {run_preview[0]} · zone 2 {run_preview[0]}–{run_preview[1] - 1} · "
        f"zone 3 {run_preview[1]}–{run_preview[2] - 1} · "
        f"zone 4 {run_preview[2]}–{run_preview[3] - 1} · "
        f"zone 5 ≥ {run_preview[3]}."
    )

    st.markdown("**🚴 Fietsen — FTP (primair) en fiets-LTHR (secundair)**")
    c1, c2 = st.columns(2)
    new_ftp = c1.number_input(
        "FTP in watt (0 = nog onbekend)", 0, 500, int(FTP or 0),
        help="Functional Threshold Power: het vermogen dat je ~een uur kunt "
             "volhouden. Zodra dit is ingevuld worden fietssessies op "
             "%FTP-vermogenszones (Coggan) beoordeeld in plaats van op "
             "hartslag. Bij een wijziging worden de vermogenszone-tijden van "
             "alle ritten met powerdata herrekend.")
    new_bike_lthr = c2.number_input(
        "Fiets-LTHR (bpm)", 100, 220, bike_lthr(config["athlete"]),
        help="De drempelhartslag op de fiets. De vuistregel '5–10 bpm onder "
             "de loop-LTHR' is maar een startpunt — je eigen duurdata weegt "
             "zwaarder. Definitief vaststellen doe je met een 20-minutentest: "
             "de gemiddelde hartslag over dat blok is je fiets-LTHR. Deze "
             "waarde geldt zolang er geen FTP is, én voor elke rit zonder "
             "vermogensdata.")
    bike_preview = bounds_from_lthr(new_bike_lthr, config["athlete"].get("zone_pct_lthr"))
    st.caption(
        f"Fiets-hartslagzones bij LTHR {new_bike_lthr} (%LTHR): "
        f"zone 1 < {bike_preview[0]} · zone 2 {bike_preview[0]}–{bike_preview[1] - 1} · "
        f"zone 3 {bike_preview[1]}–{bike_preview[2] - 1} · "
        f"zone 4 {bike_preview[2]}–{bike_preview[3] - 1} · "
        f"zone 5 ≥ {bike_preview[3]}."
    )
    if new_ftp:
        p_preview = [round(w) for w in power_zone_bounds(new_ftp)]
        st.success(
            f"Fietssessies met vermogen worden op de **vermogenszones** "
            f"beoordeeld (Coggan, FTP {new_ftp} W): "
            f"P1 < {p_preview[0]} · P2 {p_preview[0]}–{p_preview[1]} · "
            f"P3 {p_preview[1]}–{p_preview[2]} · P4 {p_preview[2]}–{p_preview[3]} · "
            f"P5 {p_preview[3]}–{p_preview[4]} · P6 > {p_preview[4]} W. "
            "Ritten zonder vermogen vallen terug op de fiets-hartslagzones "
            "hierboven."
        )
    else:
        st.warning(
            "🔧 **Tussenoplossing:** zonder FTP worden álle fietssessies op de "
            f"fiets-hartslagzones (LTHR {new_bike_lthr}) beoordeeld. Doe een "
            "ramptest op de Kickr — het dashboard herkent die op de "
            "🔍 Sessie-tab en stelt de FTP dan ter bevestiging voor "
            f"({RAMP_FTP_FACTOR:.0%} van je beste minuut)."
        )
    if not new_ftp:
        ftp_hint = cache_ftp_estimate(DATA_VERSIE)
        if ftp_hint:
            st.caption(
                f"💡 Schatting uit je data: **~{ftp_hint['ftp_watt']:.0f} W** "
                f"({FTP_EST_FACTOR:.0%} van je beste 20-minutenvermogen, "
                f"{ftp_hint['best20_watt']:.0f} W op {ftp_hint['datum']:%d-%m-%Y}). "
                "Een echte FTP-test (bijv. 20 min voluit op de Kickr) is "
                "nauwkeuriger; tot je een waarde invult wordt vermogen zonder "
                "zone-oordeel getoond."
            )
        else:
            st.caption(
                "Nog geen FTP en nog geen rit met een vol 20-minutenblok voor "
                "een schatting — vermogen wordt zonder zone-oordeel getoond."
            )

    st.markdown("**🏊 Zwemmen — geen drempel, geen zones**")
    st.caption(
        "Bewuste keuze: polshartslag onder water is onbetrouwbaar en de "
        "techniek is nog in opbouw. Zwemsessies krijgen nergens een "
        "zone-oordeel; er wordt op afstand, tempo per 100 m, slagritme en "
        "SWOLF gestuurd. De CSS (kritische zwemsnelheid) staat daar los naast "
        "als referentie op de 🏊 Zwemmen-tab."
    )

    st.subheader("🔄 Garmin-sync")
    garmin_cfg = config.get("garmin", {})
    st.caption(
        "De inloggegevens staan in `.env` (`GARMIN_EMAIL`, `GARMIN_PASSWORD`) "
        "— nooit in dit bestand. Na de eerste login bewaart de app tokens in "
        "`data/garmin_tokens/`; beide blijven buiten git."
    )
    g1, g2, g3 = st.columns(3)
    new_auto_sync = g1.toggle(
        "Automatisch syncen bij openen", value=bool(garmin_cfg.get("auto_sync", True)),
        help="Bij het laden van het dashboard stilletjes syncen als de "
             "vorige sync oud genoeg is. Werkt alleen met bewaarde tokens; "
             "er wordt nooit ongevraagd een login of MFA gestart.")
    new_sync_hours = g2.number_input(
        "Minimaal aantal uren tussen auto-syncs", min_value=1, max_value=48,
        value=int(garmin_cfg.get("auto_sync_hours", 6)))
    new_activities_days = g3.number_input(
        "Activiteiten: dagen terugkijken", min_value=1, max_value=90,
        value=int(garmin_cfg.get("activities_days", 14)),
        help="Hoe ver de activiteiten-sync terugkijkt. Alles wat al bekend "
             "is (ook handmatig geüpload of verwijderd) wordt overgeslagen.")
    email_aanwezig, ww_aanwezig = garmin_sync.credentials()
    status_delen = [
        ("✅" if email_aanwezig else "❌") + " GARMIN_EMAIL in .env",
        ("✅" if ww_aanwezig else "❌") + " GARMIN_PASSWORD in .env",
        ("✅ tokens bewaard" if garmin_sync.has_tokens() else "❌ nog geen tokens"),
    ]
    st.caption("Status: " + " · ".join(status_delen))
    if garmin_sync.has_tokens() and st.button(
            "🔌 Koppeling verwijderen (tokens wissen)",
            help="Wist de bewaarde login-tokens; bij de volgende sync moet "
                 "er opnieuw ingelogd worden."):
        garmin_sync.logout()
        st.toast("Garmin-tokens gewist.")
        st.rerun()

    st.divider()
    st.subheader("🤖 LLM")
    c1, c2, c3 = st.columns(3)
    new_host = c1.text_input("Ollama host", config["llm"]["ollama"]["host"])
    new_ollama_model = c2.text_input("Ollama model", config["llm"]["ollama"]["model"])
    new_anthropic_model = c3.text_input("Anthropic model", config["llm"]["anthropic"]["model"])
    st.caption("Routing: welk model doet welke taak?")
    routing = config["llm"]["routing"]
    rcols = st.columns(len(routing))
    new_routing = {}
    for col, (taak, provider) in zip(rcols, routing.items()):
        new_routing[taak] = col.selectbox(
            taak, ["ollama", "anthropic"],
            index=0 if provider == "ollama" else 1, key=f"route_{taak}",
        )

    if st.button("💾 Instellingen opslaan"):
        new_config = copy.deepcopy(config)
        new_config["races"] = [
            {
                "name": str(r["name"]).strip(),
                "date": pd.to_datetime(r["date"]).date(),
                "swim_m": int(r["swim_m"]) if pd.notna(r["swim_m"]) and r["swim_m"] else None,
                "bike_m": int(r["bike_m"]) if pd.notna(r["bike_m"]) and r["bike_m"] else None,
                "run_m": int(r["run_m"]) if pd.notna(r["run_m"]) and r["run_m"] else None,
                "distances": str(r["distances"] or ""),
                "goal": str(r["goal"] or ""),
                "target_time": str(r.get("target_time") or ""),
            }
            for _, r in edited_races.iterrows()
            if str(r["name"]).strip() and pd.notna(r["date"])
        ]
        new_config["athlete"]["max_hr"] = new_max_hr
        # Drempels per sport in één keer wegschrijven (wist meteen de oude
        # platte lthr/ftp-velden, zodat er één bron van waarheid is).
        set_thresholds(new_config["athlete"], new_run_lthr, new_bike_lthr,
                       int(new_ftp) or None,
                       run_lthr_source_value=new_run_source.strip() or None)
        new_config["athlete"]["training_days"] = new_training_days.strip()
        new_config["athlete"]["session_time"] = new_session_time.strip()
        new_config["athlete"].pop("zone_bounds", None)  # zones komen nu uit %LTHR
        new_config["llm"]["ollama"]["host"] = new_host.strip().rstrip("/")
        new_config["llm"]["ollama"]["model"] = new_ollama_model.strip()
        new_config["llm"]["anthropic"]["model"] = new_anthropic_model.strip()
        new_config["llm"]["routing"] = new_routing
        new_config["garmin"] = {
            **config.get("garmin", {}),
            "auto_sync": bool(new_auto_sync),
            "auto_sync_hours": int(new_sync_hours),
            "activities_days": int(new_activities_days),
        }
        save_config(new_config)

        # Profielwaarden leesbaar spiegelen naar doelen.md met changelog.
        wijzigingen = profile_mod.update_doelen(
            MEMORY_DIR, config, new_config, note="instellingen-tab")

        meldingen = []
        # Eén herberekening dekt beide hartslagdrempels: recompute_zones pakt
        # per sessie de grenzen van háár eigen sport, dus na een wijziging
        # staat de hele historie weer op één definitie.
        hr_gewijzigd = []
        for naam, oud, nieuw, soort in (
                ("loop-LTHR", run_lthr(config["athlete"]), new_run_lthr, LTHR_RUN),
                ("fiets-LTHR", bike_lthr(config["athlete"]), new_bike_lthr, LTHR_BIKE)):
            if oud != nieuw:
                hr_gewijzigd.append((naam, oud, nieuw))
                lthr_append(MEMORY_DIR, nieuw,
                            f"Aangepast via instellingen-tab (was {oud})",
                            kind=soort)
        if hr_gewijzigd:
            n = recompute_zones(conn, new_config["athlete"])
            omschrijving = " en ".join(
                f"{naam} {oud} → {nieuw}" for naam, oud, nieuw in hr_gewijzigd)
            meldingen.append(
                f"{omschrijving} vastgelegd in de drempelgeschiedenis; "
                f"zonetijden van {n} trainingen herrekend"
            )
        # FTP gezet, gewijzigd of gewist: de vermogenszone-tijden van alle
        # ritten met powerdata herrekenen (wissen = netjes terugvallen op de
        # fiets-hartslagzones); de changelog in doelen.md legt de wijziging vast.
        nieuwe_ftp = int(new_ftp) or None
        if nieuwe_ftp != (int(FTP) if FTP else None):
            n_pw = recompute_power_zones(conn, nieuwe_ftp)
            if nieuwe_ftp:
                lthr_append(MEMORY_DIR, nieuwe_ftp,
                            "Ingevuld via instellingen-tab", kind=LTHR_BIKE_FTP)
                log_ftp_determination(
                    MEMORY_DIR, float(nieuwe_ftp),
                    "handmatig ingevuld op de instellingenpagina",
                    note="Niet uit een herkende test afgeleid.")
            meldingen.append(
                f"FTP {'gewist' if not nieuwe_ftp else f'op {nieuwe_ftp} W gezet'}; "
                f"vermogenszones van {n_pw} ritten herrekend"
            )
        if meldingen:
            st.session_state["settings_flash"] = "Opgeslagen. " + \
                "; ".join(m[0].upper() + m[1:] for m in meldingen) + "."
        else:
            aantal = len(wijzigingen)
            st.session_state["settings_flash"] = (
                f"Opgeslagen in config.yaml en doelen.md"
                + (f" ({aantal} profielwijziging{'en' if aantal != 1 else ''} gelogd)." if aantal else ".")
            )
        st.rerun()

    st.divider()
    st.subheader("💰 LLM-verbruik")
    usage = usage_summary(MEMORY_DIR)
    if usage.empty:
        st.info("Nog geen LLM-aanroepen gelogd.")
    else:
        a_cfg = config["llm"]["anthropic"]
        prijs_in = a_cfg.get("cost_per_mtok_input_usd", 3.0)
        prijs_uit = a_cfg.get("cost_per_mtok_output_usd", 15.0)
        model_prices = a_cfg.get("model_prices", {})

        def kosten(row):
            """Kosten per regel; prijs per model (Haiku ≠ Sonnet), 0 voor Ollama."""
            if row["provider"] != "anthropic":
                return 0.0
            p = model_prices.get(row["model"], {})
            in_mtok = p.get("input", prijs_in)
            uit_mtok = p.get("output", prijs_uit)
            return row["prompt_tokens"] / 1e6 * in_mtok + row["antwoord_tokens"] / 1e6 * uit_mtok

        # Per model groeperen, want één taak (anthropic) kan een eigen model hebben.
        overzicht = usage.groupby(["provider", "model", "taak"], as_index=False).agg(
            aanroepen=("taak", "size"),
            prompt_tokens=("prompt_tokens", "sum"),
            antwoord_tokens=("completion_tokens", "sum"),
        )
        overzicht["kosten_usd"] = overzicht.apply(kosten, axis=1)

        totaal = overzicht["kosten_usd"].sum()
        c1, c2, c3 = st.columns(3)
        c1.metric("Totaal aanroepen", int(overzicht["aanroepen"].sum()))
        c2.metric("Waarvan Anthropic",
                  int(overzicht.loc[overzicht["provider"] == "anthropic", "aanroepen"].sum()))
        c3.metric("Kosten Anthropic", f"$ {totaal:.2f}")

        st.dataframe(
            overzicht,
            column_config={
                "provider": "Provider", "model": "Model", "taak": "Taak",
                "aanroepen": "Aanroepen",
                "prompt_tokens": st.column_config.NumberColumn("Prompt-tokens"),
                "antwoord_tokens": st.column_config.NumberColumn("Antwoord-tokens"),
                "kosten_usd": st.column_config.NumberColumn("Kosten", format="$ %.4f"),
            },
            hide_index=True, width="stretch",
        )
        st.caption(
            "Ollama is lokaal en gratis. Anthropic-kosten worden per model berekend "
            "(prijzen per miljoen tokens onder `anthropic.model_prices` in config.yaml; "
            "onbekende modellen vallen terug op de standaardprijs). Bron: memory/llm_log.md."
        )

    st.divider()
    st.subheader("🧹 Memory-review — doelen.md")
    st.caption(
        "De volledige inhoud van memory/doelen.md gaat mee in élke feedback- en "
        "adviesprompt. Loop deze tabel periodiek (bijv. maandelijks) na en werk "
        "verouderde regels bij in het bestand zelf; wijzigingen worden "
        "automatisch gedetecteerd en in de changelog van doelen.md gelogd. "
        f"Regels ouder dan {MAX_LEEFTIJD_WEKEN} weken zijn gemarkeerd."
    )
    review = review_dataframe(MEMORY_DIR)
    if review.empty:
        st.info("Geen doelen.md gevonden.")
    else:
        oud = review["Weken oud"] >= MAX_LEEFTIJD_WEKEN
        if oud.any():
            st.warning(
                f"{int(oud.sum())} regel(s) zijn {MAX_LEEFTIJD_WEKEN} weken of "
                "ouder — controleer of ze nog kloppen."
            )
        styler = review.style.apply(
            lambda _kol: ["background-color: rgba(255, 165, 0, 0.15)" if v else ""
                          for v in oud], subset=["Regel"])
        st.dataframe(
            styler,
            column_config={
                "Regel": st.column_config.TextColumn("Regel", width="large"),
                "Laatst gewijzigd": st.column_config.DateColumn(
                    "Laatst gewijzigd", format="DD-MM-YYYY",
                    help="Datum waarop deze regel voor het laatst is gewijzigd "
                         "(of waarop de wijziging voor het eerst is gezien)."),
                "Weken oud": st.column_config.NumberColumn("Weken oud"),
            },
            hide_index=True, width="stretch", height=420,
        )

    st.divider()
    st.subheader("📦 Origineel-archief")
    # Stand van het archief: hoeveel sessies hebben een bewaard origineel?
    # De inhaalslag draait ook automatisch bij elke serverstart; de knop is
    # er zodat je na het klaarzetten van oude zips niet hoeft te herstarten.
    met_origineel = conn.execute(
        "SELECT COUNT(*) FROM activities "
        "WHERE deleted_at IS NULL AND archived_path IS NOT NULL").fetchone()[0]
    zonder_origineel = conn.execute(
        "SELECT COUNT(*) FROM activities "
        "WHERE deleted_at IS NULL AND archived_path IS NULL").fetchone()[0]
    a1, a2 = st.columns(2)
    a1.metric("Met bewaard origineel", met_origineel,
              help=f"Het originele FIT-bestand staat in {UPLOADS_DIR.name}/ "
                   "en is herbruikbaar voor verificatie en terugrol.")
    a2.metric("Zonder origineel", zonder_origineel,
              help="Geïmporteerd vóór het archief bestond. Zet de oude "
                   "exportzips in garmin_import/ en klik op zoeken, dan "
                   "worden ze alsnog gearchiveerd.")
    import_map = resolve_path(config, "import_dir")
    col_zoek, col_verif = st.columns(2)
    with col_zoek:
        if st.button(f"🔎 Zoek originelen in {import_map.name}/",
                     width="stretch",
                     help="Doorzoekt de map op zips en losse FIT-bestanden, "
                          "archiveert alles wat bij een bekende sessie hoort "
                          "en haalt de 'origineel ontbreekt'-markering weg. "
                          "Byte-identieke bestanden worden nooit dubbel "
                          "opgeslagen; draaien kan dus altijd."):
            with st.spinner("Zoeken en archiveren..."):
                n = migrate_originals(conn, UPLOADS_DIR, [import_map])
            if n:
                st.toast(f"{n} origineel(en) gearchiveerd in {UPLOADS_DIR.name}/.")
            elif not import_map.exists():
                st.toast(f"Map {import_map.name}/ bestaat (nog) niet — maak "
                         "hem aan en zet de zips erin.", icon="ℹ️")
            else:
                st.toast("Geen (nieuwe) originelen gevonden die bij een "
                         "bekende sessie horen.", icon="ℹ️")
            st.rerun()
    with col_verif:
        if st.button("✅ Verificatierun archief ↔ database", width="stretch",
                     help="Parset elk bewaard origineel opnieuw en vergelijkt "
                          "sleutel, sport, duur, afstand en hartslag met de "
                          "database. Sessies zonder origineel worden "
                          "overgeslagen."):
            with st.spinner("Originelen opnieuw parsen..."):
                st.session_state["verificatie_uitkomst"] = verify_originals(conn)
    uitkomst = st.session_state.get("verificatie_uitkomst")
    if uitkomst is not None:
        telling = uitkomst["status"].value_counts().to_dict()
        st.write("Uitkomst: " + " · ".join(
            f"**{v}× {k}**" for k, v in telling.items()))
        problemen = uitkomst[uitkomst["status"].isin(["afwijking", "bestand_weg"])]
        if problemen.empty:
            st.caption("Geen afwijkingen: elk bewaard origineel levert bij "
                       "herparse dezelfde kernwaarden op als de database.")
        else:
            st.dataframe(problemen, hide_index=True, width="stretch")

    st.divider()
    # Onopvallend beheer van soft-verwijderde sessies: tonen, herstellen of
    # definitief wissen. Verwijderen zelf gebeurt op de 🔍 Sessie-tab.
    verwijderd = load_deleted_activities(conn)
    with st.expander(f"🗑️ Verwijderde sessies ({len(verwijderd)})"):
        if verwijderd.empty:
            st.caption(
                "Geen verwijderde sessies. Een sessie verwijderen kan op de "
                "🔍 Sessie-tab; hij komt dan hier terecht en kan worden "
                "hersteld of definitief gewist."
            )
        else:
            toon = verwijderd.copy()
            toon["start_time"] = toon["start_time"].dt.tz_convert(TZ)
            toon["Sport"] = toon["sport"].map(sport_label)
            toon["Afstand"] = toon["distance_m"] / 1000
            st.dataframe(
                toon[["start_time", "Sport", "Afstand", "avg_hr", "deleted_at"]],
                column_config={
                    "start_time": st.column_config.DatetimeColumn(
                        "Datum", format="DD-MM-YYYY HH:mm"),
                    "Afstand": st.column_config.NumberColumn("Afstand", format="%.2f km"),
                    "avg_hr": st.column_config.NumberColumn("Gem. HR"),
                    "deleted_at": st.column_config.DatetimeColumn(
                        "Verwijderd op", format="DD-MM-YYYY HH:mm"),
                },
                hide_index=True, width="stretch",
            )
            del_labels = {
                r["activity_key"]: (
                    f"{r['start_time'].tz_convert(TZ):%d-%m-%Y %H:%M} · "
                    f"{sport_label(r['sport'])}"
                    + (f" · {r['distance_m'] / 1000:.1f} km"
                       if pd.notna(r["distance_m"]) else "")
                )
                for _, r in verwijderd.iterrows()
            }
            herstel_keuze = st.selectbox(
                "Sessie", verwijderd["activity_key"].tolist(),
                format_func=del_labels.get, key="herstel_keuze",
            )
            col_herstel, col_wis = st.columns(2)
            with col_herstel:
                if st.button("↩️ Herstellen", width="stretch"):
                    restore_session(conn, MEMORY_DIR, herstel_keuze)
                    st.toast(f"Sessie {del_labels[herstel_keuze]} hersteld.")
                    st.rerun()
            with col_wis:
                wis_zeker = st.checkbox(
                    "Definitief wissen kan niet ongedaan worden gemaakt — "
                    "ik weet het zeker", key="wis_zeker",
                )
                if st.button("❌ Definitief wissen", disabled=not wis_zeker,
                             width="stretch"):
                    purge_session(conn, MEMORY_DIR, herstel_keuze)
                    st.toast(f"Sessie {del_labels[herstel_keuze]} definitief gewist.")
                    st.rerun()

conn.close()
