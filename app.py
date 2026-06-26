"""Triatlon Training Dashboard — Streamlit-app.

Starten met:  streamlit run app.py

De app leest de geïmporteerde sessies uit SQLite (data/training.db) en de
memory-bestanden uit memory/. Uploads gaan via de zijbalk en doorlopen
dezelfde import-pipeline als de commandline (tricoach.importer).
"""

import copy
from datetime import date, timedelta

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import dotenv

from tricoach import body
from tricoach import profile as profile_mod
from tricoach.advice import generate_advice, generate_insights, last_advice, last_insights
from tricoach.analysis import (
    aerobic_efficiency_trend,
    pace_at_hr,
    swim_per_session,
    weekly_volume,
    weekly_zone_time,
)
from tricoach.chat import answer_question
from tricoach.config import load_config, resolve_path
from tricoach.feedback import generate_feedback
from tricoach.formatting import (
    GEEN_WAARDE,
    fmt_duration,
    sessie_tempo,
    sport_label,
)
from tricoach.progress import (
    acwr_status,
    decoupling,
    efficiency_factor,
    load_curves,
    personal_records,
    progress_summary_text,
    race_prediction,
    readiness,
    swim_progression,
)
from tricoach.importer import import_zip
from tricoach.llm import LLMRouter
from tricoach.llm.log import usage_summary
from tricoach.llm.observations import session_observation
from tricoach.lthr import append_entry as lthr_append, load_history as lthr_history
from tricoach.schedule import add_note_row, load_schedule, save_schedule
from tricoach.settings import save_config
from tricoach.storage import (
    connect,
    load_activities,
    recompute_zones,
    swim_active_seconds,
)
from tricoach.weather import wind_for_activity
from tricoach.zones import bounds_from_lthr, zone_bounds

dotenv.load_dotenv()

st.set_page_config(page_title="Triatlon Coach", page_icon="🏊", layout="wide")

config = load_config()
BOUNDS = zone_bounds(config["athlete"])
Z2 = (BOUNDS[0], BOUNDS[1] - 1)  # Z2-bereik, bijv. 137-151
MEMORY_DIR = resolve_path(config, "memory_dir")
TZ = "Europe/Amsterdam"  # Garmin slaat tijden op in UTC; tonen in lokale tijd

# Vaste kleuren zodat sporten en zones in elke grafiek hetzelfde ogen.
SPORT_COLORS = {"Hardlopen": "#e45756", "Fietsen": "#4c78a8", "Zwemmen": "#72b7b2"}
ZONE_LABELS = {
    "Z1": f"Zone 1 (< {BOUNDS[0]})",
    "Z2": f"Zone 2 ({BOUNDS[0]}–{BOUNDS[1]})",
    "Z3": f"Zone 3 ({BOUNDS[1]}–{BOUNDS[2]})",
    "Z4": f"Zone 4 ({BOUNDS[2]}–{BOUNDS[3]})",
    "Z5": f"Zone 5 (> {BOUNDS[3]})",
}
ZONE_COLORS = {
    ZONE_LABELS["Z1"]: "#b8d4e8", ZONE_LABELS["Z2"]: "#54a24b",
    ZONE_LABELS["Z3"]: "#eeca3b", ZONE_LABELS["Z4"]: "#f58518",
    ZONE_LABELS["Z5"]: "#e45756",
}
STROKE_NL = {
    "breaststroke": "Schoolslag", "freestyle": "Borstcrawl",
    "backstroke": "Rugslag", "butterfly": "Vlinderslag",
    "mixed": "Gemengd", "drill": "Oefening", "im": "Wisselslag",
}


def style_fig(fig, show_legend: bool = True):
    """Huisstijl voor alle grafieken: legenda horizontaal ónder de grafiek."""
    fig.update_layout(
        legend=dict(orientation="h", yanchor="top", y=-0.22, x=0, title=None),
        showlegend=show_legend,
        margin=dict(t=30, b=10, l=10, r=10),
    )
    return fig


def date_xaxis(fig, dates):
    """Nette datum-as: alleen dd-mm-jjjj, geen uur-ticks binnen de dag.

    Bij weinig metingen zetten we de ticks precies op de meetdagen; bij veel
    metingen laten we Plotly zelf nette dag/week/maand-labels kiezen.
    """
    uniek = pd.to_datetime(pd.Series(list(dates))).dt.normalize().drop_duplicates()
    fig.update_xaxes(tickformat="%d-%m-%Y")
    if 1 <= len(uniek) <= 15:
        fig.update_xaxes(tickmode="array", tickvals=sorted(uniek))
    return fig


def pace_as_time(seconds: pd.Series) -> pd.Series:
    """Tempo in seconden -> datetime, zodat de as nette M:SS-labels krijgt."""
    return pd.to_datetime(seconds, unit="s")


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


def veilig_cel(func, *args, fallback: str = GEEN_WAARDE) -> str:
    """Voer een cel-formatter uit; bij een fout een streepje i.p.v. een crash.

    Algemeen vangnet voor de 'Recente sessies'-tabel: een probleem op één
    sessie (een ontbrekend of NaN-veld) levert hooguit een placeholder in die
    cel op, nooit een ValueError die de hele pagina onderuit haalt.
    """
    try:
        return func(*args)
    except Exception:
        return fallback


def get_conn():
    """Open een verse databaseverbinding (goedkoop; vermijdt thread-gedoe)."""
    return connect(resolve_path(config, "database"))


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
                )
                for r in results:
                    icon = "✅" if r.status == "nieuw" else "↩️"
                    st.write(f"{icon} {r.activity.start_time:%d-%m %H:%M} "
                             f"{sport_label(r.activity.sport)} — {r.status}")
                    if r.status == "nieuw" and r.wind is not None:
                        st.caption(f"🌬️ Wind: {r.wind.as_text()}")
                    # Alleen nieuwe sessies krijgen coaching-feedback (Haiku);
                    # duplicaten niet, dat zou onnodig een API-call kosten.
                    if r.status == "nieuw":
                        try:
                            fb = generate_feedback(
                                router_upload, conn, MEMORY_DIR, config,
                                r.activity, r.tiz, r.observation,
                                user_note=r.user_note, wind=r.wind,
                            )
                            verse_feedback.append(fb)
                        except Exception as e:
                            st.warning(f"Feedback overgeslagen: {e}")
        conn.close()
        if verse_feedback:
            # Bovenaan de hoofdpagina tonen (buiten de zijbalk); blijft staan tot
            # de volgende upload of tot 'sluiten'.
            st.session_state["upload_feedback"] = verse_feedback

    st.caption("Weather data by [Open-Meteo.com](https://open-meteo.com) (CC BY 4.0)")

# ------------------------------------------------------------------- data --
conn = get_conn()
acts = load_activities(conn)

if acts.empty:
    st.info("Nog geen trainingen geïmporteerd. Upload een zip via de zijbalk.")
    st.stop()

acts["start_time"] = acts["start_time"].dt.tz_convert(TZ)
acts["Sport"] = acts["sport"].map(sport_label)

router = LLMRouter(config, MEMORY_DIR)


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


render_upload_feedback()

(tab_overzicht, tab_trends, tab_voortgang, tab_lopen, tab_fietsen, tab_zwemmen,
 tab_lichaam, tab_coach, tab_chat, tab_log, tab_settings) = st.tabs(
    ["📋 Overzicht", "📈 Trends", "🚀 Voortgang", "🏃 Lopen", "🚴 Fietsen", "🏊 Zwemmen",
     "🧍 Lichaam", "🧠 Coach", "💬 Chat", "📖 Logboek", "⚙️ Instellingen"]
)

# --------------------------------------------------------------- overzicht --
with tab_overzicht:
    week_ago = pd.Timestamp.now(tz=TZ) - pd.Timedelta(days=7)
    recent = acts[acts["start_time"] >= week_ago]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Sessies (7 dagen)", len(recent))
    c2.metric("Trainingsuren (7 dagen)", f"{recent['duration_s'].sum() / 3600:.1f}")
    z2_share = recent["z2_s"].sum() / max(recent["duration_s"].sum(), 1) * 100
    c3.metric("Aandeel zone 2 (7 dagen)", f"{z2_share:.0f}%",
              help="Doel: dit omhoog krijgen — je traint structureel te hard.")
    hard = (recent["z4_s"].sum() + recent["z5_s"].sum()) / max(recent["duration_s"].sum(), 1) * 100
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
        fig.update_traces(hovertemplate="%{x} · %{fullData.name}: %{y:.1f} uur<extra></extra>")
        st.plotly_chart(style_fig(fig), width="stretch")

    with col_rechts:
        st.subheader("Tijd in hartslagzones per week")
        tz_df = weekly_zone_time(acts)
        tz_df["zone"] = tz_df["zone"].map(ZONE_LABELS)
        fig = px.bar(
            tz_df, x="week", y="minuten", color="zone",
            color_discrete_map=ZONE_COLORS,
            category_orders={
                "week": sorted(tz_df["week"].unique()),
                "zone": list(ZONE_COLORS),
            },
            labels={"week": "Week", "minuten": "Minuten", "zone": "Zone"},
        )
        fig.update_traces(hovertemplate="%{x} · %{fullData.name}: %{y:.0f} min<extra></extra>")
        st.plotly_chart(style_fig(fig), width="stretch")

    st.subheader("Mijn hartslagzones & LTHR-ontwikkeling")
    st.caption(
        f"De zones worden afgeleid van je drempelhartslag (LTHR, nu **{config['athlete']['lthr']}**). "
        "Wordt je LTHR hoger, dan schuiven alle zones mee — pas hem aan op de instellingen-tab. "
        "Zone 2 is het rustige duurtempo waar je veel wilt zitten; zone 3 de 'grijze zone'; "
        "zone 4/5 zwaar tot maximaal."
    )
    hist = lthr_history(MEMORY_DIR, config["athlete"]["lthr"])
    dates = [pd.Timestamp(d) for d in hist["datum"]]
    end = pd.Timestamp(date.today())
    if end <= dates[-1]:
        end = dates[-1] + pd.Timedelta(days=30)
    dates.append(end)
    lthrs = list(hist["lthr"]) + [int(hist["lthr"].iloc[-1])]
    pcts = config["athlete"].get("zone_pct_lthr")
    per_date = [bounds_from_lthr(l, pcts) for l in lthrs]

    fig = go.Figure()
    floor = min(b[0] for b in per_date) - 25
    fig.add_trace(go.Scatter(  # onzichtbare onderkant van de zone-1-band
        x=dates, y=[floor] * len(dates), line=dict(width=0),
        line_shape="hv", hoverinfo="skip", showlegend=False,
    ))
    tops = [
        [b[0] for b in per_date], [b[1] for b in per_date],
        [b[2] for b in per_date], [b[3] for b in per_date],
        [config["athlete"]["max_hr"]] * len(dates),
    ]
    for (label, kleur), top in zip(ZONE_COLORS.items(), tops):
        fig.add_trace(go.Scatter(
            x=dates, y=top, name=label, fill="tonexty",
            line=dict(width=0), fillcolor=kleur, line_shape="hv",
            hovertemplate=f"{label}: tot %{{y}} bpm<extra></extra>",
        ))
    fig.add_trace(go.Scatter(
        x=dates, y=lthrs, name="LTHR", line=dict(color="white", dash="dash", width=2),
        line_shape="hv", hovertemplate="LTHR: %{y} bpm<extra></extra>",
    ))
    fig.update_layout(yaxis_title="Hartslag (bpm)", xaxis_title="Datum")
    st.plotly_chart(style_fig(fig), width="stretch")

    st.subheader("Recente sessies")
    trend = aerobic_efficiency_trend(acts)
    tabel = acts.head(15).copy()
    tabel["Duur"] = pd.to_datetime(tabel["duration_s"], unit="s").dt.time
    tabel["Afstand"] = tabel["distance_m"] / 1000
    # Tempo/snelheid uit afstand en duur (altijd aanwezig), niet uit het soms
    # ontbrekende avg_speed_ms. Voor zwemmen telt de zuivere zwemtijd (som van
    # de actieve banen) als noemer; rust aan de kant valt zo weg. Elke cel is
    # afgeschermd zodat één rij met een gat in de data de tabel niet laat crashen.
    zwem_actief = swim_active_seconds(conn)
    tabel["Tempo / snelheid"] = tabel.apply(
        lambda r: veilig_cel(sessie_tempo, r["sport"], r["distance_m"],
                             r["duration_s"], zwem_actief.get(r["activity_key"])),
        axis=1)
    tabel["% Z2"] = tabel["pct_in_zone2"].map(
        lambda v: GEEN_WAARDE if pd.isna(v) else f"{v:.0f}%")
    tabel["Zones"] = tabel.apply(
        lambda r: veilig_cel(zone_bar, r["z1_s"], r["z2_s"], r["z3_s"], r["z4_s"], r["z5_s"]),
        axis=1)
    tabel["Trend"] = tabel["activity_key"].map(
        lambda k: veilig_cel(trend_cell, trend.get(k)))

    vis = tabel[["start_time", "Sport", "Duur", "Afstand", "avg_hr",
                 "Tempo / snelheid", "% Z2", "Zones", "Trend"]]
    # Kleuraccent voor de %-Z2-cel, per rij vooraf bepaald (index-gekoppeld).
    css = {idx: z2_kleur(r["pct_in_zone2"], r["z1_s"], r["z2_s"],
                         r["z3_s"], r["z4_s"], r["z5_s"])
           for idx, r in tabel.iterrows()}
    styler = vis.style.apply(
        lambda col: [css[i] for i in col.index], subset=["% Z2"])
    st.dataframe(
        styler,
        column_config={
            "start_time": st.column_config.DatetimeColumn("Datum", format="DD-MM-YYYY HH:mm"),
            "Duur": st.column_config.TimeColumn("Duur", format="H:mm:ss"),
            "Afstand": st.column_config.NumberColumn("Afstand", format="%.2f km"),
            "avg_hr": st.column_config.NumberColumn("Gem. HR"),
            "% Z2": st.column_config.TextColumn(
                "% Zone 2",
                help=f"Aandeel van de gemeten hartslagtijd in zone 2 ({Z2[0]}–{Z2[1]}). "
                     "Groen = veel rustige tijd; oranje = juist veel in Z3+."),
            "Zones": st.column_config.TextColumn(
                "Zoneverdeling",
                help="Tijd per zone in 10 blokjes: 🟦 Z1 · 🟩 Z2 · 🟨 Z3 · 🟧 Z4 · 🟥 Z5."),
            "Trend": st.column_config.TextColumn(
                "Aerobe trend",
                help="Snelheid per hartslag t.o.v. de vorige vergelijkbare sessie "
                     "(zelfde sport én intensiteit). ▲ efficiënter · ▼ minder · "
                     "▬ gelijk (±2%). ⚠ = vergeleken met de dichtstbijzijnde i.p.v. een "
                     "gelijke-intensiteit sessie. Zwemmen krijgt geen pijl."),
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

# ------------------------------------------------------------------ trends --
with tab_trends:
    st.subheader(f"Tempo bij gelijke hartslag (zone 2: {Z2[0]}–{Z2[1]})")
    st.caption(
        "De belangrijkste grafiek: gemiddeld tempo van alle meetpunten binnen zone 2, "
        "per sessie. Sneller worden bij dezelfde hartslag = grotere aerobe basis. "
        "Sessies met minder dan 5 minuten in zone 2 worden weggelaten."
    )

    col_run, col_bike = st.columns(2)
    with col_run:
        run_trend = pace_at_hr(conn, acts, "running", Z2)
        n_runs = (acts["sport"] == "running").sum()
        if run_trend.empty:
            st.info("Nog geen loopsessies met ≥5 min in zone 2 — dat zegt op zich al iets 😉")
        else:
            run_trend["tempo"] = pace_as_time(1000 / run_trend["speed_ms"])
            fig = px.line(
                run_trend, x="start_time", y="tempo", markers=True,
                labels={"start_time": "Datum", "tempo": "Tempo (min/km)"},
            )
            fig.update_yaxes(tickformat="%M:%S", autorange="reversed")
            fig.update_traces(
                marker=dict(size=11),
                hovertemplate="%{x|%d-%m-%Y} · %{y|%M:%S} min/km<extra></extra>",
            )
            fig.update_layout(title="Hardlopen — tempo in zone 2 (sneller = hoger)")
            if len(run_trend) == 1:  # bij één punt zoomt plotly extreem in
                t = pd.Timestamp(run_trend["start_time"].iloc[0])
                y = run_trend["tempo"].iloc[0]
                fig.update_xaxes(range=[t - pd.Timedelta(days=4), t + pd.Timedelta(days=4)],
                                 tickformat="%d-%m-%Y")
                fig.update_yaxes(range=[y + pd.Timedelta(seconds=30),
                                        y - pd.Timedelta(seconds=30)])
            st.plotly_chart(style_fig(fig, show_legend=False), width="stretch")
        if 0 < len(run_trend) < n_runs:
            st.caption(
                f"{n_runs - len(run_trend)} van je {n_runs} loopsessies is weggelaten: "
                "minder dan 5 minuten in zone 2 (die sessie was vrijwel volledig Z3+)."
            )

    with col_bike:
        bike_trend = pace_at_hr(conn, acts, "cycling", Z2)
        n_rides = (acts["sport"] == "cycling").sum()
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
            fig.update_layout(title="Fietsen — snelheid in zone 2")
            if len(bike_trend) == 1:
                t = pd.Timestamp(bike_trend["start_time"].iloc[0])
                y = bike_trend["snelheid_kmh"].iloc[0]
                fig.update_xaxes(range=[t - pd.Timedelta(days=4), t + pd.Timedelta(days=4)],
                                 tickformat="%d-%m-%Y")
                fig.update_yaxes(range=[y - 3, y + 3])
            st.plotly_chart(style_fig(fig, show_legend=False), width="stretch")
        if 0 < len(bike_trend) < n_rides:
            st.caption(
                f"{n_rides - len(bike_trend)} van je {n_rides} fietssessies is weggelaten: "
                "minder dan 5 minuten in zone 2."
            )

    st.subheader("Snelheid/tempo tegenover hartslag — per sport")
    st.caption(
        "Per sport een eigen grafiek met de juiste eenheid en schaal: zwemmen, "
        "lopen en fietsen liggen te ver uiteen voor één gedeelde as. Elke stip is "
        "één sessie; grotere stippen duurden langer."
    )
    sc = acts.dropna(subset=["avg_speed_ms", "avg_hr"]).copy()
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
                deel["y"] = pace_as_time(afstand / deel["avg_speed_ms"])
                y_label = "Tempo (min/km)" if sport_key == "running" else "Tempo (min/100m)"
            else:
                deel["y"] = deel["avg_speed_ms"] * 3.6
                y_label = "Snelheid (km/h)"
            fig = px.scatter(
                deel, x="avg_hr", y="y", size="duration_s", custom_data=["Datum"],
                color_discrete_sequence=[SPORT_COLORS[titel]],
                labels={"avg_hr": "Gem. hartslag", "y": y_label},
            )
            fig.update_traces(hovertemplate=hover)
            if soort == "tempo":
                fig.update_yaxes(tickformat="%M:%S", autorange="reversed")
            fig.update_layout(title=titel)
            st.plotly_chart(style_fig(fig, show_legend=False), width="stretch")

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

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=curves["datum"], y=curves["trimp"], name="Dagbelasting (TRIMP)",
        marker_color="rgba(120,120,140,0.45)",
        hovertemplate="%{x|%d-%m-%Y}: %{y:.0f} TRIMP<extra></extra>"))
    fig.add_trace(go.Scatter(
        x=curves["datum"], y=curves["ctl"], name="Fitheid (CTL)",
        line=dict(color="#54a24b", width=3),
        hovertemplate="%{x|%d-%m-%Y}: fitheid %{y:.1f}<extra></extra>"))
    fig.add_trace(go.Scatter(
        x=curves["datum"], y=curves["atl"], name="Vermoeidheid (ATL)",
        line=dict(color="#f58518", width=2, dash="dot"),
        hovertemplate="%{x|%d-%m-%Y}: vermoeidheid %{y:.1f}<extra></extra>"))
    fig.update_layout(yaxis_title="Belasting", xaxis_title="Datum")
    st.plotly_chart(style_fig(fig), width="stretch")
    st.caption(
        "Elke sessie krijgt een TRIMP-score uit de tijd per hartslagzone "
        "(zone 1 telt 1×, zone 5 telt 5× per minuut). De fitheidslijn moet "
        "richting september gestaag stijgen, met de vermoeidheidslijn er niet "
        "te ver bovenuit. De eerste weken is dit beeld nog onbetrouwbaar — "
        "de lijnen moeten 'inlopen'."
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
    ef = efficiency_factor(acts)
    c1, c2 = st.columns(2)
    for col, sport_key, titel in ((c1, "running", "Hardlopen"), (c2, "cycling", "Fietsen")):
        with col:
            ef_sport = ef[ef["sport"] == sport_key]
            if ef_sport.empty:
                st.info(f"Nog geen {titel.lower()}-sessies.")
                continue
            fig = px.line(
                ef_sport, x="start_time", y="ef", markers=True,
                labels={"start_time": "Datum", "ef": "EF (m/min per hartslag)"},
            )
            fig.update_traces(
                marker=dict(size=11), line=dict(color=SPORT_COLORS[titel]),
                hovertemplate="%{x|%d-%m-%Y} · EF %{y:.2f}<extra></extra>")
            fig.update_layout(title=f"{titel} — efficiency factor (hoger = beter)")
            if len(ef_sport) == 1:
                t = pd.Timestamp(ef_sport["start_time"].iloc[0])
                fig.update_xaxes(range=[t - pd.Timedelta(days=4), t + pd.Timedelta(days=4)],
                                 tickformat="%d-%m-%Y")
            st.plotly_chart(style_fig(fig, show_legend=False), width="stretch")

    dec = decoupling(conn, acts)
    if not dec.empty:
        dec = dec.copy().sort_values("start_time")
        dec["Sport"] = dec["sport"].map(sport_label)
        dec["label"] = dec["start_time"].dt.strftime("%d-%m")
        fig = px.bar(
            dec, x="label", y="decoupling_pct", color="Sport",
            color_discrete_map=SPORT_COLORS,
            labels={"label": "Sessie", "decoupling_pct": "Decoupling (%)"},
            category_orders={"label": dec["label"].tolist()},
        )
        # Forceer een categorie-as: plotly parst anders labels als "31-05" als
        # een datum (→ mei 2031). Zo blijft het gewoon dag-maand, chronologisch.
        fig.update_xaxes(type="category")
        fig.add_hline(y=5, line_dash="dash", line_color="#e45756",
                      annotation_text="richtwaarde 5%")
        fig.update_traces(hovertemplate="%{x} · %{fullData.name}: %{y:.1f}%<extra></extra>")
        fig.update_layout(title="HR-decoupling per sessie (lager = betere aerobe basis)")
        st.plotly_chart(style_fig(fig), width="stretch")

    st.divider()
    # -- Racevoorspelling -----------------------------------------------------
    st.subheader("🏁 Racevoorspelling — standaard (1,5 km / 40 km / 10 km)")
    pred = race_prediction(acts)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Zwemmen 1,5 km", fmt_duration(pred["zwem_1500"]))
    c2.metric("Fietsen 40 km", fmt_duration(pred["fiets_40k"]))
    c3.metric("Lopen 10 km", fmt_duration(pred["loop_10k"]))
    c4.metric("Wissels (T1+T2)", fmt_duration(pred["wissels"]))
    c5.metric("Totaal (schatting)", fmt_duration(pred["totaal"]))
    st.caption(
        "Ruwe schatting: lopen via Riegel-schaling vanaf je beste recente loop, fietsen via je "
        "snelste rit (≥15 km), zwemmen via het tempo van je laatste zwemsessie — dat verandert "
        "nu het snelst, dus deze voorspelling wordt elke zwemsessie beter. Racedag-effecten "
        "(wetsuit, drafting, spanning) zitten er niet in."
    )
    for emoji, tekst in readiness(acts):
        st.markdown(f"{emoji} {tekst}")

    st.divider()
    # -- Records & zwemprogressie ---------------------------------------------
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("🏆 Persoonlijke records")
        prs = personal_records(conn, acts)
        if prs.empty:
            st.info("Nog geen records — importeer trainingen.")
        else:
            st.dataframe(prs, hide_index=True, width="stretch")
    with c2:
        st.subheader("🏊 Zwemprogressie")
        aandeel, swolf_slag = swim_progression(conn, acts)
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
            if len(aandeel) == 1:
                t = pd.Timestamp(aandeel["start_time"].iloc[0])
                fig.update_xaxes(range=[t - pd.Timedelta(days=4), t + pd.Timedelta(days=4)],
                                 tickformat="%d-%m-%Y")
            st.plotly_chart(style_fig(fig, show_legend=False), width="stretch")
        if not swolf_slag.empty:
            swolf_slag = swolf_slag.copy()
            swolf_slag["Slag"] = swolf_slag["slag"].map(lambda s: STROKE_NL.get(s, s))
            fig = px.line(
                swolf_slag, x="start_time", y="swolf", color="Slag", markers=True,
                labels={"start_time": "Datum", "swolf": "SWOLF"},
            )
            fig.update_traces(
                marker=dict(size=11),
                hovertemplate="%{x|%d-%m-%Y} · %{fullData.name}: SWOLF %{y:.0f}<extra></extra>")
            fig.update_layout(title="SWOLF per slagtype (lager = efficiënter)")
            if swolf_slag["start_time"].nunique() == 1:
                t = pd.Timestamp(swolf_slag["start_time"].iloc[0])
                fig.update_xaxes(range=[t - pd.Timedelta(days=4), t + pd.Timedelta(days=4)],
                                 tickformat="%d-%m-%Y")
            st.plotly_chart(style_fig(fig), width="stretch")

# ------------------------------------------------------- discipline-tabs --
with tab_lopen:
    runs = acts[acts["sport"] == "running"].sort_values("start_time")
    if runs.empty:
        st.info("Nog geen loopsessies.")
    else:
        runs = runs.copy()
        runs["tempo"] = pace_as_time(1000 / runs["avg_speed_ms"])
        runs["cadans_spm"] = runs["avg_cadence"] * 2  # Garmin telt één been
        c1, c2 = st.columns(2)
        with c1:
            fig = px.scatter(
                runs, x="start_time", y="tempo", color="avg_hr",
                color_continuous_scale="RdYlGn_r",
                labels={"start_time": "Datum", "tempo": "Tempo (min/km)",
                        "avg_hr": "Gem. HR"},
            )
            fig.update_yaxes(tickformat="%M:%S", autorange="reversed")
            fig.update_traces(
                mode="lines+markers",
                marker=dict(size=14, line=dict(width=1, color="rgba(255,255,255,0.7)")),
                line=dict(color="rgba(150,150,150,0.4)"),
                hovertemplate="%{x|%d-%m-%Y} · %{y|%M:%S} min/km · HR %{marker.color}<extra></extra>",
            )
            fig.update_layout(title="Tempo per sessie (kleur = gemiddelde hartslag)")
            st.plotly_chart(style_fig(fig, show_legend=False), width="stretch")
        with c2:
            fig = px.line(
                runs, x="start_time", y="cadans_spm", markers=True,
                labels={"start_time": "Datum", "cadans_spm": "Cadans (stappen/min)"},
            )
            fig.update_traces(
                marker=dict(size=11),
                hovertemplate="%{x|%d-%m-%Y} · %{y:.0f} stappen/min<extra></extra>",
            )
            fig.update_layout(title="Cadans per sessie")
            st.plotly_chart(style_fig(fig, show_legend=False), width="stretch")

with tab_fietsen:
    rides = acts[acts["sport"] == "cycling"].sort_values("start_time")
    if rides.empty:
        st.info("Nog geen fietssessies.")
    else:
        rides = rides.copy()
        rides["snelheid_kmh"] = rides["avg_speed_ms"] * 3.6
        c1, c2 = st.columns(2)
        with c1:
            fig = px.scatter(
                rides, x="start_time", y="snelheid_kmh", color="avg_hr",
                color_continuous_scale="RdYlGn_r",
                labels={"start_time": "Datum", "snelheid_kmh": "Snelheid (km/h)",
                        "avg_hr": "Gem. HR"},
            )
            fig.update_traces(
                mode="lines+markers",
                marker=dict(size=14, line=dict(width=1, color="rgba(255,255,255,0.7)")),
                line=dict(color="rgba(150,150,150,0.4)"),
                hovertemplate="%{x|%d-%m-%Y} · %{y:.1f} km/h · HR %{marker.color}<extra></extra>",
            )
            fig.update_layout(title="Snelheid per rit (kleur = gemiddelde hartslag)")
            st.plotly_chart(style_fig(fig, show_legend=False), width="stretch")
        with c2:
            fig = px.bar(
                rides, x="start_time", y="total_ascent",
                labels={"start_time": "Datum", "total_ascent": "Hoogtemeters"},
            )
            fig.update_traces(hovertemplate="%{x|%d-%m-%Y} · %{y:.0f} m<extra></extra>")
            fig.update_layout(title="Hoogtemeters per rit")
            st.plotly_chart(style_fig(fig, show_legend=False), width="stretch")

with tab_zwemmen:
    swim = swim_per_session(conn, acts)
    if swim.empty:
        st.info("Nog geen zwemsessies.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            fig = px.line(
                swim, x="start_time", y="swolf", markers=True,
                labels={"start_time": "Datum", "swolf": "SWOLF"},
            )
            fig.update_traces(
                marker=dict(size=11),
                hovertemplate="%{x|%d-%m-%Y} · SWOLF %{y:.0f}<extra></extra>",
            )
            fig.update_layout(title="Gemiddelde SWOLF per sessie (lager = efficiënter)")
            st.plotly_chart(style_fig(fig, show_legend=False), width="stretch")
        with c2:
            swim = swim.copy()
            swim["tempo"] = pace_as_time(swim["tempo_s_per_100m"])
            fig = px.line(
                swim, x="start_time", y="tempo", markers=True,
                labels={"start_time": "Datum", "tempo": "Tempo (min/100m)"},
            )
            fig.update_yaxes(tickformat="%M:%S", autorange="reversed")
            fig.update_traces(
                marker=dict(size=11),
                hovertemplate="%{x|%d-%m-%Y} · %{y|%M:%S} /100m<extra></extra>",
            )
            fig.update_layout(title="Tempo per 100 meter (sneller = hoger)")
            st.plotly_chart(style_fig(fig, show_legend=False), width="stretch")

        # Slagverdeling van de laatste zwemsessie.
        laatste = acts[acts["sport"] == "swimming"].iloc[0]
        lengths = pd.read_sql_query(
            "SELECT swim_stroke, COUNT(*) AS banen FROM lengths "
            "WHERE activity_key = ? GROUP BY swim_stroke",
            conn, params=(laatste["activity_key"],))
        if not lengths.empty:
            lengths["Slag"] = lengths["swim_stroke"].map(lambda s: STROKE_NL.get(s, s))
            st.subheader(f"Slagverdeling laatste sessie ({laatste['start_time']:%d-%m-%Y})")
            fig = px.pie(lengths, names="Slag", values="banen", hole=0.4)
            fig.update_traces(hovertemplate="%{label}: %{value} banen (%{percent})<extra></extra>")
            st.plotly_chart(style_fig(fig), width="stretch")

# ----------------------------------------------------------------- lichaam --
with tab_lichaam:
    st.subheader("🧍 Lichaamssamenstelling")
    st.caption(
        "Neutrale data voor je sportprestatie — geen afval- of dieetcoach. "
        "De nadruk ligt op de **trend over weken**, vooral vet% en spiermassa. "
        "BMI zegt weinig bij veel spiermassa en krijgt daarom weinig gewicht."
    )

    body.ensure_table(conn)

    # -- Invoer (optioneel voorgevuld via screenshot) -----------------------
    with st.expander("➕ Nieuwe meting invoeren", expanded=False):
        st.caption(
            "Vul in wat je hebt; lege velden worden niet opgeslagen. Eventueel "
            "eerst een screenshot van de Fitdays-app uploaden om de velden "
            "automatisch voor te vullen (lokaal gemma-model, gratis)."
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
        with st.form("body_form"):
            meetdatum = st.date_input("Datum", value=date.today())
            cols = st.columns(3)
            ingevuld = {}
            for idx, (col, label, eenheid, stap) in enumerate(body.FIELDS):
                with cols[idx % 3]:
                    label_txt = f"{label}{f' ({eenheid})' if eenheid else ''}"
                    val = st.number_input(
                        label_txt, value=float(prefill.get(col, 0.0)),
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
                    body.summarize_trend(router, conn, MEMORY_DIR)
                st.session_state.pop("body_prefill", None)
                st.success(f"Meting van {meetdatum:%d-%m-%Y} opgeslagen.")
                st.rerun()

    metingen = body.load_measurements(conn)

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

        # -- Losse trendgrafieken -------------------------------------------
        st.subheader("Trends per maat")
        trend_meta = [
            ("weight_kg", "Gewicht (kg)", "#4c78a8"),
            ("fat_pct", "Lichaamsvet (%)", "#e45756"),
            ("muscle_mass_kg", "Spiermassa (kg)", "#54a24b"),
            ("visceral_fat", "Visceraal vet", "#f58518"),
        ]
        g1, g2 = st.columns(2)
        for idx, (col, titel, kleur) in enumerate(trend_meta):
            serie = sel[["measured_on", col]].dropna()
            doel = g1 if idx % 2 == 0 else g2
            with doel:
                if serie.empty:
                    st.info(f"Geen data voor {titel.lower()}.")
                    continue
                fig = px.line(serie, x="measured_on", y=col, markers=True,
                              labels={"measured_on": "Datum", col: titel})
                fig.update_traces(line=dict(color=kleur), marker=dict(size=10),
                                  hovertemplate="%{x|%d-%m-%Y} · %{y:.1f}<extra></extra>")
                fig.update_layout(title=titel)
                date_xaxis(fig, serie["measured_on"])
                if len(serie) == 1:
                    t = pd.Timestamp(serie["measured_on"].iloc[0])
                    fig.update_xaxes(range=[t - pd.Timedelta(days=7), t + pd.Timedelta(days=7)])
                st.plotly_chart(style_fig(fig, show_legend=False), width="stretch")

        # -- Gecombineerd (genormaliseerd) ----------------------------------
        combo = body.normalized_trends(sel, body.TREND_FIELDS)
        if not combo.empty and combo["measured_on"].nunique() > 1:
            st.subheader("Gecombineerd (relatieve verandering)")
            st.caption("Elke reeks geïndexeerd op de eerste meting in het bereik (=100), "
                       "zodat maten met verschillende eenheden samen vergelijkbaar zijn.")
            fig = px.line(combo, x="measured_on", y="index", color="reeks", markers=True,
                          labels={"measured_on": "Datum", "index": "Index (eerste = 100)",
                                  "reeks": "Maat"})
            fig.add_hline(y=100, line_dash="dot", line_color="rgba(150,150,150,0.6)")
            fig.update_traces(hovertemplate="%{fullData.name}: %{y:.1f}<extra></extra>")
            fig.update_layout(hovermode="x unified")
            fig.update_xaxes(hoverformat="%d-%m-%Y")
            date_xaxis(fig, combo["measured_on"])
            st.plotly_chart(style_fig(fig), width="stretch")

        # -- Kruising met trainingsdata -------------------------------------
        kruising = body.weight_vs_cycling(conn, acts)
        if not kruising.empty and len(kruising) > 1:
            st.subheader("Gewicht naast fietsvolume (richting power-to-weight)")
            st.caption(
                "Vermogen wordt niet opgeslagen, dus fietskilometers per week dienen "
                "als prestatieproxy. Daalt het gewicht terwijl het fietsvolume op peil "
                "blijft, dan beweegt je power-to-weight de goede kant op."
            )
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=kruising["week"], y=kruising["fiets_km"], name="Fietskilometers/week",
                marker_color="rgba(76,120,168,0.4)", yaxis="y2",
                hovertemplate="%{x}: %{y:.0f} km<extra></extra>"))
            fig.add_trace(go.Scatter(
                x=kruising["week"], y=kruising["weight_kg"], name="Gewicht (kg)",
                line=dict(color="#4c78a8", width=3), mode="lines+markers",
                hovertemplate="%{x}: %{y:.1f} kg<extra></extra>"))
            fig.update_layout(
                yaxis=dict(title="Gewicht (kg)"),
                yaxis2=dict(title="Fiets-km/week", overlaying="y", side="right", showgrid=False),
                xaxis_title="Week",
            )
            st.plotly_chart(style_fig(fig), width="stretch")

    st.divider()
    log_path = MEMORY_DIR / "lichaamssamenstelling.md"
    if log_path.exists():
        with st.expander("📖 Logboek & trendduiding"):
            st.markdown(log_path.read_text(encoding="utf-8"))

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
                generate_advice(router, conn, MEMORY_DIR)
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
                generate_insights(router, conn, MEMORY_DIR,
                                  progress_text=progress_summary_text(conn, acts))
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
                antwoord = answer_question(router, conn, MEMORY_DIR, vraag, escalate=escaleer)
        except Exception as e:
            antwoord = f"Er ging iets mis: {e}"
        st.chat_message("assistant").write(f"{antwoord}\n\n_— {bron}_")
        st.session_state.chat_history.append((vraag, antwoord, bron))

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
    races_df["date"] = pd.to_datetime(races_df["date"])
    edited_races = st.data_editor(
        races_df[["name", "date", "distances", "goal", "target_time"]],
        column_config={
            "name": st.column_config.TextColumn("Race"),
            "date": st.column_config.DateColumn("Datum", format="DD-MM-YYYY"),
            "distances": st.column_config.TextColumn("Afstanden"),
            "goal": st.column_config.TextColumn("Doel"),
            "target_time": st.column_config.TextColumn("Streeftijd"),
        },
        num_rows="dynamic", hide_index=True, width="stretch", key="races_editor",
    )

    st.subheader("🗓️ Trainingsdagen & beschikbare tijd")
    c1, c2 = st.columns(2)
    new_training_days = c1.text_input(
        "Geoormerkte trainingsdagen", str(config["athlete"].get("training_days", "")),
        help="Bijv.: zwemmen ma/vr-ochtend, lange duurtraining zondag.")
    new_session_time = c2.text_input(
        "Beschikbare tijd per sessie", str(config["athlete"].get("session_time", "")),
        help="Bijv.: 30-45 min doordeweeks, 1,5-2 uur zondag.")

    st.subheader("❤️ Hartslag & zones")
    c1, c2 = st.columns(2)
    new_max_hr = c1.number_input("Maximale hartslag", 120, 230, int(config["athlete"]["max_hr"]))
    new_lthr = c2.number_input("LTHR (drempelhartslag)", 100, 220, int(config["athlete"]["lthr"]))
    preview = bounds_from_lthr(new_lthr, config["athlete"].get("zone_pct_lthr"))
    st.caption(
        f"Zones bij LTHR {new_lthr} (automatisch afgeleid, %LTHR): "
        f"zone 1 < {preview[0]} · zone 2 {preview[0]}–{preview[1]} · "
        f"zone 3 {preview[1]}–{preview[2]} · zone 4 {preview[2]}–{preview[3]} · "
        f"zone 5 > {preview[3]}. Bij een LTHR-wijziging worden de zonetijden "
        "van alle trainingen herrekend en komt er een regel bij in "
        "memory/lthr_geschiedenis.md."
    )

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
                "distances": str(r["distances"] or ""),
                "goal": str(r["goal"] or ""),
                "target_time": str(r.get("target_time") or ""),
            }
            for _, r in edited_races.iterrows()
            if str(r["name"]).strip() and pd.notna(r["date"])
        ]
        new_config["athlete"]["max_hr"] = new_max_hr
        new_config["athlete"]["lthr"] = new_lthr
        new_config["athlete"]["training_days"] = new_training_days.strip()
        new_config["athlete"]["session_time"] = new_session_time.strip()
        new_config["athlete"].pop("zone_bounds", None)  # zones komen nu uit %LTHR
        new_config["llm"]["ollama"]["host"] = new_host.strip().rstrip("/")
        new_config["llm"]["ollama"]["model"] = new_ollama_model.strip()
        new_config["llm"]["anthropic"]["model"] = new_anthropic_model.strip()
        new_config["llm"]["routing"] = new_routing
        save_config(new_config)

        # Profielwaarden leesbaar spiegelen naar doelen.md met changelog.
        wijzigingen = profile_mod.update_doelen(
            MEMORY_DIR, config, new_config, note="instellingen-tab")

        if new_lthr != int(config["athlete"]["lthr"]):
            lthr_append(MEMORY_DIR, new_lthr, "Aangepast via instellingen-tab")
            n = recompute_zones(conn, preview)
            st.session_state["settings_flash"] = (
                f"Opgeslagen. Nieuwe LTHR {new_lthr} vastgelegd in de geschiedenis; "
                f"zonetijden van {n} trainingen herrekend."
            )
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

conn.close()
