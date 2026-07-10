"""Combinatietrainingen: bricks en triatlon-trainingen herkennen en analyseren.

Een Garmin maakt per onderdeel een apart FIT-bestand; een fiets-loop-brick of
een complete triatlon-training komt dus als losse sessies binnen. Deze module
koppelt ze weer aan elkaar:

- **Detectie** (:func:`detect_and_store_proposals`): losse sessies op dezelfde
  dag, in race-volgorde (zwemmen → fietsen → hardlopen of een deelvolgorde),
  waarbij elk volgend onderdeel binnen de wisseldrempel (config
  ``combo.max_gap_min``, standaard 25 minuten) na het einde van het vorige
  begint. Twee onderdelen = **brick**, drie = **triatlon-training**.
- **Bevestigbaar, nooit stilzwijgend**: een gevonden groep wordt als
  *voorstel* opgeslagen (status ``voorgesteld``). De gebruiker bevestigt
  (``bevestigd``) of maakt los (``losgemaakt``); een losgemaakte groep wordt
  niet opnieuw voorgesteld. Alleen bevestigde combos tellen mee in trends.
- **Wissel-analyse**: T1 (zwem→fiets) en T2 (fiets→loop) als oefenbare
  racetijd, plus de "bakstenen benen"-analyse van de loop na het fietsen
  (:func:`run_transition_analysis`): het eerste stuk (1 km, of 5 minuten als
  afstandsdata ontbreekt) tegenover de rest van de loop, in tempo en HR.
- **Race-simulatie** (:func:`race_similarity`): lijken de afstanden op een
  bekende race-opzet (sprint, olympisch, de rookie-opzet of een race uit
  config.yaml), dan wordt dat benoemd.
- **Feedback-context** (:func:`combo_block`): de wissel- en overgangsdata als
  tekstblok voor de coach, inclusief de trend over eerdere bevestigde combos.

De groepering staat in twee eigen tabellen (``combos`` + ``combo_members``)
naast de bestaande drie, zodat de activities-tabel onaangeroerd blijft en een
sessie hooguit in één actieve combo zit.
"""

import json
import sqlite3
from datetime import datetime

import pandas as pd

from tricoach.formatting import (
    fmt_duration,
    fmt_pace_per_100m,
    fmt_pace_per_km,
    fmt_speed_kmh,
    derive_speed_ms,
    local_time,
    sport_label,
)

# Race-volgorde: een combinatietraining volgt deze volgorde (of een deel ervan,
# strikt oplopend). Sporten buiten deze lijst breken een keten.
RACE_ORDER = {"swimming": 0, "cycling": 1, "running": 2}

# Standaarddrempel (minuten) tussen einde onderdeel A en start onderdeel B:
# genoeg voor een wissel, te kort voor een losse tweede training. Instelbaar
# via config.yaml -> combo.max_gap_min.
DEFAULT_MAX_GAP_MIN = 25

# Kleine klok-overlap toestaan (multisport-horloges kunnen het volgende
# onderdeel al starten terwijl het vorige nog afsluit).
MIN_GAP_S = -60.0

# "Bakstenen benen": het eerste stuk van de loop na het fietsen. Primair op
# afstand (eerste km); zonder afstandsdata op tijd (eerste 5 minuten).
EERSTE_STUK_M = 1000.0
EERSTE_STUK_S = 300.0

# Bekende race-opzetten voor de simulatie-vergelijking, naast de races uit
# config.yaml. De rookie-opzet is de geplande zelfgeorganiseerde mini-triatlon.
BEKENDE_OPZETTEN = [
    ("Rookie mini-triatlon (~500 m / 20 km / 5-7 km)",
     {"swimming": 500.0, "cycling": 20000.0, "running": 6000.0}),
    ("Sprint-triatlon (750 m / 20 km / 5 km)",
     {"swimming": 750.0, "cycling": 20000.0, "running": 5000.0}),
    ("Olympische/standaardafstand (1,5 km / 40 km / 10 km)",
     {"swimming": 1500.0, "cycling": 40000.0, "running": 10000.0}),
]
# Per onderdeel mag de afstand dit veel afwijken van de opzet (ratio-grenzen).
SIM_RATIO_MIN, SIM_RATIO_MAX = 0.6, 1.5

SCHEMA = """
CREATE TABLE IF NOT EXISTS combos (
    combo_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    status     TEXT NOT NULL,
    created_at TEXT,
    status_changed_at TEXT
);
CREATE TABLE IF NOT EXISTS combo_members (
    combo_id     INTEGER NOT NULL REFERENCES combos(combo_id),
    activity_key TEXT NOT NULL,
    positie      INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_combo_members_key ON combo_members(activity_key);
"""


def ensure_tables(conn: sqlite3.Connection) -> None:
    """Maak de combo-tabellen aan als ze nog niet bestaan."""
    conn.executescript(SCHEMA)
    conn.commit()


def max_gap_min(config: dict) -> int:
    """De wisseldrempel (minuten) uit config.yaml, met veilige standaard."""
    try:
        return int((config.get("combo") or {}).get("max_gap_min", DEFAULT_MAX_GAP_MIN))
    except (TypeError, ValueError):
        return DEFAULT_MAX_GAP_MIN


def combo_kind(n_members: int) -> str:
    """Twee onderdelen = brick, drie = triatlon-training."""
    return "triatlon" if n_members >= 3 else "brick"


# ------------------------------------------------------------------ detectie --

def _elapsed_s(row: pd.Series) -> float:
    """Verstreken duur (s) van een sessie, voor het bepalen van de eindtijd.

    Bij voorkeur ``total_elapsed_time`` uit de sessie-samenvatting (inclusief
    pauzes — de kloktijd dus, en dát bepaalt wanneer de wissel begint); valt
    terug op ``duration_s`` (timer-tijd) als die ontbreekt.
    """
    raw = row.get("summary_json")
    if isinstance(raw, str) and raw:
        try:
            elapsed = json.loads(raw).get("total_elapsed_time")
            if elapsed:
                return float(elapsed)
        except (ValueError, TypeError):
            pass
    dur = row.get("duration_s")
    return float(dur) if dur and not pd.isna(dur) else 0.0


def _end_time(row: pd.Series) -> pd.Timestamp:
    """Eindtijd van een sessie: starttijd + verstreken (klok)duur."""
    return row["start_time"] + pd.Timedelta(seconds=_elapsed_s(row))


def detect_chains(acts: pd.DataFrame, gap_min: int = DEFAULT_MAX_GAP_MIN) -> list[list[str]]:
    """Vind kandidaat-combinatietrainingen; geeft lijsten van activity_keys.

    Een keten groeit zolang: zelfde lokale dag, de volgende sessie start
    binnen ``gap_min`` minuten na het einde van de vorige, en de sporten
    strikt oplopen in race-volgorde (zwemmen → fietsen → hardlopen). Ketens
    van twee of drie onderdelen zijn kandidaten; losse sessies met uren
    ertussen halen de drempel niet en worden dus nooit samengevoegd.
    """
    if acts.empty:
        return []
    df = acts.sort_values("start_time")
    df = df[df["sport"].isin(RACE_ORDER)]

    chains: list[list[str]] = []
    keten: list[pd.Series] = []
    for _, row in df.iterrows():
        if keten:
            vorige = keten[-1]
            gap_s = (row["start_time"] - _end_time(vorige)).total_seconds()
            zelfde_dag = (local_time(row["start_time"]).date()
                          == local_time(vorige["start_time"]).date())
            oplopend = RACE_ORDER[row["sport"]] > RACE_ORDER[vorige["sport"]]
            if zelfde_dag and oplopend and MIN_GAP_S <= gap_s <= gap_min * 60:
                keten.append(row)
                continue
            if len(keten) >= 2:
                chains.append([r["activity_key"] for r in keten])
        keten = [row]
    if len(keten) >= 2:
        chains.append([r["activity_key"] for r in keten])
    return chains


def _existing_member_sets(conn: sqlite3.Connection) -> dict[frozenset, tuple[int, str]]:
    """Alle bestaande combos als {set van activity_keys: (combo_id, status)}."""
    rows = conn.execute(
        "SELECT c.combo_id, c.status, m.activity_key "
        "FROM combos c JOIN combo_members m ON m.combo_id = c.combo_id"
    ).fetchall()
    per_combo: dict[int, dict] = {}
    for combo_id, status, key in rows:
        per_combo.setdefault(combo_id, {"status": status, "keys": set()})
        per_combo[combo_id]["keys"].add(key)
    return {frozenset(v["keys"]): (cid, v["status"]) for cid, v in per_combo.items()}


def detect_and_store_proposals(conn: sqlite3.Connection, acts: pd.DataFrame,
                               gap_min: int = DEFAULT_MAX_GAP_MIN) -> int:
    """Detecteer combinatietrainingen en sla nieuwe voorstellen op.

    Niets gebeurt stilzwijgend: gevonden groepen krijgen status ``voorgesteld``
    en wachten op bevestiging in het dashboard. Regels:

    - een groepering die al bestaat (welke status ook) wordt niet opnieuw
      voorgesteld — een losgemaakte groep blijft dus los;
    - overlapt een kandidaat met een *bevestigde* combo, dan blijft de
      bevestiging leidend en wordt de kandidaat overgeslagen;
    - overlapt hij met een ouder *voorstel* (bijv. brick werd triatlon doordat
      het zwem-bestand later is geüpload), dan vervangt het nieuwe voorstel
      het oude.

    Geeft het aantal nieuw opgeslagen voorstellen terug.
    """
    ensure_tables(conn)
    bestaand = _existing_member_sets(conn)
    bevestigd_keys = {k for s, (cid, st) in bestaand.items() if st == "bevestigd" for k in s}

    nieuw = 0
    for keys in detect_chains(acts, gap_min):
        kset = frozenset(keys)
        if kset in bestaand:
            continue
        if any(k in bevestigd_keys for k in keys):
            continue
        # Overlappende oude voorstellen vervangen door dit (ruimere) voorstel.
        for oud_set, (oud_id, oud_status) in bestaand.items():
            if oud_status == "voorgesteld" and oud_set & kset:
                conn.execute("DELETE FROM combo_members WHERE combo_id = ?", (oud_id,))
                conn.execute("DELETE FROM combos WHERE combo_id = ?", (oud_id,))
        nu = datetime.now().isoformat(timespec="seconds")
        cur = conn.execute(
            "INSERT INTO combos (status, created_at, status_changed_at) "
            "VALUES ('voorgesteld', ?, ?)", (nu, nu))
        conn.executemany(
            "INSERT INTO combo_members (combo_id, activity_key, positie) VALUES (?,?,?)",
            [(cur.lastrowid, key, i) for i, key in enumerate(keys)])
        bestaand = _existing_member_sets(conn)
        nieuw += 1
    conn.commit()
    return nieuw


def set_combo_status(conn: sqlite3.Connection, combo_id: int, status: str) -> bool:
    """Zet de status van een combo: 'bevestigd' of 'losgemaakt'.

    Losmaken kan ook bij een eerder bevestigde combo; de groepering blijft
    bewaard zodat dezelfde set niet opnieuw wordt voorgesteld.
    """
    cur = conn.execute(
        "UPDATE combos SET status = ?, status_changed_at = ? WHERE combo_id = ?",
        (status, datetime.now().isoformat(timespec="seconds"), combo_id))
    conn.commit()
    return cur.rowcount > 0


def combo_membership(conn: sqlite3.Connection) -> dict[str, dict]:
    """Per activity_key de actieve combo (voorgesteld/bevestigd) waar hij in zit.

    Geeft {activity_key: {combo_id, status, kind}}; losgemaakte combos tellen
    niet mee. Voor het markeren van combo-onderdelen in de sessietabel.
    """
    ensure_tables(conn)
    rows = conn.execute(
        "SELECT m.activity_key, c.combo_id, c.status, "
        "       (SELECT COUNT(*) FROM combo_members m2 WHERE m2.combo_id = c.combo_id) "
        "FROM combo_members m JOIN combos c ON c.combo_id = m.combo_id "
        "WHERE c.status IN ('voorgesteld', 'bevestigd')"
    ).fetchall()
    return {key: {"combo_id": cid, "status": status, "kind": combo_kind(n)}
            for key, cid, status, n in rows}


# ------------------------------------------------------------ wissel-analyse --

def transition_label(van_sport: str, naar_sport: str) -> str:
    """Racelabel van een wissel: T1 = zwem→fiets, T2 = fiets→loop."""
    if van_sport == "swimming" and naar_sport == "cycling":
        return "T1"
    if van_sport == "cycling" and naar_sport == "running":
        return "T2"
    return "wissel"


def load_combos(conn: sqlite3.Connection, acts: pd.DataFrame,
                statuses: tuple = ("voorgesteld", "bevestigd")) -> list[dict]:
    """Alle combos (nieuwste eerst) met onderdelen, wisseltijden en totalen.

    Per combo: ``combo_id``, ``status``, ``kind``, ``members`` (de
    activities-rijen in racevolgorde), ``transitions`` (label, sporten en
    seconden per wissel), ``totaal_s`` (eerste start tot laatste einde, dus
    inclusief wissels) en ``wissel_s`` (som van de wisseltijden). Combos
    waarvan een onderdeel inmiddels is verwijderd, worden overgeslagen.
    """
    ensure_tables(conn)
    if acts.empty:
        return []
    rows = conn.execute(
        "SELECT c.combo_id, c.status, m.activity_key, m.positie FROM combos c "
        "JOIN combo_members m ON m.combo_id = c.combo_id "
        "WHERE c.status IN ({}) ORDER BY c.combo_id, m.positie".format(
            ",".join("?" * len(statuses))), tuple(statuses)).fetchall()

    per_combo: dict[int, dict] = {}
    for combo_id, status, key, _pos in rows:
        per_combo.setdefault(combo_id, {"status": status, "keys": []})
        per_combo[combo_id]["keys"].append(key)

    by_key = acts.set_index("activity_key", drop=False)
    combos = []
    for combo_id, info in per_combo.items():
        if not all(k in by_key.index for k in info["keys"]):
            continue  # een onderdeel is verwijderd
        members = [by_key.loc[k] for k in info["keys"]]
        members.sort(key=lambda r: r["start_time"])
        transitions = []
        for a, b in zip(members, members[1:]):
            transitions.append({
                "label": transition_label(a["sport"], b["sport"]),
                "van": a["sport"], "naar": b["sport"],
                "seconds": max(
                    (b["start_time"] - _end_time(a)).total_seconds(), 0.0),
            })
        combos.append({
            "combo_id": combo_id,
            "status": info["status"],
            "kind": combo_kind(len(members)),
            "members": members,
            "transitions": transitions,
            "totaal_s": (_end_time(members[-1]) - members[0]["start_time"])
            .total_seconds(),
            "wissel_s": sum(t["seconds"] for t in transitions),
            "start_time": members[0]["start_time"],
        })
    combos.sort(key=lambda c: c["start_time"], reverse=True)
    return combos


def run_transition_analysis(records: pd.DataFrame,
                            eerste_m: float = EERSTE_STUK_M,
                            eerste_s: float = EERSTE_STUK_S) -> dict | None:
    """De "bakstenen benen"-analyse: eerste stuk van de loop-na-fiets vs de rest.

    Splitst de loop op de eerste kilometer (of de eerste 5 minuten als
    afstandsdata ontbreekt) en geeft per deel tempo (s/km) en gemiddelde HR:
    ``{"basis": ..., "eerste": {...}, "rest": {...}}``. Trekt het tempo na het
    eerste stuk aan bij gelijke HR, dan kwamen de benen los — dé maat om over
    meerdere bricks te volgen. None bij te weinig data (de rest moet minstens
    zo lang zijn als het eerste stuk, anders is de vergelijking niet eerlijk).
    """
    if records.empty or "timestamp" not in records:
        return None
    df = records.dropna(subset=["timestamp"]).sort_values("timestamp")
    if df.empty:
        return None

    dist = df["distance_m"].astype(float) if "distance_m" in df else pd.Series(dtype=float)
    heeft_afstand = not dist.dropna().empty and dist.dropna().max() > 0
    t0 = df["timestamp"].iloc[0]

    if heeft_afstand and dist.dropna().max() >= 2 * eerste_m:
        mask = dist.ffill() >= eerste_m
        basis = f"eerste {eerste_m / 1000:.1f} km"
    else:
        mask = (df["timestamp"] - t0).dt.total_seconds() >= eerste_s
        basis = f"eerste {eerste_s / 60:.0f} min"
        totaal_s = (df["timestamp"].iloc[-1] - t0).total_seconds()
        if totaal_s < 2 * eerste_s:
            return None
    if not mask.any() or mask.all():
        return None

    def _segment(seg: pd.DataFrame) -> dict:
        duur = (seg["timestamp"].iloc[-1] - seg["timestamp"].iloc[0]).total_seconds()
        deel = {"duur_s": duur, "tempo_s_per_km": None, "gem_hr": None}
        if heeft_afstand:
            meters = (seg["distance_m"].dropna().iloc[-1]
                      - seg["distance_m"].dropna().iloc[0])
            if meters > 0 and duur > 0:
                deel["tempo_s_per_km"] = duur / meters * 1000
        if "heart_rate" in seg and not seg["heart_rate"].dropna().empty:
            deel["gem_hr"] = float(seg["heart_rate"].dropna().mean())
        return deel

    split_idx = mask.idxmax()
    pos = df.index.get_loc(split_idx)
    eerste, rest = _segment(df.iloc[:pos + 1]), _segment(df.iloc[pos:])
    if not eerste["duur_s"] or not rest["duur_s"]:
        return None
    return {"basis": basis, "eerste": eerste, "rest": rest}


# ------------------------------------------------------------ race-simulatie --

def _config_opzetten(config: dict) -> list[tuple[str, dict]]:
    """De races uit config.yaml als vergelijkbare opzetten (naam + afstanden)."""
    opzetten = []
    for race in (config or {}).get("races", []):
        afst = {}
        for veld, sport in (("swim_m", "swimming"), ("bike_m", "cycling"),
                            ("run_m", "running")):
            if race.get(veld):
                afst[sport] = float(race[veld])
        if len(afst) >= 2:
            opzetten.append((f"{race.get('name', 'race')} "
                             f"({race.get('distances', '')})".strip(), afst))
    return opzetten


def race_similarity(dist_by_sport: dict[str, float],
                    config: dict | None = None) -> str | None:
    """Lijkt deze combinatietraining op een bekende race-opzet?

    Vergelijkt de afstanden per onderdeel met de bekende opzetten (rookie,
    sprint, olympisch) en de races uit config.yaml. Elk aanwezig onderdeel
    moet binnen 60–150% van de opzet liggen; de opzet met de kleinste
    gemiddelde afwijking wint. Geeft een leesbare vergelijkingsregel, of None
    als niets in de buurt komt of er maar één onderdeel met afstand is.
    """
    aanwezig = {s: d for s, d in dist_by_sport.items()
                if d and not pd.isna(d) and d > 0}
    if len(aanwezig) < 2:
        return None

    beste: tuple[float, str, dict] | None = None
    for naam, opzet in BEKENDE_OPZETTEN + _config_opzetten(config or {}):
        if not all(sport in opzet for sport in aanwezig):
            continue
        ratios = {s: aanwezig[s] / opzet[s] for s in aanwezig}
        if not all(SIM_RATIO_MIN <= r <= SIM_RATIO_MAX for r in ratios.values()):
            continue
        score = sum(abs(r - 1) for r in ratios.values()) / len(ratios)
        if beste is None or score < beste[0]:
            beste = (score, naam, opzet)
    if beste is None:
        return None

    _score, naam, opzet = beste
    delen = []
    for sport in ("swimming", "cycling", "running"):
        if sport in aanwezig:
            delen.append(
                f"{sport_label(sport).lower()} {aanwezig[sport] / 1000:.1f} van "
                f"{opzet[sport] / 1000:.1f} km ({aanwezig[sport] / opzet[sport] * 100:.0f}%)")
    prefix = ("Deze opzet lijkt op" if len(aanwezig) >= 3
              else "Deze onderdelen lijken op die van")
    return f"{prefix} {naam}: " + ", ".join(delen)


# ------------------------------------------------ trend & feedback-context --

def combo_history(conn: sqlite3.Connection, acts: pd.DataFrame,
                  load_records_fn) -> pd.DataFrame:
    """Trendgegevens per *bevestigde* combo, oudste eerst.

    Per combo: datum, soort, T1/T2 (s), totale wisseltijd, en — als er een
    loop-na-fiets in zit — het tempoverschil (s/km) en HR-verschil van het
    eerste stuk t.o.v. de rest ("hoe zwaar was de overgang"). Hiermee zijn de
    kernvragen te beantwoorden: worden de wisseltijden korter, en wordt de
    overgang soepeler? ``load_records_fn(conn, key)`` wordt geïnjecteerd om
    een importcyclus met storage te vermijden.
    """
    rows = []
    for combo in load_combos(conn, acts, statuses=("bevestigd",)):
        rij = {
            "start_time": combo["start_time"], "kind": combo["kind"],
            "combo_id": combo["combo_id"], "t1_s": None, "t2_s": None,
            "wissel_s": combo["wissel_s"], "delta_tempo_s_per_km": None,
            "delta_hr": None,
        }
        for i, t in enumerate(combo["transitions"]):
            if t["label"] == "T1":
                rij["t1_s"] = t["seconds"]
            elif t["label"] == "T2":
                rij["t2_s"] = t["seconds"]
            if t["van"] == "cycling" and t["naar"] == "running":
                run = combo["members"][i + 1]
                analyse = run_transition_analysis(
                    load_records_fn(conn, run["activity_key"]))
                if analyse and analyse["eerste"]["tempo_s_per_km"] \
                        and analyse["rest"]["tempo_s_per_km"]:
                    rij["delta_tempo_s_per_km"] = (
                        analyse["eerste"]["tempo_s_per_km"]
                        - analyse["rest"]["tempo_s_per_km"])
                    if analyse["eerste"]["gem_hr"] and analyse["rest"]["gem_hr"]:
                        rij["delta_hr"] = (analyse["eerste"]["gem_hr"]
                                           - analyse["rest"]["gem_hr"])
        rows.append(rij)
    df = pd.DataFrame(rows)
    return df.sort_values("start_time") if not df.empty else df


def _member_line(row: pd.Series) -> str:
    """Eén onderdeel van een combo als leesbare regel."""
    speed = derive_speed_ms(row.get("distance_m"), row.get("duration_s"))
    if row["sport"] == "running":
        tempo = fmt_pace_per_km(speed)
    elif row["sport"] == "cycling":
        tempo = fmt_speed_kmh(speed)
    else:
        tempo = fmt_pace_per_100m(speed)
    deel = (f"{sport_label(row['sport'])}: "
            f"{(row.get('distance_m') or 0) / 1000:.2f} km in "
            f"{fmt_duration(row.get('duration_s'))} ({tempo})")
    hr = row.get("avg_hr")
    if hr is not None and not pd.isna(hr):
        deel += f", HR gem {hr:.0f}"
    return deel


def combo_block(conn: sqlite3.Connection, act, acts: pd.DataFrame,
                config: dict, load_records_fn) -> str | None:
    """Feedback-context: is de zojuist geüploade sessie deel van een combo?

    Zoekt (onafhankelijk van de opgeslagen voorstellen — die zijn er tijdens
    de upload nog niet) de sessies die er per detectieregels aan voorafgingen,
    en bouwt daar een tekstblok van: de onderdelen, de wisseltijden, de
    bakstenen-benen-analyse als dit een loop na het fietsen is, de
    race-simulatie-vergelijking en de trend over eerdere bevestigde combos.
    Geeft None als de sessie geen combinatietraining afsluit.
    """
    if acts.empty or act.sport not in RACE_ORDER:
        return None
    gap_min = max_gap_min(config)
    start = pd.Timestamp(act.start_time)
    if start.tzinfo is None:
        start = start.tz_localize("UTC")

    # Keten terugbouwen vanaf deze sessie: steeds het onderdeel ervoor zoeken.
    eerder = acts[acts["start_time"] < start].sort_values("start_time")
    keten: list[pd.Series] = []
    huidige_start, huidige_sport = start, act.sport
    for _, row in eerder.iloc[::-1].iterrows():
        if row["sport"] not in RACE_ORDER:
            break
        gap_s = (huidige_start - _end_time(row)).total_seconds()
        zelfde_dag = (local_time(row["start_time"]).date()
                      == local_time(huidige_start).date())
        oplopend = RACE_ORDER[row["sport"]] < RACE_ORDER[huidige_sport]
        if not (zelfde_dag and oplopend and MIN_GAP_S <= gap_s <= gap_min * 60):
            break
        keten.insert(0, row)
        huidige_start, huidige_sport = row["start_time"], row["sport"]
    if not keten:
        return None

    kind = combo_kind(len(keten) + 1)
    naam = "triatlon-training" if kind == "triatlon" else "brick"
    regels = [
        f"Deze sessie sluit een {naam} af: onderdeel {len(keten) + 1} van "
        f"{len(keten) + 1}, kort na de vorige onderdelen van vandaag."
    ]
    for row in keten:
        regels.append(f"- {_member_line(row)}")
    regels.append(
        f"- {sport_label(act.sport)}: deze sessie (zie 'Zojuist voltooide sessie')")

    vorige = keten[-1]
    wissel_s = max((start - _end_time(vorige)).total_seconds(), 0.0)
    label = transition_label(vorige["sport"], act.sport)
    regels.append(
        f"- Wisseltijd {label} ({sport_label(vorige['sport']).lower()} → "
        f"{sport_label(act.sport).lower()}): {fmt_duration(wissel_s)} — "
        "oefenbare racetijd.")
    if len(keten) == 2:
        w1 = max((keten[1]["start_time"] - _end_time(keten[0])).total_seconds(), 0.0)
        l1 = transition_label(keten[0]["sport"], keten[1]["sport"])
        regels.append(f"- Wisseltijd {l1}: {fmt_duration(w1)}")

    if act.sport == "running" and vorige["sport"] == "cycling":
        analyse = run_transition_analysis(act.records)
        if analyse:
            e, r = analyse["eerste"], analyse["rest"]
            def _deel(d):
                tekst = (fmt_duration(d['tempo_s_per_km']) + "/km"
                         if d["tempo_s_per_km"] else "tempo onbekend")
                if d["gem_hr"]:
                    tekst += f" bij HR gem {d['gem_hr']:.0f}"
                return tekst
            regels.append(
                f"- Bakstenen-benen-analyse ({analyse['basis']} na het fietsen "
                f"vs de rest van de loop): {_deel(e)} → {_deel(r)}. "
                "Trekt het tempo na het eerste stuk aan bij gelijke HR, dan "
                "kwamen de benen los.")

    afst = {row["sport"]: row.get("distance_m") for row in keten}
    afst[act.sport] = act.distance_m
    sim = race_similarity(afst, config)
    if sim:
        regels.append(f"- Race-simulatie: {sim}")

    historie = combo_history(conn, acts, load_records_fn)
    if not historie.empty:
        regels.append("Eerdere bevestigde combinatietrainingen (trend: worden "
                      "de wissels korter en de overgang soepeler?):")
        for _, h in historie.tail(5).iterrows():
            delen = [f"- {local_time(h['start_time']):%d-%m-%Y} ({h['kind']})"]
            if h["t1_s"] is not None:
                delen.append(f"T1 {fmt_duration(h['t1_s'])}")
            if h["t2_s"] is not None:
                delen.append(f"T2 {fmt_duration(h['t2_s'])}")
            if h["delta_tempo_s_per_km"] is not None:
                delen.append(
                    f"eerste stuk {h['delta_tempo_s_per_km']:+.0f} s/km "
                    "t.o.v. de rest van de loop")
            if h["delta_hr"] is not None:
                delen.append(f"HR {h['delta_hr']:+.0f}")
            regels.append(": ".join([delen[0], ", ".join(delen[1:])])
                          if len(delen) > 1 else delen[0])
    return "\n".join(regels)
