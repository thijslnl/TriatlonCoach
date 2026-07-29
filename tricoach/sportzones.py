"""Sport-afhankelijke drempels en zone-indeling.

Eén drempel per sport, niet uitwisselbaar — dat is het uitgangspunt sinds
juli 2026. Daarvóór draaide de hele app op één LTHR (171) voor álle sporten;
dat bleek te grof:

- **Hardlopen** — drempelhartslag (LTHR). Garmin schat die automatisch; de
  gebruikte waarde staat op de instellingenpagina en is handmatig te
  overschrijven. De zones volgen uit %LTHR (zie :mod:`tricoach.zones`).
- **Fietsen** — FTP in watt is de primaire maat zodra hij bekend is: de
  Coggan-vermogenszones (zie :mod:`tricoach.power`) zeggen bij fietsen meer
  dan hartslag, omdat HR traag reageert en meebeweegt met warmte en
  vermoeidheid. Zolang de FTP onbekend is — of een rit geen vermogensdata
  heeft (oude sessies, rit zonder Rally-pedalen) — vallen we terug op
  hartslagzones op de **fiets-LTHR**, die bij de meeste mensen 5–10 bpm
  lager ligt dan bij lopen.
- **Zwemmen** — geen drempel en geen zone-oordeel. Techniek is in opbouw en
  polshartslag onder water is onbetrouwbaar; er wordt op afstand, tempo per
  100 m en slagritme gestuurd (CSS staat er los naast als referentie).

Deze module is het ene punt waar "welke zone-indeling geldt voor deze sport,
en waarom" wordt beslist. Import, herberekening, de UI en de coach-prompt
lezen allemaal hiervandaan, zodat er nergens een tweede definitie ontstaat.

De drempels staan in ``config.yaml`` onder ``athlete.thresholds``::

    athlete:
      max_hr: 193
      zone_pct_lthr: [0.80, 0.89, 0.95, 1.00]
      thresholds:
        running:
          lthr: 164
          lthr_source: Garmin-schatting
        cycling:
          ftp: null
          lthr: 156

Configuraties van vóór die structuur (een platte ``athlete.lthr`` en
``athlete.ftp``) worden gewoon gelezen: elke accessor valt daarop terug, en
:func:`tricoach.config.normalize_athlete` zet ze bij het laden om.
"""

from dataclasses import dataclass

from tricoach.power import POWER_ZONE_NAMES, power_zone_bounds
from tricoach.zones import ZONE_NAMES, bounds_from_lthr

# De sporten die de app kent; alleen de eerste twee krijgen zones.
RUNNING = "running"
CYCLING = "cycling"
SWIMMING = "swimming"

# Terugval voor de loop-LTHR als er niets is ingesteld (Garmin-schatting
# juli 2026). Alleen een vangnet: normaal komt de waarde uit config.yaml.
DEFAULT_RUN_LTHR = 164

# De fiets-drempelhartslag ligt bij de meeste mensen 5–10 bpm lager dan bij
# lopen (minder spiermassa actief, geen verticale belasting). Zolang de
# gebruiker geen eigen fiets-LTHR heeft ingevuld, schatten we hem zo.
BIKE_LTHR_OFFSET = 8

# De drie manieren waarop een sessie beoordeeld kan worden.
METHOD_POWER = "power"   # %FTP, Coggan-vermogenszones (P1..P6)
METHOD_HR = "hr"         # %LTHR-hartslagzones (Z1..Z5)
METHOD_NONE = "none"     # geen zone-oordeel (zwemmen)


# ------------------------------------------------------------- accessors --

def _thresholds(athlete: dict, sport: str) -> dict:
    """Het drempelblok van één sport uit de athlete-config (leeg als het ontbreekt)."""
    return (athlete.get("thresholds") or {}).get(sport) or {}


def run_lthr(athlete: dict) -> int:
    """De drempelhartslag voor hardlopen (bpm).

    Leest ``thresholds.running.lthr``; valt terug op de oude platte
    ``athlete.lthr`` (configuraties van vóór de sport-afhankelijke drempels)
    en als laatste op :data:`DEFAULT_RUN_LTHR`.
    """
    waarde = _thresholds(athlete, RUNNING).get("lthr") or athlete.get("lthr")
    return int(waarde or DEFAULT_RUN_LTHR)


def run_lthr_source(athlete: dict) -> str | None:
    """Herkomst van de loop-LTHR ("Garmin-schatting", "handmatig"), of None.

    Puur documentatie voor de UI en de changelog: de gebruikte waarde is
    altijd die uit de config, ook als Garmin iets anders schat.
    """
    bron = _thresholds(athlete, RUNNING).get("lthr_source")
    return str(bron) if bron else None


def bike_lthr(athlete: dict) -> int:
    """De drempelhartslag voor fietsen (bpm).

    Terugval als er niets is ingesteld: de loop-LTHR min
    :data:`BIKE_LTHR_OFFSET`. Deze waarde is de secundaire maat voor fietsen —
    hij bepaalt de zones zolang er geen FTP is, en voor elke rit zonder
    vermogensdata.
    """
    waarde = _thresholds(athlete, CYCLING).get("lthr")
    return int(waarde) if waarde else run_lthr(athlete) - BIKE_LTHR_OFFSET


def bike_lthr_is_estimated(athlete: dict) -> bool:
    """Is de fiets-LTHR afgeleid van de loop-LTHR in plaats van ingesteld?"""
    return not _thresholds(athlete, CYCLING).get("lthr")


def ftp(athlete: dict) -> float | None:
    """De FTP (watt) voor fietsen, of None zolang die onbekend is.

    Leest ``thresholds.cycling.ftp``; valt terug op de oude platte
    ``athlete.ftp``. Een 0 of lege waarde geldt als "nog onbekend".
    """
    waarde = _thresholds(athlete, CYCLING).get("ftp")
    if waarde is None:
        waarde = athlete.get("ftp")
    return float(waarde) if waarde else None


def zone_pcts(athlete: dict) -> list[float] | None:
    """De %LTHR-fracties voor de hartslagzones (None = de standaardindeling)."""
    return athlete.get("zone_pct_lthr")


def lthr_for_sport(athlete: dict, sport: str) -> int | None:
    """De drempelhartslag die voor deze sport geldt; None bij zwemmen."""
    if sport == RUNNING:
        return run_lthr(athlete)
    if sport == CYCLING:
        return bike_lthr(athlete)
    return None


def hr_zone_bounds(athlete: dict, sport: str) -> list[int] | None:
    """De hartslagzonegrenzen (ondergrens Z2..Z5) voor één sport.

    ``None`` voor zwemmen: daar wordt bewust géén zone-oordeel geveld, dus er
    zijn ook geen grenzen. Voor lopen en fietsen komen de grenzen uit de eigen
    drempel van die sport en de gedeelde %LTHR-fracties.
    """
    drempel = lthr_for_sport(athlete, sport)
    if drempel is None:
        return None
    return bounds_from_lthr(drempel, zone_pcts(athlete))


def zone2_range(athlete: dict, sport: str) -> tuple[int, int] | None:
    """Het zone 2-bereik (onder-, bovengrens in bpm) voor één sport, of None.

    Dit is het bereik waarop de "tempo bij gelijke hartslag"-trend rekent —
    de belangrijkste voortgangsmaat voor lopen en fietsen.
    """
    bounds = hr_zone_bounds(athlete, sport)
    if bounds is None:
        return None
    return bounds[0], bounds[1] - 1


# ------------------------------------------------------------ zone-model --

@dataclass(frozen=True)
class ZoneModel:
    """Waarop wordt één sessie (of één sport) beoordeeld?

    Bundelt de methode, de gebruikte drempel en de zonegrenzen, zodat de UI,
    de herberekening en de coach-prompt allemaal hetzelfde verhaal vertellen.

    ``provisional`` markeert de tussenoplossing: fietsen op hartslagzones
    terwijl vermogen eigenlijk leidend hoort te zijn (FTP nog niet bekend).
    ``reason`` legt in één zin uit waarom deze indeling geldt.
    """

    sport: str
    method: str                  # METHOD_POWER / METHOD_HR / METHOD_NONE
    threshold: float | None      # FTP in W, of LTHR in bpm
    unit: str                    # "W", "bpm" of ""
    bounds: list[float]          # ondergrenzen Z2..Z5, of bovengrenzen P1..P5
    names: list[str]             # ZONE_NAMES of POWER_ZONE_NAMES
    provisional: bool = False
    reason: str = ""

    @property
    def has_zones(self) -> bool:
        """Krijgt deze sport/sessie überhaupt een zone-oordeel?"""
        return self.method != METHOD_NONE

    @property
    def method_label(self) -> str:
        """Korte omschrijving van de methode, voor labels en kopjes."""
        if self.method == METHOD_POWER:
            return "vermogen (%FTP, Coggan)"
        if self.method == METHOD_HR:
            return "hartslag (%LTHR)"
        return "geen zones"

    @property
    def threshold_label(self) -> str:
        """De gebruikte drempel als leesbare tekst, bijv. "FTP 240 W"."""
        if self.threshold is None:
            return "geen drempel"
        naam = "FTP" if self.method == METHOD_POWER else "LTHR"
        return f"{naam} {self.threshold:.0f} {self.unit}"

    def bounds_text(self) -> str:
        """De zonegrenzen als één regel, in de eenheid van de methode."""
        if self.method == METHOD_NONE:
            return "—"
        b = [round(x) for x in self.bounds]
        if self.method == METHOD_POWER:
            # Coggan-grenzen zijn bovengrenzen: alles boven de laatste is P6.
            delen = [f"{self.names[0]} <{b[0]}"]
            delen += [f"{self.names[i + 1]} {b[i]}–{b[i + 1]}"
                      for i in range(len(b) - 1)]
            delen.append(f"{self.names[-1]} >{b[-1]}")
            return " · ".join(delen) + f" {self.unit}"
        # HR-grenzen zijn ondergrenzen: alles onder de eerste is Z1.
        return (f"{self.names[1]} {b[0]}–{b[1] - 1} · "
                f"{self.names[2]} {b[1]}–{b[2] - 1} · "
                f"{self.names[3]} {b[2]}–{b[3] - 1} · "
                f"{self.names[4]} {b[3]}+")

    def summary(self) -> str:
        """Eén regel die methode, drempel, grenzen en kanttekening samenvat."""
        if self.method == METHOD_NONE:
            return f"geen zone-oordeel — {self.reason}"
        regel = f"{self.method_label}, {self.threshold_label} → {self.bounds_text()}"
        if self.reason:
            regel += f" ({self.reason})"
        return regel


def zone_model(athlete: dict, sport: str, has_power: bool = True) -> ZoneModel:
    """De zone-indeling die voor deze sport (en sessie) geldt.

    ``has_power`` zegt of de betreffende rit vermogensdata heeft; zet het op
    False voor een concrete rit zonder power (oude sessie, rit zonder
    Rally-pedalen). Voor een algemene beschrijving van "hoe beoordelen we
    fietsen?" laat je het op True staan.

    De regels:

    - **zwemmen** → geen zones, altijd;
    - **hardlopen** → hartslagzones op de loop-LTHR;
    - **fietsen met FTP én vermogensdata** → Coggan-vermogenszones (primair);
    - **fietsen zonder FTP** → hartslagzones op de fiets-LTHR, gemarkeerd als
      tussenoplossing;
    - **fietsen met FTP maar zonder vermogensdata** → hartslagzones op de
      fiets-LTHR, zonder "tussenoplossing"-melding (de rit mist de data, niet
      de instelling).
    """
    if sport == SWIMMING:
        return ZoneModel(
            sport=sport, method=METHOD_NONE, threshold=None, unit="",
            bounds=[], names=[],
            reason="bij zwemmen sturen we op techniek, afstand en tempo per "
                   "100 m; polshartslag onder water is onbetrouwbaar",
        )

    if sport == CYCLING:
        watt = ftp(athlete)
        if watt and has_power:
            return ZoneModel(
                sport=sport, method=METHOD_POWER, threshold=watt, unit="W",
                bounds=power_zone_bounds(watt), names=list(POWER_ZONE_NAMES),
                reason="vermogen is bij fietsen de primaire intensiteitsmaat",
            )
        hr = bike_lthr(athlete)
        bounds = bounds_from_lthr(hr, zone_pcts(athlete))
        if watt:
            reden = ("deze rit heeft geen vermogensdata, dus hartslagzones op "
                     "de fiets-LTHR")
        else:
            reden = ("tussenoplossing: FTP nog onbekend, dus voorlopig "
                     "hartslagzones op de fiets-LTHR")
        return ZoneModel(
            sport=sport, method=METHOD_HR, threshold=hr, unit="bpm",
            bounds=list(bounds), names=list(ZONE_NAMES),
            provisional=not watt, reason=reden,
        )

    hr = run_lthr(athlete)
    return ZoneModel(
        sport=sport, method=METHOD_HR, threshold=hr, unit="bpm",
        bounds=list(bounds_from_lthr(hr, zone_pcts(athlete))),
        names=list(ZONE_NAMES),
        reason="hartslag is bij hardlopen de leidende intensiteitsmaat",
    )


def zone_overview(athlete: dict) -> list[ZoneModel]:
    """De actuele zone-indeling per sport, voor een overzicht in de UI/prompt."""
    return [zone_model(athlete, sport) for sport in (RUNNING, CYCLING, SWIMMING)]


def zone_overview_text(athlete: dict) -> str:
    """Het zone-overzicht per sport als markdown-lijst (UI en coach-prompt)."""
    from tricoach.formatting import sport_label  # hier tegen een circulaire import

    return "\n".join(
        f"- **{sport_label(m.sport)}:** {m.summary()}" for m in zone_overview(athlete)
    )


# ------------------------------------------------------------- schrijven --

def set_thresholds(athlete: dict, run_lthr_value: int, bike_lthr_value: int,
                   ftp_value: float | None,
                   run_lthr_source_value: str | None = None) -> dict:
    """Schrijf de drempels in de genormaliseerde structuur terug in ``athlete``.

    Wist meteen de oude platte ``lthr``/``ftp``-velden, zodat er maar één bron
    van waarheid overblijft. Geeft hetzelfde (aangepaste) dict terug.
    """
    thresholds = athlete.setdefault("thresholds", {})
    running = thresholds.setdefault(RUNNING, {})
    running["lthr"] = int(run_lthr_value)
    if run_lthr_source_value:
        running["lthr_source"] = run_lthr_source_value
    cycling = thresholds.setdefault(CYCLING, {})
    cycling["lthr"] = int(bike_lthr_value)
    cycling["ftp"] = float(ftp_value) if ftp_value else None
    athlete.pop("lthr", None)
    athlete.pop("ftp", None)
    return athlete
