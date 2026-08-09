# Triatlon Training Dashboard met AI-coach

Lokaal Streamlit-dashboard dat Garmin-trainingen (FIT-bestanden) beheert,
trends toont en trainingsadvies geeft richting de standaard (olympische)
triatlon van mei 2027. Eenvoudige LLM-taken draaien gratis op een lokaal
Ollama-model; het echte coachwerk gaat naar de Anthropic API.

## Installatie

```powershell
# 1. Virtuele omgeving (eenmalig)
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt

# 2. Anthropic API-key (nodig voor advies en cloud-chat; de rest werkt zonder)
$env:ANTHROPIC_API_KEY = "sk-ant-..."
```

Voor de Garmin-koppeling (wellness-data + automatische activiteiten-sync)
komen er twee regels bij in `.env` (blijft buiten git):

```
GARMIN_EMAIL=jij@example.com
GARMIN_PASSWORD=...
```

De koppeling gebruikt de onofficiële Connect-API (python-garminconnect):
eenmalig inloggen via de sync-knop (MFA wordt in de zijbalk afgehandeld),
daarna werken de bewaarde tokens in `data/garmin_tokens/` maandenlang. Een
mislukte sync breekt het dashboard nooit — je krijgt een melding en de
bestaande data blijft gewoon staan.

Controleer in [config.yaml](config.yaml) of het Ollama-adres en de modeltag
kloppen (`llm.ollama.host` en `llm.ollama.model`, check met `ollama list`).

## Starten

```powershell
.venv\Scripts\streamlit run app.py
```

## Gebruik

1. **Upload** — exporteer een activiteit uit Garmin Connect ("Origineel
   exporteren"), kies de zip in de zijbalk, typ eventueel een **opmerking**
   (bijv. "voelde me moe", "intervaltraining bedoeld") en klik op **Uploaden en
   analyseren** — pas dán wordt er ingelezen, opgeslagen en geanalyseerd. Dubbel
   uploaden is veilig (deduplicatie op starttijd uit het FIT-bestand). Voor
   fiets- en loopsessies met GPS wordt automatisch de wind tijdens de rit
   opgehaald (Open-Meteo, gratis) als objectieve context voor de feedback;
   sessies zonder GPS (zwemmen) gaan gewoon zonder wind door.
2. **Overzicht / Trends / sporttabs** — weekvolume, tijd-in-zones en de
   belangrijkste grafiek: tempo bij gelijke hartslag (Z2) over tijd. Elke
   sport wordt tegen zijn eigen drempel gelezen — zie hieronder.
3. **Garmin-sync** (zijbalk) — haalt met één klik de dagelijkse
   wellness-data (rustpols, HRV, slaap, body battery, stress, VO2 max,
   training readiness) én nieuwe activiteiten als origineel FIT-bestand
   binnen. Activiteiten doorlopen exact dezelfde import-pipeline als een
   handmatige upload (archief, trainingslog, feedback) en worden dubbel
   gededupliceerd: op Garmin activity-ID vóór het downloaden en op de
   starttijd-sleutel in de pipeline — handmatig geüploade én verwijderde
   sessies komen dus nooit dubbel of stilletjes terug binnen. Optioneel
   synct de app automatisch bij het openen (instellingen-tab). De
   **🌙 Herstel-tab** toont de trends (7-daags gemiddelde is de maat) en de
   kruising met je trainingsbelasting; de coach-feedback krijgt de recente
   herstelcontext mee als context, niet als oordeel.
4. **Voeding** — vul in wat je gaat doen (sport, afstand óf duur, intensiteit,
   temperatuur, welke producten je bij je hebt) en krijg een concreet
   voedingsplan terug: een tijdlijn met wat je wanneer neemt, een meeneemlijst
   en waarschuwingen waar de constraints knellen. Zie
   [de voedingsplanner](#de-voedingsplanner) hieronder.
5. **Coach** — pas het weekschema aan en genereer een weekadvies
   (Anthropic API; het laatste advies blijft bewaard).
6. **Chat** — stel vragen over je data. Standaard antwoordt het lokale
   Ollama-model; zet de toggle aan om de cloud-coach te vragen.
7. **Heatmap** — voor de lol: alle GPS-tracks die je ooit hebt gefietst,
   gelopen en open water gezwommen op één donkere kaart (CartoDB Dark Matter),
   feller waar je vaker langskwam. Filterbaar op sport en periode, met een
   **privacyzone** rond je huisadres die standaard aan staat. Zie
   [de heatmap](#de-heatmap) hieronder — met name waarom de dichtheid over
   afstand en niet over tijd wordt geteld.
8. **Instellingen** — racedata (naam, datum), de **drempels per sport**
   (loop-LTHR, FTP en fiets-LTHR), Ollama-host
   en -model, en de LLM-routing per taak. Onderaan staat het verbruik:
   aanroepen, tokens en geschatte Anthropic-kosten (geparset uit
   memory/llm_log.md). Let op: opslaan herschrijft config.yaml.

Testen zonder dashboard kan ook:

```powershell
.venv\Scripts\python tests\test_parse.py    # parse de zips in garmin_import/ (alleen lezen)
.venv\Scripts\python tests\test_import.py   # importeer ze in SQLite + trainingslog
.venv\Scripts\python tests\test_voeding.py  # voedingsplanner (op verzonnen data)
```

## Projectstructuur

```
app.py                  Streamlit-dashboard (UI)
config.yaml             drempels per sport, races, LLM-routing, paden
tricoach/
  fit_parser.py         zip/FIT -> geparste activiteiten (fitdecode, incl. GPS)
  sportzones.py         drempels en zone-indeling PER SPORT (het ene beslispunt)
  zones.py              tijd-in-hartslagzones op basis van %LTHR
  ramptest.py           FTP-test herkennen + FTP-voorstel (Kickr/Zwift)
  migrate_sportzones.py eenmalige herberekening naar de drempels per sport
  storage.py            SQLite (activities, records, lengths)
  wellness.py           dagelijkse wellness-data (rustpols, HRV, slaap, ...)
  garmin_sync.py        Garmin Connect-sync: wellness + activiteiten (dedup
                        op activity-ID én starttijd; MFA; zacht falen)
  weather.py            winddata per sessie via Open-Meteo (gratis, geen key)
  importer.py           de import-pipeline (parse -> archief -> opslaan -> log)
  archive.py            origineel-archief van uploads (uploads/yyyy/mm/, versies)
  transport.py          transport-markering (ritje naar het zwembad ≠ training)
  heatmap.py            GPS-heatmap: extractie uit het FIT-archief, herbemonstering
                        op vaste afstand, dichtheid per rastercel, privacyzone
  nutrition/            voedingsplanner (🥤 Voeding-tab), volledig deterministisch
    rules.py            de richtlijnen als benoemde constanten + pure functies
    products.py         de bewerkbare productdatabase (nutrition_products)
    duration.py         duurschatting uit de eigen sessies, als bandbreedte
    plan.py             de planner: tijdlijn, meeneemlijst, waarschuwingen
    store.py            opgeslagen plannen + terugkoppeling (tolerantie)
    explain.py          optionele toelichting van 2-3 zinnen (LLM, geen cijfers)
  trainingslog.py       markdown-entries in memory/trainingslog.md
  analysis.py           weekvolumes, zonetijden, tempo-bij-HR-trends
  schedule.py           aanpasbaar weekschema (memory/weekschema.md)
  advice.py             weekadvies via Anthropic, vastgelegd in adviezen.md
  chat.py               Q&A met routing (Ollama eerst, escalatie naar API)
  llm/                  router, Ollama-client, Anthropic-client, llm_log
memory/                 het leesbare geheugen van de tool (markdown)
  doelen.md             racedoelen en voorkeuren (uit het intakegesprek)
  weekschema.md         het geplande trainingsritme (aanpasbaar in de app)
  trainingslog.md       elke sessie: kerncijfers + observatie
  adviezen.md           elk gegeven advies, met datum en onderbouwing
  llm_log.md            álle LLM-communicatie (model, prompt, antwoord, tokens)
  externe_data_log.md   elke Open-Meteo-windaanroep (locatie, uur, windwaarden)
  inzichten.md          langetermijnpatronen
  beslissingen.md       architectuurkeuzes en waarom
data/training.db        SQLite met de ruwe sessie- en seconde-data, de
                        wellness-dagen, de bewaarde coach-feedback per sessie
                        (session_feedback), de voedingsproducten en -plannen
                        (nutrition_products, nutrition_plans,
                        nutrition_feedback) en de sync-status
data/garmin_tokens/     OAuth-tokens van de Garmin-login (buiten git)
data/heatmap_privacy.json  middelpunt en straal van de privacyzone van de
                        heatmap — bewust hier en niet in config.yaml, want dat
                        staat in versiebeheer en het middelpunt is je huisadres
uploads/                onveranderlijk archief van elk geüpload FIT-origineel
                        (uploads/yyyy/mm/yyyy-mm-dd_HHmm_<activityid>.fit;
                        heruploads met andere inhoud worden _v2, _v3, ...)
garmin_import/          plek voor exportzips (tests/test_import.py en de
                        archief-inhaalslag op de instellingen-tab lezen hieruit)
tests/                  losse testscripts (python tests/test_<naam>.py, draaien
                        op tijdelijke data en raken de echte database niet aan;
                        test_parse/test_import/test_uitbreiding lezen wél
                        garmin_import/ en verwachten de projectroot als werkmap)
```

## De voedingsplanner

Vooraf invoeren wat je gaat doen, terugkrijgen hoeveel koolhydraten je wanneer
uit welk product haalt en wat er mee moet ([tricoach/nutrition/](tricoach/nutrition/),
tab **🥤 Voeding**).

> Dit is een startpunt op basis van algemene sportvoedingsrichtlijnen, bedoeld
> om in training te testen — geen voedingsadvies. Bij twijfel of maagklachten:
> raadpleeg een sportdiëtist. Diezelfde regel staat boven de tab.

**Het algoritme rekent; het taalmodel legt hooguit uit.** Elk getal — duur,
gram per uur, timing, aantallen — komt uit Python, en elke drempelwaarde is een
benoemde constante in [rules.py](tricoach/nutrition/rules.py). Dezelfde invoer
geeft altijd hetzelfde plan. Wil je er een toelichting van 2-3 zinnen bij, dan
kan dat met één klik (taak `nutrition_note`, standaard het lokale Ollama-model)
— maar dat model krijgt het al berekende plan en mag er geen cijfer aan
toevoegen.

**De duur komt uit je eigen sessies, niet uit een tabel.** Vergelijkbare
sessies worden gezocht op het vermogensvenster (fietsen, %FTP) of het
hartslagvenster (lopen), gecorrigeerd voor temperatuur en met Riegel naar de
doelafstand geschaald. Je krijgt altijd een **bereik** ("3:05-3:25") met de
onderbouwing eronder, en je kunt het overschrijven. Bij fietsen is de band het
breedst: wind is daar de dominante onzekerheid en die zit niet in je historie.
Hoeveel tempo warmte je kost, wordt uit je eigen sessies geschat; is daar te
weinig data voor, dan valt het terug op 0,4% per graad boven 18 °C en zegt de
app dat erbij.

**Twee harde constraints, in deze volgorde.** Ze kunnen elkaar tegenspreken,
dus de volgorde is een keuze:

| # | Constraint | Wat de planner doet |
|---|---|---|
| A | **Single-source limiet.** Glucose/maltodextrine gaat via één darmtransporter en loopt vast op ~60 g/uur. | Bevat je selectie alleen single-source producten, dan wordt het advies afgetopt op 60 g/uur, mét de melding een dual-source product (glucose + fructose, ~1:0,8) toe te voegen. Daarmee kan het plan naar 90 g/uur, bij een getrainde darm tot 120. |
| B | **Concentratielimiet.** Sportdrank is isotoon rond 6-8% koolhydraten; meer in dezelfde bidon maakt hem hypertoon en vertraagt juist de opname. | De planner vult eerst zoveel mogelijk uit drank (koolhydraten, vocht én natrium in één keer) maar niet verder dan 8% van het **totale** geplande vochtvolume; de rest verschuift automatisch naar gels. Is er geen gel geselecteerd, dan meldt het plan dat het doel niet gehaald wordt in plaats van de bidon stiekem te verzwaren. |

Dat de check op het *totale* volume staat en niet per product is bewust: een
geconcentreerd sachet (Beta Fuel, 80 g per 500 ml) is prima zodra je het met
extra water aanvult — zo is het ook bedoeld.

**Onbekende bron telt als single-source.** De Lidl HealthyFit noemt vier
suikers maar geen verhouding. Zo'n product staat als `onbekend` in de database
en wordt conservatief als single-source gerekend: een te laag plafond kost wat
prestatie, een te hoog plafond kost je je race.

**Vocht, natrium en cafeïne.** Vocht loopt van 400 ml/uur bij koud weer naar
800 ml/uur vanaf 30 °C; natrium van 300 mg/l naar 600 mg/l rond 25 °C, en
alleen bij hitte én langer dan drie uur tot ~1000 mg/l. De app rekent uit
hoeveel natrium jouw producten leveren en meldt een tekort. Cafeïnegels worden
bewust in de tweede helft gepland en nooit boven 3 mg per kg lichaamsgewicht
(dat gewicht komt uit je laatste meting op de Lichaam-tab).

**Timing.** Eerste inname na ~30 minuten — niet meteen bij de start — daarna
elke 20-30 minuten, en bij een brick of triatlon zijn T1 en T2 vaste
eetmomenten: dat zijn de rustigste momenten van de race. Tijdens het zwemmen
kun je niets innemen, dus de behoefte wordt over de fiets, de loop en de
wissels verdeeld.

**Terugkoppeling is het waardevolste deel.** Sla een plan op bij een geplande
sessie en vul achteraf in wat er werkelijk in ging en hoe je maag reageerde
(goed / licht ongemak / klachten). Na een paar sessies staat er zwart op wit
wat *jouw* darm aankan — "100 g/uur viel goed op 9 augustus" weegt zwaarder dan
welke algemene richtlijn ook. De opgeslagen plannen bewaren de berekening als
JSON, zodat een later gewijzigd product niet met terugwerkende kracht verandert
wat er ooit gepland was.

De onderbouwing van alle drempelwaarden staat in
[memory/beslissingen.md](memory/beslissingen.md); de verificatiescenario's
staan in `tests/test_voeding.py`.

## De heatmap

Een eigen Strava-achtige heatmap van alle GPS-tracks
([tricoach/heatmap.py](tricoach/heatmap.py), tab **🗺️ Heatmap**). Geen
trainingsanalyse — transport-ritjes tellen hier dus mee, want het gaat om waar
je komt, niet om de trainingsprikkel.

**Lijndichtheid, niet puntdichtheid.** Dit is de hele truc. Het horloge logt
per seconde, dus waar je langzaam gaat — stoplichten, een klim, wachten bij een
oversteek — stapelen de punten zich op. Een heatmap die ruwe GPS-punten telt
licht daar fel op terwijl je er niet vaker was: op de eigen data komt zo'n
enkele stilstand op **320 punten in één rastercel** uit, terwijl een tien keer
gereden route op 10 komt. Het stoplicht zou dus feller zijn dan de dagelijkse
woon-werkroute — precies verkeerd. Daarom wordt elke track vóór het tellen
**herbemonsterd naar punten op vaste afstand** (elke 10 m, met interpolatie
tussen de gelogde punten). Elke gereden meter weegt dan even zwaar; tien
minuten stilstaan levert één punt op. Sprongen groter dan 500 m (pauze,
autorit) worden niet overbrugd, zodat er geen kaarsrechte spooklijnen ontstaan.

Daarna wordt per rastercel (standaard 20 m) het aantal **passages** geteld:
opeenvolgende punten in dezelfde cel binnen één track gelden als één passage,
een latere terugkomst als een nieuwe. Daarmee maakt ook de hoek waaronder je
een cel kruist niet meer uit.

**Kleurschaal.** Van dof donkerrood (één passage) via oranje en geel naar wit,
en nooit lineair — één dagelijkse route zou anders al het andere tot vlak boven
zwart platdrukken. Twee schalen, die een andere vraag beantwoorden:

| Schaal | Wat je ziet |
|---|---|
| logaritmisch (standaard) | de verhoudingen tussen aantallen blijven herkenbaar; een route van tien keer is duidelijk feller dan een van twee keer |
| percentiel | álles wat je meer dan één keer deed springt ver naar boven; antwoordt op "waar kom ik vaker dan eens", ten koste van het onderscheid in de top |

**Cache.** De coördinaten komen uit de originele FIT-bestanden in `uploads/`
(de `records`-tabel bewaart geen posities). FIT slaat posities op in
*semicircles*: `graden = semicircles × 180 / 2³¹`. De herbemonsterde punten
gaan in `track_points`, met per activiteit een regel in `track_extract` — ook
voor sessies **zonder** GPS, zodat een zwembadsessie of Zwift-rit niet bij elke
render opnieuw wordt opengetrokken. Elk FIT-bestand wordt dus één keer geparst;
bij het openen van de tab worden alleen nieuwe activiteiten bijgewerkt.

**Beginbeeld.** De kaart opent op het gebied waar minstens 90% van je
rastercellen ligt, niet op de volledige bounding box — een enkel hardloopje in
het buitenland zou de kaart anders naar landniveau uitzoomen en je eigen
omgeving tot een vlekje maken. Die verre routes staan er nog steeds: ze vallen
alleen buiten het startbeeld, en uitzoomen brengt ze terug (de UI meldt hoeveel
cellen dat zijn). Symmetrisch quantielen afknippen werkt daar slecht — ligt een
verre groep net boven de toegestane fractie, dan schuift de grens er middenin en
blijft de kaart even ver uitgezoomd. Daarom trekt `view_bounds` per stap de rand
in die de meeste kaartbreedte per opgegeven cel oplevert: een compacte verre
groep verdwijnt zo in één keer, terwijl een gelijkmatig uitgesmeerde spreiding
grotendeels in beeld blijft.

**Privacy.** Tracks beginnen en eindigen bij de voordeur. De privacyzone
(instelbaar middelpunt, standaard 400 m straal) laat punten daarbinnen weg en
staat standaard aan; de routes lopen gewoon door tot de rand van de zone. Het
middelpunt wordt opgeslagen in `data/heatmap_privacy.json` en **niet** in
config.yaml: dat laatste staat in versiebeheer, en het middelpunt *is* het
huisadres. Is er nog niets opgeslagen, dan gebruikt de zone een schatting uit
de mediaan van alle startpunten.

De kaartachtergrond is CartoDB Dark Matter (gratis, geen API-key) via pydeck;
de vereiste bronvermelding — OpenStreetMap-contributors en CARTO — staat onder
de kaart in de UI.

## Drempels en zones per sport

Elke sport heeft een **eigen drempel**; ze zijn niet uitwisselbaar. Alles wordt
beslist in [tricoach/sportzones.py](tricoach/sportzones.py), zodat import,
herberekening, UI en coach-prompt nooit uit elkaar lopen.

| Sport | Primaire maat | Zone-indeling |
|---|---|---|
| Hardlopen | hartslag — **loop-LTHR** | %LTHR (Z1–Z5) |
| Fietsen | **vermogen — FTP** zodra bekend | %FTP, Coggan (P1–P6) |
| Fietsen zonder FTP of zonder powerdata | hartslag — **fiets-LTHR** | %LTHR, gemeld als *tussenoplossing* |
| Zwemmen | — | **geen zone-oordeel** (afstand, tempo/100 m, slagritme, SWOLF; CSS als losse referentie) |

De fiets-LTHR ligt standaard 8 bpm onder de loop-LTHR (gebruikelijk is 5–10) en
is op de instellingenpagina te overschrijven. De maximale hartslag blijft een
los profielveld: %max is alleen een terugval, %drempel is leidend.

Wijzig je een drempel op de instellingen-tab, dan worden de zonetijden van álle
sessies herrekend (elke sessie tegen de drempel van háár eigen sport) en komt de
wijziging in `memory/lthr_geschiedenis.md` en de changelog van
`memory/doelen.md`. Handmatig herrekenen kan ook:

```powershell
.venv\Scripts\python -m tricoach.migrate_sportzones --dry-run  # tonen
.venv\Scripts\python -m tricoach.migrate_sportzones            # uitvoeren
```

**FTP bepalen.** Rijd je een ramptest op de Kickr (indoor, oplopend blokkig
vermogen), dan herkent de 🔍 Sessie-tab die en stelt hij een FTP voor: 75% van
je beste minuut. Bij een 20-minuten-veldtest is het 95% van dat vermogen. Het
voorstel wordt nooit automatisch opgeslagen — je bevestigt het met één klik,
waarna de vermogenszones van alle ritten worden herrekend en de vaststelling
(datum + methode) in `memory/inzichten.md` belandt.

> ⚠️ **Trendbreuk juli 2026 — alleen voor fietsen en zwemmen.** Fietsen wordt nu
> tegen de eigen, lagere fiets-LTHR gelezen (zone 2-plafond 150 → 145), dus
> oudere fietscijfers lagen optisch gunstiger. Zwemmen heeft geen zonecijfers
> meer. Voor hardlopen is er géén breuk: die drempel is en blijft 170. Alle
> historie is per sport herrekend; de onderbouwing staat in
> `memory/beslissingen.md`.

> 🩺 **Garmin-drempeldip na een brick is een artefact.** Rijd je een brick
> (fietsen, dan lopen), dan liggen je loophartslagen bij hetzelfde tempo hoger
> doordat de benen voorbelast zijn — Garmin leest dat als een lágere
> lactaatdrempel en stelt zijn schatting tijdelijk naar beneden bij. Neem die
> dip niet over: alleen een verschuiving die terugkomt over meerdere **losse**
> loopsessies is een echt signaal. De coach-prompts weten dit en trekken er
> geen conclusie uit.

## Principes

- **Memory by design**: elke interpretatie, beslissing en advies staat in
  leesbare markdown onder `memory/`; alleen ruwe meetdata staat in SQLite.
- **Lokaal eerst**: Ollama (gratis, onbeperkt) voor samenvattingen en
  eenvoudige vragen; Anthropic (`claude-sonnet-4-6`) alleen voor advies,
  trends en geëscaleerde vragen. Routing is config (`llm.routing`).
- **Zuinig met API-calls**: adviezen worden gecachet; er gaat nooit een
  request uit bij een gewone page-load.
- **Privacy**: er gaan geen GPS-routes of persoonsgegevens naar de
  Anthropic-API; de API-key komt uitsluitend uit de environment variable
  `ANTHROPIC_API_KEY`. Voor de winddata gaat alléén de startcoördinaat + dag
  naar Open-Meteo (een aparte, gratis weerdienst); de coach krijgt enkel de
  afgeleide windregel, geen route. GPS-coördinaten worden niet in de database
  bewaard. Bronvermelding "Weather data by Open-Meteo.com" (CC BY 4.0) staat in
  de zijbalk.
