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
3. **Coach** — pas het weekschema aan en genereer een weekadvies
   (Anthropic API; het laatste advies blijft bewaard).
4. **Chat** — stel vragen over je data. Standaard antwoordt het lokale
   Ollama-model; zet de toggle aan om de cloud-coach te vragen.
5. **Instellingen** — racedata (naam, datum), de **drempels per sport**
   (loop-LTHR, FTP en fiets-LTHR), Ollama-host
   en -model, en de LLM-routing per taak. Onderaan staat het verbruik:
   aanroepen, tokens en geschatte Anthropic-kosten (geparset uit
   memory/llm_log.md). Let op: opslaan herschrijft config.yaml.

Testen zonder dashboard kan ook:

```powershell
.venv\Scripts\python tests\test_parse.py    # parse de zips in garmin_import/ (alleen lezen)
.venv\Scripts\python tests\test_import.py   # importeer ze in SQLite + trainingslog
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
  weather.py            winddata per sessie via Open-Meteo (gratis, geen key)
  importer.py           de import-pipeline (parse -> archief -> opslaan -> log)
  archive.py            origineel-archief van uploads (uploads/yyyy/mm/, versies)
  transport.py          transport-markering (ritje naar het zwembad ≠ training)
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
data/training.db        SQLite met de ruwe sessie- en seconde-data
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
