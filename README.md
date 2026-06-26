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
   belangrijkste grafiek: tempo bij gelijke hartslag (Z2) over tijd.
3. **Coach** — pas het weekschema aan en genereer een weekadvies
   (Anthropic API; het laatste advies blijft bewaard).
4. **Chat** — stel vragen over je data. Standaard antwoordt het lokale
   Ollama-model; zet de toggle aan om de cloud-coach te vragen.
5. **Instellingen** — racedata (naam, datum), hartslagzones, Ollama-host
   en -model, en de LLM-routing per taak. Onderaan staat het verbruik:
   aanroepen, tokens en geschatte Anthropic-kosten (geparset uit
   memory/llm_log.md). Let op: opslaan herschrijft config.yaml.

Testen zonder dashboard kan ook:

```powershell
.venv\Scripts\python test_parse.py    # parse de zips in garmin_import/ (alleen lezen)
.venv\Scripts\python test_import.py   # importeer ze in SQLite + trainingslog
```

## Projectstructuur

```
app.py                  Streamlit-dashboard (UI)
config.yaml             zones, races, LLM-routing, paden
tricoach/
  fit_parser.py         zip/FIT -> geparste activiteiten (fitdecode, incl. GPS)
  zones.py              tijd-in-zones op basis van %LTHR
  storage.py            SQLite (activities, records, lengths)
  weather.py            winddata per sessie via Open-Meteo (gratis, geen key)
  importer.py           de import-pipeline (parse -> opslaan -> log)
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
garmin_import/          plek voor exportzips (test_import.py leest hieruit)
```

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
