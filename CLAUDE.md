# Instructies voor Claude — Triatlon Coach

## App herstarten

Streamlit's runOnSave herlaadt alleen `app.py` zelf; wijzigingen in
`tricoach/*.py` (of andere modules) worden pas actief na een container-
herstart. **Herstart daarom altijd zelf** na het aanpassen van code, zonder
daar apart om te vragen:

    docker restart triatlon-coach

## Git: committen en pushen

**Commit wijzigingen altijd** met een beschrijvende boodschap (de "waarom",
niet alleen de "wat") zodra een taak/wijziging klaar is, en **push ze ook
meteen** naar `origin` — de remote is al SSH
(`git@github.com:thijslnl/TriatlonCoach.git`), dus dat is gewoon een gewone
`git push`. Vraag hier niet apart om bevestiging voor.

Blijft gelden: geen `--force`, geen `--amend` op al gepushte commits, en bij
destructieve operaties (reset --hard, branch verwijderen e.d.) wél eerst
overleggen — dit is alleen een pre-autorisatie voor het gewone
commit-en-push-ritme.
