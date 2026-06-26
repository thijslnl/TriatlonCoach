# Triatlon Coach — Streamlit-dashboard.
#
# Bewust minimaal: in de image zitten alléén de Python-dependencies. De
# applicatiecode (app.py, tricoach/, config.yaml, memory/, data/) wordt bij het
# draaien via een bind-mount ingehangen (zie docker-compose.yml). Daardoor is een
# codewijziging meteen actief en hoeft de image niet opnieuw gebouwd te worden —
# alleen bij een wijziging in requirements.txt is een rebuild nodig.

FROM python:3.12-slim

WORKDIR /app

# Dependencies apart installeren zodat deze laag in de cache blijft.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8501

# 0.0.0.0 zodat de app buiten de container bereikbaar is; runOnSave + poll-watcher
# zorgen dat een opgeslagen wijziging op schijf automatisch herladen wordt (poll
# is betrouwbaar op bind-mounts/netwerkshares waar inotify niet altijd afgaat).
CMD ["streamlit", "run", "app.py", \
     "--server.address=0.0.0.0", \
     "--server.port=8501", \
     "--server.headless=true", \
     "--server.runOnSave=true", \
     "--server.fileWatcherType=poll"]
