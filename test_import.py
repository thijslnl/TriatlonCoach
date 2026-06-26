"""Testscript voor stap 3: importeer alle zips uit garmin_import/ in SQLite + trainingslog.

Gebruik:  python test_import.py
Twee keer draaien is veilig: de tweede keer is alles 'duplicaat'.
"""

from tricoach.config import load_config, resolve_path
from tricoach.formatting import sport_label
from tricoach.importer import import_zip
from tricoach.llm import LLMRouter
from tricoach.llm.observations import session_observation
from tricoach.storage import connect, load_activities


def main() -> None:
    config = load_config()
    conn = connect(resolve_path(config, "database"))
    memory_dir = resolve_path(config, "memory_dir")
    import_dir = resolve_path(config, "import_dir")
    router = LLMRouter(config, memory_dir)

    def observe(act, tiz):
        return session_observation(router, act, tiz)

    for zip_path in sorted(import_dir.glob("*.zip")):
        for r in import_zip(zip_path, conn, config, memory_dir, observation_fn=observe):
            a = r.activity
            print(f"[{r.status:9}] {a.start_time:%Y-%m-%d %H:%M}  "
                  f"{sport_label(a.sport):10} {a.distance_m/1000:6.2f} km  ({zip_path.name})")

    df = load_activities(conn)
    print(f"\nDatabase bevat nu {len(df)} activiteiten:")
    print(df[["start_time", "sport", "distance_m", "avg_hr"]].to_string(index=False))


if __name__ == "__main__":
    main()
