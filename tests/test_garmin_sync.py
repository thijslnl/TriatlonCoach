"""Testscript voor de activiteiten-sync en de dedup (tricoach.garmin_sync).

Gebruik:  python tests/test_garmin_sync.py   (vanuit de projectroot)

Gebruikt een échte exportzip uit garmin_import/ (alleen lezen) met een fake
Garmin-client, op een tijdelijke database en een tijdelijke memory-map — de
echte data blijft onaangeroerd. Controleert de verificatie-eisen van de
sync-uitbreiding:

1. eerste sync: activiteit wordt gedownload, geïmporteerd, gearchiveerd en
   de Garmin activity-ID wordt geregistreerd;
2. tweede sync: geen download meer (dedup op activity-ID vóór het downloaden);
3. handmatige upload eerst, dan sync: ensure_garmin_ids herkent de ID uit de
   bestandsnaam en de sync slaat de activiteit over;
4. soft-verwijderde sessie: komt via de sync niet stilletjes terug;
5. verkeerde credentials / geen tokens: connect_client geeft een nette
   GarminSyncError (zacht falen), geen crash.
"""

# De tests staan in tests/; zet de projectroot op sys.path zodat
# `python tests/test_<naam>.py` het tricoach-package kan importeren.
import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))


import os
import shutil
import tempfile
from pathlib import Path

from tricoach import garmin_sync
from tricoach.config import PROJECT_ROOT, load_config
from tricoach.importer import import_zip
from tricoach.storage import connect, is_deleted, load_activities, soft_delete_activity

IMPORT_DIR = PROJECT_ROOT / "garmin_import"


class FakeActivityClient:
    """Serveert één echte exportzip alsof hij van de Garmin-API komt."""

    def __init__(self, aid: str, zip_bytes: bytes):
        self.aid = aid
        self.zip_bytes = zip_bytes
        self.downloads = 0

    def get_activities_by_date(self, start, eind=None, *args, **kwargs):
        return [{"activityId": int(self.aid)}]

    def download_activity(self, aid, dl_fmt=None):
        self.downloads += 1
        return self.zip_bytes


def main() -> None:
    zips = sorted(IMPORT_DIR.glob("*.zip"))
    assert zips, "geen exportzips in garmin_import/ — deze test heeft er één nodig"
    zip_path = zips[0]
    aid = zip_path.stem  # Garmin-zips heten <activityid>.zip
    zip_bytes = zip_path.read_bytes()

    tmp = Path(tempfile.mkdtemp(prefix="tricoach_sync_"))
    uploads_root = PROJECT_ROOT / f".test_uploads_{tmp.name}"
    memory_dir = tmp / "memory"
    memory_dir.mkdir()
    config = load_config()
    conn = connect(tmp / "test.db")
    client = FakeActivityClient(aid, zip_bytes)

    try:
        # 1. eerste sync: nieuw + ID geregistreerd + origineel gearchiveerd --
        res = garmin_sync.sync_activities(
            client, conn, config, memory_dir, uploads_dir=uploads_root,
            tmp_dir=tmp / "sync_tmp")
        assert len(res.new) == 1 and not res.errors, vars(res)
        acts = load_activities(conn)
        assert len(acts) == 1
        key = acts.iloc[0]["activity_key"]
        geregistreerd = conn.execute(
            "SELECT garmin_activity_id FROM activities WHERE activity_key = ?",
            (key,)).fetchone()[0]
        assert geregistreerd == aid, f"ID niet geregistreerd: {geregistreerd}"
        archief = acts.iloc[0]["archived_path"]
        assert archief and aid in archief and archief.endswith(".fit"), archief
        print(f"1. eerste sync importeert en archiveert ({archief}) OK")

        # 2. tweede sync: dedup vóór de download -----------------------------
        res2 = garmin_sync.sync_activities(
            client, conn, config, memory_dir, uploads_dir=uploads_root,
            tmp_dir=tmp / "sync_tmp")
        assert res2.skipped_known == 1 and not res2.new, vars(res2)
        assert client.downloads == 1, "tweede sync had niets mogen downloaden"
        print("2. tweede sync downloadt niets (dedup op activity-ID) OK")

        # 3. handmatige upload eerst, dan sync -------------------------------
        conn2 = connect(tmp / "test2.db")
        memory2 = tmp / "memory2"
        memory2.mkdir()
        import_zip(zip_path, conn2, config, memory2, uploads_dir=None)
        client2 = FakeActivityClient(aid, zip_bytes)
        res3 = garmin_sync.sync_activities(
            client2, conn2, config, memory2, uploads_dir=uploads_root,
            tmp_dir=tmp / "sync_tmp")
        assert res3.skipped_known == 1 and client2.downloads == 0, vars(res3)
        assert len(load_activities(conn2)) == 1, "handmatige upload dubbel binnengekomen"
        print("3. handmatig geüploade activiteit komt niet dubbel binnen OK")

        # 4. soft-verwijderde sessie blijft verwijderd ------------------------
        soft_delete_activity(conn2, key)
        # Wis de ID-registratie én de herkenbare bestandsnaam, zodat de sync
        # de download écht opnieuw doet en de tweede dedup-laag — de
        # starttijd-sleutel in de import-pipeline — aan de beurt komt.
        conn2.execute("UPDATE activities SET garmin_activity_id = NULL, "
                      "source_file = 'handmatig-hernoemd.fit'")
        conn2.commit()
        res4 = garmin_sync.sync_activities(
            client2, conn2, config, memory2, uploads_dir=uploads_root,
            tmp_dir=tmp / "sync_tmp")
        assert client2.downloads == 1, "de download had nu wél moeten gebeuren"
        assert not res4.new and res4.deleted_kept == 1, vars(res4)
        assert is_deleted(conn2, key), "verwijderde sessie is teruggekomen!"
        assert load_activities(conn2).empty
        print("4. soft-verwijderde sessie keert niet terug via de sync OK")

        # 5. zacht falen zonder tokens/credentials ---------------------------
        oud_tokens = garmin_sync.TOKEN_DIR
        oud_email = os.environ.pop("GARMIN_EMAIL", None)
        oud_ww = os.environ.pop("GARMIN_PASSWORD", None)
        garmin_sync.TOKEN_DIR = tmp / "geen_tokens"
        try:
            garmin_sync.connect_client()
            raise AssertionError("connect_client had moeten falen")
        except garmin_sync.GarminSyncError as e:
            assert "GARMIN_EMAIL" in str(e)
            print(f"5. nette foutmelding zonder credentials OK ({e})")
        finally:
            garmin_sync.TOKEN_DIR = oud_tokens
            if oud_email:
                os.environ["GARMIN_EMAIL"] = oud_email
            if oud_ww:
                os.environ["GARMIN_PASSWORD"] = oud_ww

        print("\nAlle sync-tests geslaagd.")
    finally:
        conn.close()
        shutil.rmtree(uploads_root, ignore_errors=True)
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
