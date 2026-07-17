"""Testscript voor het origineel-archief van uploads (tricoach.archive).

Gebruik:  python test_archief.py

Draait volledig op een tijdelijke database en een tijdelijke archiefmap;
raakt de echte data niet aan. Controleert:

1. naamgeving: uploads/yyyy/mm/yyyy-mm-dd_HHmm_<activityid>.fit, met de
   lokale starttijd (Europe/Amsterdam) en de activity-ID uit de bestandsnaam;
2. versienummering: byte-identieke herupload = géén nieuw bestand; gewijzigde
   inhoud = _v2 en de database wijst de nieuwe versie als actief aan;
3. migratie: een sluimerend origineel in een oude map wordt gearchiveerd en
   geregistreerd; sessies zonder origineel houden original_missing;
4. verificatierun: originelen worden opnieuw geparst en vergeleken met de
   database — ok / afwijking / overgeslagen (original_missing) doen wat ze
   beloven.

Voor de verificatierun wordt de parser geïnjecteerd (er is geen echte
FIT-writer beschikbaar); de echte parser draait bij elke echte upload.
"""

# De tests staan in tests/; zet de projectroot op sys.path zodat
# `python tests/test_<naam>.py` het tricoach-package kan importeren.
import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))


import io
import tempfile
from datetime import datetime
from pathlib import Path

import pandas as pd

from tricoach import archive
from tricoach.fit_parser import ParsedActivity
from tricoach.storage import connect, save_activity

BOUNDS = [136, 152, 162, 171]


def _act(uur_utc: int, minuut: int = 22, aid: str = "23627119348",
         afstand: float = 6000.0) -> ParsedActivity:
    """Fietssessie met UTC-starttijd, zoals uit een echt FIT-bestand."""
    start = pd.Timestamp(datetime(2026, 7, 17, uur_utc, minuut, 0), tz="UTC")
    return ParsedActivity(
        activity_key=start.isoformat(),
        sport="cycling", sub_sport=None, start_time=start,
        summary={"total_timer_time": 1080.0, "total_distance": afstand,
                 "avg_heart_rate": 128, "sport": "cycling"},
        records=pd.DataFrame(), lengths=pd.DataFrame(),
        source_file=f"{aid}_ACTIVITY.fit",
    )


def main() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="tricoach_archief_"))
    conn = connect(tmp / "test.db")
    # Archief binnen de projectroot (vereist voor relative_to), wel uniek
    # per testrun; wordt aan het einde opgeruimd.
    root = archive.PROJECT_ROOT / f".test_uploads_{tmp.name}"

    try:
        # 1. Naamgeving: lokale tijd (06:22 UTC = 08:22 Amsterdam), geen
        # dubbele punten, activity-ID uit de bestandsnaam.
        act = _act(6)
        save_activity(conn, act, BOUNDS)
        res = archive.archive_and_register(conn, root, act, b"FIT-origineel-v1")
        assert res.rel_path.endswith(
            "2026/07/2026-07-17_0822_23627119348.fit"), res.rel_path
        assert res.version == 1 and res.is_new_file
        assert ":" not in Path(res.rel_path).name
        rij = conn.execute(
            "SELECT archived_path, original_missing FROM activities "
            "WHERE activity_key = ?", (act.activity_key,)).fetchone()
        assert rij[0] == res.rel_path and rij[1] is None
        print("1. archiefpad: lokale starttijd + activity-ID, geregistreerd: OK")

        # 2. Versienummering: identiek = geen nieuw bestand; gewijzigd = _v2
        # en de database wijst v2 aan als actief.
        res2 = archive.archive_original(root, act, b"FIT-origineel-v1")
        assert not res2.is_new_file and res2.version == 1
        res3 = archive.archive_and_register(conn, root, act,
                                            b"FIT-origineel-v2-gecorrigeerd")
        assert res3.is_new_file and res3.version == 2
        assert res3.rel_path.endswith("_23627119348_v2.fit")
        assert (archive.PROJECT_ROOT / res.rel_path).read_bytes() \
            == b"FIT-origineel-v1", "v1 mag nooit worden overschreven"
        rij = conn.execute(
            "SELECT archived_path FROM activities WHERE activity_key = ?",
            (act.activity_key,)).fetchone()
        assert rij[0] == res3.rel_path, "database hoort v2 als actief aan te wijzen"
        print("2. versienummering: dedupe op inhoud, _v2 bij correctie: OK")

        # 3. Migratie: sessie van vóór het archief heeft original_missing=1
        # (schemamigratie); duikt het origineel op in een oude map, dan wordt
        # het gearchiveerd en gaat de markering eraf. Een sessie zonder
        # origineel houdt de markering.
        oud = _act(9, minuut=3, aid="23627200000")
        zonder = _act(15, minuut=0, aid="23627300000")
        save_activity(conn, oud, BOUNDS)
        save_activity(conn, zonder, BOUNDS)
        conn.execute("UPDATE activities SET original_missing = 1 "
                     "WHERE activity_key IN (?, ?)",
                     (oud.activity_key, zonder.activity_key))
        conn.commit()
        oude_map = tmp / "garmin_import"
        oude_map.mkdir()
        (oude_map / "23627200000_ACTIVITY.fit").write_bytes(b"oud-origineel")

        # parse_fit injecteren kan hier niet (migratie parset zelf), dus
        # simuleer de vondst: parser vervangen door een stub die de sessie
        # herkent aan de bytes.
        echte_parser = archive.parse_fit
        archive.parse_fit = (
            lambda stream, source_name: oud
            if stream.read() == b"oud-origineel" else None)
        try:
            n = archive.migrate_originals(conn, root, [oude_map, tmp / "bestaat_niet"])
        finally:
            archive.parse_fit = echte_parser
        assert n == 1
        rij = conn.execute(
            "SELECT archived_path, original_missing FROM activities "
            "WHERE activity_key = ?", (oud.activity_key,)).fetchone()
        assert rij[0] and rij[0].endswith("_23627200000.fit") and rij[1] is None
        rij = conn.execute(
            "SELECT archived_path, original_missing FROM activities "
            "WHERE activity_key = ?", (zonder.activity_key,)).fetchone()
        assert rij[0] is None and rij[1] == 1, \
            "zonder origineel blijft original_missing staan"
        print("3. migratie archiveert vondsten, ontbrekend blijft gemarkeerd: OK")

        # 4. Verificatierun met geïnjecteerde parser: ok bij overeenstemming,
        # afwijking bij een gemanipuleerde databaserij, overgeslagen bij
        # original_missing.
        conn.execute("UPDATE activities SET distance_m = 9999 "
                     "WHERE activity_key = ?", (oud.activity_key,))
        conn.commit()
        per_bestand = {
            b"FIT-origineel-v2-gecorrigeerd": act,
            b"oud-origineel": oud,
        }

        def stub_parser(stream, source_name):
            return per_bestand.get(stream.read())

        uit = archive.verify_originals(conn, parser=stub_parser)
        status = dict(zip(uit["activity_key"], uit["status"]))
        assert status[act.activity_key] == "ok"
        assert status[oud.activity_key] == "afwijking"
        detail = uit.set_index("activity_key").loc[oud.activity_key, "detail"]
        assert "afstand" in detail
        assert status[zonder.activity_key] == "overgeslagen"
        print("4. verificatierun: ok / afwijking / overgeslagen kloppen: OK")

        print("\nAlle controles geslaagd.")
    finally:
        import shutil
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    main()
