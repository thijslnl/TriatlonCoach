"""Testscript voor de GPS-heatmap (tricoach.heatmap).

Gebruik:  python tests/test_heatmap.py

Draait volledig op een tijdelijke database met gefabriceerde tracks; raakt de
echte data niet aan. Controleert de dingen die bepalen of de kaart klopt:

1. herbemonstering op afstand: een track met een lange stilstand levert per
   afgelegde meter één punt op, niet per seconde — en een sprong (pauze +
   verderop hervat) wordt niet overbrugd met een spooklijn;
2. de stoplichttest: bij puntdichtheid licht een stilstand feller op dan een
   route die tien keer is gereden; bij passage-dichtheid niet;
3. frequentie: een tien keer gereden route krijgt een hogere intensiteit dan
   een route die één keer is gereden;
4. de kleurschalen: nooit lineair, één passage altijd op 0, en de legenda
   gebruikt exact dezelfde omzetting als de kaart;
5. de cache: zwembadsessies worden zonder foutmelding overgeslagen en niet
   opnieuw geparst, open water zwemmen komt er wél in, en een tweede
   verversing doet geen werk meer;
6. de privacyzone: punten binnen de straal verdwijnen, de rest blijft;
7. de filters: sport, periode, transport en soft-deletes doen wat ze beloven.

De FIT-extractie zelf wordt geïnjecteerd (er is geen FIT-writer beschikbaar);
het pad van semicircles naar graden zit in tricoach.fit_parser en wordt door
tests/test_parse.py gedekt.
"""

# De tests staan in tests/; zet de projectroot op sys.path zodat
# `python tests/test_<naam>.py` het tricoach-package kan importeren.
import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))


import math
import tempfile
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from tricoach import heatmap as hm
from tricoach.fit_parser import ParsedActivity
from tricoach.storage import connect, save_activity, soft_delete_activity

BOUNDS = [136, 152, 162, 171]

# Een rechte lijn oost-west door Eemnes: handig ijkpunt, want op deze breedte
# is één graad lengte ongeveer 68 km.
HUIS_LAT, HUIS_LON = 52.2529, 5.2649


def _meters_per_deg_lon(lat: float) -> float:
    return hm.METERS_PER_DEG_LAT * math.cos(math.radians(lat))


def _rechte_lijn(lengte_m: float, n: int, lat: float = HUIS_LAT,
                 lon0: float = HUIS_LON) -> tuple[np.ndarray, np.ndarray]:
    """``n`` gelijk verdeelde punten op een oost-west lijn van ``lengte_m``."""
    dlon = lengte_m / _meters_per_deg_lon(lat)
    return np.full(n, lat), np.linspace(lon0, lon0 + dlon, n)


def test_resample_op_afstand() -> None:
    """Herbemonstering levert punten op vaste afstand, niet op vaste tijd."""
    # 1000 m in 101 punten: elke 10 m een punt. Herbemonsteren op 10 m moet
    # dus ~100 punten geven, ongeacht hoe de bronpunten verdeeld zijn.
    lat, lon = _rechte_lijn(1000.0, 101)
    uit = hm.resample_track(lat, lon, interval_m=10.0)
    assert 99 <= len(uit) <= 101, len(uit)

    # Dezelfde 1000 m, maar met 500 extra punten stilstand halverwege: het
    # aantal herbemonsterde punten mag daar niet van veranderen.
    helft = len(lon) // 2
    lon_stil = np.concatenate([lon[:helft], np.full(500, lon[helft]), lon[helft:]])
    lat_stil = np.full(lon_stil.size, HUIS_LAT)
    uit_stil = hm.resample_track(lat_stil, lon_stil, interval_m=10.0)
    assert abs(len(uit_stil) - len(uit)) <= 1, (len(uit_stil), len(uit))
    print(f"1. herbemonstering: 1000 m -> {len(uit)} punten, met 500 punten "
          f"stilstand -> {len(uit_stil)}: OK")

    # De onderlinge afstand is inderdaad ~10 m.
    x = uit["lon"].to_numpy() * _meters_per_deg_lon(HUIS_LAT)
    stappen = np.diff(x)
    assert np.allclose(stappen, 10.0, atol=0.5), stappen[:5]

    # Een sprong groter dan MAX_GAP_M wordt niet overbrugd: twee stukken van
    # 100 m met 5 km ertussen geven ~20 punten, geen 500.
    ver_lon = HUIS_LON + 5000.0 / _meters_per_deg_lon(HUIS_LAT)
    lat_a, lon_a = _rechte_lijn(100.0, 11)
    lat_b, lon_b = _rechte_lijn(100.0, 11, lon0=ver_lon)
    uit_gat = hm.resample_track(np.concatenate([lat_a, lat_b]),
                                np.concatenate([lon_a, lon_b]), interval_m=10.0)
    assert len(uit_gat) <= 25, len(uit_gat)
    # ... en er ligt niets in het niemandsland tussen de twee stukken.
    midden = HUIS_LON + 2500.0 / _meters_per_deg_lon(HUIS_LAT)
    assert not ((uit_gat["lon"] > midden - 0.005)
                & (uit_gat["lon"] < midden + 0.005)).any()
    print(f"   sprong van 5 km niet overbrugd ({len(uit_gat)} punten, geen "
          "spooklijn): OK")

    # Nulpunten (bericht vóór de eerste satellietfix) vallen weg.
    lat_nul = np.concatenate([[0.0, 0.0], lat])
    lon_nul = np.concatenate([[0.0, 0.0], lon])
    assert hm.resample_track(lat_nul, lon_nul)["lat"].min() > 50.0
    print("   (0, 0)-punten zonder satellietfix weggefilterd: OK")


def _punten_frame(tracks: dict[str, tuple[np.ndarray, np.ndarray]]) -> pd.DataFrame:
    """Bouw een punten-DataFrame zoals load_track_points die teruggeeft."""
    delen = []
    for key, (lat, lon) in tracks.items():
        delen.append(pd.DataFrame({
            "activity_key": key, "sport": "cycling", "sub_sport": "road",
            "seq": np.arange(len(lat)), "lat": lat, "lon": lon,
            "start_time": pd.Timestamp("2026-07-01", tz="Europe/Amsterdam"),
            "datum": date(2026, 7, 1), "categorie": "Fietsen",
            "is_transport": False, "is_deleted": False,
        }))
    return pd.concat(delen, ignore_index=True)


def test_stoplichttest() -> None:
    """Puntdichtheid laat een stilstand feller oplichten dan een drukke route;
    passage-dichtheid niet. Dit is de kern van de hele feature."""
    lat, lon = _rechte_lijn(1000.0, 1001)  # 1 punt per meter, als bij 1 Hz
    # Eén rit met 300 s stilstand bij een stoplicht op 500 m.
    stop = len(lon) // 2
    lon_stop = np.concatenate([lon[:stop], np.full(300, lon[stop]), lon[stop:]])
    lat_stop = np.full(lon_stop.size, HUIS_LAT)

    # Naïef: ruwe punten per rastercel tellen.
    ruw = _punten_frame({"stoplichtrit": (lat_stop, lon_stop)})
    naief = ruw.assign(
        ix=np.floor(ruw.lon / (20.0 / _meters_per_deg_lon(HUIS_LAT))).astype(int),
    ).groupby("ix").size()

    # Tien keer dezelfde route zonder stoplicht, netjes herbemonsterd.
    tien = {}
    for i in range(10):
        res = hm.resample_track(lat, lon, interval_m=10.0)
        tien[f"rit{i}"] = (res["lat"].to_numpy(), res["lon"].to_numpy())
    raster_tien = hm.density_grid(_punten_frame(tien), cell_m=20.0)

    assert naief.max() > raster_tien["count"].max(), (
        f"naief {naief.max()} vs 10 ritten {raster_tien['count'].max()}")
    print(f"2. naïeve puntdichtheid: één stilstand = {naief.max()} punten in een "
          f"cel, tien ritten = {raster_tien['count'].max()} — de kaart zou het "
          "stoplicht feller tonen dan de dagelijkse route: OK (dat vermijden we)")

    # Met herbemonstering + passages telt die stilstand als één passage.
    res_stop = hm.resample_track(lat_stop, lon_stop, interval_m=10.0)
    raster_stop = hm.density_grid(
        _punten_frame({"stoplichtrit": (res_stop["lat"].to_numpy(),
                                        res_stop["lon"].to_numpy())}),
        cell_m=20.0)
    assert raster_stop["count"].max() <= 2, raster_stop["count"].max()
    assert raster_stop["count"].max() < raster_tien["count"].max()
    print(f"   passage-dichtheid: stilstand = {raster_stop['count'].max()}× "
          f"langsgekomen, tien ritten = {raster_tien['count'].max()}×: OK")


def test_frequentie_en_schaal() -> None:
    """Een vaak gereden route licht feller op dan een eenmalige, en de
    kleurschaal is niet lineair."""
    # Woon-werk: 10 keer dezelfde 2 km. Eenmalige route: 2 km ernaast.
    lat_ww, lon_ww = _rechte_lijn(2000.0, 201)
    lat_los, lon_los = _rechte_lijn(2000.0, 201, lat=HUIS_LAT + 0.05)
    tracks = {}
    for i in range(10):
        res = hm.resample_track(lat_ww, lon_ww)
        tracks[f"woonwerk{i}"] = (res["lat"].to_numpy(), res["lon"].to_numpy())
    res = hm.resample_track(lat_los, lon_los)
    tracks["eenmalig"] = (res["lat"].to_numpy(), res["lon"].to_numpy())

    cellen = hm.heatmap_cells(_punten_frame(tracks), cell_m=20.0)
    ww = cellen[cellen["lat"] < HUIS_LAT + 0.01]
    los = cellen[cellen["lat"] > HUIS_LAT + 0.01]
    assert ww["count"].median() >= 9, ww["count"].median()
    assert los["count"].max() == 1, los["count"].max()
    assert ww["t"].median() == 1.0 and los["t"].max() == 0.0
    # Feller = hoger in elk kleurkanaal; de eenmalige route blijft dof rood.
    assert ww["r"].median() > los["r"].median()
    assert ww["g"].median() > los["g"].median()
    print(f"3. woon-werk (10×) t={ww['t'].median():.2f} "
          f"rgb({ww['r'].median():.0f},{ww['g'].median():.0f},{ww['b'].median():.0f}) "
          f"vs eenmalig t={los['t'].max():.2f} "
          f"rgb({los['r'].max()},{los['g'].max()},{los['b'].max()}): OK")

    # De schalen: één passage altijd op 0, en niet lineair.
    counts = np.array([1] * 100 + [2] * 30 + [5] * 8 + [40])
    for methode in (hm.SCALE_LOG, hm.SCALE_PERCENTILE):
        t = hm.scale_counts(counts, methode)
        assert t[counts == 1].max() == 0.0, methode
        assert t.max() == 1.0, methode
        lineair = (5 - 1) / (40 - 1)
        assert t[counts == 5][0] > lineair * 2, (methode, t[counts == 5][0])
        # De legenda gebruikt dezelfde omzetting als de kaart.
        schaal = hm.IntensityScale(counts, methode)
        stappen = hm.legend_stops(counts, methode)
        for aantal, hex_kleur in stappen:
            verwacht = hm.gradient_colors(schaal([aantal]))[0]
            assert hex_kleur == "#%02x%02x%02x" % tuple(verwacht[:3]), stappen
        print(f"   schaal {methode}: 1×={t[counts == 1].max():.2f}, "
              f"5×={t[counts == 5][0]:.2f} (lineair zou {lineair:.2f} zijn), "
              f"40×={t.max():.2f}, legenda consistent: OK")

    # Eén niveau: geen uitspraak doen, dus middenintensiteit.
    assert (hm.scale_counts(np.ones(50)) == 0.5).all()
    print("   selectie met één niveau krijgt middenintensiteit: OK")


def _act(key: str, sport: str, sub_sport: str | None,
         dag: int = 1) -> ParsedActivity:
    """Minimale activiteit voor de opslag (de GPS komt uit de injectie)."""
    start = pd.Timestamp(datetime(2026, 7, dag, 6, 0, 0), tz="UTC")
    return ParsedActivity(
        activity_key=key, sport=sport, sub_sport=sub_sport, start_time=start,
        summary={"total_timer_time": 1800.0, "total_distance": 5000.0,
                 "avg_heart_rate": 130, "sport": sport},
        records=pd.DataFrame(), lengths=pd.DataFrame(),
        source_file=f"{key}_ACTIVITY.fit",
    )


def test_cache_en_filters(tmp: Path) -> None:
    """De trackcache slaat sessies zonder GPS netjes over, werkt alleen nieuwe
    activiteiten bij, en de filters doen wat ze beloven."""
    conn = connect(tmp / "heatmap.db")
    sessies = {
        "fiets-woonwerk": ("cycling", "road", 1),
        "fiets-transport": ("cycling", "road", 2),
        "loop": ("running", "generic", 3),
        "bad": ("swimming", "lap_swimming", 4),      # geen GPS
        "openwater": ("swimming", hm.OPEN_WATER, 5),
        "zwift": ("cycling", "virtual_activity", 6),  # geen GPS
        "verwijderd": ("cycling", "road", 7),
    }
    archief = tmp / "uploads"
    archief.mkdir(parents=True, exist_ok=True)
    for key, (sport, sub, dag) in sessies.items():
        save_activity(conn, _act(key, sport, sub, dag), BOUNDS)
        # Een leeg bestand op een absoluut pad in de tijdelijke map: de
        # extractie moet het bestand kunnen vínden (dat controleert ze), de
        # inhoud komt van de geïnjecteerde parser hieronder.
        origineel = archief / f"{key}.fit"
        origineel.write_bytes(b"")
        conn.execute("UPDATE activities SET archived_path = ? "
                     "WHERE activity_key = ?", (str(origineel), key))
    conn.execute("UPDATE activities SET excluded_reason = 'transport' "
                 "WHERE activity_key = 'fiets-transport'")
    conn.commit()
    soft_delete_activity(conn, "verwijderd")

    # Injecteer de FIT-extractie: elke sessie krijgt een eigen rechte lijn.
    geparst: list[str] = []
    lijnen = {
        key: _rechte_lijn(1000.0, 101, lat=HUIS_LAT + i * 0.02)
        for i, key in enumerate(sessies)
    }
    echte_gps_uit_fit = hm.gps_from_fit

    def nep_gps(pad: Path) -> pd.DataFrame:
        key = pad.stem
        geparst.append(key)
        lat, lon = lijnen[key]
        return pd.DataFrame({"lat": lat, "lon": lon,
                             "timestamp": pd.date_range("2026-07-01", periods=len(lat),
                                                        freq="1s")})

    hm.gps_from_fit = nep_gps
    try:
        telling = hm.refresh_track_cache(conn)
        # Banenzwemmen en Zwift zijn niet eens geparst.
        assert "bad" not in geparst and "zwift" not in geparst, geparst
        assert telling[hm.STATUS_NO_GPS] == 2, telling
        assert telling[hm.STATUS_OK] == 5, telling
        assert telling[hm.STATUS_ERROR] == 0, telling
        print(f"5. cache: {telling[hm.STATUS_OK]} tracks ingelezen, "
              f"{telling[hm.STATUS_NO_GPS]} zonder GPS overgeslagen zonder fout "
              f"(banenzwemmen + Zwift niet geparst): OK")

        # Tweede verversing: niets meer te doen — de cache doet zijn werk.
        geparst.clear()
        opnieuw = hm.refresh_track_cache(conn)
        assert sum(v for k, v in opnieuw.items() if k != "punten") == 0, opnieuw
        assert geparst == [], geparst
        assert hm.pending_activities(conn).empty
        print("   tweede verversing parst geen enkel FIT-bestand opnieuw: OK")

        # Een onleesbaar bestand wordt een status, geen crash.
        conn.execute("DELETE FROM track_extract WHERE activity_key = 'loop'")
        conn.commit()

        def stukke_gps(pad: Path):
            raise ValueError("afgekapt FIT-bestand")

        hm.gps_from_fit = stukke_gps
        assert hm.refresh_track_cache(conn)[hm.STATUS_ERROR] == 1
        hm.gps_from_fit = nep_gps
        conn.execute("DELETE FROM track_extract WHERE activity_key = 'loop'")
        conn.commit()
        hm.refresh_track_cache(conn)
        print("   onleesbaar bestand levert status 'fout' op, geen crash: OK")
    finally:
        hm.gps_from_fit = echte_gps_uit_fit

    # -- filters ---------------------------------------------------------
    punten = hm.load_track_points(conn)
    categorieen = set(punten["categorie"])
    assert categorieen == {"Fietsen", "Hardlopen", "Open water zwemmen"}, categorieen
    assert punten[punten["categorie"] == "Open water zwemmen"]["activity_key"] \
        .nunique() == 1
    print(f"   open water zwemmen staat wél op de kaart; categorieën: "
          f"{sorted(categorieen)}: OK")

    def keys(**kwargs) -> set[str]:
        return set(hm.filter_points(punten, **kwargs)["activity_key"])

    # Soft-deletes standaard eruit, transport standaard erin.
    standaard = keys()
    assert "verwijderd" not in standaard, standaard
    assert "fiets-transport" in standaard, standaard
    assert "verwijderd" in keys(include_deleted=True)
    assert "fiets-transport" not in keys(include_transport=False)
    print("6. filters: soft-deletes standaard weg, transport standaard mee: OK")

    assert keys(categories=["Hardlopen"]) == {"loop"}
    # Periode: alleen de sessie van 3 juli (de loop).
    assert keys(start=date(2026, 7, 3), end=date(2026, 7, 3)) == {"loop"}
    assert keys(start=date(2026, 7, 8), end=date(2026, 7, 9)) == set()
    print("   sport- en datumfilter: OK")

    # -- privacyzone -----------------------------------------------------
    # De fietstrack loopt langs HUIS_LAT; een zone van 300 m rond het beginpunt
    # snijdt daar het eerste stuk af en laat de rest staan.
    fiets = punten[punten["activity_key"] == "fiets-woonwerk"]
    over = hm.apply_privacy_zone(fiets, HUIS_LAT, HUIS_LON, 300.0)
    assert 0 < len(over) < len(fiets), (len(over), len(fiets))
    afstand = hm.distance_to_m(over, HUIS_LAT, HUIS_LON)
    assert afstand.min() > 300.0, afstand.min()
    weg = len(fiets) - len(over)
    assert 28 <= weg <= 32, weg  # 300 m bij 10 m per punt
    print(f"7. privacyzone 300 m: {weg} punten rond het huis weggelaten, "
          f"{len(over)} blijven staan (dichtstbij nu {afstand.min():.0f} m): OK")
    # Zonder middelpunt verandert er niets; uitzetten kan dus.
    assert len(hm.apply_privacy_zone(fiets, None, None, 300.0)) == len(fiets)
    print("   zonder middelpunt (zone uit) blijven alle punten staan: OK")

    # Huis-schatting uit de startpunten: alle tracks starten op lon HUIS_LON.
    voorstel = hm.suggest_home(punten)
    assert abs(voorstel[1] - HUIS_LON) < 1e-6, voorstel
    print(f"   huis-schatting uit startpunten: {voorstel[0]:.4f}, "
          f"{voorstel[1]:.4f}: OK")

    # De zone-instellingen komen náást de database te staan, niet in
    # config.yaml — dat bestand staat in versiebeheer.
    # Absoluut pad, zodat het bestand in de tijdelijke map belandt en niet in
    # de projectmap (resolve_path hangt relatieve paden aan PROJECT_ROOT).
    nep_config = {"paths": {"database": str(tmp / "heatmap.db")}}
    pad = hm.privacy_path(nep_config)
    assert pad.name == hm.PRIVACY_FILE and pad.parent == tmp, pad
    assert "config.yaml" not in str(pad)
    # Standaard: zone aan, nog geen middelpunt, straal 400 m.
    leeg = hm.privacy_settings(nep_config)
    assert leeg["enabled"] and leeg["lat"] is None
    assert leeg["radius_m"] == hm.DEFAULT_PRIVACY_RADIUS_M
    hm.store_privacy_settings(nep_config, True, HUIS_LAT, HUIS_LON, 500.0)
    terug = hm.privacy_settings(nep_config)
    assert (terug["lat"], terug["lon"], terug["radius_m"]) == (
        round(HUIS_LAT, 6), round(HUIS_LON, 6), 500.0), terug
    # Onleesbaar bestand valt terug op de standaarden in plaats van te crashen.
    pad.write_text("{kapot", encoding="utf-8")
    assert hm.privacy_settings(nep_config)["lat"] is None
    print(f"   zone-instellingen bewaard in {pad.name} (buiten versiebeheer), "
          "onleesbaar bestand valt terug op standaarden: OK")

    # -- opruimen na een purge -------------------------------------------
    conn.execute("DELETE FROM activities WHERE activity_key = 'loop'")
    conn.commit()
    assert hm.prune_track_cache(conn) == 1
    assert "loop" not in set(hm.load_track_points(conn)["activity_key"])
    print("   trackcache van een definitief gewiste sessie opgeruimd: OK")
    conn.close()


def test_kaartpositie() -> None:
    """De beginpositie omvat het kerngebied en laat zich niet uitzoomen door
    een enkele verre rit."""
    cellen = pd.DataFrame({"lat": [52.25, 52.40], "lon": [4.85, 5.33]})
    view = hm.fit_view(cellen, width_px=1100, height_px=640)
    assert 52.25 < view["latitude"] < 52.40
    assert 4.85 < view["longitude"] < 5.33
    assert 8.0 <= view["zoom"] <= 12.0, view
    # Zonder data: midden van Nederland, ver uitgezoomd.
    leeg = hm.fit_view(pd.DataFrame(columns=["lat", "lon"]))
    assert leeg["zoom"] < 9.0
    print(f"8. kaartpositie: zoom {view['zoom']:.1f} op een bbox van "
          f"Amsterdam tot Eemnes: OK")

    # 2000 cellen rond huis + 100 cellen in Spanje (5% van het totaal).
    rng = np.random.default_rng(7)
    thuis = pd.DataFrame({"lat": HUIS_LAT + rng.normal(0, 0.02, 2000),
                          "lon": HUIS_LON + rng.normal(0, 0.03, 2000)})
    ver = pd.DataFrame({"lat": 39.47 + rng.normal(0, 0.01, 100),
                        "lon": -0.38 + rng.normal(0, 0.01, 100)})
    alles = pd.concat([thuis, ver], ignore_index=True)

    ruim = hm.fit_view(alles, coverage=1.0)
    kern = hm.fit_view(alles, coverage=0.90)
    la0, la1, lo0, lo1 = hm.view_bounds(alles, 0.90)
    binnen = (alles.lat.between(la0, la1) & alles.lon.between(lo0, lo1)).mean()
    # De eis wordt gehaald ...
    assert binnen >= 0.90, binnen
    # ... de verre groep valt buiten het startbeeld ...
    assert lo0 > 0.0, (lo0, lo1)
    # ... en dat scheelt flink in de zoom (elke stap = 2× zo dicht).
    assert kern["zoom"] > ruim["zoom"] + 3, (ruim["zoom"], kern["zoom"])
    assert abs(kern["latitude"] - HUIS_LAT) < 0.05, kern
    print(f"   met 5% van de cellen in Spanje: zoom {ruim['zoom']:.1f} "
          f"(alles) -> {kern['zoom']:.1f} (kerngebied), dekking "
          f"{binnen:.1%} — de verre rit blijft op de kaart, buiten het "
          "startbeeld: OK")

    # Een gelijkmatige spreiding mag niet zomaar worden weggeknipt: de dekking
    # blijft gehaald en de box krimpt maar beperkt.
    egaal = pd.DataFrame({"lat": HUIS_LAT + rng.normal(0, 0.05, 3000),
                          "lon": HUIS_LON + rng.normal(0, 0.05, 3000)})
    v_alles, v_kern = hm.fit_view(egaal, coverage=1.0), hm.fit_view(egaal, coverage=0.90)
    assert v_kern["zoom"] < v_alles["zoom"] + 1.5, (v_alles, v_kern)
    print(f"   bij een egale spreiding blijft de zoom vergelijkbaar "
          f"({v_alles['zoom']:.1f} -> {v_kern['zoom']:.1f}): OK")


def main() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="tricoach_heatmap_"))
    try:
        test_resample_op_afstand()
        test_stoplichttest()
        test_frequentie_en_schaal()
        test_cache_en_filters(tmp)
        test_kaartpositie()
        print("\nAlle controles geslaagd.")
    finally:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
