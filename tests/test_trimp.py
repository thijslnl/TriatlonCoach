"""Testscript voor de TRIMP-berekening, met name de zwem-TRIMP (Banister).

Gebruik:  python tests/test_trimp.py

Controleert:

1. sporten met zones: TRIMP blijft minuten per zone × zonegewicht;
2. zwemmen met hartslag: Banister-TRIMP op de gemiddelde hartslag over de
   actieve tijd, geijkt met de factor uit swim_trimp_params;
3. zwemmen zonder actieve tijd valt terug op de totale duur;
4. zwemmen zonder hartslagdata blijft op 0;
5. swim_trimp_params levert bruikbare waarden op uit de echte config, en de
   ijking klopt: op de Z2/Z3-grens van het lopen telt zwemmen 2,5 per minuut.
"""

# De tests staan in tests/; zet de projectroot op sys.path zodat
# `python tests/test_<naam>.py` het tricoach-package kan importeren.
import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))


import math

import numpy as np
import pandas as pd

import tricoach.progress as progress
from tricoach.progress import SWIM_TRIMP_MIDWEIGHT, _banister_per_min


def main() -> None:
    # Vaste parameters zodat de test niet van config/wellness afhangt.
    rust, max_hr, factor = 60.0, 190.0, 1.4
    echte_params = progress.swim_trimp_params
    progress.swim_trimp_params = lambda: (rust, max_hr, factor)
    try:
        acts = pd.DataFrame({
            "start_time": pd.to_datetime(["2026-08-01", "2026-08-02",
                                          "2026-08-03", "2026-08-04"]),
            "sport": ["running", "swimming", "swimming", "swimming"],
            "duration_s": [3600.0, 2100.0, 1800.0, 2400.0],
            "active_s": [3600.0, 1800.0, np.nan, 2400.0],
            "avg_hr": [150.0, 127.0, 140.0, np.nan],
            "z1_s": [600, 0, 0, 0], "z2_s": [1800, 0, 0, 0],
            "z3_s": [1200, 0, 0, 0], "z4_s": [0, 0, 0, 0], "z5_s": [0, 0, 0, 0],
        })
        trimp = progress.trimp_per_session(acts)["trimp"]

        assert math.isclose(trimp[0], 600 / 60 + 1800 / 60 * 2 + 1200 / 60 * 3), \
            f"loop-TRIMP hoort zonaal te blijven, kreeg {trimp[0]}"
        print("1. zonale TRIMP voor lopen ongewijzigd: OK")

        def banister(minuten: float, avg_hr: float) -> float:
            hrr = (avg_hr - rust) / (max_hr - rust)
            return minuten * _banister_per_min(hrr) * factor

        assert math.isclose(trimp[1], banister(1800 / 60, 127.0)), \
            f"zwem-TRIMP wijkt af van Banister op actieve tijd: {trimp[1]}"
        assert trimp[1] > 0, "zwemsessie met hartslag hoort mee te tellen"
        print(f"2. zwem-TRIMP via Banister op actieve tijd ({trimp[1]:.1f}): OK")

        assert math.isclose(trimp[2], banister(1800 / 60, 140.0)), \
            f"zonder active_s hoort duration_s te gelden: {trimp[2]}"
        print("3. terugval op duration_s zonder actieve tijd: OK")

        assert trimp[3] == 0, "zwemmen zonder hartslagdata hoort 0 te blijven"
        print("4. zwemmen zonder hartslag blijft 0: OK")
    finally:
        progress.swim_trimp_params = echte_params

    rust, max_hr, factor = progress.swim_trimp_params()
    assert 30 <= rust < max_hr <= 230, f"onzinnige HR-parameters: {rust}, {max_hr}"
    assert factor > 0, f"ijkfactor hoort positief te zijn: {factor}"
    # De ijking zelf: op de Z2/Z3-grens van het lopen moet de zwemscore per
    # minuut precies het tussengewicht zijn.
    from tricoach.config import load_config
    from tricoach.sportzones import run_lthr, zone_pcts
    from tricoach.zones import bounds_from_lthr
    athlete = load_config().get("athlete") or {}
    grens = bounds_from_lthr(run_lthr(athlete), zone_pcts(athlete))[1]
    per_min = _banister_per_min((grens - rust) / (max_hr - rust)) * factor
    assert math.isclose(per_min, SWIM_TRIMP_MIDWEIGHT, rel_tol=1e-6), \
        f"ijking op de Z2/Z3-grens klopt niet: {per_min}"
    print(f"5. swim_trimp_params uit config (rust {rust:.0f}, max {max_hr:.0f}, "
          f"factor {factor:.2f}) en ijking op de Z2/Z3-grens: OK")

    print("\nAlle controles geslaagd.")


if __name__ == "__main__":
    main()
