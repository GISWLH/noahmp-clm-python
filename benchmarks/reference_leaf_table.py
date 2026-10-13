"""Independent Decimal evaluation of published CLM5 equations at 25 °C.

No imports from the production model. Fixed ci means no stomatal-model fit.
This is a published-equation reference, not measured leaf data or CTSM output.
"""
from decimal import Decimal as D, getcontext
import json


def main():
    getcontext().prec = 60
    rows = []
    for ci in (50, 280, 1000):
        for par in (0, 50, 100, 200, 500, 1000, 2000):
            light = D("0.425") * par
            transport = ((light + 100) - ((light + 100)**2 - D("2.8")*light*100).sqrt()) / D("1.4")
            ac = 60 * (D(ci)-D("42.75")) / (D(ci) + D("404.9")*(1+D(210000)/278400))
            aj = transport*(D(ci)-D("42.75"))/(4*D(ci)+8*D("42.75"))
            gross = min(ac, aj, D(30))
            rows.append({"ci_ppm": ci, "absorbed_par": par, "gross": float(gross),
                "net": float(gross-D("0.9")), "rubisco": float(ac), "rubp": float(aj)})
    print(json.dumps({"source": "CLM5 technical note 2.9 equations 2.9.2-2.9.7",
        "url": "https://escomp.github.io/CTSM/release-clm5.0/tech_note/Photosynthesis/CLM50_Tech_Note_Photosynthesis.html",
        "reference_kind": "published_equations_independent_decimal_60_digits_not_observations",
        "parameters": {"leaf_temperature_K": 298.15, "vcmax25": 60., "jmax25": 100.,
            "tpu25": 10., "respiration25": .9, "oxygen_ppm": 210000.}, "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
