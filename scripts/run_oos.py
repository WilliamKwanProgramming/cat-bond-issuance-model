#!/usr/bin/env python3
"""Out-of-sample check on real data: fit on 1900-1979, predict 1980-2019 counts by category.

Writes results/hazard_oos/oos_by_category.csv and MANIFEST.txt. The analysis (windows, groups,
intensity model, interval levels, bootstrap size) was fixed before the script was run.

Usage: python3 scripts/run_oos.py [--seed 3001] [--n-boot 2000]
"""

from __future__ import annotations

import argparse
import sys

from catbond.hazard.validation import out_of_sample_check
from catbond.manifest import repo_root, write_manifest
from catbond.pipeline import OOS_FIT, OOS_TEST, load_inputs

OUT = repo_root() / "results" / "hazard_oos"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seed", type=int, default=3001)
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()
    inp = load_inputs()
    out = out_of_sample_check(
        inp.landfalls, OOS_FIT, OOS_TEST, "gpd", level=0.90, n_boot=args.n_boot, seed=args.seed
    )
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT / "oos_by_category.csv", index=False, float_format="%.4f")
    write_manifest(
        OUT,
        args.seed,
        {
            "data": "real (HURDAT2 via AL.rda), hybrid landfall definition",
            "fit_years": f"{OOS_FIT[0]}-{OOS_FIT[1]}",
            "test_years": f"{OOS_TEST[0]}-{OOS_TEST[1]}",
            "lambda_fit": f"{out.attrs['lambda_fit']:.6f}",
            "bootstrap": f"{args.n_boot} resamples",
        },
    )
    print(out.to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
