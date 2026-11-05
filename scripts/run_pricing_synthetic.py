#!/usr/bin/env python3
"""Check the spread regression on SYNTHETIC deal tables with a known generating relation.

No real deal data are used. For each scenario, R replications of a table of n synthetic tranches
(``catbond.pricing.synthetic``) are fitted with log(spread) ~ log(EL) + issue_year + trigger, and
one extra tranche per replication is predicted. Recorded per scenario: mean estimate and bias of
the log-EL slope, the empirical SD of the slope, the mean OLS / HC3 / cluster standard errors,
95% interval coverage of the slope for each SE, coverage of the 90% prediction interval, and the
mean leave-one-out RMSE against the true error SD.

Writes results/pricing_synthetic/coverage.csv and MANIFEST.txt.

Usage: python3 scripts/run_pricing_synthetic.py [--reps 1000] [--n 60] [--seed 4001]
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from catbond.manifest import repo_root, write_manifest
from catbond.pricing.fit import fit_log_spread
from catbond.pricing.synthetic import TRUE_COEF, synthetic_deals

OUT = repo_root() / "results" / "pricing_synthetic"
COVS = ("issue_year", "trigger")
SCENARIOS = {
    "iid errors": dict(sigma=0.25),
    "heteroskedastic (SD x e^{0.5 (log EL - log 2)})": dict(sigma=0.25, hetero=0.5),
    "sponsor effects (SD 0.25, 15 sponsors)": dict(sigma=0.15, sponsor_sd=0.25, n_sponsors=15),
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--reps", type=int, default=1000)
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--seed", type=int, default=4001)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    rows = []
    for name, kw in SCENARIOS.items():
        est, se = [], {"ols": [], "hc3": [], "cluster": []}
        pi_hit, loo = [], []
        for _ in range(args.reps):
            d = synthetic_deals(args.n + 1, rng, **kw)
            f = fit_log_spread(d.iloc[: args.n], COVS)
            i = f.names.index("log_el")
            est.append(f.coef[i])
            se["ols"].append(f.se_ols[i])
            se["hc3"].append(f.se_hc3[i])
            se["cluster"].append(f.se_cluster[i])
            p = f.predict(d.iloc[[args.n]], level=0.90)
            s = d["spread_pct"].iloc[args.n]
            pi_hit.append(p["pi_lo_pct"].item() <= s <= p["pi_hi_pct"].item())
            loo.append(f.loo_rmse)
        est_a = np.array(est)
        true = TRUE_COEF["log_el"]
        row = {
            "scenario": name,
            "data": "synthetic",
            "n_deals": args.n,
            "reps": args.reps,
            "slope_true": true,
            "slope_mean": est_a.mean(),
            "slope_bias": est_a.mean() - true,
            "slope_sd": est_a.std(ddof=1),
        }
        for k, v in se.items():
            v_a = np.array(v)
            row[f"se_{k}_mean"] = v_a.mean()
            row[f"cover95_{k}"] = float(np.mean(np.abs(est_a - true) <= 1.96 * v_a))
        row["pi90_coverage"] = float(np.mean(pi_hit))
        row["loo_rmse_mean"] = float(np.mean(loo))
        rows.append(row)
        print(row, flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / "coverage.csv", index=False, float_format="%.4f")
    write_manifest(
        OUT,
        args.seed,
        {
            "data": "synthetic deal tables only (no real deals)",
            "reps": str(args.reps),
            "n_deals": str(args.n),
        },
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
