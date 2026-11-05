#!/usr/bin/env python3
"""Monte Carlo convergence: spread over independent seeds of the 1-in-100 OEP loss and layer EL.

For each N in --n-years, simulates the base-case event set with --n-seeds independent seeds
(seeds start, start + 1, ...) and records the 1-in-100 OEP loss with its within-run bootstrap SE,
the AAL, and the reference layer's AP and EL with the within-run SE of EL (standard error of a
mean of independent years). Comparing the across-seed SD with the average within-run SE checks
the SE formulas. Needs results/eventset/reference_layer.csv (run scripts/run_eventset.py first).

Writes results/convergence/convergence_N<N>.csv (one row per seed), summary_N<N>.csv, MANIFEST.

Usage: python3 scripts/run_convergence.py --n-years 10000 [--n-seeds 20] [--seed-start 1001]
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from catbond.eventset import simulate_event_set
from catbond.hazard.model import fit_hazard
from catbond.layers import annual_principal_loss, layer_metrics
from catbond.losses import annual_max_and_sum, return_period_losses, return_period_losses_se
from catbond.manifest import repo_root, write_manifest
from catbond.pipeline import CALIBRATION, RP_BOOTSTRAP, TIV_USD, load_inputs

OUT = repo_root() / "results" / "convergence"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n-years", type=int, required=True)
    ap.add_argument("--n-seeds", type=int, default=20)
    ap.add_argument("--seed-start", type=int, default=1001)
    args = ap.parse_args()
    ref = pd.read_csv(repo_root() / "results" / "eventset" / "reference_layer.csv").iloc[0]
    a, e = float(ref["attachment_usd"]), float(ref["exhaustion_usd"])
    inp = load_inputs()
    model = fit_hazard(inp.landfalls, *CALIBRATION)
    rows = []
    for k in range(args.n_seeds):
        seed = args.seed_start + k
        es = simulate_event_set(model, inp.region.coast_edges, inp.exposure, args.n_years, seed)
        y, loss = es.events["year"], es.events["loss_usd"]
        mx, sm = annual_max_and_sum(y, loss, es.n_years)
        oep100 = return_period_losses(mx, (100,))[0]
        se100 = return_period_losses_se(mx, (100,), RP_BOOTSTRAP, seed=seed)[0]
        m = layer_metrics(y, loss, es.n_years, a, e)
        p = annual_principal_loss(y, loss, es.n_years, a, e) / (e - a)
        rows.append(
            {
                "seed": seed,
                "n_years": args.n_years,
                "oep100_pct_tiv": 100 * oep100 / TIV_USD,
                "oep100_boot_se_pct_tiv": 100 * se100 / TIV_USD,
                "aal_pct_tiv": 100 * sm.mean() / TIV_USD,
                "ref_ap_pct": 100 * m.attachment_prob,
                "ref_el_pct": 100 * m.expected_loss,
                "ref_el_se_pct": 100 * p.std(ddof=1) / np.sqrt(es.n_years),
            }
        )
        print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / f"convergence_N{args.n_years}.csv", index=False, float_format="%.6g")
    summ = []
    for col, se_col in [
        ("oep100_pct_tiv", "oep100_boot_se_pct_tiv"),
        ("aal_pct_tiv", None),
        ("ref_ap_pct", None),
        ("ref_el_pct", "ref_el_se_pct"),
    ]:
        summ.append(
            {
                "n_years": args.n_years,
                "statistic": col,
                "n_seeds": len(df),
                "mean": df[col].mean(),
                "sd_across_seeds": df[col].std(ddof=1),
                "min": df[col].min(),
                "max": df[col].max(),
                "mean_within_run_se": df[se_col].mean() if se_col else np.nan,
            }
        )
    pd.DataFrame(summ).to_csv(
        OUT / f"summary_N{args.n_years}.csv", index=False, float_format="%.6g"
    )
    write_manifest(
        OUT,
        args.seed_start,
        {
            f"N{args.n_years}": f"seeds {args.seed_start}..{args.seed_start + args.n_seeds - 1}",
            "data": "hazard fitted on real HURDAT2 landfalls; hypothetical exposure",
            "reference_layer_usd": f"{a:.0f} to {e:.0f}",
        },
        name=f"MANIFEST_N{args.n_years}.txt",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
