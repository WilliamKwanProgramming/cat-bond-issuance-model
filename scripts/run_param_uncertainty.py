#!/usr/bin/env python3
"""Parameter uncertainty of layer EL: bootstrap the calibration years, refit, re-simulate.

Replicate b (b = 0..B-1) refits the hazard model on a bootstrap sample of the 1900-2019 years
(``catbond.uncertainty``) and simulates --n-years years with seed --seed + 1 + b. The spread of
the replicate results combines parameter uncertainty and Monte Carlo noise at --n-years; the
Monte Carlo part is measured separately by scripts/run_convergence.py. Needs
results/eventset/reference_layer.csv.

Writes results/param_uncertainty/replicates.csv, summary.csv, MANIFEST.txt, and for every layer
of the candidate grid (catbond.pipeline.GRID_*) grid_replicates.csv (AP, EP, EL per replicate)
and grid_summary.csv (mean and quantiles of the EL across replicates); rp_replicates.csv and
rp_summary.csv hold the OEP and AEP return-period losses (percent of TIV) of every replicate at
LEC_RETURN_PERIODS and their quantiles (the band around the loss exceedance curve).

Usage: python3 scripts/run_param_uncertainty.py [--n-boot 200] [--n-years 10000] [--seed 5001]
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from catbond.eventset import simulate_event_set
from catbond.hazard.model import fit_hazard
from catbond.layers import layer_metrics, layer_table
from catbond.losses import annual_max_and_sum, return_period_losses
from catbond.manifest import repo_root, write_manifest
from catbond.pipeline import (
    CALIBRATION,
    GRID_ATTACHMENTS,
    GRID_LIMITS,
    TERM_YEARS,
    TIV_USD,
    load_inputs,
)
from catbond.uncertainty import bootstrap_hazard_models

OUT = repo_root() / "results" / "param_uncertainty"
QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)
LEC_RETURN_PERIODS = (2, 5, 10, 25, 50, 100, 200, 250, 500)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n-boot", type=int, default=200)
    ap.add_argument("--n-years", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=5001)
    args = ap.parse_args()
    ref = pd.read_csv(repo_root() / "results" / "eventset" / "reference_layer.csv").iloc[0]
    a, e = float(ref["attachment_usd"]), float(ref["exhaustion_usd"])
    inp = load_inputs()
    base = fit_hazard(inp.landfalls, *CALIBRATION)
    models = bootstrap_hazard_models(inp.landfalls, base, args.n_boot, args.seed)
    rows, grid, rps = [], [], []
    for b, m in enumerate(models):
        es = simulate_event_set(
            m, inp.region.coast_edges, inp.exposure, args.n_years, args.seed + 1 + b
        )
        y, loss = es.events["year"], es.events["loss_usd"]
        mx, sm = annual_max_and_sum(y, loss, es.n_years)
        lm = layer_metrics(y, loss, es.n_years, a, e)
        rows.append(
            {
                "replicate": b,
                "freq_mean": m.freq_mean,
                "gpd_xi": m.intensity.shape,
                "gpd_sigma": m.intensity.scale,
                "n_pool": m.pool_lon.size,
                "aal_pct_tiv": 100 * sm.mean() / TIV_USD,
                "oep100_pct_tiv": 100 * return_period_losses(mx, (100,))[0] / TIV_USD,
                "ref_ap_pct": 100 * lm.attachment_prob,
                "ref_el_pct": 100 * lm.expected_loss,
            }
        )
        for curve, annual in (("OEP", mx), ("AEP", sm)):
            for t_rp, v in zip(
                LEC_RETURN_PERIODS, return_period_losses(annual, LEC_RETURN_PERIODS), strict=True
            ):
                rps.append(
                    {
                        "replicate": b,
                        "curve": curve,
                        "return_period": t_rp,
                        "loss_pct_tiv": 100 * v / TIV_USD,
                    }
                )
        t = layer_table(
            y, loss, es.n_years, GRID_ATTACHMENTS * TIV_USD, GRID_LIMITS * TIV_USD, TERM_YEARS
        )
        grid.append(
            pd.DataFrame(
                {
                    "replicate": b,
                    "attachment_pct_tiv": 100 * t["attachment_usd"] / TIV_USD,
                    "limit_pct_tiv": 100 * t["limit_usd"] / TIV_USD,
                    "ap_pct": 100 * t["attachment_prob"],
                    "ep_pct": 100 * t["exhaustion_prob"],
                    "el_pct": 100 * t["expected_loss"],
                }
            )
        )
        if b % 20 == 0:
            print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "replicates.csv", index=False, float_format="%.6g")
    summ = []
    for col in ["freq_mean", "gpd_xi", "aal_pct_tiv", "oep100_pct_tiv", "ref_ap_pct", "ref_el_pct"]:
        q = np.quantile(df[col], QUANTILES)
        summ.append(
            {
                "statistic": col,
                "n_replicates": len(df),
                "mean": df[col].mean(),
                "sd": df[col].std(ddof=1),
            }
            | {f"q{int(100 * p):02d}": v for p, v in zip(QUANTILES, q, strict=True)}
        )
    pd.DataFrame(summ).to_csv(OUT / "summary.csv", index=False, float_format="%.6g")
    g = pd.concat(grid, ignore_index=True)
    g.to_csv(OUT / "grid_replicates.csv", index=False, float_format="%.6g")
    keys = ["attachment_pct_tiv", "limit_pct_tiv"]
    gs = g.groupby(keys)["el_pct"]
    gsum = gs.agg(n_replicates="size", el_mean="mean", el_sd="std").reset_index()
    for p in QUANTILES:
        gsum[f"el_q{int(100 * p):02d}"] = gs.quantile(p).to_numpy()
    gsum.to_csv(OUT / "grid_summary.csv", index=False, float_format="%.6g")
    r = pd.DataFrame(rps)
    r.to_csv(OUT / "rp_replicates.csv", index=False, float_format="%.6g")
    rq = r.groupby(["curve", "return_period"])["loss_pct_tiv"]
    rsum = rq.agg(n_replicates="size", mean="mean").reset_index()
    for p in QUANTILES:
        rsum[f"q{int(100 * p):02d}"] = rq.quantile(p).to_numpy()
    rsum.to_csv(OUT / "rp_summary.csv", index=False, float_format="%.6g")
    write_manifest(
        OUT,
        args.seed,
        {
            "data": "real HURDAT2 landfalls 1900-2019, resampled by year; hypothetical exposure",
            "n_boot": str(args.n_boot),
            "n_years_per_replicate": str(args.n_years),
            "replicate_seeds": f"{args.seed + 1}..{args.seed + args.n_boot}",
            "reference_layer_usd": f"{a:.0f} to {e:.0f}",
        },
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
