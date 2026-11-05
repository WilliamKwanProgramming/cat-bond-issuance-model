#!/usr/bin/env python3
"""Base event set, return-period losses with bootstrap SEs, reference layer and sensitivities.

The hazard model is fitted on real data (HURDAT2, Florida landfalls 1900-2019); the portfolio is
hypothetical. Teaching-grade model: not comparable to a vendor model's expected loss.

Writes to results/eventset/:
  event_table.csv.gz   base-case events (git-ignored: regenerate with this script)
  rp_losses.csv        OEP and AEP return-period losses, base case, with bootstrap SEs
  reference_layer.csv  the pre-specified reference layer (1-in-50 to 1-in-100 OEP, rounded)
  sensitivity.csv      AAL, 1-in-100 / 1-in-250 OEP and reference-layer metrics per variant
  sensitivity_layers.csv  AP, EP and EL of every layer of the candidate grid, per variant
  MANIFEST.txt
and results/exposure/county_exposure.csv (the hypothetical portfolio by county).

Usage: python3 scripts/run_eventset.py [--seed 1] [--n-years 100000]
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace

import numpy as np
import pandas as pd

from catbond.eventset import LossSettings, simulate_event_set
from catbond.hazard.model import fit_hazard
from catbond.layers import layer_metrics, layer_table
from catbond.losses import (
    RETURN_PERIODS,
    annual_max_and_sum,
    return_period_losses,
    return_period_losses_se,
)
from catbond.manifest import repo_root, write_manifest
from catbond.pipeline import (
    ALT_WINDOW,
    BASE_N_YEARS,
    BASE_SEED,
    CALIBRATION,
    EXTENDED,
    GRID_ATTACHMENTS,
    GRID_LIMITS,
    RP_BOOTSTRAP,
    TERM_YEARS,
    TIV_USD,
    load_inputs,
    reference_layer,
)
from catbond.vulnerability import EMANUEL_V_THRESH_KT

OUT = repo_root() / "results" / "eventset"
EXPOSURE_OUT = repo_root() / "results" / "exposure"
CLIMADA_V_HALF_KT = 74.7 / 0.514444  # CLIMADA from_emanuel_usa default, converted to kt


def summary_row(name: str, es, a: float, e: float) -> dict:
    ev = es.events
    mx, sm = annual_max_and_sum(ev["year"], ev["loss_usd"], es.n_years)
    rp = return_period_losses(mx, (100, 250))
    m = layer_metrics(ev["year"], ev["loss_usd"], es.n_years, a, e, TERM_YEARS)
    return {
        "variant": name,
        "n_years": es.n_years,
        "events_per_year": len(ev) / es.n_years,
        "aal_pct_tiv": 100 * sm.mean() / TIV_USD,
        "oep100_pct_tiv": 100 * rp[0] / TIV_USD,
        "oep250_pct_tiv": 100 * rp[1] / TIV_USD,
        "ref_ap_pct": 100 * m.attachment_prob,
        "ref_ep_pct": 100 * m.exhaustion_prob,
        "ref_el_pct": 100 * m.expected_loss,
    }


def grid_rows(name: str, es) -> pd.DataFrame:
    """AP, EP and EL (percent) of every candidate-grid layer for one variant."""
    t = layer_table(
        es.events["year"],
        es.events["loss_usd"],
        es.n_years,
        GRID_ATTACHMENTS * TIV_USD,
        GRID_LIMITS * TIV_USD,
        TERM_YEARS,
    )
    return pd.DataFrame(
        {
            "variant": name,
            "attachment_pct_tiv": 100 * t["attachment_usd"] / TIV_USD,
            "limit_pct_tiv": 100 * t["limit_usd"] / TIV_USD,
            "ap_pct": 100 * t["attachment_prob"],
            "ep_pct": 100 * t["exhaustion_prob"],
            "el_pct": 100 * t["expected_loss"],
        }
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seed", type=int, default=BASE_SEED)
    ap.add_argument("--n-years", type=int, default=BASE_N_YEARS)
    args = ap.parse_args()
    inp = load_inputs()
    coast, ex = inp.region.coast_edges, inp.exposure
    OUT.mkdir(parents=True, exist_ok=True)
    EXPOSURE_OUT.mkdir(parents=True, exist_ok=True)
    ex.to_csv(EXPOSURE_OUT / "county_exposure.csv", index=False, float_format="%.6f")
    write_manifest(
        EXPOSURE_OUT,
        None,
        {
            "data": "HYPOTHETICAL portfolio: USD 100 bn TIV spread "
            "by 2020 Census households (UF BEBR table)"
        },
    )

    base_model = fit_hazard(inp.landfalls, *CALIBRATION)
    base = simulate_event_set(base_model, coast, ex, args.n_years, args.seed)
    base.events.to_csv(OUT / "event_table.csv.gz", index=False, float_format="%.6g")

    mx, sm = annual_max_and_sum(base.events["year"], base.events["loss_usd"], base.n_years)
    rows = []
    for kind, annual in (("OEP", mx), ("AEP", sm)):
        est = return_period_losses(annual, RETURN_PERIODS)
        se = return_period_losses_se(annual, RETURN_PERIODS, RP_BOOTSTRAP, seed=args.seed)
        for t, v, s in zip(RETURN_PERIODS, est, se, strict=True):
            rows.append(
                {
                    "curve": kind,
                    "return_period": t,
                    "loss_usd": v,
                    "se_usd": s,
                    "loss_pct_tiv": 100 * v / TIV_USD,
                    "se_pct_tiv": 100 * s / TIV_USD,
                }
            )
    rows.append(
        {
            "curve": "AAL",
            "return_period": np.nan,
            "loss_usd": sm.mean(),
            "se_usd": sm.std(ddof=1) / np.sqrt(base.n_years),
            "loss_pct_tiv": 100 * sm.mean() / TIV_USD,
            "se_pct_tiv": 100 * sm.std(ddof=1) / np.sqrt(base.n_years) / TIV_USD,
        }
    )
    rp = pd.DataFrame(rows)
    rp.to_csv(OUT / "rp_losses.csv", index=False, float_format="%.6g")

    oep = dict(zip(RETURN_PERIODS, return_period_losses(mx, RETURN_PERIODS), strict=True))
    a, e = reference_layer(oep)
    m = layer_metrics(base.events["year"], base.events["loss_usd"], base.n_years, a, e, TERM_YEARS)
    pd.DataFrame(
        [
            {
                "attachment_usd": a,
                "exhaustion_usd": e,
                "attachment_pct_tiv": 100 * a / TIV_USD,
                "exhaustion_pct_tiv": 100 * e / TIV_USD,
                "ap_pct": 100 * m.attachment_prob,
                "ep_pct": 100 * m.exhaustion_prob,
                "ep_occurrence_pct": 100 * m.ep_occurrence,
                "el_pct": 100 * m.expected_loss,
                f"prob_any_loss_{TERM_YEARS}y_pct": 100 * m.term_prob_any_loss,
            }
        ]
    ).to_csv(OUT / "reference_layer.csv", index=False, float_format="%.6g")

    base_settings = LossSettings()
    variants = [
        ("base", base_model, base_settings),
        ("frequency x 0.8", base_model.scaled(0.8), base_settings),
        ("frequency x 1.2", base_model.scaled(1.2), base_settings),
        (
            "intensity: KDE instead of GPD",
            fit_hazard(inp.landfalls, *CALIBRATION, "kde"),
            base_settings,
        ),
        ("window 1950-2019", fit_hazard(inp.landfalls, *ALT_WINDOW), base_settings),
        (
            "window 1900-2024 (2020-24 geometric)",
            fit_hazard(inp.landfalls_extended, *EXTENDED),
            base_settings,
        ),
        (
            "V_half 110 kt (Emanuel 2011 sensitivity)",
            base_model,
            replace(base_settings, v_half_kt=110.0),
        ),
        (
            "V_half 145.2 kt (CLIMADA default)",
            base_model,
            replace(base_settings, v_half_kt=CLIMADA_V_HALF_KT),
        ),
        (
            "V_thresh 40 kt",
            base_model,
            replace(base_settings, v_thresh_kt=EMANUEL_V_THRESH_KT - 10),
        ),
        ("deductible 0%", base_model, replace(base_settings, deductible=0.0)),
        ("deductible 5%", base_model, replace(base_settings, deductible=0.05)),
        ("RMW x 0.8", base_model, replace(base_settings, rmw_scale=0.8)),
        ("RMW x 1.2", base_model, replace(base_settings, rmw_scale=1.2)),
        ("no inland decay", base_model, replace(base_settings, decay=False)),
    ]
    sens, grids = [], []
    for name, model, st in variants:
        es = (
            base
            if name == "base"
            else simulate_event_set(model, coast, ex, args.n_years, args.seed, st)
        )
        sens.append(
            summary_row(name, es, a, e)
            | {
                "frequency_model": "poisson" if model.nb_size is None else "negbin",
                "freq_mean": model.freq_mean,
            }
        )
        print(sens[-1], flush=True)
        grids.append(grid_rows(name, es))
    pd.DataFrame(sens).to_csv(OUT / "sensitivity.csv", index=False, float_format="%.6g")
    pd.concat(grids, ignore_index=True).to_csv(
        OUT / "sensitivity_layers.csv", index=False, float_format="%.6g"
    )
    write_manifest(
        OUT,
        args.seed,
        {
            "data": "hazard fitted on real HURDAT2 landfalls; exposure hypothetical "
            "(USD 100 bn TIV over Florida counties by 2020 Census households)",
            "n_years": str(args.n_years),
            "rp_bootstrap": f"{RP_BOOTSTRAP} resamples of years",
            "common_random_numbers": "every variant uses the same seed",
        },
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
