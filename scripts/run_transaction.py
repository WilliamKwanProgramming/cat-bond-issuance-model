#!/usr/bin/env python3
"""Hypothetical three-layer transaction: layer metrics, EL band, regression spread range.

HYPOTHETICAL sponsor and portfolio; the layer metrics come from the teaching-grade loss model
(hazard fitted to real HURDAT2 landfalls), the spread mapping from M1 and M2 fitted on the REAL
public deal table. The spread range is a regression output, not a quote. Rules fixed in
docs/pricing_protocol.md ("Hypothetical transaction").

Needs results/layers/layer_grid.csv, results/param_uncertainty/grid_summary.csv and
results/eventset/sensitivity_layers.csv. Writes results/transaction/: layers.csv, spreads.csv,
comparables.csv, sensitivity.csv, MANIFEST.txt.

Usage: python3 scripts/run_transaction.py
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from catbond.manifest import repo_root, write_manifest
from catbond.pipeline import TIV_USD
from catbond.pricing.comparables import comparable_deals
from catbond.pricing.fit import fit_log_spread
from catbond.pricing.sample import regression_sample
from catbond.transaction import choose_stack, spread_at

R = repo_root() / "results"
OUT = R / "transaction"
DEALS = repo_root() / "data" / "deals" / "cat_bond_deals.csv"
MODELS = {"M1": (), "M2": ("year_fe",)}
YEAR = 2025
PI_LEVEL = 0.90
K_COMPARABLES = 5
PROTOCOL_VARIANTS = (
    "base",
    "frequency x 0.8",
    "frequency x 1.2",
    "V_half 110 kt (Emanuel 2011 sensitivity)",
    "V_half 145.2 kt (CLIMADA default)",
    "V_thresh 40 kt",
    "window 1950-2019",
)
KEYS = ["attachment_pct_tiv", "limit_pct_tiv"]


def _match(df: pd.DataFrame, a: float, lim: float) -> pd.DataFrame:
    return df[np.isclose(df[KEYS[0]], a) & np.isclose(df[KEYS[1]], lim)]


def main() -> int:
    grid = pd.read_csv(R / "layers" / "layer_grid.csv")
    band = pd.read_csv(R / "param_uncertainty" / "grid_summary.csv")
    sens = pd.read_csv(R / "eventset" / "sensitivity_layers.csv")
    stack = choose_stack(grid)
    deals = regression_sample(pd.read_csv(DEALS))
    fits = {m: fit_log_spread(deals, covs, cluster="deal_name") for m, covs in MODELS.items()}

    layers, spreads, comps, sens_rows = [], [], [], []
    for r in stack.itertuples():
        a, lim = r.attachment_pct_tiv, r.limit_pct_tiv
        b = _match(band, a, lim).iloc[0]
        name = (
            f"Layer {r.layer} ({'bottom' if r.layer == 1 else 'middle' if r.layer == 2 else 'top'})"
        )
        layers.append(
            {
                "layer": name,
                "target_el_pct": r.target_el_pct,
                "attachment_pct_tiv": a,
                "exhaustion_pct_tiv": a + lim,
                "limit_pct_tiv": lim,
                "attachment_usd_m": a / 100 * TIV_USD / 1e6,
                "limit_usd_m": lim / 100 * TIV_USD / 1e6,
                "ap_pct": r.ap_pct,
                "ep_pct": r.ep_pct,
                "ep_occurrence_pct": r.ep_occurrence_pct,
                "el_pct": r.el_pct,
                "el_mc_se_pct": r.el_se_pct,
                "el_band_q05_pct": b["el_q05"],
                "el_band_q50_pct": b["el_q50"],
                "el_band_q95_pct": b["el_q95"],
                "n_band_replicates": int(b["n_replicates"]),
                "prob_any_loss_3y_pct": r.prob_any_loss_3y_pct,
            }
        )
        for m, fit in fits.items():
            for case, el in (
                ("EL point", r.el_pct),
                ("EL band q05", b["el_q05"]),
                ("EL band q95", b["el_q95"]),
            ):
                spreads.append(
                    {"layer": name, "model": m, "issue_year": YEAR, "el_case": case, "el_pct": el}
                    | spread_at(fit, float(el), YEAR, PI_LEVEL)
                )
        c = comparable_deals(deals, float(r.el_pct), YEAR, k=K_COMPARABLES)
        for rank, cr in enumerate(c.itertuples(), start=1):
            comps.append(
                {
                    "layer": name,
                    "rank": rank,
                    "deal_name": cr.deal_name,
                    "tranche": cr.tranche,
                    "sponsor": cr.sponsor,
                    "issue_year": cr.issue_year,
                    "expected_loss_pct": cr.expected_loss_pct,
                    "spread_pct": cr.spread_pct,
                    "multiple": cr.multiple,
                    "distance": cr.distance,
                    "source_url": cr.source_url,
                }
            )
        base_el = None
        for variant, sv in sens.groupby("variant", sort=False):
            row = _match(sv, a, lim).iloc[0]
            el = float(row["el_pct"])
            base_el = el if variant == "base" else base_el
            rec = {
                "layer": name,
                "variant": variant,
                "in_protocol_list": variant in PROTOCOL_VARIANTS,
                "ap_pct": row["ap_pct"],
                "ep_pct": row["ep_pct"],
                "el_pct": el,
            }
            for m, fit in fits.items():
                rec[f"{m}_spread_median_pct"] = spread_at(fit, el, YEAR)["spread_median_pct"]
            sens_rows.append(rec)
    sens_df = pd.DataFrame(sens_rows)
    base = sens_df[sens_df["variant"] == "base"].set_index("layer")
    sens_df["el_ratio_to_base"] = sens_df["el_pct"] / sens_df["layer"].map(base["el_pct"])
    for m in fits:
        col = f"{m}_spread_median_pct"
        sens_df[f"{m}_change_pp"] = sens_df[col] - sens_df["layer"].map(base[col])

    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(layers).to_csv(OUT / "layers.csv", index=False, float_format="%.6g")
    pd.DataFrame(spreads).to_csv(OUT / "spreads.csv", index=False, float_format="%.6g")
    pd.DataFrame(comps).to_csv(OUT / "comparables.csv", index=False, float_format="%.6g")
    sens_df.to_csv(OUT / "sensitivity.csv", index=False, float_format="%.6g")
    write_manifest(
        OUT,
        None,
        {
            "data": "HYPOTHETICAL portfolio and sponsor; hazard from real HURDAT2; spread "
            "mapping fitted on REAL public deal terms (56 distinct layers)",
            "layer_choice": f"choose_stack on results/layers/layer_grid.csv, criterion "
            f"{stack.attrs['criterion']:.6g}",
            "models": "M1, M2 (year fixed effects evaluated at 2025); 90% prediction intervals",
            "inputs": "results/layers/layer_grid.csv; results/param_uncertainty/grid_summary.csv;"
            " results/eventset/sensitivity_layers.csv; data/deals/cat_bond_deals.csv",
        },
    )
    print(pd.DataFrame(layers).T.to_string())
    print(pd.DataFrame(spreads).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
