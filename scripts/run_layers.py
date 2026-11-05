#!/usr/bin/env python3
"""Candidate layer grid from the base-case event set (run scripts/run_eventset.py first).

For every attachment (0.25 to 4.00 percent of TIV in steps of 0.25) and limit (0.5, 1, 2 and 3
percent of TIV): attachment probability, exhaustion probability (annual, aggregate and
occurrence-only), expected loss as percent of the limit with its Monte Carlo standard error (SE of
a mean of independent simulated years), and the probability of any principal loss over a 3-year
term. Teaching-grade model with a hypothetical portfolio: not comparable to a vendor EL.

Writes results/layers/layer_grid.csv and MANIFEST.txt.

Usage: python3 scripts/run_layers.py
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from catbond.layers import annual_principal_loss, layer_table
from catbond.manifest import repo_root, write_manifest
from catbond.pipeline import GRID_ATTACHMENTS, GRID_LIMITS, TERM_YEARS, TIV_USD

OUT = repo_root() / "results" / "layers"
EVENTS = repo_root() / "results" / "eventset" / "event_table.csv.gz"


def main() -> int:
    ev = pd.read_csv(EVENTS)
    manifest = (EVENTS.parent / "MANIFEST.txt").read_text()
    n_years = int(
        next(ln.split(":", 1)[1] for ln in manifest.splitlines() if ln.startswith("n_years:"))
    )
    t = layer_table(
        ev["year"],
        ev["loss_usd"],
        n_years,
        GRID_ATTACHMENTS * TIV_USD,
        GRID_LIMITS * TIV_USD,
        TERM_YEARS,
    )
    se = []
    for a, e in zip(t["attachment_usd"], t["exhaustion_usd"], strict=True):
        p = annual_principal_loss(ev["year"], ev["loss_usd"], n_years, a, e) / (e - a)
        se.append(p.std(ddof=1) / np.sqrt(n_years))
    out = pd.DataFrame(
        {
            "attachment_pct_tiv": 100 * t["attachment_usd"] / TIV_USD,
            "limit_pct_tiv": 100 * t["limit_usd"] / TIV_USD,
            "exhaustion_pct_tiv": 100 * t["exhaustion_usd"] / TIV_USD,
            "attachment_usd": t["attachment_usd"],
            "exhaustion_usd": t["exhaustion_usd"],
            "ap_pct": 100 * t["attachment_prob"],
            "ep_pct": 100 * t["exhaustion_prob"],
            "ep_occurrence_pct": 100 * t["ep_occurrence"],
            "el_pct": 100 * t["expected_loss"],
            "el_se_pct": 100 * np.array(se),
            f"prob_any_loss_{TERM_YEARS}y_pct": 100 * t["term_prob_any_loss"],
            "n_years": n_years,
        }
    )
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT / "layer_grid.csv", index=False, float_format="%.6g")
    write_manifest(
        OUT,
        None,
        {
            "input": "results/eventset/event_table.csv.gz (base case)",
            "data": "hazard from real HURDAT2; exposure hypothetical",
        },
    )
    print(out.head(12).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
