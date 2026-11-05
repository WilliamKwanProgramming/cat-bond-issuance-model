#!/usr/bin/env python3
"""POST HOC diagnostics of the spread regression on the public deal table (REAL data).

Not part of docs/pricing_protocol.md: these were written after the protocol's results had been
seen, in answer to an independent review, and are reported whatever they show. They do not
change the pre-specified test of proportionality (scripts/run_pricing.py).

Writes results/pricing_posthoc/:
  composition.csv        regions and perils of the 56 regression layers
  r2_partition.csv       R-squared of log spread on log EL only, issue year only, and both
  specifications.csv     slope b with region-group and multi-peril indicators; Florida layers only
  florida_layer_spreads.csv  median spread of the three hypothetical layers (Florida only), protocol
                         M2 against M2 with region-group indicators (needs results/transaction/)
  implied_error_sd.csv   error SD in the stated EL that would turn a true slope of 1 into the
                         estimated slopes if the error is classical
  attenuation.csv        simulation: mean slope under classical and under Berkson error
  MANIFEST.txt

Usage: python3 scripts/run_pricing_posthoc.py [--reps 2000] [--seed 7001] [--deals CSV] [--out DIR]
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import pandas as pd

from catbond.manifest import repo_root, write_manifest
from catbond.pricing.fit import fit_log_spread
from catbond.pricing.posthoc import (
    attenuation_simulation,
    composition,
    florida_layer_spreads,
    implied_error_sd,
    r2_partition,
    specifications,
)
from catbond.pricing.sample import regression_sample

DEALS = repo_root() / "data" / "deals" / "cat_bond_deals.csv"
OUT = repo_root() / "results" / "pricing_posthoc"
TRANSACTION = repo_root() / "results" / "transaction"
SIGMAS = (0.0, 0.1, 0.2, 0.25)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--reps", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=7001)
    ap.add_argument("--deals", type=pathlib.Path, default=DEALS)
    ap.add_argument("--out", type=pathlib.Path, default=OUT)
    args = ap.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    sample = regression_sample(pd.read_csv(args.deals))
    composition(sample).to_csv(out / "composition.csv", index=False)
    r2_partition(sample).to_csv(out / "r2_partition.csv", index=False, float_format="%.6g")
    specifications(sample).to_csv(out / "specifications.csv", index=False, float_format="%.6g")

    layers = pd.read_csv(TRANSACTION / "layers.csv")
    florida_layer_spreads(sample, layers).to_csv(
        out / "florida_layer_spreads.csv", index=False, float_format="%.6g"
    )

    fits = {
        "M1": fit_log_spread(sample, (), cluster="deal_name"),
        "M2": fit_log_spread(sample, ("year_fe",), cluster="deal_name"),
    }
    b_hat = {m: float(f.coef[f.names.index("log_el")]) for m, f in fits.items()}
    implied_error_sd(sample, b_hat).to_csv(
        out / "implied_error_sd.csv", index=False, float_format="%.6g"
    )
    attenuation_simulation(
        sample, SIGMAS, resid_sd=fits["M2"].sigma, n_rep=args.reps, seed=args.seed
    ).to_csv(out / "attenuation.csv", index=False, float_format="%.6g")
    write_manifest(
        out,
        args.seed,
        {
            "data": "REAL: data/deals/cat_bond_deals.csv, main regression sample (56 layers)",
            "status": "POST HOC; not in docs/pricing_protocol.md",
            "simulation": f"{args.reps} replications per error SD; residual SD of M2 "
            f"({fits['M2'].sigma:.4f}); true slope 1; no intercept, no year effects",
        },
    )
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
