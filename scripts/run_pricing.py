#!/usr/bin/env python3
"""Spread regression on the public deal table, exactly as fixed in docs/pricing_protocol.md.

REAL DATA: data/deals/cat_bond_deals.csv (terms of public cat bond tranches read from Artemis
deal pages). Models (OLS, log spread on log EL, both in percent):

  M1  log(spread) = a + b log(EL)
  M2  M1 + issue-year fixed effects
  M3  M2 + trigger type + log(size)

on five pre-specified samples (catbond.pricing.sample.SAMPLES). Standard errors: classical,
HC3, CR1 clustered by note series (deal_name) and CR1 clustered by sponsor. For b in M1 and M2:
restricted wild cluster bootstrap-t interval by test inversion (Rademacher weights, --n-boot
draws), clustered by series (primary) and by sponsor. Leave-one-series-out and
leave-one-sponsor-out prediction errors for every model and sample. By-year spread/EL multiples.

Writes results/pricing/: sample.csv, coefficients.csv, fit_stats.csv, slope_intervals.csv,
proportionality.csv, cv.csv, cv_rows.csv, year_multiple.csv, fitted_main.csv, MANIFEST.txt.

Usage: python3 scripts/run_pricing.py [--n-boot 9999] [--seed 6001] [--deals CSV] [--out DIR]
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import pandas as pd
from scipy import stats

from catbond.manifest import repo_root, write_manifest
from catbond.pricing.fit import cluster_labels, fit_log_spread
from catbond.pricing.multiple import spread_multiple
from catbond.pricing.sample import SAMPLES, flag_rows, regression_sample
from catbond.pricing.validation import leave_group_out, summarize_errors
from catbond.pricing.wild import WildClusterBootstrap

DEALS = repo_root() / "data" / "deals" / "cat_bond_deals.csv"
OUT = repo_root() / "results" / "pricing"
MODELS = {"M1": (), "M2": ("year_fe",), "M3": ("year_fe", "trigger", "log_size")}
WILD_MODELS = ("M1", "M2")
CLUSTERS = {"series": "deal_name", "sponsor": "sponsor"}
PRIMARY = "series"
LEVEL = 0.95
MAIN = "main (56 distinct layers)"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n-boot", type=int, default=9999)
    ap.add_argument("--seed", type=int, default=6001)
    ap.add_argument("--deals", type=pathlib.Path, default=DEALS)
    ap.add_argument("--out", type=pathlib.Path, default=OUT)
    args = ap.parse_args()
    out = args.out
    deals = pd.read_csv(args.deals)
    out.mkdir(parents=True, exist_ok=True)

    flagged = flag_rows(deals)
    for name, flags in SAMPLES.items():
        keep = ~flagged["duplicate"]
        for f in flags:
            keep &= ~flagged[f]
        flagged[f"in: {name}"] = keep
    flagged.drop(columns=["source_url", "access_date", "notes"]).to_csv(
        out / "sample.csv", index=False
    )

    coef_rows, stat_rows, slope_rows, cv_rows, cv_detail = [], [], [], [], []
    stream = 0
    for s_name, flags in SAMPLES.items():
        rows = regression_sample(deals, flags)
        for m_name, covs in MODELS.items():
            fit = fit_log_spread(rows, covs, cluster=CLUSTERS[PRIMARY])
            se_sp = fit.cluster_se(CLUSTERS["sponsor"])
            n_sp = int(cluster_labels(fit.data, CLUSTERS["sponsor"]).max()) + 1
            for i, term in enumerate(fit.names):
                coef_rows.append(
                    {
                        "sample": s_name,
                        "model": m_name,
                        "term": term,
                        "coef": fit.coef[i],
                        "se_ols": fit.se_ols[i],
                        "se_hc3": fit.se_hc3[i],
                        "se_cr1_series": fit.se_cluster[i],
                        "se_cr1_sponsor": se_sp[i],
                    }
                )
            stat_rows.append(
                {
                    "sample": s_name,
                    "model": m_name,
                    "n": fit.n,
                    "k": len(fit.coef),
                    "n_series": fit.n_clusters,
                    "n_sponsors": n_sp,
                    "sigma_log": fit.sigma,
                    "r2": fit.r2,
                    "dropped": "; ".join(f"{k}: {v}" for k, v in fit.report.dropped.items() if v),
                }
            )
            j = fit.names.index("log_el")
            b, df_res = fit.coef[j], fit.n - len(fit.coef)
            t_hc3 = stats.t.ppf(0.5 + LEVEL / 2, df_res)
            slope_rows.append(
                {
                    "sample": s_name,
                    "model": m_name,
                    "method": "HC3, t(n-k)",
                    "b": b,
                    "se": fit.se_hc3[j],
                    "lo": b - t_hc3 * fit.se_hc3[j],
                    "hi": b + t_hc3 * fit.se_hc3[j],
                    "n_clusters": "",
                    "n_boot": "",
                    "p_b_eq_1": 2 * stats.t.sf(abs(b - 1) / fit.se_hc3[j], df_res),
                }
            )
            for c_name, col in CLUSTERS.items():
                groups = cluster_labels(fit.data, col)
                g = int(groups.max()) + 1
                se = fit.se_cluster[j] if c_name == PRIMARY else se_sp[j]
                tq = stats.t.ppf(0.5 + LEVEL / 2, g - 1)
                slope_rows.append(
                    {
                        "sample": s_name,
                        "model": m_name,
                        "method": f"CR1 {c_name}, t(G-1)",
                        "b": b,
                        "se": se,
                        "lo": b - tq * se,
                        "hi": b + tq * se,
                        "n_clusters": g,
                        "n_boot": "",
                        "p_b_eq_1": 2 * stats.t.sf(abs(b - 1) / se, g - 1),
                    }
                )
                if m_name in WILD_MODELS:
                    rng = np.random.default_rng([args.seed, stream])
                    stream += 1
                    w = WildClusterBootstrap(fit.x, fit.y, groups, j, args.n_boot, rng)
                    r = w.interval(LEVEL, null_values=(1.0,))
                    slope_rows.append(
                        {
                            "sample": s_name,
                            "model": m_name,
                            "method": f"wild cluster bootstrap {c_name} (WCR, Rademacher)",
                            "b": r.estimate,
                            "se": r.se_cluster,
                            "lo": r.lo,
                            "hi": r.hi,
                            "n_clusters": r.n_clusters,
                            "n_boot": r.n_boot,
                            "p_b_eq_1": r.p_value_at[1.0],
                        }
                    )
            for scheme in ("series", "sponsor"):
                e = leave_group_out(rows, covs, CLUSTERS[scheme])
                cv_rows.append(
                    {"sample": s_name, "model": m_name, "held_out": scheme} | summarize_errors(e)
                )
                cv_detail.append(e.assign(sample=s_name, model=m_name, held_out=scheme))
            if s_name == MAIN:
                rows = rows.assign(**{f"fitted_{m_name}_pct": np.exp(fit.x @ fit.coef)})
        if s_name == MAIN:
            main_rows = rows

    slopes = pd.DataFrame(slope_rows)
    slopes.to_csv(out / "slope_intervals.csv", index=False, float_format="%.6g")
    pd.DataFrame(coef_rows).to_csv(out / "coefficients.csv", index=False, float_format="%.6g")
    pd.DataFrame(stat_rows).to_csv(out / "fit_stats.csv", index=False, float_format="%.6g")
    pd.DataFrame(cv_rows).to_csv(out / "cv.csv", index=False, float_format="%.6g")
    pd.concat(cv_detail, ignore_index=True).to_csv(
        out / "cv_rows.csv", index=False, float_format="%.6g"
    )

    # the pre-specified finding rule: proportionality (b = 1) is rejected only if the primary
    # (series) wild bootstrap interval excludes 1 in both M1 and M2
    prop = []
    primary = slopes["method"] == f"wild cluster bootstrap {PRIMARY} (WCR, Rademacher)"
    for s_name in SAMPLES:
        sub = slopes[primary & (slopes["sample"] == s_name)].set_index("model")
        excl = {m: not (sub.loc[m, "lo"] <= 1.0 <= sub.loc[m, "hi"]) for m in WILD_MODELS}
        prop.append(
            {
                "sample": s_name,
                "M1_b": sub.loc["M1", "b"],
                "M1_lo": sub.loc["M1", "lo"],
                "M1_hi": sub.loc["M1", "hi"],
                "M2_b": sub.loc["M2", "b"],
                "M2_lo": sub.loc["M2", "lo"],
                "M2_hi": sub.loc["M2", "hi"],
                "proportionality_rejected": all(excl.values()),
            }
        )
    pd.DataFrame(prop).to_csv(out / "proportionality.csv", index=False, float_format="%.6g")

    keep_cols = [
        "deal_name",
        "tranche",
        "sponsor",
        "issue_year",
        "expected_loss_pct",
        "spread_pct",
        "fitted_M1_pct",
        "fitted_M2_pct",
        "fitted_M3_pct",
    ]
    main_rows[keep_cols].to_csv(out / "fitted_main.csv", index=False, float_format="%.6g")
    mult = main_rows.assign(
        multiple=spread_multiple(main_rows["spread_pct"], main_rows["expected_loss_pct"])
    )
    yr = (
        mult.groupby("issue_year")
        .agg(
            n=("multiple", "size"),
            median_el_pct=("expected_loss_pct", "median"),
            median_spread_pct=("spread_pct", "median"),
            median_multiple=("multiple", "median"),
            min_multiple=("multiple", "min"),
            max_multiple=("multiple", "max"),
        )
        .reset_index()
    )
    allrow = {
        "issue_year": "all",
        "n": len(mult),
        "median_el_pct": mult["expected_loss_pct"].median(),
        "median_spread_pct": mult["spread_pct"].median(),
        "median_multiple": mult["multiple"].median(),
        "min_multiple": mult["multiple"].min(),
        "max_multiple": mult["multiple"].max(),
    }
    yr = pd.concat([yr, pd.DataFrame([allrow])], ignore_index=True)
    yr.to_csv(out / "year_multiple.csv", index=False, float_format="%.4g")
    write_manifest(
        out,
        args.seed,
        {
            "data": "REAL public deal terms, data/deals/cat_bond_deals.csv (Artemis pages)"
            if args.deals.resolve() == DEALS.resolve()
            else "deal table given by --deals (not the committed table)",
            "protocol": "docs/pricing_protocol.md (committed before this script was first run)",
            "n_boot": str(args.n_boot),
            "bootstrap_streams": "numpy default_rng([seed, i]), i = 0, 1, ... in the order "
            "sample x model (M1, M2) x cluster (series, sponsor)",
        },
    )
    print(pd.DataFrame(prop).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
