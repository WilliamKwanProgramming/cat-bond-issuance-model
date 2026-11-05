#!/usr/bin/env python3
"""Build the Markdown tables quoted in README.md and docs/ from the files in results/.

Writes results/tables.md. Every number in it is read from a results file; nothing is typed in.

Usage: python3 scripts/make_tables.py
"""

from __future__ import annotations

import sys

import pandas as pd

from catbond.manifest import repo_root

R = repo_root() / "results"


def md(df: pd.DataFrame, fmt: dict[str, str] | None = None) -> str:
    fmt = fmt or {}
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if isinstance(v, float) and pd.isna(v):
                cells.append("")
            elif c in fmt:
                cells.append(format(v, fmt[c]))
            else:
                cells.append(str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def section(title: str, body: str, source: str) -> str:
    return f"### {title}\n\n{body}\n\nSource: `{source}`\n"


def ci(b: float, lo: float, hi: float, f: str = ".3f") -> str:
    """'b [lo, hi]' with the given number format."""
    return f"{b:{f}} [{lo:{f}}, {hi:{f}}]"


def pricing_sections() -> list[str]:
    """Tables of the spread regression on the real deal table (results/pricing/)."""
    d = R / "pricing"
    if not (d / "slope_intervals.csv").exists():
        return []
    real = "REAL data: public deal terms from Artemis pages"
    out = []
    sl = pd.read_csv(d / "slope_intervals.csv")
    main = sl[sl["sample"] == sl["sample"].iloc[0]]
    t = pd.DataFrame(
        {
            "model": main["model"],
            "method": main["method"],
            "b [95% interval]": [ci(r.b, r.lo, r.hi) for r in main.itertuples()],
            "p (b = 1)": main["p_b_eq_1"].map(lambda v: f"{v:.4f}"),
            "clusters": main["n_clusters"].map(lambda v: "" if pd.isna(v) else str(int(v))),
            "bootstrap draws": main["n_boot"].map(lambda v: "" if pd.isna(v) else str(int(v))),
        }
    )
    out.append(
        section(
            f"Slope b of log spread on log EL, main sample of 56 distinct layers ({real})",
            md(t),
            "results/pricing/slope_intervals.csv",
        )
    )
    pr = pd.read_csv(d / "proportionality.csv")
    t = pd.DataFrame(
        {
            "sample": pr["sample"],
            "M1 b [wild 95%, series]": [ci(r.M1_b, r.M1_lo, r.M1_hi) for r in pr.itertuples()],
            "M2 b [wild 95%, series]": [ci(r.M2_b, r.M2_lo, r.M2_hi) for r in pr.itertuples()],
            "b = 1 rejected (both exclude 1)": pr["proportionality_rejected"].map(
                {True: "yes", False: "no"}
            ),
        }
    )
    out.append(
        section(
            f"Pre-specified test of proportionality (b = 1) in every sample ({real})",
            md(t),
            "results/pricing/proportionality.csv",
        )
    )
    m3 = sl[(sl["model"] == "M3") & (sl["method"] == "CR1 series, t(G-1)")]
    t = pd.DataFrame(
        {
            "sample": m3["sample"],
            "M3 b [CR1 series, t(G-1) 95%]": [ci(r.b, r.lo, r.hi) for r in m3.itertuples()],
        }
    )
    out.append(
        section(
            f"M3 slope in every sample ({real}; no bootstrap for M3, as pre-specified)",
            md(t),
            "results/pricing/slope_intervals.csv",
        )
    )
    fs = pd.read_csv(d / "fit_stats.csv")
    out.append(
        section(
            f"Fit statistics ({real})",
            md(fs.drop(columns=["dropped"]), {"sigma_log": ".3f", "r2": ".3f"}),
            "results/pricing/fit_stats.csv",
        )
    )
    cv = pd.read_csv(d / "cv.csv")
    out.append(
        section(
            f"Leave-one-series-out and leave-one-sponsor-out prediction errors ({real})",
            md(
                cv,
                {
                    "rmse_log": ".3f",
                    "mae_log": ".3f",
                    "rmse_pp": ".2f",
                    "mae_pp": ".2f",
                    "median_abs_pp": ".2f",
                },
            ),
            "results/pricing/cv.csv",
        )
    )
    co = pd.read_csv(d / "coefficients.csv")
    co = co[(co["sample"] == co["sample"].iloc[0]) & (co["model"] != "M1")]
    out.append(
        section(
            f"Coefficients of M2 and M3, main sample ({real}; standard errors, no tests)",
            md(
                co.drop(columns=["sample"]),
                {c: ".3f" for c in ("coef", "se_ols", "se_hc3", "se_cr1_series", "se_cr1_sponsor")},
            ),
            "results/pricing/coefficients.csv",
        )
    )
    ym = pd.read_csv(d / "year_multiple.csv", dtype={"issue_year": str})
    out.append(
        section(
            f"Spread / EL multiple by issue year, 56 distinct layers ({real})",
            md(
                ym,
                {
                    "median_el_pct": ".2f",
                    "median_spread_pct": ".2f",
                    "median_multiple": ".2f",
                    "min_multiple": ".2f",
                    "max_multiple": ".2f",
                },
            ),
            "results/pricing/year_multiple.csv",
        )
    )
    return out


def posthoc_sections() -> list[str]:
    """Tables of the POST HOC diagnostics (results/pricing_posthoc/), labelled as such."""
    d = R / "pricing_posthoc"
    if not (d / "composition.csv").exists():
        return []
    tag = "POST HOC, not in the protocol; REAL data: public deal terms from Artemis pages"
    out = []
    comp = pd.read_csv(d / "composition.csv")
    cross = comp[comp["dimension"] == "region x peril"].pivot(
        index="group", columns="peril_group", values="n_layers"
    )
    cross = cross.loc[comp[comp["dimension"] == "region"]["group"]]
    cross["all"] = cross.sum(axis=1)
    cross = cross.reset_index().rename(columns={"group": "region group"})
    cross.loc[len(cross)] = ["all", *cross.drop(columns="region group").sum().astype(int)]
    out.append(
        section(
            f"Composition of the 56 regression layers by region and peril ({tag})",
            md(cross),
            "results/pricing_posthoc/composition.csv",
        )
    )
    r2 = pd.read_csv(d / "r2_partition.csv")
    out.append(
        section(
            f"R-squared of log spread: log EL only, issue year only, both ({tag})",
            md(r2, {"r2": ".3f", "adj_r2": ".3f"}),
            "results/pricing_posthoc/r2_partition.csv",
        )
    )
    sp = pd.read_csv(d / "specifications.csv")

    def hc3(r) -> str:
        return "undefined" if pd.isna(r.lo_hc3) else f"[{r.lo_hc3:.3f}, {r.hi_hc3:.3f}]"

    t = pd.DataFrame(
        {
            "specification": sp["specification"],
            "n (series)": [f"{r.n} ({r.n_series})" for r in sp.itertuples()],
            "b [95% CR1 series, t(G-1)]": [ci(r.b, r.lo_cr1, r.hi_cr1) for r in sp.itertuples()],
            "HC3 interval": [hc3(r) for r in sp.itertuples()],
            "indicator terms: coefficient (CR1 se)": sp["indicator_terms"].fillna(""),
        }
    )
    out.append(
        section(
            f"Slope b with region and peril indicators, and on Florida layers only ({tag}; "
            "reference levels: Florida and other states, named storm and other perils)",
            md(t),
            "results/pricing_posthoc/specifications.csv",
        )
    )
    fl = pd.read_csv(d / "florida_layer_spreads.csv")
    t = pd.DataFrame(
        {
            "layer": fl["layer"],
            "model": fl["model"],
            "EL %": fl["el_pct"].map(lambda v: f"{v:.2f}"),
            "median spread % [90% PI]": [
                ci(r.spread_median_pct, r.pi_lo_pct, r.pi_hi_pct, ".2f") for r in fl.itertuples()
            ],
            "multiple": fl["multiple"].map(lambda v: f"{v:.2f}"),
        }
    )
    out.append(
        section(
            f"Median spread of the hypothetical layers (issue year 2025, Florida only): protocol "
            f"M2 against M2 with region-group indicators ({tag}; HYPOTHETICAL layers)",
            md(t),
            "results/pricing_posthoc/florida_layer_spreads.csv",
        )
    )
    im = pd.read_csv(d / "implied_error_sd.csv")
    out.append(
        section(
            f"Error SD in the stated log EL that would turn a true slope of 1 into the estimate, "
            f"if the error is classical ({tag})",
            md(im, {"b_hat": ".3f", "var_stated_log_el": ".4f", "implied_error_sd_log": ".3f"}),
            "results/pricing_posthoc/implied_error_sd.csv",
        )
    )
    at = pd.read_csv(d / "attenuation.csv")
    wide = at.pivot_table(
        index="error_sd_log_el", columns=["scenario", "model"], values="mean_b_hat"
    )
    t = pd.DataFrame({"error SD (log EL)": [f"{v:.2f}" for v in wide.index]})
    for sc, m in wide.columns:
        t[f"{sc} {m}"] = [f"{v:.3f}" for v in wide[(sc, m)]]
    out.append(
        section(
            f"Mean estimated slope when the true slope is 1, with classical and with Berkson "
            f"error in the stated EL (SIMULATION on the real table's design; {at['n_rep'].iloc[0]} "
            f"replications per row; {tag})",
            md(t),
            "results/pricing_posthoc/attenuation.csv",
        )
    )
    return out


def transaction_sections() -> list[str]:
    """Tables of the hypothetical transaction (results/transaction/)."""
    d = R / "transaction"
    if not (d / "layers.csv").exists():
        return []
    tag = "HYPOTHETICAL sponsor and portfolio; hazard fitted to real HURDAT2"
    out = []
    la = pd.read_csv(d / "layers.csv")
    t = pd.DataFrame(
        {
            "layer": la["layer"],
            "attach - exhaust (% TIV)": [
                f"{r.attachment_pct_tiv:.2f} - {r.exhaustion_pct_tiv:.2f}" for r in la.itertuples()
            ],
            "attach - exhaust (USD m)": [
                f"{r.attachment_usd_m:,.0f} - {r.attachment_usd_m + r.limit_usd_m:,.0f}"
                for r in la.itertuples()
            ],
            "AP %": la["ap_pct"].map(lambda v: f"{v:.2f}"),
            "EP %": la["ep_pct"].map(lambda v: f"{v:.2f}"),
            "EL % (MC SE)": [f"{r.el_pct:.2f} ({r.el_mc_se_pct:.2f})" for r in la.itertuples()],
            "EL hazard-parameter band 5-95%": [
                f"{r.el_band_q05_pct:.2f} - {r.el_band_q95_pct:.2f}" for r in la.itertuples()
            ],
            "P(any loss in 3 years) %": la["prob_any_loss_3y_pct"].map(lambda v: f"{v:.1f}"),
        }
    )
    out.append(
        section(
            f"Hypothetical three-layer transaction ({tag}; 100,000 simulated years, seed 1; "
            "band from 200 bootstrap refits x 10,000 years)",
            md(t),
            "results/transaction/layers.csv",
        )
    )
    sp = pd.read_csv(d / "spreads.csv")
    t = pd.DataFrame(
        {
            "layer": sp["layer"],
            "model": sp["model"],
            "EL case": sp["el_case"],
            "EL %": sp["el_pct"].map(lambda v: f"{v:.2f}"),
            "median spread % [90% PI]": [
                ci(r.spread_median_pct, r.pi_lo_pct, r.pi_hi_pct, ".2f") for r in sp.itertuples()
            ],
            "multiple": sp["multiple"].map(lambda v: f"{v:.2f}"),
        }
    )
    out.append(
        section(
            "Regression spread range for the hypothetical layers (M1, M2 at issue year 2025, "
            "fitted on the REAL deal table; a regression output, not a quote)",
            md(t),
            "results/transaction/spreads.csv",
        )
    )
    co = pd.read_csv(d / "comparables.csv")
    out.append(
        section(
            "Five nearest public layers by EL and issue year (REAL data: Artemis pages)",
            md(
                co.drop(columns=["source_url", "sponsor"]),
                {"spread_pct": ".2f", "multiple": ".2f", "distance": ".3f"},
            ),
            "results/transaction/comparables.csv",
        )
    )
    se = pd.read_csv(d / "sensitivity.csv")
    out.append(
        section(
            f"Sensitivity of layer EL and implied spread ({tag}; same seed, 100,000 years "
            "per variant; in_protocol_list marks the variants named in the protocol)",
            md(
                se.drop(columns=["ap_pct", "ep_pct"]),
                {
                    "el_pct": ".2f",
                    "M1_spread_median_pct": ".2f",
                    "M2_spread_median_pct": ".2f",
                    "el_ratio_to_base": ".2f",
                    "M1_change_pp": ".2f",
                    "M2_change_pp": ".2f",
                },
            ),
            "results/transaction/sensitivity.csv",
        )
    )
    return out


def main() -> int:
    parts = [
        "# Generated tables\n\nGenerated by `scripts/make_tables.py` from `results/`; "
        "do not edit by hand.\n"
    ]

    ag = pd.read_csv(R / "hazard" / "definition_agreement.csv")
    parts.append(
        section(
            "Landfall definitions compared (real data: HURDAT2)",
            md(
                ag.rename(
                    columns={"n_first": "n (first)", "n_second": "n (second)", "n_both": "n (both)"}
                ),
                {"median_position_diff_km": ".1f", "max_abs_wind_diff_kt": ".1f"},
            ),
            "results/hazard/definition_agreement.csv",
        )
    )

    nm = pd.read_csv(R / "hazard" / "near_miss_hurricanes.csv")
    n_lf = len(pd.read_csv(R / "hazard" / "landfalls.csv").query("1900 <= year <= 2019"))
    parts.append(
        section(
            f"Hurricane-strength storms within 50 km of the Florida coastline in 1900-2019 that "
            f"are not counted as Florida landfalls ({len(nm)} storms, against {n_lf} counted "
            "landfalls; real data: HURDAT2; not in the simulated event set)",
            md(
                nm.drop(columns=["storm_id", "time"]),
                {"lat": ".1f", "lon": ".1f", "wind": ".0f", "distance_florida_km": ".0f",
                 "distance_other_coast_km": ".0f"},
            ),
            "results/hazard/near_miss_hurricanes.csv",
        )
    )

    fr = pd.read_csv(R / "hazard" / "frequency.csv")
    cols = [
        "window",
        "n_years",
        "n_landfalls",
        "mean_per_year",
        "mean_ci95_lo",
        "mean_ci95_hi",
        "variance",
        "dispersion_p",
        "nb_size",
        "nb_lr_p",
        "chosen",
        "trend_rate_ratio_per_century",
        "trend_p",
        "decade_dispersion_p",
        "ljung_box_p",
    ]
    parts.append(
        section(
            "Annual Florida hurricane landfall counts (real data: HURDAT2)",
            md(
                fr[cols],
                {
                    "mean_per_year": ".3f",
                    "mean_ci95_lo": ".3f",
                    "mean_ci95_hi": ".3f",
                    "variance": ".3f",
                    "dispersion_p": ".3f",
                    "nb_size": ".2f",
                    "nb_lr_p": ".3f",
                    "trend_rate_ratio_per_century": ".2f",
                    "trend_p": ".3f",
                    "decade_dispersion_p": ".3f",
                    "ljung_box_p": ".3f",
                },
            ),
            "results/hazard/frequency.csv",
        )
    )

    it = pd.read_csv(R / "hazard" / "intensity.csv")
    prim = it[it["window"] == "1900-2019 (primary)"]
    head = prim.iloc[0]
    body = md(
        prim[["group", "observed", "observed_share", "gpd_prob", "kde_prob"]],
        {"observed_share": ".3f", "gpd_prob": ".3f", "kde_prob": ".3f"},
    )
    body += (
        f"\n\nGPD fit: xi = {head['gpd_xi']:.3f}, sigma = {head['gpd_sigma']:.1f} kt; "
        f"KDE bandwidth {head['kde_bandwidth']:.1f} kt; Kolmogorov-Smirnov distance "
        f"{head['ks_gpd']:.3f} (GPD), {head['ks_kde']:.3f} (KDE); n = {int(head['n'])}."
    )
    parts.append(
        section(
            "Landfall intensity, 1900-2019 (real data: HURDAT2)",
            body,
            "results/hazard/intensity.csv",
        )
    )

    mo = pd.read_csv(R / "hazard" / "motion_by_coast.csv")
    parts.append(
        section(
            "Heading and translation speed at landfall by coast segment "
            "(real data: HURDAT2, 1900-2019)",
            md(
                mo,
                {
                    "circular_mean_heading_deg": ".0f",
                    "mean_resultant_length": ".2f",
                    "median_speed_kmh": ".1f",
                    "speed_q25": ".1f",
                    "speed_q75": ".1f",
                },
            ),
            "results/hazard/motion_by_coast.csv",
        )
    )

    oos = pd.read_csv(R / "hazard_oos" / "oos_by_category.csv")
    parts.append(
        section(
            "Out-of-sample check: fit 1900-1979, landfalls in 1980-2019 (real data: HURDAT2)",
            md(
                oos,
                {
                    "predicted_mean": ".1f",
                    "plugin_lo": ".0f",
                    "plugin_hi": ".0f",
                    "boot_lo": ".0f",
                    "boot_hi": ".0f",
                    "observed_ci95_lo": ".1f",
                    "observed_ci95_hi": ".1f",
                    "p_two_sided": ".2f",
                },
            ),
            "results/hazard_oos/oos_by_category.csv",
        )
    )

    rp = pd.read_csv(R / "eventset" / "rp_losses.csv")
    rp["return_period"] = rp["return_period"].map(lambda v: "" if pd.isna(v) else f"{int(v)}")
    parts.append(
        section(
            "Portfolio losses, base case (hazard fitted on real data; hypothetical portfolio, "
            "USD 100 bn TIV; 100,000 simulated years)",
            md(
                rp[["curve", "return_period", "loss_pct_tiv", "se_pct_tiv"]],
                {"loss_pct_tiv": ".3f", "se_pct_tiv": ".3f"},
            ),
            "results/eventset/rp_losses.csv",
        )
    )

    ref = pd.read_csv(R / "eventset" / "reference_layer.csv")
    parts.append(
        section(
            "Reference layer, base case (hypothetical portfolio)",
            md(
                ref.drop(columns=["attachment_usd", "exhaustion_usd"]),
                {c: ".3f" for c in ref.columns},
            ),
            "results/eventset/reference_layer.csv",
        )
    )

    se = pd.read_csv(R / "eventset" / "sensitivity.csv")
    parts.append(
        section(
            "Sensitivity runs (same seed, 100,000 years each; reference layer fixed in USD)",
            md(
                se[
                    [
                        "variant",
                        "frequency_model",
                        "events_per_year",
                        "aal_pct_tiv",
                        "oep100_pct_tiv",
                        "oep250_pct_tiv",
                        "ref_ap_pct",
                        "ref_ep_pct",
                        "ref_el_pct",
                    ]
                ],
                {
                    "events_per_year": ".3f",
                    "aal_pct_tiv": ".3f",
                    "oep100_pct_tiv": ".2f",
                    "oep250_pct_tiv": ".2f",
                    "ref_ap_pct": ".2f",
                    "ref_ep_pct": ".2f",
                    "ref_el_pct": ".2f",
                },
            ),
            "results/eventset/sensitivity.csv",
        )
    )

    conv = pd.concat([pd.read_csv(p) for p in sorted((R / "convergence").glob("summary_N*.csv"))])
    parts.append(
        section(
            "Monte Carlo convergence over 20 independent seeds",
            md(
                conv,
                {
                    "mean": ".4f",
                    "sd_across_seeds": ".4f",
                    "min": ".4f",
                    "max": ".4f",
                    "mean_within_run_se": ".4f",
                },
            ),
            "results/convergence/summary_N*.csv",
        )
    )

    pu = pd.read_csv(R / "param_uncertainty" / "summary.csv")
    parts.append(
        section(
            "Parameter uncertainty: 200 bootstrap refits of the hazard model, 10,000 years each",
            md(pu, {c: ".4f" for c in pu.columns if c not in ("statistic", "n_replicates")}),
            "results/param_uncertainty/summary.csv",
        )
    )

    lg = pd.read_csv(R / "layers" / "layer_grid.csv")
    sub = lg[lg["limit_pct_tiv"].round(3) == 1.0]
    parts.append(
        section(
            "Candidate layers with a limit of 1% of TIV (hypothetical portfolio; full grid in CSV)",
            md(
                sub[
                    [
                        "attachment_pct_tiv",
                        "exhaustion_pct_tiv",
                        "ap_pct",
                        "ep_pct",
                        "el_pct",
                        "el_se_pct",
                        "prob_any_loss_3y_pct",
                    ]
                ],
                {
                    "attachment_pct_tiv": ".2f",
                    "exhaustion_pct_tiv": ".2f",
                    "ap_pct": ".2f",
                    "ep_pct": ".2f",
                    "el_pct": ".3f",
                    "el_se_pct": ".3f",
                    "prob_any_loss_3y_pct": ".2f",
                },
            ),
            "results/layers/layer_grid.csv",
        )
    )

    pr = pd.read_csv(R / "pricing_synthetic" / "coverage.csv")
    parts.append(
        section(
            "Spread regression on SYNTHETIC deal tables (known generating relation)",
            md(
                pr.drop(columns=["data"]),
                {c: ".3f" for c in pr.columns if c not in ("scenario", "data", "n_deals", "reps")},
            ),
            "results/pricing_synthetic/coverage.csv",
        )
    )

    parts.extend(pricing_sections())
    parts.extend(posthoc_sections())
    parts.extend(transaction_sections())

    ex = pd.read_csv(R / "exposure" / "county_exposure.csv", dtype={"fips": str})
    ex = ex.assign(share_pct=100 * ex["share"], tiv_usd_m=ex["tiv_usd"] / 1e6)
    parts.append(
        section(
            "Hypothetical portfolio by county (households: 2020 Census via UF BEBR; "
            "TIV USD 100 bn, HYPOTHETICAL)",
            md(
                ex[["fips", "county", "households_2020", "share_pct", "tiv_usd_m", "lon", "lat"]],
                {"share_pct": ".3f", "tiv_usd_m": ".1f", "lon": ".3f", "lat": ".3f"},
            ),
            "results/exposure/county_exposure.csv",
        )
    )

    (R / "tables.md").write_text("\n".join(parts))
    print(f"wrote {(R / 'tables.md').relative_to(repo_root())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
