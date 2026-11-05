#!/usr/bin/env python3
"""Build deliverables/catbond_pitchbook.pdf from deliverables/pitchbook/catbond_pitchbook.tex.

ILLUSTRATIVE, HYPOTHETICAL SPONSOR, NOT AN OFFERING. Every number in the document is read here
from a file in results/ (or the deal table) and written to deliverables/pitchbook/build/ as a
LaTeX macro (numbers.tex), a LaTeX table (tab_*.tex) or a figure (fig_*.pdf); the .tex file
types no numbers of its own. Then latexmk compiles the document and the PDF is copied to
deliverables/catbond_pitchbook.pdf. The build directory is git-ignored.

Needs: results/ (scripts/reproduce.sh), data/raw/us_counties.geojson (scripts/fetch_data.py),
pdflatex and latexmk.

Usage: python3 scripts/make_pitchbook.py [--no-compile]
"""

from __future__ import annotations

import argparse
import pathlib
import shutil
import subprocess
import sys

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from catbond.manifest import repo_root  # noqa: E402
from catbond.pipeline import raw_dir  # noqa: E402

ROOT = repo_root()
R = ROOT / "results"
SRC = ROOT / "deliverables" / "pitchbook"
BUILD = SRC / "build"
PDF = ROOT / "deliverables" / "catbond_pitchbook.pdf"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "legend.frameon": False,
        "savefig.bbox": "tight",
    }
)


def tex_escape(s: str) -> str:
    """Escape the LaTeX special characters that occur in deal names and labels."""
    for a, b in (
        ("\\", r"\textbackslash{}"),
        ("&", r"\&"),
        ("%", r"\%"),
        ("_", r"\_"),
        ("#", r"\#"),
        ("$", r"\$"),
    ):
        s = s.replace(a, b)
    return s


def table(path: pathlib.Path, header: list[str], rows: list[list[str]], spec: str) -> None:
    """Write a booktabs tabular."""
    lines = [rf"\begin{{tabular}}{{{spec}}}", r"\toprule", " & ".join(header) + r" \\", r"\midrule"]
    lines += [" & ".join(r) + r" \\" for r in rows]
    lines += [r"\bottomrule", r"\end{tabular}"]
    path.write_text("\n".join(lines) + "\n")


def short_sample(name: str) -> str:
    """Sample names short enough for a table column."""
    return name.split(" (")[0].replace("without ", "w/o ")


def f2(v: float) -> str:
    return f"{v:.2f}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--no-compile", action="store_true")
    args = ap.parse_args()
    BUILD.mkdir(parents=True, exist_ok=True)
    num: dict[str, str] = {}

    # ------------------------------------------------------------ transaction layers
    la = pd.read_csv(R / "transaction" / "layers.csv")
    sp = pd.read_csv(R / "transaction" / "spreads.csv")
    pt = sp[sp["el_case"] == "EL point"].set_index(["layer", "model"])
    rows = []
    for r in la.itertuples():
        m2 = pt.loc[(r.layer, "M2")]
        m1 = pt.loc[(r.layer, "M1")]
        rows.append(
            [
                r.layer,
                f"{r.attachment_usd_m:,.0f}--{r.attachment_usd_m + r.limit_usd_m:,.0f}",
                f"{r.attachment_pct_tiv:.2f}--{r.exhaustion_pct_tiv:.2f}",
                f2(r.ap_pct),
                f2(r.ep_pct),
                f2(r.el_pct),
                f"{r.el_band_q05_pct:.2f}--{r.el_band_q95_pct:.2f}",
                f"{m1.spread_median_pct:.1f} [{m1.pi_lo_pct:.1f}, {m1.pi_hi_pct:.1f}]",
                f"{m2.spread_median_pct:.1f} [{m2.pi_lo_pct:.1f}, {m2.pi_hi_pct:.1f}]",
            ]
        )
    table(
        BUILD / "tab_summary.tex",
        [
            "Layer",
            "USD m",
            r"\% TIV",
            r"AP \%",
            r"EP \%",
            r"EL \%",
            r"EL band \%",
            r"M1 spread \%",
            r"M2 spread \%",
        ],
        rows,
        "lrrrrrrrr",
    )
    num["TotalLimitUSDm"] = f"{la['limit_usd_m'].sum():,.0f}"
    num["LayerLimitUSDm"] = f"{la['limit_usd_m'].iloc[0]:,.0f}"
    num["NBand"] = str(int(la["n_band_replicates"].iloc[0]))
    for key, r in zip(("Bot", "Mid", "Top"), la.itertuples(), strict=True):
        num[f"EL{key}"] = f2(r.el_pct)
        num[f"AP{key}"] = f2(r.ap_pct)
        num[f"EP{key}"] = f2(r.ep_pct)
        num[f"Att{key}"] = f2(r.attachment_pct_tiv)
        num[f"Exh{key}"] = f2(r.exhaustion_pct_tiv)
        num[f"PThree{key}"] = f"{r.prob_any_loss_3y_pct:.1f}"
        num[f"BandLo{key}"] = f2(r.el_band_q05_pct)
        num[f"BandHi{key}"] = f2(r.el_band_q95_pct)
        num[f"ELse{key}"] = f2(r.el_mc_se_pct)

    # ------------------------------------------------------------ exposure
    ex = pd.read_csv(R / "exposure" / "county_exposure.csv", dtype={"fips": str})
    top = ex.sort_values("tiv_usd", ascending=False).head(10)
    table(
        BUILD / "tab_exposure.tex",
        ["County", "Households 2020", r"Share \%", "TIV USD m"],
        [
            [r.county, f"{r.households_2020:,}", f"{100 * r.share:.1f}", f"{r.tiv_usd / 1e6:,.0f}"]
            for r in top.itertuples()
        ],
        "lrrr",
    )
    num["NCounties"] = str(len(ex))
    num["TopTenShare"] = f"{100 * top['share'].sum():.1f}"
    num["TIVbn"] = f"{ex['tiv_usd'].sum() / 1e9:,.0f}"
    from catbond.geo import load_region

    region = load_region(raw_dir() / "us_counties.geojson")
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    for _, _, rings in region.counties:
        for ring in rings:
            ax.fill(ring[:, 0], ring[:, 1], color="#f0efec", ec="#c3c2b7", lw=0.4)
    lf = pd.read_csv(R / "hazard" / "landfalls.csv")
    lf = lf[(lf["year"] >= 1900) & (lf["year"] <= 2019)]
    sizes = 1200 * ex["share"].to_numpy()
    ax.scatter(
        ex["lon"],
        ex["lat"],
        s=sizes,
        color=BLUE,
        alpha=0.75,
        ec="white",
        lw=0.8,
        label="TIV by county (area proportional)",
    )
    ax.scatter(
        lf["lon"],
        lf["lat"],
        s=28,
        marker="^",
        color=ORANGE,
        ec="white",
        lw=0.5,
        label="Hurricane landfalls 1900-2019 (HURDAT2)",
    )
    ax.set_aspect(1 / np.cos(np.radians(28)))
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(False)
    ax.legend(loc="lower left", fontsize=9)
    fig.savefig(BUILD / "fig_exposure.pdf")
    plt.close(fig)

    # ------------------------------------------------------------ hazard
    fr = pd.read_csv(R / "hazard" / "frequency.csv").set_index("window")
    f0 = fr.iloc[0]
    num["FreqMean"] = f"{f0['mean_per_year']:.3f}"
    num["FreqLo"] = f"{f0['mean_ci95_lo']:.3f}"
    num["FreqHi"] = f"{f0['mean_ci95_hi']:.3f}"
    num["NLandfalls"] = str(int(f0["n_landfalls"]))
    num["NBp"] = f"{f0['nb_lr_p']:.3f}"
    num["TrendP"] = f"{f0['trend_p']:.2f}"
    f1 = fr.iloc[1]
    num["AltWindowChoice"] = str(f1["chosen"]).replace("negbin", "negative binomial")
    num["AltWindowP"] = f"{f1['nb_lr_p']:.3f}"
    num["AltWindowMean"] = f"{f1['mean_per_year']:.3f}"
    it = pd.read_csv(R / "hazard" / "intensity.csv")
    it0 = it[it["window"] == it["window"].iloc[0]]
    num["GPDxi"] = f"{it0['gpd_xi'].iloc[0]:.3f}"
    num["GPDsigma"] = f"{it0['gpd_sigma'].iloc[0]:.1f}"
    num["KSgpd"] = f"{it0['ks_gpd'].iloc[0]:.3f}"
    oos = pd.read_csv(R / "hazard_oos" / "oos_by_category.csv")
    table(
        BUILD / "tab_oos.tex",
        ["Category", "Predicted", "Bootstrap 90\\% PI", "Observed", "p (two-sided)"],
        [
            [
                r.group,
                f"{r.predicted_mean:.1f}",
                f"{r.boot_lo:.0f}--{r.boot_hi:.0f}",
                str(int(r.observed)),
                f"{r.p_two_sided:.2f}",
            ]
            for r in oos.itertuples()
        ],
        "lrrrr",
    )
    allh = oos[oos["group"] == "All hurricanes"].iloc[0]
    num["OOSpred"] = f"{allh['predicted_mean']:.1f}"
    num["OOSobs"] = str(int(allh["observed"]))
    num["OOSp"] = f"{allh['p_two_sided']:.2f}"
    da = pd.read_csv(R / "hazard" / "definition_agreement.csv")
    num["DefAgree"] = str(int(da["n_both"].iloc[0]))
    num["DefN"] = str(int(da["n_first"].iloc[0]))
    syn = da[da["comparison"].str.contains("synoptic fixes only")].iloc[0]
    num["DefSynAgree"] = str(int(syn["n_both"]))
    num["DefSynWind"] = syn["detail"].split(" kt;")[0].rsplit(" ", 1)[1].lstrip("-")
    nm = pd.read_csv(R / "hazard" / "near_miss_hurricanes.csv")
    num["NearMiss"] = str(len(nm))
    lfall = pd.read_csv(R / "hazard" / "landfalls.csv")
    num["NLandfalls"] = str(int(lfall["year"].between(1900, 2019).sum()))
    comp = pd.read_csv(R / "pricing_posthoc" / "composition.csv")
    by_region = comp[comp["dimension"] == "region"].set_index("group")["n_layers"]
    by_peril = comp[comp["dimension"] == "peril"].set_index("group")["n_layers"]
    num["NFloridaOnly"] = str(int(by_region["Florida only"]))
    num["NNoFlorida"] = str(int(by_region["no Florida listed"]))
    num["NMultiPeril"] = str(int(by_peril["named storm and other perils"]))

    # ------------------------------------------------------------ loss exceedance curve
    rp = pd.read_csv(R / "eventset" / "rp_losses.csv")
    band = pd.read_csv(R / "param_uncertainty" / "rp_summary.csv")
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    for curve, col in (("OEP", BLUE), ("AEP", ORANGE)):
        b = band[band["curve"] == curve].sort_values("return_period")
        if curve == "OEP":
            ax.fill_between(
                b["return_period"],
                b["q05"],
                b["q95"],
                color=BLUE,
                alpha=0.18,
                lw=0,
                label="OEP hazard-parameter band (5-95%)",
            )
        c = rp[rp["curve"] == curve].sort_values("return_period")
        ax.plot(
            c["return_period"],
            c["loss_pct_tiv"],
            color=col,
            lw=2,
            marker="o",
            ms=6,
            label=f"{curve}, base case (100,000 years)",
        )
    for r, key in zip(la.itertuples(), ("bottom", "middle", "top"), strict=True):
        ax.axhspan(r.attachment_pct_tiv, r.exhaustion_pct_tiv, color=GRID, alpha=0.5, lw=0)
        ax.text(
            2.1,
            (r.attachment_pct_tiv + r.exhaustion_pct_tiv) / 2,
            f"{key} layer",
            va="center",
            fontsize=9,
            color=MUTED,
        )
    ax.set_xscale("log")
    ax.set_xlabel("Return period (years)")
    ax.set_ylabel("Loss (% of TIV)")
    ax.set_xticks([2, 5, 10, 25, 50, 100, 250, 500])
    ax.set_xticklabels(["2", "5", "10", "25", "50", "100", "250", "500"])
    ax.legend(loc="upper left", fontsize=9)
    fig.savefig(BUILD / "fig_lec.pdf")
    plt.close(fig)
    oep = rp[rp["curve"] == "OEP"].set_index("return_period")
    aep = rp[rp["curve"] == "AEP"].set_index("return_period")
    num["OEPhundred"] = f"{oep.loc[100, 'loss_pct_tiv']:.2f}"
    num["OEPhundredSE"] = f"{oep.loc[100, 'se_pct_tiv']:.3f}"
    bo = band[(band["curve"] == "OEP") & (band["return_period"] == 100)].iloc[0]
    num["OEPhundredLo"] = f"{bo['q05']:.2f}"
    num["OEPhundredHi"] = f"{bo['q95']:.2f}"
    num["AAL"] = f"{rp[rp['curve'] == 'AAL']['loss_pct_tiv'].iloc[0]:.3f}"
    table(
        BUILD / "tab_rp.tex",
        ["Return period", r"OEP \% TIV", "SE", r"5--95\% band", r"AEP \% TIV"],
        [
            [
                str(int(t)),
                f"{oep.loc[t, 'loss_pct_tiv']:.2f}",
                f"{oep.loc[t, 'se_pct_tiv']:.3f}",
                "{:.2f}--{:.2f}".format(
                    *band[(band["curve"] == "OEP") & (band["return_period"] == t)][
                        ["q05", "q95"]
                    ].iloc[0]
                ),
                f2(aep.loc[t, "loss_pct_tiv"]),
            ]
            for t in (10, 25, 50, 100, 250)
        ],
        "rrrrr",
    )

    # ------------------------------------------------------------ layer grid
    gr = pd.read_csv(R / "layers" / "layer_grid.csv")
    g5 = gr[np.isclose(gr["limit_pct_tiv"], 0.5)]
    table(
        BUILD / "tab_grid.tex",
        [
            r"Attach \% TIV",
            r"Exhaust \% TIV",
            r"AP \%",
            r"EP \%",
            r"EL \%",
            "MC SE",
            r"P(loss in 3 y) \%",
        ],
        [
            [
                f"{r.attachment_pct_tiv:.2f}",
                f"{r.exhaustion_pct_tiv:.2f}",
                f2(r.ap_pct),
                f2(r.ep_pct),
                f2(r.el_pct),
                f"{r.el_se_pct:.3f}",
                f"{r.prob_any_loss_3y_pct:.1f}",
            ]
            for r in g5[g5["attachment_pct_tiv"] <= 3.0].itertuples()
        ],
        "rrrrrrr",
    )
    ref = pd.read_csv(R / "eventset" / "reference_layer.csv").iloc[0]
    num["RefAtt"] = f2(ref["attachment_pct_tiv"])
    num["RefExh"] = f2(ref["exhaustion_pct_tiv"])
    num["RefEL"] = f2(ref["el_pct"])

    # ------------------------------------------------------------ pricing
    sl = pd.read_csv(R / "pricing" / "slope_intervals.csv")
    main_name = sl["sample"].iloc[0]
    msl = sl[sl["sample"] == main_name]
    table(
        BUILD / "tab_slope.tex",
        ["Model", "Interval method", "b", r"95\% interval", "p (b = 1)"],
        [
            [
                r.model,
                tex_escape(r.method.replace(" (WCR, Rademacher)", "")),
                f"{r.b:.3f}",
                f"[{r.lo:.3f}, {r.hi:.3f}]",
                f"{r.p_b_eq_1:.3f}",
            ]
            for r in msl.itertuples()
        ],
        "llrrr",
    )
    prop = pd.read_csv(R / "pricing" / "proportionality.csv")
    table(
        BUILD / "tab_prop.tex",
        ["Sample", "M1 b [wild, series]", "M2 b [wild, series]", "b = 1 rejected"],
        [
            [
                tex_escape(short_sample(r.sample)),
                f"{r.M1_b:.3f} [{r.M1_lo:.3f}, {r.M1_hi:.3f}]",
                f"{r.M2_b:.3f} [{r.M2_lo:.3f}, {r.M2_hi:.3f}]",
                "yes" if r.proportionality_rejected else "no",
            ]
            for r in prop.itertuples()
        ],
        "lrrl",
    )
    num["NRejected"] = str(int(prop["proportionality_rejected"].sum()))
    num["NSamples"] = str(len(prop))
    w = msl[msl["method"].str.startswith("wild cluster bootstrap series")].set_index("model")
    for m, word in (("M1", "One"), ("M2", "Two")):
        num[f"b{word}"] = f"{w.loc[m, 'b']:.3f}"
        num[f"b{word}Lo"] = f"{w.loc[m, 'lo']:.3f}"
        num[f"b{word}Hi"] = f"{w.loc[m, 'hi']:.3f}"
    num["NBoot"] = f"{int(w.loc['M1', 'n_boot']):,}"
    fs = pd.read_csv(R / "pricing" / "fit_stats.csv")
    fsm = fs[fs["sample"] == main_name].set_index("model")
    num["RsqOne"] = f"{fsm.loc['M1', 'r2']:.2f}"
    num["RsqTwo"] = f"{fsm.loc['M2', 'r2']:.2f}"
    num["NLayers"] = str(int(fsm.loc["M1", "n"]))
    num["NSeries"] = str(int(fsm.loc["M1", "n_series"]))
    num["NSponsors"] = str(int(fsm.loc["M1", "n_sponsors"]))
    cv = pd.read_csv(R / "pricing" / "cv.csv")
    cvm = cv[(cv["sample"] == main_name)].set_index(["model", "held_out"])
    num["CVseriesLog"] = f"{cvm.loc[('M2', 'series'), 'rmse_log']:.3f}"
    num["CVseriesPP"] = f"{cvm.loc[('M2', 'series'), 'rmse_pp']:.2f}"
    num["CVsponsorPP"] = f"{cvm.loc[('M2', 'sponsor'), 'rmse_pp']:.2f}"
    ym = pd.read_csv(R / "pricing" / "year_multiple.csv", dtype={"issue_year": str})
    yy = ym[ym["issue_year"] != "all"]
    num["MultMin"] = f2(yy["median_multiple"].min())
    num["MultMinYear"] = yy.loc[yy["median_multiple"].idxmin(), "issue_year"]
    num["MultMax"] = f2(yy["median_multiple"].max())
    num["MultMaxYear"] = yy.loc[yy["median_multiple"].idxmax(), "issue_year"]
    fm = pd.read_csv(R / "pricing" / "fitted_main.csv")
    co = pd.read_csv(R / "pricing" / "coefficients.csv")
    cm = co[co["sample"] == main_name]
    fig, ax = plt.subplots(figsize=(6.0, 4.6))
    recent = fm["issue_year"] >= 2023
    ax.scatter(
        fm.loc[~recent, "expected_loss_pct"],
        fm.loc[~recent, "spread_pct"],
        s=40,
        color=AQUA,
        ec="white",
        lw=0.8,
        label="Public layers, issued 2017-2022",
    )
    ax.scatter(
        fm.loc[recent, "expected_loss_pct"],
        fm.loc[recent, "spread_pct"],
        s=40,
        color=BLUE,
        ec="white",
        lw=0.8,
        label="Public layers, issued 2023-2025",
    )
    xg = np.linspace(0.4, 5.6, 100)
    c1 = cm[cm["model"] == "M1"].set_index("term")["coef"]
    c2 = cm[cm["model"] == "M2"].set_index("term")["coef"]
    ax.plot(
        xg, np.exp(c1["const"] + c1["log_el"] * np.log(xg)), color=MUTED, lw=2, label="M1 median"
    )
    ax.plot(
        xg,
        np.exp(c2["const"] + c2["year[2025]"] + c2["log_el"] * np.log(xg)),
        color=ORANGE,
        lw=2,
        label="M2 median, issue year 2025",
    )
    m2pt = pt.xs("M2", level="model")
    ax.errorbar(
        la["el_pct"],
        m2pt.loc[la["layer"], "spread_median_pct"],
        yerr=[
            m2pt.loc[la["layer"], "spread_median_pct"] - m2pt.loc[la["layer"], "pi_lo_pct"],
            m2pt.loc[la["layer"], "pi_hi_pct"] - m2pt.loc[la["layer"], "spread_median_pct"],
        ],
        fmt="D",
        color=INK,
        ms=7,
        capsize=4,
        lw=1.5,
        label="Hypothetical layers: M2 median, 90% PI",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks([0.5, 1, 2, 3, 5])
    ax.set_xticklabels(["0.5", "1", "2", "3", "5"])
    ax.set_yticks([3, 5, 10, 20])
    ax.set_yticklabels(["3", "5", "10", "20"])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel("Expected loss (% per year, deal's own model)")
    ax.set_ylabel("Spread (% per year)")
    ax.legend(loc="upper left", fontsize=8.5)
    fig.savefig(BUILD / "fig_pricing.pdf")
    plt.close(fig)
    cp = pd.read_csv(R / "transaction" / "comparables.csv")
    table(
        BUILD / "tab_comps.tex",
        ["Layer", "Deal", "Class", "Year", r"EL \%", r"Spread \%", "Multiple"],
        [
            [
                r.layer.split(" (")[1].rstrip(")"),
                tex_escape(r.deal_name.replace(" Ltd.", "").replace(" Pte.", "")),
                r.tranche.replace("Class ", ""),
                str(r.issue_year),
                f2(r.expected_loss_pct),
                f2(r.spread_pct),
                f2(r.multiple),
            ]
            for r in cp.itertuples()
        ],
        "lllrrrr",
    )

    # ------------------------------------------------------------ sensitivities
    se = pd.read_csv(R / "transaction" / "sensitivity.csv")
    order = list(dict.fromkeys(se["variant"]))
    piv = se.pivot(index="variant", columns="layer", values="el_pct").loc[order]
    m2c = se.pivot(index="variant", columns="layer", values="M2_spread_median_pct").loc[order]
    lay = list(la["layer"])
    table(
        BUILD / "tab_sens.tex",
        ["Variant (* in protocol)", "EL bottom", "EL middle", "EL top", r"M2 spread middle \%"],
        [
            [
                tex_escape(v.split(" (")[0])
                + (" *" if se[se["variant"] == v]["in_protocol_list"].iloc[0] else ""),
                f2(piv.loc[v, lay[0]]),
                f2(piv.loc[v, lay[1]]),
                f2(piv.loc[v, lay[2]]),
                f2(m2c.loc[v, lay[1]]),
            ]
            for v in order
        ],
        "lrrrr",
    )
    fig, ax = plt.subplots(figsize=(5.4, 4.8))
    mid = piv[lay[1]]
    proto = [v for v in order if se[se["variant"] == v]["in_protocol_list"].iloc[0]]
    vals = mid.loc[proto]
    yy_ = np.arange(len(proto))[::-1]
    ax.barh(yy_, vals, color=[MUTED if v == "base" else BLUE for v in proto], height=0.6, zorder=2)
    ax.axvline(mid.loc["base"], color=INK, lw=1, zorder=3)
    ax.axvspan(
        la["el_band_q05_pct"].iloc[1],
        la["el_band_q95_pct"].iloc[1],
        color=YELLOW,
        alpha=0.25,
        lw=0,
        zorder=0,
        label="hazard-parameter band (5-95%)",
    )
    for y, v in zip(yy_, vals, strict=True):
        ax.text(max(v, 3.0) + 0.2, y, f"{v:.2f}", va="center", fontsize=10, color=INK)
    ax.set_yticks(yy_)
    ax.set_yticklabels(
        [
            p.replace(" (Emanuel 2011 sensitivity)", "").replace(" (CLIMADA default)", "")
            for p in proto
        ]
    )
    ax.set_xlabel("Middle layer EL (% of limit)")
    ax.grid(axis="y", visible=False)
    ax.set_ylim(-0.6, len(proto) + 0.4)
    ax.legend(loc="upper right", fontsize=10)
    fig.savefig(BUILD / "fig_sens.pdf")
    plt.close(fig)
    num["VhalfMid"] = f2(piv.loc["V_half 110 kt (Emanuel 2011 sensitivity)", lay[1]])
    num["WindowMid"] = f2(piv.loc["window 1950-2019", lay[1]])
    num["DealELmin"] = f2(fm["expected_loss_pct"].min())
    num["DealELmax"] = f2(fm["expected_loss_pct"].max())

    # ------------------------------------------------------------ macros
    lines = ["% generated by scripts/make_pitchbook.py from results/; do not edit"]
    for k, v in num.items():
        if not k.isalpha():
            raise ValueError(f"macro name {k!r} must be letters only")
        lines.append(rf"\newcommand{{\num{k}}}{{{tex_escape(v)}}}")
    (BUILD / "numbers.tex").write_text("\n".join(lines) + "\n")

    if args.no_compile:
        return 0
    subprocess.run(
        [
            "latexmk",
            "-pdf",
            "-interaction=nonstopmode",
            "-halt-on-error",
            "-outdir=build",
            "catbond_pitchbook.tex",
        ],
        cwd=SRC,
        check=True,
        capture_output=True,
        timeout=600,
    )
    shutil.copy(BUILD / "catbond_pitchbook.pdf", PDF)
    print(f"wrote {PDF.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
