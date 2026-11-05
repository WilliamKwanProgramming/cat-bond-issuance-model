#!/usr/bin/env python3
"""Hazard calibration on real HURDAT2 data: landfall sets, frequency tests, intensity fits.

Writes to results/hazard/:
  landfalls.csv              Florida hurricane landfalls 1900-2024 (one row per storm)
  definition_agreement.csv   record against geometric definition, and the two data files
  near_miss_hurricanes.csv   hurricane-strength storms within 50 km of the Florida coastline in
                             1900-2019 that are not counted as Florida landfalls (not modelled)
  frequency.csv              count models and tests for the three pre-specified windows
  intensity.csv              GPD and KDE fits against observed category frequencies
  motion_by_coast.csv        heading and translation speed by coast segment
  l_flags_by_period.csv      number of "L" records in AL.rda per five-year period (all basins)
  MANIFEST.txt

Usage: python3 scripts/run_hazard.py
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from scipy import stats

from catbond.hazard.frequency import (
    compare_poisson_negbin,
    dispersion_test,
    low_frequency_test,
    poisson_mean_ci,
    poisson_trend_test,
)
from catbond.hazard.intensity import fit_gpd, fit_kde
from catbond.hazard.landfall import (
    FLAG_GAP_YEARS,
    annual_counts,
    category,
    landfalls_from_records,
    landfalls_geometric,
    near_misses,
    synoptic_fixes,
)
from catbond.hazard.validation import GROUPS, group_probability
from catbond.manifest import repo_root, write_manifest
from catbond.pipeline import ALT_WINDOW, CALIBRATION, EXTENDED, load_inputs

OUT = repo_root() / "results" / "hazard"


def windows(inp):
    return {
        "1900-2019 (primary)": (inp.landfalls, CALIBRATION),
        "1950-2019": (inp.landfalls, ALT_WINDOW),
        "1900-2024 (2020-24 geometric)": (inp.landfalls_extended, EXTENDED),
    }


def frequency_table(inp) -> pd.DataFrame:
    rows = []
    for name, (lf, (y0, y1)) in windows(inp).items():
        c = annual_counts(lf, y0, y1).to_numpy()
        lo, hi = poisson_mean_ci(int(c.sum()), c.size)
        d = dispersion_test(c)
        nb = compare_poisson_negbin(c)
        tr = poisson_trend_test(c, np.arange(y0, y1 + 1))
        lf_test = low_frequency_test(c, block=10, lags=10)
        rows.append(
            {
                "window": name,
                "n_years": c.size,
                "n_landfalls": int(c.sum()),
                "mean_per_year": c.mean(),
                "mean_ci95_lo": lo,
                "mean_ci95_hi": hi,
                "variance": d.variance,
                "dispersion_D": d.statistic,
                "dispersion_p": d.p_value,
                "nb_size": nb.nb_size,
                "nb_lr": nb.lr_statistic,
                "nb_lr_p": nb.lr_p_value,
                "aic_poisson": nb.aic_poisson,
                "aic_negbin": nb.aic_nb,
                "chosen": nb.chosen,
                "trend_rate_ratio_per_century": tr.rate_ratio_per_century,
                "trend_slope_se": tr.slope_se,
                "trend_p": tr.p_value,
                "decade_dispersion_p": lf_test.block_p_value,
                "ljung_box_q10": lf_test.ljung_box_q,
                "ljung_box_p": lf_test.ljung_box_p_value,
            }
        )
    return pd.DataFrame(rows)


def intensity_table(inp) -> pd.DataFrame:
    rows = []
    for name, (lf, (y0, y1)) in windows(inp).items():
        w = lf.loc[lf["year"].between(y0, y1), "wind"].to_numpy(dtype=float)
        gpd, kde = fit_gpd(w), fit_kde(w)
        ks_g = stats.kstest(w, lambda v, m=gpd: m.cdf(v)).statistic
        ks_k = stats.kstest(w, lambda v, m=kde: m.cdf(v)).statistic
        for g, (lo, hi) in GROUPS.items():
            obs = int(((w >= lo) & (w < hi)).sum())
            rows.append(
                {
                    "window": name,
                    "group": g,
                    "observed": obs,
                    "observed_share": obs / w.size,
                    "gpd_prob": group_probability(gpd, lo, hi),
                    "kde_prob": group_probability(kde, lo, hi),
                    "n": w.size,
                    "gpd_xi": gpd.shape,
                    "gpd_sigma": gpd.scale,
                    "kde_bandwidth": kde.bandwidth,
                    "ks_gpd": ks_g,
                    "ks_kde": ks_k,
                }
            )
    return pd.DataFrame(rows)


def agreement_table(inp) -> pd.DataFrame:
    t = inp.tracks[inp.tracks["year"].between(*CALIBRATION)]
    gap = t["year"].between(*FLAG_GAP_YEARS)
    rec = landfalls_from_records(t[~gap], inp.region)
    geo_flag = landfalls_geometric(t[~gap], inp.region)
    geo_gap = landfalls_geometric(t[gap], inp.region)
    merged = rec.merge(geo_flag, on="storm_id", suffixes=("_rec", "_geo"))
    dist = (
        np.hypot(
            (merged["lon_rec"] - merged["lon_geo"]) * np.cos(np.deg2rad(merged["lat_rec"])),
            merged["lat_rec"] - merged["lat_geo"],
        )
        * 111.195
    )
    # the geometric definition on the synoptic fixes only, as it must work in the flag gap
    geo_syn = landfalls_geometric(synoptic_fixes(t[~gap]), inp.region)
    m_syn = rec.merge(geo_syn, on="storm_id", suffixes=("_rec", "_geo"))
    dist_syn = (
        np.hypot(
            (m_syn["lon_rec"] - m_syn["lon_geo"]) * np.cos(np.deg2rad(m_syn["lat_rec"])),
            m_syn["lat_rec"] - m_syn["lat_geo"],
        )
        * 111.195
    )
    wind_syn = m_syn["wind_geo"] - m_syn["wind_rec"]
    missed = rec[~rec["storm_id"].isin(geo_syn["storm_id"])]
    # storms.csv (geometric) against AL.rda (geometric) on the overlap 1975-2019, by name and year
    s = inp.storms[inp.storms["year"].between(1975, CALIBRATION[1])]
    geo_s = landfalls_geometric(s, inp.region)
    geo_a = landfalls_geometric(t[t["year"] >= 1975], inp.region)
    ka = set(zip(geo_a["name"].str.upper(), geo_a["year"], strict=True))
    ks = set(zip(geo_s["name"].str.upper(), geo_s["year"], strict=True))
    rows = [
        {
            "comparison": "record vs geometric, AL.rda, years with L flags",
            "n_first": len(rec),
            "n_second": len(geo_flag),
            "n_both": len(merged),
            "median_position_diff_km": float(np.median(dist)),
            "max_abs_wind_diff_kt": float((merged["wind_rec"] - merged["wind_geo"]).abs().max()),
            "detail": "",
        },
        {
            "comparison": "record vs geometric on synoptic fixes only, AL.rda, years with L flags",
            "n_first": len(rec),
            "n_second": len(geo_syn),
            "n_both": len(m_syn),
            "median_position_diff_km": float(np.median(dist_syn)),
            "max_abs_wind_diff_kt": float(wind_syn.abs().max()),
            "detail": f"mean wind difference (geometric - record) {wind_syn.mean():.1f} kt; "
            "record only: "
            + ", ".join(f"{n} {y}" for n, y in zip(missed["name"], missed["year"], strict=True)),
        },
        {
            "comparison": "geometric only, AL.rda, flag gap "
            f"{FLAG_GAP_YEARS[0]}-{FLAG_GAP_YEARS[1]}",
            "n_first": 0,
            "n_second": len(geo_gap),
            "n_both": 0,
            "median_position_diff_km": np.nan,
            "max_abs_wind_diff_kt": np.nan,
            "detail": "; ".join(
                f"{n} {y}" for n, y in zip(geo_gap["name"], geo_gap["year"], strict=True)
            ),
        },
        {
            "comparison": "geometric: AL.rda vs storms.csv, 1975-2019",
            "n_first": len(ka),
            "n_second": len(ks),
            "n_both": len(ka & ks),
            "median_position_diff_km": np.nan,
            "max_abs_wind_diff_kt": np.nan,
            "detail": "AL only: "
            + ", ".join(f"{n} {y}" for n, y in sorted(ka - ks))
            + "; storms.csv only: "
            + ", ".join(f"{n} {y}" for n, y in sorted(ks - ka)),
        },
    ]
    return pd.DataFrame(rows)


def coast_segment(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Pre-specified coast segments: South (lat < 25.9), Northwest (lat >= 29, lon < -82.5),
    East (lon > -81.5), West (the rest)."""
    seg = np.where(lon > -81.5, "East", "West")
    seg = np.where((lat >= 29.0) & (lon < -82.5), "Northwest", seg)
    return np.where(lat < 25.9, "South", seg)


def motion_table(inp) -> pd.DataFrame:
    lf = inp.landfalls[inp.landfalls["year"].between(*CALIBRATION)]
    seg = coast_segment(lf["lat"].to_numpy(), lf["lon"].to_numpy())
    rows = []
    for s in ["Northwest", "West", "South", "East", "All"]:
        g = lf if s == "All" else lf[seg == s]
        h = np.deg2rad(g["heading_deg"].to_numpy())
        rows.append(
            {
                "segment": s,
                "n": len(g),
                "circular_mean_heading_deg": float(
                    np.rad2deg(np.arctan2(np.sin(h).mean(), np.cos(h).mean())) % 360
                ),
                "mean_resultant_length": float(np.hypot(np.sin(h).mean(), np.cos(h).mean())),
                "median_speed_kmh": float(g["speed_kmh"].median()),
                "speed_q25": float(g["speed_kmh"].quantile(0.25)),
                "speed_q75": float(g["speed_kmh"].quantile(0.75)),
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    inp = load_inputs()
    OUT.mkdir(parents=True, exist_ok=True)
    lf = inp.landfalls_extended.copy()
    lf["category"] = category(lf["wind"].to_numpy())
    lf.to_csv(OUT / "landfalls.csv", index=False, float_format="%.4f")
    agreement_table(inp).to_csv(OUT / "definition_agreement.csv", index=False, float_format="%.3f")
    cal = inp.tracks[inp.tracks["year"].between(*CALIBRATION)]
    near_misses(cal, inp.landfalls, inp.region).to_csv(
        OUT / "near_miss_hurricanes.csv", index=False, float_format="%.4f"
    )
    frequency_table(inp).to_csv(OUT / "frequency.csv", index=False, float_format="%.6g")
    intensity_table(inp).to_csv(OUT / "intensity.csv", index=False, float_format="%.6g")
    motion_table(inp).to_csv(OUT / "motion_by_coast.csv", index=False, float_format="%.4g")
    lrec = inp.tracks[inp.tracks["record"] == "L"]
    per = (lrec["year"] // 5 * 5).value_counts().sort_index()
    per = per.reindex(range(1850, 2020, 5), fill_value=0)
    g0, g1 = FLAG_GAP_YEARS
    flags = pd.DataFrame(
        {"period": [f"{y}-{y + 4}" for y in per.index], "n_L_records": per.to_numpy()}
    )
    gap_row = pd.DataFrame(
        {
            "period": [f"{g0}-{g1} (flag gap)"],
            "n_L_records": [int(lrec["year"].between(g0, g1).sum())],
        }
    )
    pd.concat([flags, gap_row]).to_csv(OUT / "l_flags_by_period.csv", index=False)
    write_manifest(
        OUT,
        None,
        {
            "data": "data/raw/AL.rda, data/raw/storms.csv, "
            "data/raw/us_counties.geojson (real, public)"
        },
    )
    print(f"wrote {OUT.relative_to(repo_root())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
