"""Post hoc diagnostics of the spread regression. NOT part of docs/pricing_protocol.md.

These were written after the protocol's results had been seen (an independent review raised
them), so they are reported as post hoc, whatever they show, and none of them changes the
pre-specified test of proportionality.

* ``composition``: regions and perils of the 56 regression layers (the regression has no region
  or peril term);
* ``r2_partition``: how much of log spread issue year alone and log EL alone explain;
* ``specifications``: the slope b when region-group or multi-peril indicators are added, and on
  the Florida layers only;
* ``implied_error_sd`` and ``attenuation_simulation``: what an error in the stated EL would have
  to be to turn a true slope of 1 into the estimated slopes, if the error is classical (stated
  = true + noise, which attenuates b) and what happens if it is Berkson (true = stated + noise,
  which does not).
"""

from __future__ import annotations

import re
from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy import stats

from catbond.pricing.fit import fit_log_spread

REGION_GROUPS = (
    "Florida only",
    "Florida and other states",
    "no Florida listed",
    "whole US, Florida not named",
)
PERIL_GROUPS = ("named storm only", "named storm and other perils")


def region_group(region: str) -> str:
    """Group of a ``region`` string of the deal table (rules in the order written).

    ``Florida only``: the string is exactly "Florida". ``no Florida listed``: it says "excluding
    Florida" or names neither Florida nor FL. ``whole US, Florida not named``: it starts with
    "US," (no states listed). ``Florida and other states``: any other string naming Florida or
    FL (for example "Florida, South Carolina" or "US Southeast (FL, GA, NC, SC)").
    """
    r = str(region).strip()
    if r == "Florida":
        return REGION_GROUPS[0]
    if "excluding florida" in r.lower():
        return REGION_GROUPS[2]
    tokens = re.split(r"[\s(),]+", r)
    if "FL" in tokens or "florida" in r.lower():
        return REGION_GROUPS[1]
    if r.startswith("US,"):
        return REGION_GROUPS[3]
    return REGION_GROUPS[2]


def peril_group(peril: str) -> str:
    """"named storm only" if the peril string is exactly "named storm", else the other group."""
    return PERIL_GROUPS[0] if str(peril).strip() == "named storm" else PERIL_GROUPS[1]


def with_groups(sample: pd.DataFrame) -> pd.DataFrame:
    """Copy of the regression sample with ``region_group`` and ``peril_group`` columns."""
    out = sample.copy()
    out["region_group"] = out["region"].map(region_group)
    out["peril_group"] = out["peril"].map(peril_group)
    return out


def composition(sample: pd.DataFrame) -> pd.DataFrame:
    """Layers of the regression sample by region group, by peril group and by both."""
    d = with_groups(sample)
    rows = []
    for g in REGION_GROUPS:
        rows.append(("region", g, "all", int((d["region_group"] == g).sum())))
    for g in PERIL_GROUPS:
        rows.append(("peril", g, "all", int((d["peril_group"] == g).sum())))
    for r in REGION_GROUPS:
        for p in PERIL_GROUPS:
            n = int(((d["region_group"] == r) & (d["peril_group"] == p)).sum())
            rows.append(("region x peril", r, p, n))
    return pd.DataFrame(rows, columns=["dimension", "group", "peril_group", "n_layers"])


def _r2(y: np.ndarray, x: np.ndarray) -> tuple[float, float]:
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    e = y - x @ beta
    n, k = x.shape
    r2 = 1.0 - float(e @ e) / float(np.sum((y - y.mean()) ** 2))
    return r2, 1.0 - (1.0 - r2) * (n - 1) / (n - k)


def r2_partition(sample: pd.DataFrame) -> pd.DataFrame:
    """R-squared and adjusted R-squared of log spread on EL only, issue year only, and both."""
    y = np.log(sample["spread_pct"].to_numpy(dtype=float))
    x = np.log(sample["expected_loss_pct"].to_numpy(dtype=float))
    years = sample["issue_year"].astype(int).to_numpy()
    one = np.ones((len(y), 1))
    dummies = (years[:, None] == np.unique(years)[1:]).astype(float)
    rows = []
    for name, design in (
        ("log EL only (M1)", np.column_stack([one, x])),
        ("issue-year effects only", np.column_stack([one, dummies])),
        ("log EL and issue-year effects (M2)", np.column_stack([one, x, dummies])),
    ):
        r2, adj = _r2(y, design)
        rows.append({"model": name, "n": len(y), "k": design.shape[1], "r2": r2, "adj_r2": adj})
    return pd.DataFrame(rows)


def _interval(b: float, se: float, df: float, level: float = 0.95) -> tuple[float, float]:
    if not np.isfinite(se):
        return float("nan"), float("nan")
    t = stats.t.ppf(0.5 + level / 2, df)
    return b - t * se, b + t * se


def _terms(fit, prefixes: Sequence[str]) -> str:
    out = []
    for i, name in enumerate(fit.names):
        if name.startswith(tuple(prefixes)):
            out.append(f"{name} {fit.coef[i]:.3f} ({fit.se_cluster[i]:.3f})")
    return "; ".join(out)


def specifications(sample: pd.DataFrame) -> pd.DataFrame:
    """Slope on log EL under post hoc specifications (CR1 by series, HC3 where it is defined).

    The region and peril indicators are treatment dummies; the reference level is the first
    in alphabetical order ("Florida and other states" for the region group, "named storm and
    other perils" for peril), so each coefficient is a difference in log spread from that group
    at the same log EL and issue year.
    """
    d = with_groups(sample)
    d_reg = d.assign(region=d["region_group"])
    d_per = d.assign(peril=d["peril_group"])
    d_both = d.assign(region=d["region_group"], peril=d["peril_group"])
    florida_only = d[d["region_group"] == REGION_GROUPS[0]]
    florida_named = d[d["region_group"].isin(REGION_GROUPS[:2])]
    specs = [
        ("M2 (protocol), for reference", "56 layers", d, ("year_fe",), ()),
        ("M2 + region group", "56 layers", d_reg, ("year_fe", "region"), ("region[",)),
        ("M2 + multi-peril indicator", "56 layers", d_per, ("year_fe", "peril"), ("peril[",)),
        (
            "M2 + region group + multi-peril indicator",
            "56 layers",
            d_both,
            ("year_fe", "region", "peril"),
            ("region[", "peril["),
        ),
        ("M1, Florida-only layers", "Florida only", florida_only, (), ()),
        ("M1, layers naming Florida", "Florida only + Florida and other", florida_named, (), ()),
    ]
    rows = []
    for name, which, frame, covs, show in specs:
        fit = fit_log_spread(frame, covs, cluster="deal_name")
        j = fit.names.index("log_el")
        b = float(fit.coef[j])
        lo, hi = _interval(b, float(fit.se_cluster[j]), fit.n_clusters - 1)
        lo_h, hi_h = _interval(b, float(fit.se_hc3[j]), fit.n - len(fit.coef))
        rows.append(
            {
                "specification": name,
                "sample": which,
                "n": fit.n,
                "n_series": fit.n_clusters,
                "k": len(fit.coef),
                "b": b,
                "se_cr1": float(fit.se_cluster[j]),
                "lo_cr1": lo,
                "hi_cr1": hi,
                "lo_hc3": lo_h,
                "hi_hc3": hi_h,
                "indicator_terms": _terms(fit, show) if show else "",
            }
        )
    return pd.DataFrame(rows)


def florida_layer_spreads(
    sample: pd.DataFrame, layers: pd.DataFrame, year: int = 2025, level: float = 0.90
) -> pd.DataFrame:
    """Median spread of Florida-only layers: protocol M2 against M2 with region-group dummies.

    ``layers`` has a ``layer`` name and ``el_pct`` (the model EL in percent) per row. Both fits
    use issue-year effects and cluster by series; the second adds the region-group indicators.
    """
    d = with_groups(sample)
    fit_m2 = fit_log_spread(d, ("year_fe",), cluster="deal_name")
    fit_reg = fit_log_spread(d.assign(region=d["region_group"]), ("year_fe", "region"),
                             cluster="deal_name")
    new = pd.DataFrame(
        {
            "expected_loss_pct": layers["el_pct"].to_numpy(dtype=float),
            "issue_year": year,
            "region": REGION_GROUPS[0],
        }
    )
    rows = []
    for name, fit in (("M2 (protocol)", fit_m2), ("M2 + region group (post hoc)", fit_reg)):
        p = fit.predict(new, level=level)
        for layer, r in zip(layers["layer"], p.itertuples(), strict=True):
            rows.append(
                {
                    "layer": layer,
                    "model": name,
                    "el_pct": r.expected_loss_pct,
                    "spread_median_pct": r.spread_median_pct,
                    "pi_lo_pct": r.pi_lo_pct,
                    "pi_hi_pct": r.pi_hi_pct,
                    "multiple": r.multiple,
                }
            )
    return pd.DataFrame(rows)


def implied_error_sd(
    sample: pd.DataFrame, b_hat: dict[str, float], true_b: float = 1.0
) -> pd.DataFrame:
    """Error SD (in log EL) that would turn a true slope ``true_b`` into the estimates ``b_hat``.

    With classical error (stated = true + noise, noise independent of the true value) the slope
    is multiplied by Var(true) / Var(stated) = 1 - s^2 / Var(stated), so ``s^2 = (1 - b_hat /
    true_b) Var(stated)``. For M1 Var(stated) is the variance of the stated log EL, for M2 its
    variance within issue years (population variances, divided by n); a large-sample
    approximation, which ignores that fixed effects absorb part of the noise.
    """
    x = np.log(sample["expected_loss_pct"].to_numpy(dtype=float))
    years = sample["issue_year"].astype(int).to_numpy()
    within = x - pd.Series(x).groupby(years).transform("mean").to_numpy()
    rows = []
    for model, var in (("M1", float(np.var(x))), ("M2", float(np.var(within)))):
        b = b_hat[model]
        rows.append(
            {
                "model": model,
                "b_hat": b,
                "true_b": true_b,
                "var_stated_log_el": var,
                "implied_error_sd_log": float(np.sqrt(max(1.0 - b / true_b, 0.0) * var)),
            }
        )
    return pd.DataFrame(rows)


def attenuation_simulation(
    sample: pd.DataFrame,
    sigmas: Sequence[float],
    resid_sd: float,
    true_b: float = 1.0,
    n_rep: int = 2000,
    seed: int = 7001,
) -> pd.DataFrame:
    """Mean OLS slope on the stated log EL under classical and Berkson error in the stated EL.

    Design: the sample's stated log ELs (variance V) and issue years. Outcome = true log EL
    times ``true_b`` plus N(0, resid_sd^2), with no intercept and no year effects.

    * classical: the true log EL is the sample mean plus c times the stated deviation, with
      c^2 = 1 - s^2 / V, so that Var(true) + s^2 = V; the stated value is the true value plus
      N(0, s^2). The simulated stated EL then has the variance of the observed one;
    * Berkson: the stated log EL is the observed one and the true value is stated plus
      N(0, s^2).

    The slope is fitted by OLS on the stated EL (M1) and with issue-year effects (M2).
    """
    x_obs = np.log(sample["expected_loss_pct"].to_numpy(dtype=float))
    years = sample["issue_year"].astype(int).to_numpy()
    n = x_obs.size
    dummies = (years[:, None] == np.unique(years)[1:]).astype(float)
    ones = np.ones((n, 1))
    designs = {"M1": ones, "M2": np.column_stack([ones, dummies])}
    v = float(np.var(x_obs))
    rng = np.random.default_rng(seed)

    def slope(stated: np.ndarray, y: np.ndarray, model: str) -> float:
        x = np.column_stack([designs[model][:, :1], stated, designs[model][:, 1:]])
        return float(np.linalg.lstsq(x, y, rcond=None)[0][1])

    rows = []
    for s in sigmas:
        if s * s >= v:
            raise ValueError(f"error SD {s} is not below the SD of the stated log EL")
        c = np.sqrt(1.0 - s * s / v)
        draws: dict[tuple[str, str], list[float]] = {
            (sc, m): [] for sc in ("classical", "Berkson") for m in ("M1", "M2")
        }
        for _ in range(n_rep):
            eps = rng.normal(0.0, resid_sd, n)
            truth = x_obs.mean() + c * (x_obs - x_obs.mean())
            stated = truth + rng.normal(0.0, s, n)
            y_cl = true_b * truth + eps
            x_true = x_obs + rng.normal(0.0, s, n)
            y_bk = true_b * x_true + eps
            for m in ("M1", "M2"):
                draws[("classical", m)].append(slope(stated, y_cl, m))
                draws[("Berkson", m)].append(slope(x_obs, y_bk, m))
        for (sc, m), v_ in draws.items():
            rows.append(
                {
                    "scenario": sc,
                    "model": m,
                    "error_sd_log_el": s,
                    "true_b": true_b,
                    "mean_b_hat": float(np.mean(v_)),
                    "sd_b_hat": float(np.std(v_, ddof=1)),
                    "n_rep": n_rep,
                }
            )
    return pd.DataFrame(rows)
