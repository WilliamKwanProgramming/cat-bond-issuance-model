"""Regression of log spread on log expected loss over a table of cat bond tranches.

Model (ordinary least squares on the rows with complete data):

    log(spread_pct_i) = b0 + b1 log(expected_loss_pct_i) + c' z_i + e_i,

with optional covariates z: ``issue_year`` (years after 2000), ``log_size`` (log of
size_usd_m), and the categorical ``trigger``, ``peril`` and ``region`` (treatment dummies, the
alphabetically first level is the reference). Reported standard errors:

* classical OLS, s^2 (X'X)^-1 with s^2 = e'e / (n - k);
* HC3, (X'X)^-1 X' diag(e_i^2 / (1 - h_ii)^2) X (X'X)^-1 (heteroskedasticity-robust);
* cluster-robust by sponsor (CR1), c (X'X)^-1 (sum_g X_g' e_g e_g' X_g) (X'X)^-1 with
  c = G / (G - 1) (n - 1) / (n - k); a row with no sponsor is its own cluster.

Leave-one-out prediction errors use e_i / (1 - h_ii) (exact for OLS). The prediction interval
for a new tranche is the classical normal-theory interval on the log scale,
x0'b +/- t_{n-k, 1-a/2} s sqrt(1 + x0'(X'X)^-1 x0), exponentiated (so the point prediction is
the conditional median spread, not the mean).

Input schema: the deals interface in docs/model.md (one row per tranche). Missing values are
handled explicitly: rows are dropped for named reasons and every drop is counted in
``SampleReport``; nothing is imputed.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

DEAL_COLUMNS = [
    "deal_name",
    "tranche",
    "sponsor",
    "issue_year",
    "issue_month",
    "size_usd_m",
    "peril",
    "region",
    "trigger",
    "expected_loss_pct",
    "attachment_prob_pct",
    "exhaustion_prob_pct",
    "spread_pct",
    "spread_basis",
    "rating",
    "source_url",
    "access_date",
    "notes",
]

NUMERIC_COVARIATES = {"issue_year": "issue_year", "log_size": "size_usd_m"}
CATEGORICAL_COVARIATES = ("trigger", "peril", "region")
YEAR_ORIGIN = 2000


@dataclass(frozen=True)
class SampleReport:
    """How many input rows were used, and how many were dropped for each reason (in order)."""

    n_input: int
    n_used: int
    dropped: dict[str, int] = field(default_factory=dict)


def _blank(s: pd.Series) -> pd.Series:
    return s.isna() | (s.astype(str).str.strip() == "")


def prepare_deals(
    deals: pd.DataFrame,
    covariates: Sequence[str] = (),
    spread_basis: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, SampleReport]:
    """Select usable rows, recording every drop by reason.

    Reasons, applied in this order: spread_basis not in ``spread_basis`` (if given); missing or
    non-positive expected_loss_pct; missing or non-positive spread_pct; then, for each covariate,
    missing (or non-positive size for ``log_size``).
    """
    missing_cols = [
        c for c in ("expected_loss_pct", "spread_pct", "sponsor") if c not in deals.columns
    ]
    for cov in covariates:
        col = NUMERIC_COVARIATES.get(cov, cov)
        if cov not in NUMERIC_COVARIATES and cov not in CATEGORICAL_COVARIATES:
            raise ValueError(f"unknown covariate {cov!r}")
        if col not in deals.columns:
            missing_cols.append(col)
    if spread_basis is not None and "spread_basis" not in deals.columns:
        missing_cols.append("spread_basis")
    if missing_cols:
        raise ValueError(f"deals table lacks columns: {sorted(set(missing_cols))}")

    df = deals.copy()
    n_input = len(df)
    dropped: dict[str, int] = {}

    def drop(mask: pd.Series, reason: str) -> None:
        nonlocal df
        dropped[reason] = int(mask.sum())
        df = df[~mask]

    if spread_basis is not None:
        drop(
            ~df["spread_basis"].astype(str).str.strip().isin(list(spread_basis)),
            f"spread_basis not in {list(spread_basis)}",
        )
    for col in ("expected_loss_pct", "spread_pct"):
        v = pd.to_numeric(df[col], errors="coerce")
        drop(v.isna() | (v <= 0), f"{col} missing or <= 0")
        df = df.assign(**{col: pd.to_numeric(df[col], errors="coerce")})
    for cov in covariates:
        col = NUMERIC_COVARIATES.get(cov, cov)
        if cov in NUMERIC_COVARIATES:
            v = pd.to_numeric(df[col], errors="coerce")
            bad = v.isna() | ((v <= 0) if cov == "log_size" else False)
            drop(bad, f"{col} missing" + (" or <= 0" if cov == "log_size" else ""))
            df = df.assign(**{col: pd.to_numeric(df[col], errors="coerce")})
        else:
            drop(_blank(df[col]), f"{col} missing")
    return df.reset_index(drop=True), SampleReport(n_input, len(df), dropped)


def _design(
    df: pd.DataFrame, covariates: Sequence[str], levels: dict[str, list[str]] | None
) -> tuple[np.ndarray, list[str], dict[str, list[str]]]:
    cols = [np.ones(len(df)), np.log(df["expected_loss_pct"].to_numpy(dtype=float))]
    names = ["const", "log_el"]
    levels = dict(levels or {})
    for cov in covariates:
        if cov == "issue_year":
            cols.append(df["issue_year"].to_numpy(dtype=float) - YEAR_ORIGIN)
            names.append("issue_year")
        elif cov == "log_size":
            cols.append(np.log(df["size_usd_m"].to_numpy(dtype=float)))
            names.append("log_size")
        else:
            vals = df[cov].astype(str).str.strip()
            if cov not in levels:
                levels[cov] = sorted(vals.unique())
            unknown = set(vals) - set(levels[cov])
            if unknown:
                raise ValueError(f"{cov}: levels not seen in the fit: {sorted(unknown)}")
            for lev in levels[cov][1:]:
                cols.append((vals == lev).to_numpy(dtype=float))
                names.append(f"{cov}[{lev}]")
    return np.column_stack(cols), names, levels


@dataclass(frozen=True)
class SpreadFit:
    """Fitted log-spread regression (see module docstring)."""

    names: list[str]
    coef: np.ndarray
    se_ols: np.ndarray
    se_hc3: np.ndarray
    se_cluster: np.ndarray
    n: int
    n_clusters: int
    sigma: float
    xtx_inv: np.ndarray
    loo_errors: np.ndarray
    covariates: tuple[str, ...]
    levels: dict[str, list[str]]
    report: SampleReport

    @property
    def loo_rmse(self) -> float:
        """Root mean squared leave-one-out prediction error on the log-spread scale."""
        return float(np.sqrt(np.mean(self.loo_errors**2)))

    def coef_table(self) -> pd.DataFrame:
        """Coefficients with OLS, HC3 and cluster standard errors."""
        return pd.DataFrame(
            {
                "term": self.names,
                "coef": self.coef,
                "se_ols": self.se_ols,
                "se_hc3": self.se_hc3,
                "se_cluster": self.se_cluster,
            }
        )

    def predict(self, new: pd.DataFrame, level: float = 0.90) -> pd.DataFrame:
        """Median spread and prediction interval (pct) for new tranches; adds the multiple."""
        x, _, _ = _design(new.reset_index(drop=True), self.covariates, self.levels)
        mu = x @ self.coef
        se = self.sigma * np.sqrt(1.0 + np.einsum("ij,jk,ik->i", x, self.xtx_inv, x))
        t = stats.t.ppf(0.5 + level / 2, self.n - len(self.coef))
        el = new["expected_loss_pct"].to_numpy(dtype=float)
        return pd.DataFrame(
            {
                "expected_loss_pct": el,
                "spread_median_pct": np.exp(mu),
                "pi_lo_pct": np.exp(mu - t * se),
                "pi_hi_pct": np.exp(mu + t * se),
                "multiple": np.exp(mu) / el,
            }
        )


def fit_log_spread(
    deals: pd.DataFrame,
    covariates: Sequence[str] = (),
    spread_basis: Sequence[str] | None = None,
) -> SpreadFit:
    """Fit the log-spread regression on the usable rows of ``deals``."""
    df, report = prepare_deals(deals, covariates, spread_basis)
    x, names, levels = _design(df, covariates, None)
    n, k = x.shape
    if n <= k:
        raise ValueError(f"{n} usable rows for {k} coefficients")
    y = np.log(df["spread_pct"].to_numpy(dtype=float))
    xtx_inv = np.linalg.inv(x.T @ x)
    coef = xtx_inv @ x.T @ y
    e = y - x @ coef
    h = np.einsum("ij,jk,ik->i", x, xtx_inv, x)
    s2 = float(e @ e / (n - k))
    se_ols = np.sqrt(np.diag(s2 * xtx_inv))
    meat = (x * (e / (1 - h))[:, None] ** 2).T @ x
    se_hc3 = np.sqrt(np.diag(xtx_inv @ meat @ xtx_inv))
    sponsor = df["sponsor"].astype(str).str.strip()
    sponsor = sponsor.where(~_blank(df["sponsor"]), "__row" + df.index.astype(str))
    groups = pd.factorize(sponsor)[0]
    g = int(groups.max()) + 1
    scores = np.zeros((g, k))
    np.add.at(scores, groups, x * e[:, None])
    c = g / (g - 1) * (n - 1) / (n - k) if g > 1 else np.nan
    se_cl = np.sqrt(np.diag(c * xtx_inv @ (scores.T @ scores) @ xtx_inv))
    return SpreadFit(
        names,
        coef,
        se_ols,
        se_hc3,
        se_cl,
        n,
        g,
        float(np.sqrt(s2)),
        xtx_inv,
        e / (1 - h),
        tuple(covariates),
        levels,
        report,
    )
