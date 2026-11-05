"""Exceedance curves and return-period losses from an event loss table.

Definitions (per simulated year, N years, years without events have zero loss):

* occurrence loss  M_y = max of the event losses in year y (0 if none); OEP(x) = P(M > x);
* aggregate loss   S_y = sum of the event losses in year y;          AEP(x) = P(S > x);
* return-period loss for T years: the empirical (1 - 1/T) quantile of M (OEP) or S (AEP),
  computed with numpy's default linear interpolation.

Standard errors are bootstrap standard errors over simulated years: resample the N annual
values with replacement B times and take the standard deviation of the statistic.
"""

from __future__ import annotations

import numpy as np

RETURN_PERIODS = (10, 25, 50, 100, 250)


def annual_max_and_sum(
    year: np.ndarray, loss: np.ndarray, n_years: int
) -> tuple[np.ndarray, np.ndarray]:
    """Per-year maximum and sum of event losses over years 0..n_years-1 (zeros for empty years)."""
    year = np.asarray(year, dtype=np.int64)
    loss = np.asarray(loss, dtype=float)
    mx = np.zeros(n_years)
    np.maximum.at(mx, year, loss)
    sm = np.bincount(year, weights=loss, minlength=n_years).astype(float)
    return mx, sm


def exceedance_probability(annual: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Empirical P(annual > x) for each threshold in ``x``."""
    a = np.sort(np.asarray(annual, dtype=float))
    return 1.0 - np.searchsorted(a, np.asarray(x, dtype=float), side="right") / a.size


def return_period_losses(
    annual: np.ndarray, periods: tuple[int, ...] = RETURN_PERIODS
) -> np.ndarray:
    """Empirical (1 - 1/T) quantiles of the annual losses for each T in ``periods``."""
    q = 1.0 - 1.0 / np.asarray(periods, dtype=float)
    return np.quantile(np.asarray(annual, dtype=float), q)


def return_period_losses_se(
    annual: np.ndarray,
    periods: tuple[int, ...] = RETURN_PERIODS,
    n_boot: int = 200,
    seed: int = 0,
) -> np.ndarray:
    """Bootstrap-over-years standard error of each return-period loss (``n_boot`` resamples)."""
    a = np.asarray(annual, dtype=float)
    rng = np.random.default_rng(seed)
    est = np.empty((n_boot, len(periods)))
    for b in range(n_boot):
        est[b] = return_period_losses(a[rng.integers(0, a.size, size=a.size)], periods)
    return est.std(axis=0, ddof=1)
