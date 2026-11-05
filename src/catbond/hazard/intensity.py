"""Landfall intensity (maximum 1-minute wind, kt, given a hurricane landfall, i.e. V >= 64 kt).

Two models, compared in ``scripts/run_hazard.py``:

* ``gpd``: the excess y = V - 64 follows a generalized Pareto distribution with shape xi and scale
  sigma, F(y) = 1 - (1 + xi y / sigma)^(-1/xi) (exponential when xi = 0), fitted by maximum
  likelihood (scipy.stats.genpareto with location fixed at 0).
* ``kde``: Gaussian kernel density of the observed winds with Silverman's bandwidth
  h = 0.9 min(sd, IQR / 1.34) n^(-1/5), reflected at 64 kt so that no mass falls below it.

Both are truncated at ``V_CAP_KT`` = 180 kt (stated assumption: above the strongest Atlantic
HURDAT2 wind in the data used here). Truncation is applied by sampling the CDF on (0, F(cap)).

HURDAT2 winds are reported in 5-kt steps (Landsea and Franklin 2013); both models treat them as
continuous.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import optimize, stats

THRESHOLD_KT = 64.0
V_CAP_KT = 180.0


@dataclass(frozen=True)
class IntensityModel:
    """Fitted landfall-intensity distribution (``kind`` is "gpd" or "kde")."""

    kind: str
    shape: float = 0.0  # GPD xi
    scale: float = 1.0  # GPD sigma (kt)
    data: tuple[float, ...] = ()  # observed winds for the KDE
    bandwidth: float = 0.0  # KDE h (kt)

    def cdf(self, v: np.ndarray) -> np.ndarray:
        """Untruncated CDF at wind ``v`` (kt)."""
        v = np.asarray(v, dtype=float)
        if self.kind == "gpd":
            return stats.genpareto.cdf(v - THRESHOLD_KT, self.shape, loc=0.0, scale=self.scale)
        d = np.asarray(self.data)
        h = self.bandwidth
        z = (v[..., None] - d) / h
        zr = (v[..., None] - (2 * THRESHOLD_KT - d)) / h
        # reflected density f(v) = mean[phi_h(v - d) + phi_h(v - (2T - d))] for v >= T
        c = (stats.norm.cdf(z) + stats.norm.cdf(zr)).mean(axis=-1) - 1.0
        return np.where(v < THRESHOLD_KT, 0.0, c)

    def sf_truncated(self, v: np.ndarray) -> np.ndarray:
        """P(V > v) under the distribution truncated at ``V_CAP_KT``."""
        fcap = float(self.cdf(np.array(V_CAP_KT)))
        return np.clip((fcap - self.cdf(v)) / fcap, 0.0, 1.0)

    def sample(self, rng: np.random.Generator, n: int) -> np.ndarray:
        """Draw ``n`` winds (kt) from the truncated distribution."""
        if self.kind == "gpd":
            fcap = float(self.cdf(np.array(V_CAP_KT)))
            u = rng.uniform(0.0, fcap, size=n)
            return THRESHOLD_KT + stats.genpareto.ppf(u, self.shape, loc=0.0, scale=self.scale)
        d = np.asarray(self.data)
        out = np.empty(n)
        filled = 0
        while filled < n:
            m = n - filled
            v = d[rng.integers(0, d.size, size=m)] + self.bandwidth * rng.standard_normal(m)
            v = np.where(v < THRESHOLD_KT, 2 * THRESHOLD_KT - v, v)
            v = v[v <= V_CAP_KT]
            out[filled : filled + v.size] = v
            filled += v.size
        return out


def _gpd_profile(theta: float, y: np.ndarray) -> tuple[float, float, float]:
    """Profile log-likelihood of the GPD in theta = xi / sigma: (loglik, xi, sigma).

    For fixed theta the MLE of xi is mean(log(1 + theta y)) and sigma = xi / theta, giving
    l*(theta) = -n log(xi / theta) - n (1 + xi). theta = 0 is the exponential limit
    (sigma = mean(y), l* = -n log(mean y) - n).
    """
    n = y.size
    if abs(theta) < 1e-12:
        s = float(y.mean())
        return -n * np.log(s) - n, 0.0, s
    xi = float(np.mean(np.log1p(theta * y)))
    sigma = xi / theta
    return -n * np.log(sigma) - n * (1.0 + xi), xi, sigma


def fit_gpd(winds: np.ndarray) -> IntensityModel:
    """Maximum-likelihood GPD fit to the excess over 64 kt (location fixed at 0).

    Maximises the profile likelihood (see ``_gpd_profile``) over theta, restricted to xi >= -1
    (below it the likelihood is unbounded as theta -> -1 / max(y)). The test suite checks the
    result against scipy.stats.genpareto.fit.
    """
    y = np.asarray(winds, dtype=float) - THRESHOLD_KT
    if np.any(y < 0):
        raise ValueError("all winds must be >= 64 kt")
    y = np.maximum(y, 1e-9)
    ymax, ybar = float(y.max()), float(y.mean())
    lo_edge = -1.0 / ymax

    def xi_of(theta: float) -> float:
        return float(np.mean(np.log1p(theta * y)))

    # smallest theta with xi(theta) >= -1 (xi is increasing in theta)
    a = lo_edge * (1 - 1e-12)
    lo = a if xi_of(a) >= -1 else optimize.brentq(lambda t: xi_of(t) + 1.0, a, 0.0, xtol=1e-14)
    hi = 50.0 / ybar
    res = optimize.minimize_scalar(
        lambda t: -_gpd_profile(t, y)[0],
        bounds=(lo, hi),
        method="bounded",
        options={"xatol": 1e-10 / ybar},
    )
    best = min([res.x, lo, 0.0], key=lambda t: -_gpd_profile(t, y)[0])
    _, xi, sigma = _gpd_profile(best, y)
    return IntensityModel("gpd", shape=float(xi), scale=float(sigma))


def fit_kde(winds: np.ndarray) -> IntensityModel:
    """Reflected Gaussian KDE with Silverman's rule-of-thumb bandwidth."""
    v = np.asarray(winds, dtype=float)
    iqr = np.subtract(*np.percentile(v, [75, 25]))
    h = 0.9 * min(v.std(ddof=1), iqr / 1.34) * v.size ** (-0.2)
    return IntensityModel("kde", data=tuple(float(x) for x in v), bandwidth=float(h))


def fit_intensity(winds: np.ndarray, kind: str) -> IntensityModel:
    """Fit the intensity model named ``kind`` ("gpd" or "kde")."""
    if kind == "gpd":
        return fit_gpd(winds)
    if kind == "kde":
        return fit_kde(winds)
    raise ValueError(f"unknown intensity model {kind!r}")
