"""Annual landfall-count models and tests: Poisson, negative binomial, trend, low-frequency signal.

All tests take a vector of annual counts (zeros included) for consecutive years.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import optimize, special, stats


@dataclass(frozen=True)
class DispersionTest:
    """Index-of-dispersion test of the Poisson hypothesis.

    D = sum (x_i - xbar)^2 / xbar is approximately chi-square with n - 1 degrees of freedom under
    a Poisson model; the p-value is the upper tail (over-dispersion).
    """

    mean: float
    variance: float
    statistic: float
    dof: int
    p_value: float


def dispersion_test(counts: np.ndarray) -> DispersionTest:
    """Index-of-dispersion test (see ``DispersionTest``); variance uses divisor n - 1."""
    x = np.asarray(counts, dtype=float)
    n = x.size
    m = x.mean()
    d = float(((x - m) ** 2).sum() / m)
    return DispersionTest(m, float(x.var(ddof=1)), d, n - 1, float(stats.chi2.sf(d, n - 1)))


def poisson_loglik(counts: np.ndarray, lam: float) -> float:
    """Poisson log-likelihood sum(x log lam - lam - log x!)."""
    x = np.asarray(counts, dtype=float)
    return float((x * np.log(lam) - lam - special.gammaln(x + 1)).sum())


def negbin_loglik(counts: np.ndarray, mean: float, size: float) -> float:
    """Negative binomial (NB2) log-likelihood with mean mu and size k (variance mu + mu^2 / k).

    P(x) = Gamma(x + k) / (Gamma(k) x!) (k / (k + mu))^k (mu / (k + mu))^x.
    """
    x = np.asarray(counts, dtype=float)
    k, mu = size, mean
    return float(
        (
            special.gammaln(x + k)
            - special.gammaln(k)
            - special.gammaln(x + 1)
            + k * np.log(k / (k + mu))
            + x * np.log(mu / (k + mu))
        ).sum()
    )


@dataclass(frozen=True)
class FrequencyComparison:
    """Poisson against negative binomial by maximum likelihood.

    The NB mean MLE equals the sample mean; the size k is found numerically. The likelihood-ratio
    statistic tests k = infinity (Poisson), a boundary point, so its p-value is
    0.5 P(chi2_1 > LR) (50:50 mixture of chi2_0 and chi2_1 for one parameter on the boundary);
    LR = 0 when the MLE is at the boundary.
    """

    poisson_mean: float
    poisson_loglik: float
    nb_size: float
    nb_loglik: float
    lr_statistic: float
    lr_p_value: float
    aic_poisson: float
    aic_nb: float
    chosen: str


def compare_poisson_negbin(counts: np.ndarray, alpha: float = 0.05) -> FrequencyComparison:
    """Fit both models and choose NB only if the boundary LR test rejects Poisson at ``alpha``."""
    x = np.asarray(counts, dtype=float)
    mu = float(x.mean())
    ll_p = poisson_loglik(x, mu)
    res = optimize.minimize_scalar(
        lambda lk: -negbin_loglik(x, mu, float(np.exp(lk))),
        bounds=(np.log(1e-3), np.log(1e6)),
        method="bounded",
    )
    k = float(np.exp(res.x))
    ll_nb = max(negbin_loglik(x, mu, k), ll_p)
    lr = max(2.0 * (ll_nb - ll_p), 0.0)
    p = 0.5 * float(stats.chi2.sf(lr, 1)) if lr > 0 else 1.0
    return FrequencyComparison(
        mu,
        ll_p,
        k,
        ll_nb,
        lr,
        p,
        2 * 1 - 2 * ll_p,
        2 * 2 - 2 * ll_nb,
        "negbin" if p < alpha else "poisson",
    )


@dataclass(frozen=True)
class TrendTest:
    """Poisson log-linear trend: log lambda_t = a + b (t - tbar); LR test of b = 0."""

    slope_per_year: float
    slope_se: float
    rate_ratio_per_century: float
    lr_statistic: float
    p_value: float


def poisson_trend_test(counts: np.ndarray, years: np.ndarray) -> TrendTest:
    """Fit the log-linear Poisson trend by Newton-Raphson and test b = 0 by likelihood ratio."""
    x = np.asarray(counts, dtype=float)
    t = np.asarray(years, dtype=float)
    t = t - t.mean()
    design = np.column_stack([np.ones_like(t), t])
    beta = np.array([np.log(x.mean()), 0.0])
    for _ in range(100):
        mu = np.exp(design @ beta)
        grad = design.T @ (x - mu)
        hess = design.T @ (design * mu[:, None])
        step = np.linalg.solve(hess, grad)
        beta = beta + step
        if np.max(np.abs(step)) < 1e-12:
            break
    mu = np.exp(design @ beta)
    cov = np.linalg.inv(design.T @ (design * mu[:, None]))
    ll1 = float((x * np.log(mu) - mu - special.gammaln(x + 1)).sum())
    ll0 = poisson_loglik(x, x.mean())
    lr = max(2 * (ll1 - ll0), 0.0)
    return TrendTest(
        float(beta[1]),
        float(np.sqrt(cov[1, 1])),
        float(np.exp(100 * beta[1])),
        lr,
        float(stats.chi2.sf(lr, 1)),
    )


@dataclass(frozen=True)
class LowFrequencyTest:
    """Two checks for multi-decadal clustering of counts.

    * Block dispersion: index-of-dispersion test on non-overlapping ``block``-year sums (the last
      incomplete block is dropped); a low-frequency signal inflates the variance of block sums.
    * Ljung-Box Q = n (n + 2) sum_{k=1}^{m} rho_k^2 / (n - k) on annual counts, chi-square with m
      degrees of freedom.
    """

    block_years: int
    n_blocks: int
    block_dispersion: float
    block_p_value: float
    ljung_box_lags: int
    ljung_box_q: float
    ljung_box_p_value: float


def low_frequency_test(counts: np.ndarray, block: int = 10, lags: int = 10) -> LowFrequencyTest:
    """Block-sum dispersion and Ljung-Box tests (see ``LowFrequencyTest``)."""
    x = np.asarray(counts, dtype=float)
    nb = x.size // block
    sums = x[: nb * block].reshape(nb, block).sum(axis=1)
    bd = dispersion_test(sums)
    n = x.size
    xc = x - x.mean()
    denom = (xc**2).sum()
    rho = np.array([(xc[k:] * xc[:-k]).sum() / denom for k in range(1, lags + 1)])
    q = float(n * (n + 2) * (rho**2 / (n - np.arange(1, lags + 1))).sum())
    return LowFrequencyTest(
        block, nb, bd.statistic, bd.p_value, lags, q, float(stats.chi2.sf(q, lags))
    )


def poisson_mean_ci(total: int, exposure: float, level: float = 0.95) -> tuple[float, float]:
    """Exact (Garwood) confidence interval for a Poisson rate: total events over ``exposure``.

    lower = chi2_{a/2}(2 total) / 2, upper = chi2_{1 - a/2}(2 total + 2) / 2, divided by exposure.
    """
    a = 1 - level
    lo = 0.0 if total == 0 else stats.chi2.ppf(a / 2, 2 * total) / 2
    hi = stats.chi2.ppf(1 - a / 2, 2 * total + 2) / 2
    return float(lo / exposure), float(hi / exposure)


def sample_counts(
    rng: np.random.Generator, n_years: int, mean: float, nb_size: float | None
) -> np.ndarray:
    """Annual counts: Poisson(mean) if ``nb_size`` is None, else NB2 with that size.

    NB2 is drawn as a gamma-Poisson mixture: lambda ~ Gamma(k, mean / k), x ~ Poisson(lambda).
    """
    if nb_size is None:
        return rng.poisson(mean, size=n_years)
    lam = rng.gamma(nb_size, mean / nb_size, size=n_years)
    return rng.poisson(lam)
