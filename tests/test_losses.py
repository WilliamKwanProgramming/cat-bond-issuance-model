"""Analytic checks: Poisson(lambda) events per year, exponential(mu) event losses.

Closed forms used:
* OEP(x) = P(max > x) = 1 - exp(-lambda e^{-x/mu});
* AEP(x) = P(sum > x) = sum_{n>=1} Pois(n; lambda) P(Gamma(n, mu) > x);
* return-period occurrence loss x_T = -mu ln(-ln(1 - 1/T) / lambda).
"""

import numpy as np
import pytest
from scipy import stats

from catbond.losses import (
    annual_max_and_sum,
    exceedance_probability,
    return_period_losses,
    return_period_losses_se,
)

LAM, MU, N = 0.6, 1.0, 200_000


@pytest.fixture(scope="module")
def poisson_exponential():
    rng = np.random.default_rng(2024)
    counts = rng.poisson(LAM, N)
    year = np.repeat(np.arange(N), counts)
    loss = rng.exponential(MU, year.size)
    return year, loss


def oep(x):
    return 1 - np.exp(-LAM * np.exp(-x / MU))


def aep(x):
    n = np.arange(1, 60)
    return float((stats.poisson.pmf(n, LAM) * stats.gamma.sf(x, n, scale=MU)).sum())


def test_annual_max_and_sum_by_hand():
    mx, sm = annual_max_and_sum(np.array([0, 0, 2]), np.array([1.0, 3.0, 2.0]), 4)
    assert mx.tolist() == [3.0, 0.0, 2.0, 0.0] and sm.tolist() == [4.0, 0.0, 2.0, 0.0]


def test_exceedance_is_strict():
    assert exceedance_probability(np.array([0.0, 1.0, 1.0, 2.0]), np.array([1.0])).item() == 0.25


def test_oep_and_aep_match_closed_form(poisson_exponential):
    year, loss = poisson_exponential
    mx, sm = annual_max_and_sum(year, loss, N)
    for x in (0.5, 1.0, 2.0, 3.0, 4.0):
        p = oep(x)
        tol = 4 * np.sqrt(p * (1 - p) / N)
        assert exceedance_probability(mx, np.array([x])).item() == pytest.approx(p, abs=tol)
        q = aep(x)
        assert exceedance_probability(sm, np.array([x])).item() == pytest.approx(
            q, abs=4 * np.sqrt(q * (1 - q) / N)
        )


def test_return_period_losses_match_closed_form(poisson_exponential):
    year, loss = poisson_exponential
    mx, _ = annual_max_and_sum(year, loss, N)
    periods = (10, 25, 50, 100, 250)
    est = return_period_losses(mx, periods)
    se = return_period_losses_se(mx, periods, n_boot=100, seed=1)
    truth = -MU * np.log(-np.log(1 - 1 / np.array(periods)) / LAM)
    assert np.all(np.abs(est - truth) < 4 * se)
    # bootstrap SE against the asymptotic quantile SE sqrt(p(1-p)/N) / f(x_T)
    p = 1 / np.array(periods)
    dens = LAM / MU * np.exp(-truth / MU) * np.exp(-LAM * np.exp(-truth / MU))
    asym = np.sqrt(p * (1 - p) / N) / dens
    np.testing.assert_allclose(se, asym, rtol=0.35)
