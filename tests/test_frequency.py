import numpy as np
import pytest
from scipy import stats

from catbond.hazard.frequency import (
    compare_poisson_negbin,
    dispersion_test,
    low_frequency_test,
    negbin_loglik,
    poisson_loglik,
    poisson_mean_ci,
    poisson_trend_test,
    sample_counts,
)


def test_dispersion_statistic_by_hand():
    x = np.array([0, 1, 2, 1, 1])
    r = dispersion_test(x)
    # mean 1, sum of squares 2 -> D = 2, dof 4
    assert r.statistic == pytest.approx(2.0) and r.dof == 4
    assert r.p_value == pytest.approx(stats.chi2.sf(2.0, 4))
    assert r.variance == pytest.approx(0.5)


def test_dispersion_test_size_and_power():
    rng = np.random.default_rng(3)
    p_pois = [dispersion_test(rng.poisson(0.6, 120)).p_value for _ in range(400)]
    assert 0.02 < np.mean(np.array(p_pois) < 0.05) < 0.09  # nominal 5%
    p_nb = [dispersion_test(sample_counts(rng, 120, 0.6, 0.5)).p_value for _ in range(100)]
    assert np.mean(np.array(p_nb) < 0.05) > 0.8


def test_loglik_against_scipy():
    x = np.array([0, 0, 1, 3, 2, 0, 5])
    assert poisson_loglik(x, 1.3) == pytest.approx(stats.poisson.logpmf(x, 1.3).sum())
    mu, k = 1.3, 0.7
    assert negbin_loglik(x, mu, k) == pytest.approx(stats.nbinom.logpmf(x, k, k / (k + mu)).sum())


def test_negbin_chosen_only_when_overdispersed():
    rng = np.random.default_rng(5)
    nb = compare_poisson_negbin(sample_counts(rng, 5000, 0.6, 1.0))
    assert nb.chosen == "negbin" and nb.nb_size == pytest.approx(1.0, rel=0.25)
    assert nb.aic_nb < nb.aic_poisson
    po = compare_poisson_negbin(rng.poisson(0.6, 5000))
    assert po.nb_size > 5 or po.chosen == "poisson"
    assert po.poisson_mean == pytest.approx(0.6, abs=0.05)


def test_trend_matches_statsmodels():
    sm = pytest.importorskip("statsmodels.api")
    rng = np.random.default_rng(7)
    years = np.arange(1900, 2020)
    x = rng.poisson(np.exp(-0.6 + 0.004 * (years - years.mean())))
    r = poisson_trend_test(x, years)
    design = sm.add_constant(years - years.mean())
    fit = sm.GLM(x, design, family=sm.families.Poisson()).fit()
    assert r.slope_per_year == pytest.approx(fit.params[1], rel=1e-6)
    assert r.slope_se == pytest.approx(fit.bse[1], rel=1e-5)
    null = sm.GLM(x, np.ones((x.size, 1)), family=sm.families.Poisson()).fit()
    assert r.lr_statistic == pytest.approx(2 * (fit.llf - null.llf), rel=1e-6)


def test_ljung_box_matches_statsmodels():
    diag = pytest.importorskip("statsmodels.stats.diagnostic")
    rng = np.random.default_rng(11)
    x = rng.poisson(0.6, 120)
    r = low_frequency_test(x, block=10, lags=10)
    lb = diag.acorr_ljungbox(x, lags=[10])
    assert r.ljung_box_q == pytest.approx(float(lb["lb_stat"].iloc[0]), rel=1e-9)
    assert r.ljung_box_p_value == pytest.approx(float(lb["lb_pvalue"].iloc[0]), rel=1e-9)
    assert r.n_blocks == 12


def test_block_dispersion_detects_low_frequency_regime():
    rng = np.random.default_rng(13)
    # 30-year alternating regimes 0.3 / 0.9 per year
    lam = np.repeat(np.tile([0.3, 0.9], 2), 30)
    hits = [low_frequency_test(rng.poisson(lam)).block_p_value < 0.05 for _ in range(100)]
    assert np.mean(hits) > 0.5


def test_garwood_interval_known_values():
    lo, hi = poisson_mean_ci(0, 1.0)
    assert lo == 0.0 and hi == pytest.approx(3.6889, abs=1e-3)
    lo, hi = poisson_mean_ci(10, 1.0)
    assert lo == pytest.approx(4.7954, abs=1e-3) and hi == pytest.approx(18.3904, abs=1e-3)
    lo2, hi2 = poisson_mean_ci(10, 40.0)
    assert lo2 == pytest.approx(lo / 40) and hi2 == pytest.approx(hi / 40)


def test_sample_counts_moments():
    rng = np.random.default_rng(17)
    p = sample_counts(rng, 200_000, 0.6, None)
    assert p.mean() == pytest.approx(0.6, abs=0.01) and p.var() == pytest.approx(0.6, abs=0.015)
    n = sample_counts(rng, 200_000, 0.6, 2.0)
    assert n.mean() == pytest.approx(0.6, abs=0.01)
    assert n.var() == pytest.approx(0.6 + 0.36 / 2.0, abs=0.02)
