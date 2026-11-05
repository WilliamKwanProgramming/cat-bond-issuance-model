import numpy as np
import pytest
from scipy import stats

from catbond.hazard.intensity import V_CAP_KT, IntensityModel, fit_gpd, fit_intensity, fit_kde


def test_gpd_recovers_parameters():
    rng = np.random.default_rng(1)
    y = stats.genpareto.rvs(-0.3, scale=35.0, size=4000, random_state=rng)
    m = fit_gpd(64.0 + y)
    assert m.shape == pytest.approx(-0.3, abs=0.04) and m.scale == pytest.approx(35.0, rel=0.06)


def test_gpd_fit_is_a_likelihood_maximum():
    rng = np.random.default_rng(2)
    v = 64.0 + stats.genpareto.rvs(-0.2, scale=30.0, size=300, random_state=rng)
    m = fit_gpd(v)

    def ll(xi, s):
        return stats.genpareto.logpdf(v - 64.0, xi, scale=s).sum()

    best = ll(m.shape, m.scale)
    for dxi, ds in [(0.02, 0), (-0.02, 0), (0, 1.0), (0, -1.0)]:
        assert best >= ll(m.shape + dxi, m.scale + ds)


@pytest.mark.parametrize("xi,n", [(-0.35, 67), (-0.1, 40), (0.0, 200), (0.2, 100), (-0.6, 30)])
def test_gpd_fit_matches_scipy(xi, n):
    rng = np.random.default_rng(int(100 * (xi + 1)) + n)
    y = stats.genpareto.rvs(xi, scale=30.0, size=n, random_state=rng)
    m = fit_gpd(64.0 + y)
    xs, _, ss = stats.genpareto.fit(y, floc=0.0)

    def ll(a, s):
        return stats.genpareto.logpdf(y, a, scale=s).sum()

    # our profile MLE is at least as good as scipy's generic optimiser, and agrees with it
    assert ll(m.shape, m.scale) >= ll(xs, ss) - 1e-6
    if xs > -1:
        assert m.shape == pytest.approx(xs, abs=0.02) and m.scale == pytest.approx(ss, rel=0.02)


def test_rejects_winds_below_threshold():
    with pytest.raises(ValueError):
        fit_gpd(np.array([70.0, 60.0]))
    with pytest.raises(ValueError):
        fit_intensity(np.array([70.0]), "weibull")


@pytest.mark.parametrize(
    "model",
    [
        IntensityModel("gpd", shape=-0.35, scale=39.0),
        IntensityModel("gpd", shape=0.1, scale=20.0),  # unbounded tail, truncated at the cap
        fit_kde(np.array([65, 70, 75, 80, 90, 100, 115, 125, 140, 160], dtype=float)),
    ],
)
def test_sampling_matches_truncated_distribution(model):
    rng = np.random.default_rng(3)
    s = model.sample(rng, 20_000)
    assert s.min() >= 64.0 and s.max() <= V_CAP_KT
    for v in (70.0, 83.0, 96.0, 113.0, 137.0):
        assert np.mean(s > v) == pytest.approx(float(model.sf_truncated(np.array(v))), abs=0.012)


def test_kde_cdf_endpoints_and_monotone():
    m = fit_kde(np.array([65, 70, 75, 80, 90, 100, 115, 125, 140, 160], dtype=float))
    grid = np.linspace(50, 260, 300)
    c = m.cdf(grid)
    assert c[0] == 0.0 and c[-1] == pytest.approx(1.0, abs=1e-6)
    assert np.all(np.diff(c) >= -1e-12)
    assert float(m.cdf(np.array(64.0))) == pytest.approx(0.0, abs=1e-12)
    # Silverman bandwidth by hand
    v = np.array([65, 70, 75, 80, 90, 100, 115, 125, 140, 160], dtype=float)
    iqr = np.subtract(*np.percentile(v, [75, 25]))
    assert m.bandwidth == pytest.approx(0.9 * min(v.std(ddof=1), iqr / 1.34) * 10**-0.2)
