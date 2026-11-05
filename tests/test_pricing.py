import numpy as np
import pandas as pd
import pytest

from catbond.pricing.comparables import comparable_deals
from catbond.pricing.fit import DEAL_COLUMNS, fit_log_spread, prepare_deals
from catbond.pricing.multiple import spread_multiple
from catbond.pricing.synthetic import TRUE_COEF, synthetic_deals

COVS = ("issue_year", "trigger")


def test_multiple():
    np.testing.assert_allclose(
        spread_multiple([6.0, 4.0, 1.0], [2.0, 0.0, np.nan]), [3.0, np.nan, np.nan]
    )


def test_against_statsmodels():
    sm = pytest.importorskip("statsmodels.api")
    rng = np.random.default_rng(1)
    deals = synthetic_deals(80, rng, sponsor_sd=0.2, n_sponsors=15)
    f = fit_log_spread(deals, COVS)
    x = np.column_stack(
        [
            np.ones(80),
            np.log(deals["expected_loss_pct"]),
            deals["issue_year"] - 2000,
            (deals["trigger"] == "parametric") * 1.0,
        ]
    )
    y = np.log(deals["spread_pct"])
    ols = sm.OLS(y, x).fit()
    np.testing.assert_allclose(f.coef, ols.params, rtol=1e-10)
    np.testing.assert_allclose(f.se_ols, ols.bse, rtol=1e-10)
    np.testing.assert_allclose(f.se_hc3, sm.OLS(y, x).fit(cov_type="HC3").bse, rtol=1e-10)
    groups = pd.factorize(deals["sponsor"])[0]
    cl = sm.OLS(y, x).fit(cov_type="cluster", cov_kwds={"groups": groups})
    np.testing.assert_allclose(f.se_cluster, cl.bse, rtol=1e-10)
    assert f.names == ["const", "log_el", "issue_year", "trigger[parametric]"]
    assert f.n_clusters == len(set(groups))


def test_leave_one_out_errors_by_refitting():
    rng = np.random.default_rng(2)
    deals = synthetic_deals(25, rng)
    f = fit_log_spread(deals, ("trigger",))
    for i in (0, 7, 24):
        g = fit_log_spread(deals.drop(index=i), ("trigger",))
        pred = g.predict(deals.iloc[[i]])["spread_median_pct"].item()
        err = np.log(deals["spread_pct"].iloc[i]) - np.log(pred)
        assert f.loo_errors[i] == pytest.approx(err, rel=1e-9)


def test_recovers_generating_coefficients():
    rng = np.random.default_rng(3)
    f = fit_log_spread(synthetic_deals(4000, rng, sigma=0.25), COVS)
    for name, true in TRUE_COEF.items():
        i = f.names.index(name)
        assert abs(f.coef[i] - true) < 4 * f.se_ols[i]
    assert f.sigma == pytest.approx(0.25, rel=0.05)


def test_prediction_interval_coverage():
    rng = np.random.default_rng(4)
    hits = []
    for _ in range(1000):
        d = synthetic_deals(41, rng, sigma=0.3)
        f = fit_log_spread(d.iloc[:40], COVS)
        p = f.predict(d.iloc[[40]], level=0.90)
        s = d["spread_pct"].iloc[40]
        hits.append(p["pi_lo_pct"].item() <= s <= p["pi_hi_pct"].item())
    assert np.mean(hits) == pytest.approx(0.90, abs=0.03)


def test_coefficient_interval_coverage_hc3_and_cluster():
    rng = np.random.default_rng(5)
    cover_hc3, cover_cl, cover_ols = [], [], []
    for _ in range(300):
        d = synthetic_deals(120, rng, sigma=0.15, n_sponsors=20, sponsor_sd=0.25)
        f = fit_log_spread(d)
        i = f.names.index("log_el")
        b = f.coef[i] - TRUE_COEF["log_el"]
        cover_hc3.append(abs(b) < 1.96 * f.se_hc3[i])
        cover_cl.append(abs(b) < 1.96 * f.se_cluster[i])
        cover_ols.append(abs(b) < 1.96 * f.se_ols[i])
    # sponsor effects are independent of EL here, so all three cover roughly at 95% for the
    # slope; the check guards against grossly wrong variance formulas
    for c in (cover_hc3, cover_cl, cover_ols):
        assert 0.88 < np.mean(c) < 0.99


def test_cluster_se_detects_within_sponsor_correlation():
    # sponsor effects shared by ~7 tranches each: the intercept's true sampling SD is far above
    # the OLS and HC3 standard errors; the cluster SE should track it
    rng = np.random.default_rng(6)
    est, se_ols, se_hc3, se_cl = [], [], [], []
    for _ in range(300):
        f = fit_log_spread(synthetic_deals(200, rng, sigma=0.05, n_sponsors=30, sponsor_sd=0.4))
        est.append(f.coef[0])
        se_ols.append(f.se_ols[0])
        se_hc3.append(f.se_hc3[0])
        se_cl.append(f.se_cluster[0])
    sd = np.std(est)
    assert np.median(se_cl) == pytest.approx(sd, rel=0.2)
    assert np.median(se_ols) < 0.6 * sd and np.median(se_hc3) < 0.6 * sd


def test_missing_values_reported_not_silently_dropped():
    rng = np.random.default_rng(7)
    d = synthetic_deals(30, rng)
    d.loc[0, "expected_loss_pct"] = np.nan
    d.loc[1, "spread_pct"] = np.nan
    d.loc[2, "expected_loss_pct"] = 0.0
    d.loc[3, "trigger"] = ""
    d.loc[4, "issue_year"] = np.nan
    d.loc[5, "spread_basis"] = "guidance"
    d.loc[6, "sponsor"] = np.nan  # kept, its own cluster
    used, rep = prepare_deals(d, COVS, spread_basis=["final"])
    assert rep.n_input == 30 and rep.n_used == 24 and len(used) == 24
    assert rep.n_used + sum(rep.dropped.values()) == rep.n_input
    assert rep.dropped == {
        "spread_basis not in ['final']": 1,
        "expected_loss_pct missing or <= 0": 2,
        "spread_pct missing or <= 0": 1,
        "issue_year missing": 1,
        "trigger missing": 1,
    }
    f = fit_log_spread(d, COVS, spread_basis=["final"])
    assert f.n == 24 and f.report == rep
    with pytest.raises(ValueError, match="lacks columns"):
        fit_log_spread(d.drop(columns=["sponsor"]))
    with pytest.raises(ValueError, match="unknown covariate"):
        fit_log_spread(d, ("colour",))


def test_predict_rejects_unseen_level():
    rng = np.random.default_rng(8)
    d = synthetic_deals(30, rng)
    f = fit_log_spread(d, ("trigger",))
    new = d.iloc[[0]].assign(trigger="industry index")
    with pytest.raises(ValueError, match="not seen"):
        f.predict(new)


def test_comparables():
    deals = pd.DataFrame(
        {
            "deal_name": list("ABCDEF"),
            "expected_loss_pct": [2.0, 2.1, 6.0, 2.0, 2.0, np.nan],
            "spread_pct": [6.0, 7.0, 12.0, 5.0, 6.5, 8.0],
            "size_usd_m": [200.0, 400.0, 200.0, 200.0, np.nan, 200.0],
            "peril": [
                "US named storm",
                "US named storm",
                "US named storm",
                "Japan quake",
                "us named storm ",
                "US named storm",
            ],
            "region": ["Florida", "Texas", "Florida", "Japan", "Florida", "Florida"],
        }
    )
    out = comparable_deals(deals, 2.0, 200.0, "US named storm", "Florida", k=4)
    assert out["deal_name"].tolist()[0] == "A" and out["distance"].iloc[0] == 0.0
    assert out.attrs["n_usable"] == 5 and "F" not in out["deal_name"].tolist()
    d = out.set_index("deal_name")["distance"]
    assert d["E"] == pytest.approx(1.0)  # missing size: one unit
    assert d["B"] == pytest.approx(np.hypot(np.log(2.1 / 2.0) / 0.5, np.log(2.0)) + 1.0)
    assert d["C"] == pytest.approx(np.log(3.0) / 0.5)
    assert out.set_index("deal_name")["multiple"]["A"] == pytest.approx(3.0)
    assert list(deals.columns) != DEAL_COLUMNS  # the screen needs only a subset of columns
