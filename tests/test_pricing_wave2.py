"""Tests of the wave-2 pricing code: year fixed effects, cluster choice, wild cluster bootstrap,
leave-group-out errors and the deal-table sample rules. Synthetic data unless stated."""

import pathlib

import numpy as np
import pandas as pd
import pytest

from catbond.pricing.fit import cluster_labels, cluster_robust_se, fit_log_spread
from catbond.pricing.sample import SAMPLES, flag_rows, regression_sample
from catbond.pricing.synthetic import synthetic_deals
from catbond.pricing.validation import leave_group_out, summarize_errors
from catbond.pricing.wild import WildClusterBootstrap, rademacher

DEALS = pathlib.Path(__file__).resolve().parents[1] / "data" / "deals" / "cat_bond_deals.csv"


def _with_series(d: pd.DataFrame, n_series: int, rng: np.random.Generator) -> pd.DataFrame:
    return d.assign(deal_name=[f"Series {j}" for j in rng.integers(0, n_series, len(d))])


def test_year_fixed_effects_against_statsmodels():
    sm = pytest.importorskip("statsmodels.formula.api")
    rng = np.random.default_rng(11)
    d = _with_series(synthetic_deals(90, rng, sponsor_sd=0.2, n_sponsors=12), 25, rng)
    f = fit_log_spread(d, ("year_fe", "trigger", "log_size"), cluster="deal_name")
    df = d.assign(
        ly=np.log(d["spread_pct"]),
        lel=np.log(d["expected_loss_pct"]),
        lsize=np.log(d["size_usd_m"]),
    )
    ref = sm.ols("ly ~ lel + C(issue_year) + C(trigger) + lsize", df).fit(
        cov_type="cluster", cov_kwds={"groups": pd.factorize(d["deal_name"])[0]}
    )
    years = sorted(d["issue_year"].unique())
    assert f.names[:3] == ["const", "log_el", f"year[{years[1]}]"]
    assert len(f.names) == 2 + (len(years) - 1) + 1 + 1
    order = (
        ["Intercept", "lel"]
        + [f"C(issue_year)[T.{y}]" for y in years[1:]]
        + ["C(trigger)[T.parametric]", "lsize"]
    )
    np.testing.assert_allclose(f.coef, ref.params[order], rtol=1e-9)
    np.testing.assert_allclose(f.se_cluster, ref.bse[order], rtol=1e-9)
    assert f.n_clusters == d["deal_name"].nunique()
    np.testing.assert_allclose(f.cluster_se("deal_name"), f.se_cluster, rtol=1e-12)
    g = fit_log_spread(d, ("year_fe", "trigger", "log_size"))  # default: sponsor
    np.testing.assert_allclose(f.cluster_se("sponsor"), g.se_cluster, rtol=1e-12)
    assert f.r2 == pytest.approx(ref.rsquared, rel=1e-9)


def test_year_fe_prediction_needs_a_seen_year():
    rng = np.random.default_rng(12)
    d = synthetic_deals(40, rng)
    f = fit_log_spread(d, ("year_fe",))
    with pytest.raises(ValueError, match="not seen"):
        f.predict(d.iloc[[0]].assign(issue_year=1999))
    p = f.predict(d.iloc[[3]])
    assert np.log(p["spread_median_pct"].item()) == pytest.approx(f.x[3] @ f.coef, rel=1e-12)


def test_cluster_labels_blank_is_own_cluster():
    df = pd.DataFrame({"s": ["a", "a", "", None, "b"]})
    codes = cluster_labels(df, "s")
    assert codes[0] == codes[1] and len(set(codes)) == 4


def test_rademacher_weights():
    w = rademacher(np.random.default_rng(1), 4000, 30)
    assert set(np.unique(w)) == {-1.0, 1.0}
    assert abs(w.mean()) < 0.01 and w.var() == pytest.approx(1.0, abs=0.01)
    # independent across clusters
    assert abs(np.corrcoef(w[:, 0], w[:, 1])[0, 1]) < 0.05


def _clustered_data(rng, g=12, beta=1.0):
    sizes = rng.integers(2, 9, size=g)
    groups = np.repeat(np.arange(g), sizes)
    n = groups.size
    xc = rng.standard_normal(g)[groups]  # regressor correlated within clusters
    x1 = xc + 0.5 * rng.standard_normal(n)
    u = 0.8 * rng.standard_normal(g)[groups] * (1 + np.abs(xc)) + 0.3 * rng.standard_normal(n)
    x = np.column_stack([np.ones(n), x1])
    return x, 0.5 + beta * x1 + u, groups


def test_wild_bootstrap_se_matches_cr1_and_p_is_one_at_estimate():
    rng = np.random.default_rng(13)
    x, y, groups = _clustered_data(rng)
    w = WildClusterBootstrap(x, y, groups, 1, 999, np.random.default_rng(1))
    xtx_inv = np.linalg.inv(x.T @ x)
    b = xtx_inv @ x.T @ y
    assert w.estimate == pytest.approx(b[1], rel=1e-12)
    assert w.se == pytest.approx(cluster_robust_se(x, y - x @ b, xtx_inv, groups)[1], rel=1e-12)
    assert w.p_value(w.estimate) == 1.0
    r = w.interval(0.95, null_values=(w.estimate + 10 * w.se,))
    assert r.lo < r.estimate < r.hi
    assert r.p_value_at[w.estimate + 10 * w.se] < 0.01
    r80 = w.interval(0.80)
    assert r.lo < r80.lo < r80.hi < r.hi
    # the end points are where p crosses alpha
    assert w.p_value(r.lo - 0.01 * w.se) <= 0.05 < w.p_value(r.lo + 0.01 * w.se)


def test_wild_interval_shifts_with_the_data():
    # adding c * x_j to y shifts the estimate and the interval by exactly c (same weights)
    rng = np.random.default_rng(14)
    x, y, groups = _clustered_data(rng)
    r0 = WildClusterBootstrap(x, y, groups, 1, 499, np.random.default_rng(2)).interval()
    r1 = WildClusterBootstrap(x, y + 0.7 * x[:, 1], groups, 1, 499, np.random.default_rng(2))
    r1 = r1.interval()
    assert r1.estimate == pytest.approx(r0.estimate + 0.7, abs=1e-9)
    assert r1.lo == pytest.approx(r0.lo + 0.7, abs=1e-3 * r0.se_cluster)
    assert r1.hi == pytest.approx(r0.hi + 0.7, abs=1e-3 * r0.se_cluster)


def test_wild_interval_coverage_with_few_clusters():
    # 12 clusters with correlated regressor and errors: the CR1 t-interval (normal quantile)
    # undercovers; the wild cluster bootstrap interval should be close to 95%
    rng = np.random.default_rng(15)
    hit_wild, hit_cr1 = [], []
    for _ in range(300):
        x, y, groups = _clustered_data(rng)
        w = WildClusterBootstrap(x, y, groups, 1, 199, rng)
        r = w.interval(0.95, tol=1e-2)
        hit_wild.append(r.lo <= 1.0 <= r.hi)
        hit_cr1.append(abs(w.estimate - 1.0) <= 1.96 * w.se)
    assert 0.91 <= np.mean(hit_wild) <= 0.99
    assert np.mean(hit_cr1) < np.mean(hit_wild) - 0.02


def test_leave_one_out_equals_hat_matrix_formula():
    rng = np.random.default_rng(16)
    d = synthetic_deals(30, rng).assign(row_id=[f"r{i}" for i in range(30)])
    f = fit_log_spread(d, ("trigger",))
    e = leave_group_out(d, ("trigger",), "row_id")
    assert e["predictable"].all()
    np.testing.assert_allclose(e["err_log"], f.loo_errors, rtol=1e-9)


def test_leave_group_out_by_refitting_and_unseen_levels():
    rng = np.random.default_rng(17)
    d = _with_series(synthetic_deals(40, rng), 8, rng).reset_index(drop=True)
    d["issue_year"] = rng.choice([2020, 2021, 2022], size=40)
    d.loc[d["deal_name"] == "Series 3", "issue_year"] = 2030  # a year only this series has
    e = leave_group_out(d, ("year_fe",), "deal_name")
    held = d["deal_name"] == "Series 3"
    assert (~e.loc[held, "predictable"]).all() and e.loc[~held, "predictable"].all()
    g = "Series 5"
    fit = fit_log_spread(d[d["deal_name"] != g], ("year_fe",))
    pred = fit.predict(d[d["deal_name"] == g])["spread_median_pct"].to_numpy()
    np.testing.assert_allclose(e.loc[d["deal_name"] == g, "pred_spread_pct"], pred, rtol=1e-12)
    np.testing.assert_allclose(
        e.loc[d["deal_name"] == g, "err_pp"], d.loc[d["deal_name"] == g, "spread_pct"] - pred
    )
    s = summarize_errors(e)
    assert s["n_rows"] == 40 and s["n_unpredictable"] == int(held.sum())
    assert s["n_predicted"] + s["n_unpredictable"] == 40
    ok = e[e["predictable"]]
    assert s["rmse_log"] == pytest.approx(np.sqrt(np.mean(ok["err_log"] ** 2)))
    assert s["mae_pp"] == pytest.approx(np.mean(np.abs(ok["err_pp"])))


@pytest.mark.skipif(not DEALS.exists(), reason="deal table not present")
def test_sample_rules_on_the_deal_table():
    deals = pd.read_csv(DEALS)
    f = flag_rows(deals)
    assert len(deals) == 58 and int(f["duplicate"].sum()) == 2
    assert int(f["guidance"].sum()) == 3 and int(f["discount"].sum()) == 4
    assert int(f["launch_el"].sum()) == 2
    sizes = {name: len(regression_sample(deals, flags)) for name, flags in SAMPLES.items()}
    assert sizes == {
        "main (56 distinct layers)": 56,
        "without guidance prices": 53,
        "without discount notes": 52,
        "without launch-plan EL (Alamo Re 2021-1, 2023-1)": 54,
        "without all flagged rows": 48,  # Alamo Re 2021-1 is both guidance and launch-plan EL
    }
    kept = regression_sample(deals)
    assert {"Class A1", "Class B1"} <= set(kept["tranche"])
    assert not {"Class A2", "Class B2"} & set(
        kept.loc[kept["deal_name"].str.startswith("Integrity Re III"), "tranche"]
    )


@pytest.mark.skipif(not DEALS.exists(), reason="deal table not present")
def test_sample_rules_fail_loudly_on_a_changed_table():
    deals = pd.read_csv(DEALS)
    bad = deals.copy()
    i = bad.index[(bad["tranche"] == "Class A2") & bad["deal_name"].str.startswith("Integrity")]
    bad.loc[i, "spread_pct"] += 0.25
    with pytest.raises(ValueError, match="not a duplicate"):
        flag_rows(bad)
    gone = deals[deals["tranche"] != "Class C1"]
    with pytest.raises(ValueError, match="occurs 0 times"):
        flag_rows(gone)
    extra = deals.copy()
    extra.loc[0, "spread_basis"] = "final"
    with pytest.raises(ValueError, match="guidance rows"):
        flag_rows(extra)
