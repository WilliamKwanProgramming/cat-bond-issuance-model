"""Tests of the post hoc diagnostics of the spread regression (catbond.pricing.posthoc)."""

import pathlib

import numpy as np
import pandas as pd
import pytest

from catbond.pricing.fit import fit_log_spread
from catbond.pricing.posthoc import (
    REGION_GROUPS,
    attenuation_simulation,
    composition,
    florida_layer_spreads,
    implied_error_sd,
    peril_group,
    r2_partition,
    region_group,
    specifications,
)
from catbond.pricing.sample import regression_sample
from catbond.pricing.synthetic import synthetic_deals
from catbond.transaction import spread_at

DEALS = pathlib.Path(__file__).resolve().parents[1] / "data" / "deals" / "cat_bond_deals.csv"
F_ONLY, F_PLUS, NO_F, US = REGION_GROUPS


@pytest.fixture(scope="module")
def real_sample() -> pd.DataFrame:
    return regression_sample(pd.read_csv(DEALS))


@pytest.mark.parametrize(
    "region, expected",
    [
        ("Florida", F_ONLY),
        ("Florida, South Carolina", F_PLUS),
        ("US Gulf and Florida (FL, LA, AL, MS, TX)", F_PLUS),
        ("US (AL, FL, GA, MS, NC, SC)", F_PLUS),
        ("US Southeast (FL, GA, NC, SC)", F_PLUS),
        ("US (AL, LA, MS, NY, NC, SC, TX)", NO_F),
        ("US excluding Florida and New Jersey", NO_F),
        ("Texas", NO_F),
        ("North Carolina", NO_F),
        ("Louisiana", NO_F),
        ("Hawaii", NO_F),
        ("US Northeast", NO_F),
        ("US, Puerto Rico, US Virgin Islands, Canada", US),
    ],
)
def test_region_group_rules(region, expected):
    assert region_group(region) == expected


def test_peril_group_and_the_composition_of_the_real_sample(real_sample):
    assert peril_group("named storm") == "named storm only"
    assert peril_group("named storm; severe thunderstorm") == "named storm and other perils"
    c = composition(real_sample)
    by_region = c[c["dimension"] == "region"].set_index("group")["n_layers"]
    assert by_region.to_dict() == {F_ONLY: 21, F_PLUS: 15, NO_F: 19, US: 1}
    by_peril = c[c["dimension"] == "peril"].set_index("group")["n_layers"]
    assert by_peril.to_dict() == {"named storm only": 41, "named storm and other perils": 15}
    # the cross table adds up to the sample, whatever the cell
    assert c[c["dimension"] == "region x peril"]["n_layers"].sum() == len(real_sample) == 56


def test_r2_partition_matches_the_fits_and_orders_the_models(real_sample):
    t = r2_partition(real_sample).set_index("model")
    m1 = fit_log_spread(real_sample, (), cluster="deal_name")
    m2 = fit_log_spread(real_sample, ("year_fe",), cluster="deal_name")
    assert t.loc["log EL only (M1)", "r2"] == pytest.approx(m1.r2, abs=1e-12)
    assert t.loc["log EL and issue-year effects (M2)", "r2"] == pytest.approx(m2.r2, abs=1e-12)
    both = t.loc["log EL and issue-year effects (M2)"]
    for only in ("log EL only (M1)", "issue-year effects only"):
        assert both["r2"] >= t.loc[only, "r2"]
    # the claim the README makes: the issue year alone explains more than EL alone
    assert t.loc["issue-year effects only", "r2"] > t.loc["log EL only (M1)", "r2"]
    assert t.loc["issue-year effects only", "adj_r2"] > t.loc["log EL only (M1)", "adj_r2"]


def test_implied_error_sd_closed_form():
    rng = np.random.default_rng(3)
    d = synthetic_deals(60, rng)
    x = np.log(d["expected_loss_pct"].to_numpy())
    got = implied_error_sd(d, {"M1": 0.5, "M2": 1.0}).set_index("model")
    assert got.loc["M1", "implied_error_sd_log"] == pytest.approx(np.sqrt(0.5 * np.var(x)))
    assert got.loc["M2", "implied_error_sd_log"] == 0.0  # no attenuation to explain
    within = x - pd.Series(x).groupby(d["issue_year"].to_numpy()).transform("mean").to_numpy()
    assert got.loc["M2", "var_stated_log_el"] == pytest.approx(np.var(within))


def test_attenuation_simulation_classical_attenuates_and_berkson_does_not():
    rng = np.random.default_rng(4)
    d = synthetic_deals(60, rng)  # log EL is uniform on [log 0.5, log 8]: variance (log 16)^2 / 12
    v = float(np.var(np.log(d["expected_loss_pct"].to_numpy())))
    s = 0.4
    sim = attenuation_simulation(d, [s], resid_sd=0.25, n_rep=600, seed=1)
    got = sim.set_index(["scenario", "model"])["mean_b_hat"]
    assert got[("classical", "M1")] == pytest.approx(1.0 - s * s / v, abs=0.06)
    assert got[("Berkson", "M1")] == pytest.approx(1.0, abs=0.06)
    assert got[("Berkson", "M2")] == pytest.approx(1.0, abs=0.08)
    assert got[("classical", "M2")] < got[("Berkson", "M2")] - 0.15
    # reproducible with the same seed, different with another
    again = attenuation_simulation(d, [s], resid_sd=0.25, n_rep=600, seed=1)
    pd.testing.assert_frame_equal(sim, again)
    other = attenuation_simulation(d, [s], resid_sd=0.25, n_rep=600, seed=2)
    assert not np.allclose(sim["mean_b_hat"], other["mean_b_hat"])
    # no error: both scenarios estimate the true slope without bias
    zero = attenuation_simulation(d, [0.0], resid_sd=0.25, n_rep=300, seed=1)
    assert zero["mean_b_hat"].between(0.95, 1.05).all()
    with pytest.raises(ValueError, match="not below the SD"):
        attenuation_simulation(d, [v**0.5], resid_sd=0.25, n_rep=2)


def test_specifications_on_the_real_sample(real_sample):
    t = specifications(real_sample).set_index("specification")
    ref = fit_log_spread(real_sample, ("year_fe",), cluster="deal_name")
    assert t.loc["M2 (protocol), for reference", "b"] == pytest.approx(ref.coef[1], abs=1e-12)
    assert t["n"].to_dict() == {
        "M2 (protocol), for reference": 56,
        "M2 + region group": 56,
        "M2 + multi-peril indicator": 56,
        "M2 + region group + multi-peril indicator": 56,
        "M1, Florida-only layers": 21,
        "M1, layers naming Florida": 36,
    }
    terms = t.loc["M2 + region group", "indicator_terms"]
    assert "region[no Florida listed]" in terms and "region[Florida only]" in terms
    assert "Florida and other states" not in terms  # the reference level has no coefficient
    # layers that name no Florida are priced below the reference at the same EL and year
    assert float(terms.split("region[no Florida listed] ")[1].split(" ")[0]) < -0.3
    # intervals are ordered where defined
    assert (t["lo_cr1"] < t["b"]).all() and (t["b"] < t["hi_cr1"]).all()


def test_florida_layer_spreads_reproduce_the_protocol_m2(real_sample):
    layers = pd.DataFrame({"layer": ["a", "b"], "el_pct": [3.9, 1.1]})
    t = florida_layer_spreads(real_sample, layers)
    m2 = fit_log_spread(real_sample, ("year_fe",), cluster="deal_name")
    for r in t[t["model"] == "M2 (protocol)"].itertuples():
        want = spread_at(m2, r.el_pct, 2025)
        assert r.spread_median_pct == pytest.approx(want["spread_median_pct"], rel=1e-9)
        assert r.pi_lo_pct == pytest.approx(want["pi_lo_pct"], rel=1e-9)
    post = t[t["model"] == "M2 + region group (post hoc)"]
    assert len(post) == 2 and (post["spread_median_pct"] > 0).all()
