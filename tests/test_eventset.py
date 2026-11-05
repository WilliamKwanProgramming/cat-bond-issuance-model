import numpy as np
import pandas as pd
import pytest
from conftest import toy_hazard

from catbond.eventset import EVENT_COLUMNS, LossSettings, event_losses, simulate_event_set
from catbond.hazard.windfield import (
    holland_b_willoughby_rahn,
    peak_site_winds,
    rmw_willoughby_rahn,
)
from catbond.vulnerability import emanuel_damage, insured_loss_ratio

COAST = np.array([[-80.0, 25.0, -80.0, 28.0]])  # a straight north-south "coast" at lon -80


def sites(n=12, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "lon": -80.2 - rng.uniform(0, 2.5, n),
            "lat": 25.5 + rng.uniform(0, 2, n),
            "tiv_usd": rng.uniform(1e8, 5e9, n),
        }
    )


def test_seed_reproducibility_and_columns():
    m, ex = toy_hazard(), sites()
    a = simulate_event_set(m, COAST, ex, 3000, seed=42)
    b = simulate_event_set(m, COAST, ex, 3000, seed=42)
    c = simulate_event_set(m, COAST, ex, 3000, seed=43)
    assert list(a.events.columns) == EVENT_COLUMNS
    pd.testing.assert_frame_equal(a.events, b.events)
    assert not a.events["loss_usd"].equals(c.events["loss_usd"])
    assert a.events["year"].between(0, 2999).all() and a.events["event_id"].is_unique


def test_event_attributes_do_not_depend_on_exposure_or_settings():
    m = toy_hazard()
    a = simulate_event_set(m, COAST, sites(seed=1), 2000, seed=7)
    b = simulate_event_set(m, COAST, sites(n=5, seed=2), 2000, seed=7, settings=LossSettings(0.0))
    cols = ["year", "lon", "lat", "wind_kt", "heading_deg", "speed_kmh"]
    pd.testing.assert_frame_equal(a.events[cols], b.events[cols])


def test_frequency_mean_and_multiplier():
    m = toy_hazard(freq=0.6)
    es = simulate_event_set(m, COAST, sites(), 50_000, seed=3)
    assert len(es.events) / 50_000 == pytest.approx(0.6, abs=4 * np.sqrt(0.6 / 50_000))
    es2 = simulate_event_set(m.scaled(1.2), COAST, sites(), 50_000, seed=3)
    assert len(es2.events) / 50_000 == pytest.approx(0.72, abs=4 * np.sqrt(0.72 / 50_000))


def test_event_loss_is_sum_of_county_losses():
    ex = sites()
    params = pd.DataFrame(
        {
            "lon": [-80.0, -80.0],
            "lat": [26.0, 27.0],
            "heading_deg": [270.0, 250.0],
            "speed_kmh": [20.0, 35.0],
            "wind_kt": [130.0, 90.0],
        }
    )
    st = LossSettings(deductible=0.02)
    out = event_losses(params, ex, st)
    for i in range(2):
        p = params.iloc[i]
        r = rmw_willoughby_rahn(p["wind_kt"], p["lat"])
        b = holland_b_willoughby_rahn(p["wind_kt"], p["lat"])
        w = peak_site_winds(
            np.array([p["lon"]]),
            np.array([p["lat"]]),
            np.array([p["heading_deg"]]),
            np.array([p["speed_kmh"]]),
            np.array([p["wind_kt"]]),
            np.array([r]),
            np.array([b]),
            ex["lon"].to_numpy(),
            ex["lat"].to_numpy(),
        )[0]
        expected = (insured_loss_ratio(emanuel_damage(w), 0.02) * ex["tiv_usd"]).sum()
        assert out["loss_usd"].iloc[i] == pytest.approx(expected, rel=1e-12)
        assert out["rmw_km"].iloc[i] == pytest.approx(r)


def test_zero_loss_below_damage_threshold():
    ex = sites()
    params = pd.DataFrame(
        {
            "lon": [-80.0],
            "lat": [26.0],
            "heading_deg": [90.0],
            "speed_kmh": [20.0],
            "wind_kt": [70.0],
        }
    )
    # heading east, away from all sites (which lie west of the coast): far-field winds only
    w_far = pd.DataFrame({"lon": [-82.5], "lat": [26.0], "tiv_usd": [1e9]})
    assert event_losses(params, w_far, LossSettings())["loss_usd"].item() == 0.0
    # threshold just above the event's peak wind: no site can be damaged
    st = LossSettings(v_thresh_kt=70.0, v_half_kt=150.0, deductible=0.0)
    assert event_losses(params, ex, st)["loss_usd"].item() == 0.0


def test_deductible_and_decay_reduce_losses():
    m, ex = toy_hazard(), sites()
    base = simulate_event_set(m, COAST, ex, 4000, seed=5).events["loss_usd"]
    no_ded = simulate_event_set(
        m, COAST, ex, 4000, seed=5, settings=LossSettings(deductible=0.0)
    ).events["loss_usd"]
    no_decay = simulate_event_set(
        m, COAST, ex, 4000, seed=5, settings=LossSettings(decay=False)
    ).events["loss_usd"]
    assert np.all(no_ded >= base) and no_ded.sum() > base.sum()
    assert np.all(no_decay >= base - 1e-6) and no_decay.sum() > 1.1 * base.sum()
