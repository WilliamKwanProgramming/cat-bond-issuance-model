import numpy as np
import pytest

from catbond.hazard.windfield import (
    KD_VB_KT,
    holland_b_willoughby_rahn,
    holland_shape,
    kaplan_demaria,
    peak_site_winds,
    rmw_willoughby_rahn,
)
from catbond.vulnerability import emanuel_damage


def test_holland_shape_limits():
    r = np.array([1e-6, 30.0, 1e6])
    h = holland_shape(r, 30.0, 1.4)
    assert h[0] == pytest.approx(0.0, abs=1e-12)
    assert h[1] == pytest.approx(1.0)
    assert h[2] < 2e-3 and holland_shape(np.array(1e9), 30.0, 1.4) < 1e-5
    grid = np.linspace(1, 300, 3000)
    hh = holland_shape(grid, 30.0, 1.4)
    assert grid[np.argmax(hh)] == pytest.approx(30.0, abs=0.2)
    # larger B: faster decay outside R
    assert holland_shape(np.array(100.0), 30.0, 2.0) < holland_shape(np.array(100.0), 30.0, 1.0)


def test_holland_shape_closed_form_value():
    # r = 2R, B = 1: x = 1/2, h = sqrt(0.5 e^0.5)
    assert holland_shape(np.array(60.0), 30.0, 1.0) == pytest.approx(np.sqrt(0.5 * np.exp(0.5)))


def test_willoughby_rahn_relations_by_hand():
    v_kt, lat = 100.0, 25.0
    v_ms = 100.0 * 0.514444
    assert rmw_willoughby_rahn(v_kt, lat) == pytest.approx(
        46.29 * np.exp(-0.0153 * v_ms + 0.0166 * lat)
    )
    assert rmw_willoughby_rahn(v_kt, lat) == pytest.approx(31.91, abs=0.02)
    assert holland_b_willoughby_rahn(v_kt, lat) == pytest.approx(
        0.886 + 0.0177 * v_ms - 0.0094 * lat
    )
    assert holland_b_willoughby_rahn(v_kt, lat) == pytest.approx(1.5615, abs=1e-3)
    assert (
        holland_b_willoughby_rahn(300.0, 0.0) == 2.5 and holland_b_willoughby_rahn(0.0, 60) == 1.0
    )


def test_kaplan_demaria_decay():
    assert kaplan_demaria(120.0, 0.0) == 120.0
    assert kaplan_demaria(120.0, -5.0) == 120.0
    assert kaplan_demaria(120.0, 1e-9) == pytest.approx(108.0)
    assert kaplan_demaria(120.0, 1e4) == pytest.approx(KD_VB_KT)
    assert kaplan_demaria(120.0, 10.0) == pytest.approx(26.7 + (108.0 - 26.7) * np.exp(-0.95))
    assert kaplan_demaria(120.0, 10.0, decay=False) == 120.0


def one_event(**kw):
    ev = dict(
        lon0=np.array([-80.0]),
        lat0=np.array([26.0]),
        heading_deg=np.array([270.0]),
        speed_kmh=np.array([20.0]),
        vmax_kt=np.array([120.0]),
        rmw_km=np.array([30.0]),
        b=np.array([1.4]),
    )
    ev.update(kw)
    return ev


def site_at(dx_km, dy_km, lon0=-80.0, lat0=26.0):
    k = np.pi / 180 * 6371.0088
    return np.array([lon0 + dx_km / (k * np.cos(np.deg2rad(lat0)))]), np.array([lat0 + dy_km / k])


def test_peak_wind_right_and_left_of_track():
    ev = one_event()
    # moving west: right of track is north. Site 30 km north of the landfall point.
    lon, lat = site_at(0.0, 30.0)
    right = peak_site_winds(**ev, site_lon=lon, site_lat=lat, decay=False)[0, 0]
    lon, lat = site_at(0.0, -30.0)
    left = peak_site_winds(**ev, site_lon=lon, site_lat=lat, decay=False)[0, 0]
    vt = 20.0 / 1.852
    assert right == pytest.approx(120.0, rel=1e-6)
    # at closest approach the left side has 120 - 2 V_t; slightly earlier or later the site is
    # off the beam (cos theta > -1), so the maximum over time is a little higher
    assert 120.0 - 2 * vt <= left < right - vt


def test_decay_reduces_inland_wind():
    ev = one_event()
    lon, lat = site_at(-200.0, 30.0)  # 200 km down-track (west), 30 km right of track
    with_decay = peak_site_winds(**ev, site_lon=lon, site_lat=lat)[0, 0]
    without = peak_site_winds(**ev, site_lon=lon, site_lat=lat, decay=False)[0, 0]
    t = 200.0 / 20.0
    assert without == pytest.approx(120.0, rel=1e-6)
    assert with_decay == pytest.approx(kaplan_demaria(120.0, t), rel=0.02)
    assert with_decay < 0.6 * without


def test_candidate_maximum_matches_dense_search():
    rng = np.random.default_rng(4)
    n = 200
    ev = one_event(
        lon0=-80 - rng.uniform(0, 2, n),
        lat0=25 + rng.uniform(0, 5, n),
        heading_deg=rng.uniform(0, 360, n),
        speed_kmh=rng.uniform(5, 60, n),
        vmax_kt=rng.uniform(64, 160, n),
        rmw_km=rng.uniform(10, 60, n),
        b=rng.uniform(1, 2.5, n),
    )
    site_lon = -82 + rng.uniform(-2, 2, 30)
    site_lat = 27 + rng.uniform(-2, 3, 30)
    fast = peak_site_winds(**ev, site_lon=site_lon, site_lat=site_lat)
    dense = peak_site_winds(
        **ev, site_lon=site_lon, site_lat=site_lat, offsets=np.linspace(-15, 15, 3001)
    )
    # both are maxima over finite sets of real track positions (lower bounds of the true peak)
    assert np.all(fast <= dense * 1.001 + 1e-9)
    # where damage can occur (above the 50 kt threshold) the shortfall is under 1 percent of the
    # peak wind and the summed damage ratio is within 0.2 percent
    big = dense > 50
    assert np.max(1 - fast[big] / dense[big]) < 0.01
    assert emanuel_damage(fast).sum() / emanuel_damage(dense).sum() > 0.998
