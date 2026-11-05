import numpy as np
import pytest

from catbond.geo import (
    distance_to_edges_km,
    haversine_km,
    local_xy,
    points_in_rings,
    ring_area_centroid,
    segment_edge_crossings,
    snap_to_edges,
)

SQUARE = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]], dtype=float)
HOLE = np.array([[0.4, 0.4], [0.6, 0.4], [0.6, 0.6], [0.4, 0.6], [0.4, 0.4]], dtype=float)


def test_local_xy_matches_haversine_within_one_percent():
    rng = np.random.default_rng(1)
    lon0, lat0 = -81.0, 27.0
    lon = lon0 + rng.uniform(-3, 3, 200)
    lat = lat0 + rng.uniform(-3, 3, 200)
    x, y = local_xy(lon, lat, lon0, lat0)
    d = haversine_km(lon0, lat0, lon, lat)
    assert np.all(np.abs(np.hypot(x, y) / d - 1) < 0.01)


def test_one_degree_of_latitude_is_111_km():
    assert haversine_km(0.0, 0.0, 0.0, 1.0) == pytest.approx(111.195, abs=0.01)
    x, y = local_xy(np.array(10.0), np.array(61.0), 10.0, 60.0)
    assert y == pytest.approx(111.195, abs=0.01) and x == 0


def test_points_in_rings_square_and_hole():
    lon = np.array([0.5, 0.2, 1.5, -0.1, 0.99])
    lat = np.array([0.5, 0.2, 0.5, 0.5, 0.01])
    assert points_in_rings(lon, lat, [SQUARE]).tolist() == [True, True, False, False, True]
    # with a hole the centre is outside
    assert points_in_rings(lon, lat, [SQUARE, HOLE]).tolist() == [False, True, False, False, True]


def test_points_in_two_disjoint_rings():
    other = SQUARE + np.array([3.0, 0.0])
    got = points_in_rings(np.array([0.5, 3.5, 2.0]), np.array([0.5, 0.5, 0.5]), [SQUARE, other])
    assert got.tolist() == [True, True, False]


def test_segment_crossings_through_square():
    t = segment_edge_crossings((-1.0, 0.5), (2.0, 0.5), np.c_[SQUARE[:-1], SQUARE[1:]])
    np.testing.assert_allclose(t, [1 / 3, 2 / 3])
    assert segment_edge_crossings((0.2, 0.2), (0.8, 0.8), np.c_[SQUARE[:-1], SQUARE[1:]]).size == 0


def test_distance_and_snap_to_edges():
    edges = np.array([[-81.0, 27.0, -80.0, 27.0]])
    d, idx = distance_to_edges_km(np.array([-80.5, -79.0]), np.array([28.0, 27.0]), edges)
    assert d[0] == pytest.approx(111.195, rel=1e-3)
    assert d[1] == pytest.approx(111.195 * np.cos(np.deg2rad(27.0)), rel=1e-3)
    assert idx.tolist() == [0, 0]
    lon, lat = snap_to_edges(np.array([-80.5, -79.0, -82.0]), np.array([28.0, 27.0, 26.5]), edges)
    np.testing.assert_allclose(lon, [-80.5, -80.0, -81.0], atol=1e-9)
    np.testing.assert_allclose(lat, [27.0, 27.0, 27.0], atol=1e-9)


def test_area_centroid_of_rectangle_and_l_shape():
    rect = np.array([[-82, 26], [-80, 26], [-80, 26.2], [-82, 26.2], [-82, 26]], dtype=float)
    lon, lat = ring_area_centroid([rect])
    assert lon == pytest.approx(-81.0, abs=1e-6) and lat == pytest.approx(26.1, abs=1e-3)
    # L-shape of two unit squares (0..2 x 0..1) and (0..1 x 1..2) near the equator: centroid
    # (5/6, 5/6) of a degree in the planar limit
    ell = np.array([[0, 0], [2, 0], [2, 1], [1, 1], [1, 2], [0, 2], [0, 0]], dtype=float) * 0.01
    lon, lat = ring_area_centroid([ell])
    assert lon == pytest.approx(5 / 6 * 0.01, rel=1e-4) and lat == pytest.approx(
        5 / 6 * 0.01, rel=1e-4
    )


def test_region_coast_excludes_internal_and_land_border_edges(square_region):
    e = square_region.coast_edges
    # outer box perimeter is 6 unit edges; the shared west/east edge is internal and the
    # northern edge of West is shared with state 13: 5 coastal unit edges remain
    assert len(e) == 5
    keys = {tuple(sorted([(a, b), (c, d)])) for a, b, c, d in e.tolist()}
    assert ((-82.0, 27.0), (-81.0, 27.0)) not in keys
    assert ((-81.0, 26.0), (-81.0, 27.0)) not in keys
    assert len(square_region.counties) == 2
    assert points_in_rings(
        np.array([-81.5, -80.5, -79.5]), np.array([26.5, 26.5, 26.5]), square_region.land_rings
    ).tolist() == [True, True, False]


def test_real_florida_geometry(florida):
    assert len(florida.counties) == 67
    inside = points_in_rings(
        np.array([-80.19, -84.28, -81.38, -84.0, -84.39]),
        np.array([25.77, 30.44, 28.54, 26.0, 33.75]),
        florida.land_rings,
    )
    # Miami, Tallahassee, Orlando inside; a Gulf point and Atlanta outside
    assert inside.tolist() == [True, True, True, False, False]
    d, _ = distance_to_edges_km(np.array([-81.38]), np.array([28.54]), florida.coast_edges)
    assert 40 < d[0] < 120  # Orlando is inland
