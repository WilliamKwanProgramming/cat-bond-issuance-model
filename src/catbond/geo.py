"""Small planar geometry helpers on longitude/latitude data (no GIS dependency).

Distances use a local equirectangular projection about a reference point,
x = R cos(lat0) (lon - lon0) pi/180, y = R (lat - lat0) pi/180, with R = 6371.0088 km (mean
Earth radius). Over the few hundred kilometres used here the error against great-circle distance
is well under 1 percent, which is far below the other approximations in the model.

The Florida land polygon and coastline are built from a US county boundary GeoJSON file
(``data/raw/us_counties.geojson``, see ``scripts/fetch_data.py``): the land polygon is the set of
Florida county rings, and the coastline is the set of Florida county edges that no other county
in the file shares. Edges shared with Georgia or Alabama counties cancel, so the land border is
not counted as coast.
"""

from __future__ import annotations

import json
import pathlib
from collections import Counter
from dataclasses import dataclass

import numpy as np

EARTH_RADIUS_KM = 6371.0088
FLORIDA_FIPS = "12"


def local_xy(
    lon: np.ndarray, lat: np.ndarray, lon0: np.ndarray | float, lat0: np.ndarray | float
) -> tuple[np.ndarray, np.ndarray]:
    """Project lon/lat (degrees) to km east/north of (lon0, lat0), equirectangular about lat0."""
    k = np.pi / 180.0 * EARTH_RADIUS_KM
    x = k * np.cos(np.deg2rad(lat0)) * (np.asarray(lon) - lon0)
    y = k * (np.asarray(lat) - lat0)
    return x, y


def haversine_km(
    lon1: np.ndarray, lat1: np.ndarray, lon2: np.ndarray, lat2: np.ndarray
) -> np.ndarray:
    """Great-circle distance (km).

    d = 2 R asin(sqrt(sin^2(dlat / 2) + cos(lat1) cos(lat2) sin^2(dlon / 2))).
    """
    p1, p2 = np.deg2rad(lat1), np.deg2rad(lat2)
    dphi = p2 - p1
    dlmb = np.deg2rad(np.asarray(lon2) - np.asarray(lon1))
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def points_in_rings(lon: np.ndarray, lat: np.ndarray, rings: list[np.ndarray]) -> np.ndarray:
    """Even-odd (ray casting) point-in-polygon test against a set of closed rings.

    Each ring is an (n, 2) array of lon/lat vertices. A point is inside if a ray towards +lon
    crosses an odd number of ring edges in total, so holes and multi-part polygons are handled
    when all their rings are passed together.
    """
    lon = np.atleast_1d(np.asarray(lon, dtype=float))
    lat = np.atleast_1d(np.asarray(lat, dtype=float))
    inside = np.zeros(lon.shape, dtype=bool)
    for ring in rings:
        x1, y1 = ring[:-1, 0], ring[:-1, 1]
        x2, y2 = ring[1:, 0], ring[1:, 1]
        straddle = (y1[None, :] > lat[:, None]) != (y2[None, :] > lat[:, None])
        with np.errstate(divide="ignore", invalid="ignore"):
            xcross = x1 + (lat[:, None] - y1) * (x2 - x1) / (y2 - y1)
        hits = straddle & (lon[:, None] < xcross)
        inside ^= (hits.sum(axis=1) % 2).astype(bool)
    return inside


def segment_edge_crossings(
    p0: tuple[float, float], p1: tuple[float, float], edges: np.ndarray
) -> np.ndarray:
    """Parameters t in [0, 1] at which the segment p0 -> p1 crosses any of ``edges``.

    ``edges`` is (m, 4): lon_a, lat_a, lon_b, lat_b. Coordinates are treated as planar, which is
    adequate for 6-hourly track segments of a few hundred km. Returns the sorted crossing
    parameters along the segment (empty if none). Parallel edges are ignored.
    """
    px, py = p0
    rx, ry = p1[0] - px, p1[1] - py
    qx, qy = edges[:, 0], edges[:, 1]
    sx, sy = edges[:, 2] - qx, edges[:, 3] - qy
    denom = rx * sy - ry * sx
    with np.errstate(divide="ignore", invalid="ignore"):
        t = ((qx - px) * sy - (qy - py) * sx) / denom
        u = ((qx - px) * ry - (qy - py) * rx) / denom
    ok = (denom != 0) & (t >= 0) & (t <= 1) & (u >= 0) & (u < 1)
    return np.sort(t[ok])


def distance_to_edges_km(
    lon: np.ndarray, lat: np.ndarray, edges: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Distance (km) from each point to the nearest edge, and the index of that edge.

    Each point is projected with ``local_xy`` about itself, then the point-to-segment distance is
    the distance to the clamped orthogonal projection onto the segment.
    """
    lon = np.atleast_1d(np.asarray(lon, dtype=float))
    lat = np.atleast_1d(np.asarray(lat, dtype=float))
    dist = np.empty(lon.shape)
    idx = np.empty(lon.shape, dtype=int)
    for i in range(lon.size):
        ax, ay = local_xy(edges[:, 0], edges[:, 1], lon[i], lat[i])
        bx, by = local_xy(edges[:, 2], edges[:, 3], lon[i], lat[i])
        dx, dy = bx - ax, by - ay
        ll = dx * dx + dy * dy
        with np.errstate(divide="ignore", invalid="ignore"):
            s = np.where(ll > 0, -(ax * dx + ay * dy) / ll, 0.0)
        s = np.clip(s, 0.0, 1.0)
        d = np.hypot(ax + s * dx, ay + s * dy)
        idx[i] = int(np.argmin(d))
        dist[i] = d[idx[i]]
    return dist, idx


def snap_to_edges(
    lon: np.ndarray, lat: np.ndarray, edges: np.ndarray, chunk: int = 1000
) -> tuple[np.ndarray, np.ndarray]:
    """Lon/lat of the nearest point on ``edges`` for every input point (vectorised in chunks).

    Distances are measured in the local projection of each point (see ``local_xy``).
    """
    lon = np.atleast_1d(np.asarray(lon, dtype=float))
    lat = np.atleast_1d(np.asarray(lat, dtype=float))
    out_lon, out_lat = np.empty_like(lon), np.empty_like(lat)
    for i0 in range(0, lon.size, chunk):
        sl = slice(i0, i0 + chunk)
        p_lon, p_lat = lon[sl, None], lat[sl, None]
        ax, ay = local_xy(edges[None, :, 0], edges[None, :, 1], p_lon, p_lat)
        bx, by = local_xy(edges[None, :, 2], edges[None, :, 3], p_lon, p_lat)
        dx, dy = bx - ax, by - ay
        ll = dx * dx + dy * dy
        with np.errstate(divide="ignore", invalid="ignore"):
            s = np.where(ll > 0, -(ax * dx + ay * dy) / ll, 0.0)
        s = np.clip(s, 0.0, 1.0)
        k = np.argmin(np.hypot(ax + s * dx, ay + s * dy), axis=1)
        rows = np.arange(k.size)
        sk = s[rows, k]
        e = edges[k]
        out_lon[sl] = e[:, 0] + sk * (e[:, 2] - e[:, 0])
        out_lat[sl] = e[:, 1] + sk * (e[:, 3] - e[:, 1])
    return out_lon, out_lat


def ring_area_centroid(rings: list[np.ndarray]) -> tuple[float, float]:
    """Area-weighted centroid (lon, lat) of a set of outer rings (shoelace formula).

    The rings are projected with ``local_xy`` about their mean vertex so that areas are in km^2;
    signed areas are used, so a hole listed with opposite orientation is subtracted.
    """
    allv = np.vstack(rings)
    lon0, lat0 = float(allv[:, 0].mean()), float(allv[:, 1].mean())
    a_tot = cx_tot = cy_tot = 0.0
    for ring in rings:
        x, y = local_xy(ring[:, 0], ring[:, 1], lon0, lat0)
        cross = x[:-1] * y[1:] - x[1:] * y[:-1]
        a = 0.5 * cross.sum()
        a_tot += a
        cx_tot += ((x[:-1] + x[1:]) * cross).sum() / 6.0
        cy_tot += ((y[:-1] + y[1:]) * cross).sum() / 6.0
    cx, cy = cx_tot / a_tot, cy_tot / a_tot
    k = np.pi / 180.0 * EARTH_RADIUS_KM
    return lon0 + cx / (k * np.cos(np.deg2rad(lat0))), lat0 + cy / k


@dataclass(frozen=True)
class Region:
    """Florida land rings, coastline edges and county table from the county GeoJSON file."""

    land_rings: list[np.ndarray]
    coast_edges: np.ndarray  # (m, 4) lon_a, lat_a, lon_b, lat_b
    counties: list[tuple[str, str, list[np.ndarray]]]  # (fips, name, rings)
    other_coast_edges: np.ndarray  # unshared edges of other states' counties near the region


def _rings_of(geometry: dict) -> list[np.ndarray]:
    polys = [geometry["coordinates"]] if geometry["type"] == "Polygon" else geometry["coordinates"]
    return [np.asarray(r, dtype=float) for p in polys for r in p]


def _edge_key(a: np.ndarray, b: np.ndarray) -> tuple:
    pa = (round(float(a[0]), 6), round(float(a[1]), 6))
    pb = (round(float(b[0]), 6), round(float(b[1]), 6))
    return (pa, pb) if pa <= pb else (pb, pa)


def load_region(
    path: pathlib.Path, state_fips: str = FLORIDA_FIPS, margin_deg: float = 3.0
) -> Region:
    """Build the land rings, coastline and county list of one state from the county GeoJSON.

    Coastline = edges of the state's county rings that occur exactly once in the whole file
    (vertices rounded to 1e-6 degrees for matching). ``other_coast_edges`` are the unshared edges
    of other states' counties within ``margin_deg`` degrees of the state's bounding box; they are
    used to decide whether a coastal point is nearer this state's coast than a neighbour's.
    """
    features = json.loads(pathlib.Path(path).read_text())["features"]
    counts: Counter = Counter()
    for f in features:
        for r in _rings_of(f["geometry"]):
            for a, b in zip(r[:-1], r[1:], strict=True):
                counts[_edge_key(a, b)] += 1
    counties = []
    land: list[np.ndarray] = []
    coast: list[list[float]] = []
    other: list[list[float]] = []
    for f in features:
        own = str(f["id"]).startswith(state_fips)
        rings = _rings_of(f["geometry"])
        if own:
            counties.append((str(f["id"]), str(f["properties"]["NAME"]), rings))
            land.extend(rings)
        for r in rings:
            for a, b in zip(r[:-1], r[1:], strict=True):
                if counts[_edge_key(a, b)] == 1:
                    (coast if own else other).append([a[0], a[1], b[0], b[1]])
    coast_arr = np.asarray(coast)
    other_arr = np.asarray(other)
    lo_lon, hi_lon = (
        coast_arr[:, [0, 2]].min() - margin_deg,
        coast_arr[:, [0, 2]].max() + margin_deg,
    )
    lo_lat, hi_lat = (
        coast_arr[:, [1, 3]].min() - margin_deg,
        coast_arr[:, [1, 3]].max() + margin_deg,
    )
    keep = (
        (other_arr[:, 0] >= lo_lon)
        & (other_arr[:, 0] <= hi_lon)
        & (other_arr[:, 1] >= lo_lat)
        & (other_arr[:, 1] <= hi_lat)
    )
    counties.sort(key=lambda c: c[0])
    return Region(
        land_rings=land, coast_edges=coast_arr, counties=counties, other_coast_edges=other_arr[keep]
    )
