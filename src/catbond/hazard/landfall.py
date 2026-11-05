"""Florida hurricane landfalls from best-track data: two pre-specified definitions.

**Record definition** (used for calibration, 1900-2019): a HURDAT2 fix with record identifier
"L" ("center of system crossing a coastline", Landsea and Franklin 2013) whose position is at most
``max_coast_km`` (25 km) from the Florida coastline and not nearer to another state's coast, with
maximum wind >= 64 kt at that fix.

**Geometric definition** (used where no record identifier exists, i.e. the 1975-2024 file): the
straight segment between consecutive fixes crosses the Florida coastline from water to land, and
the wind linearly interpolated to the crossing is >= 64 kt.

A storm counts once per year however many qualifying Florida landfalls it makes; its event
attributes are those of its most intense qualifying landfall (the first one on a tie). Heading
and translation speed are taken from the fixes before and after the landfall time (great-circle
bearing and distance divided by elapsed time).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from catbond.geo import (
    Region,
    distance_to_edges_km,
    haversine_km,
    points_in_rings,
    segment_edge_crossings,
)

HURRICANE_KT = 64.0
LANDFALL_COLUMNS = [
    "storm_id",
    "name",
    "year",
    "time",
    "lat",
    "lon",
    "wind",
    "heading_deg",
    "speed_kmh",
    "n_fl_landfalls",
    "definition",
]


def bearing_deg(
    lon1: np.ndarray, lat1: np.ndarray, lon2: np.ndarray, lat2: np.ndarray
) -> np.ndarray:
    """Initial great-circle bearing from point 1 to point 2, degrees clockwise from north."""
    p1, p2 = np.deg2rad(lat1), np.deg2rad(lat2)
    dl = np.deg2rad(np.asarray(lon2) - np.asarray(lon1))
    y = np.sin(dl) * np.cos(p2)
    x = np.cos(p1) * np.sin(p2) - np.sin(p1) * np.cos(p2) * np.cos(dl)
    return np.rad2deg(np.arctan2(y, x)) % 360.0


def _motion(track: pd.DataFrame, when: pd.Timestamp) -> tuple[float, float]:
    """Heading (deg) and speed (km/h) between the last fix before and the first fix after ``when``.

    If there is no fix on one side, the fix at ``when`` itself is used on that side.
    """
    before = track[track["time"] < when]
    after = track[track["time"] > when]
    at = track[track["time"] == when]
    a = before.iloc[-1] if len(before) else (at.iloc[0] if len(at) else None)
    b = after.iloc[0] if len(after) else (at.iloc[-1] if len(at) else None)
    if a is None or b is None or a["time"] == b["time"]:
        return float("nan"), float("nan")
    hours = (b["time"] - a["time"]).total_seconds() / 3600.0
    dist = float(haversine_km(a["lon"], a["lat"], b["lon"], b["lat"]))
    return float(bearing_deg(a["lon"], a["lat"], b["lon"], b["lat"])), dist / hours


def _one_per_storm(cands: pd.DataFrame, definition: str) -> pd.DataFrame:
    if cands.empty:
        return pd.DataFrame(columns=LANDFALL_COLUMNS)
    cands = cands.sort_values(["storm_id", "time"], kind="stable")
    n = cands.groupby("storm_id")["wind"].transform("size")
    best = (
        cands.assign(n_fl_landfalls=n)
        .sort_values(["storm_id", "wind", "time"], ascending=[True, False, True], kind="stable")
        .groupby("storm_id", sort=False)
        .head(1)
    )
    best = best.assign(definition=definition)
    return best[LANDFALL_COLUMNS].sort_values("time", kind="stable").reset_index(drop=True)


def landfalls_from_records(
    tracks: pd.DataFrame,
    region: Region,
    max_coast_km: float = 25.0,
    min_wind: float = HURRICANE_KT,
) -> pd.DataFrame:
    """Florida hurricane landfalls by the record definition (one row per storm, see module doc)."""
    rec = tracks[(tracks["record"] == "L") & tracks["lon"].notna()].copy()
    if rec.empty:
        return _one_per_storm(rec, "record")
    d_fl, _ = distance_to_edges_km(rec["lon"].to_numpy(), rec["lat"].to_numpy(), region.coast_edges)
    d_other, _ = distance_to_edges_km(
        rec["lon"].to_numpy(), rec["lat"].to_numpy(), region.other_coast_edges
    )
    keep = (d_fl <= max_coast_km) & (d_fl <= d_other) & (rec["wind"].to_numpy() >= min_wind)
    rec = rec[keep]
    motion = [
        _motion(tracks[tracks["storm_id"] == s], t)
        for s, t in zip(rec["storm_id"], rec["time"], strict=True)
    ]
    rec["heading_deg"] = [m[0] for m in motion]
    rec["speed_kmh"] = [m[1] for m in motion]
    return _one_per_storm(rec, "record")


def landfalls_geometric(
    tracks: pd.DataFrame, region: Region, min_wind: float = HURRICANE_KT
) -> pd.DataFrame:
    """Florida hurricane landfalls by the geometric definition (one row per storm)."""
    lo_lon = region.coast_edges[:, [0, 2]].min() - 1.0
    hi_lon = region.coast_edges[:, [0, 2]].max() + 1.0
    lo_lat = region.coast_edges[:, [1, 3]].min() - 1.0
    hi_lat = region.coast_edges[:, [1, 3]].max() + 1.0
    near = tracks["lon"].between(lo_lon, hi_lon) & tracks["lat"].between(lo_lat, hi_lat)
    rows = []
    for sid in tracks.loc[near, "storm_id"].unique():
        tr = tracks[tracks["storm_id"] == sid].dropna(subset=["lon", "lat"])
        lon, lat = tr["lon"].to_numpy(), tr["lat"].to_numpy()
        wind, time = tr["wind"].to_numpy(), tr["time"].to_numpy()
        if len(tr) < 2:
            continue
        # land/water state just after each segment start, so that a fix lying exactly on the
        # coastline is not counted twice (once as a segment end, once as the next start)
        eps = 1e-6
        inside = points_in_rings(
            lon[:-1] + eps * np.diff(lon), lat[:-1] + eps * np.diff(lat), region.land_rings
        )
        for i in range(len(tr) - 1):
            ts = segment_edge_crossings(
                (lon[i], lat[i]), (lon[i + 1], lat[i + 1]), region.coast_edges
            )
            state = bool(inside[i])
            for t in ts[ts > eps]:
                state = not state
                if not state:
                    continue  # exiting crossing
                w = wind[i] + t * (wind[i + 1] - wind[i])
                if not w >= min_wind:
                    continue
                hours = (time[i + 1] - time[i]) / np.timedelta64(1, "h")
                dist = float(haversine_km(lon[i], lat[i], lon[i + 1], lat[i + 1]))
                rows.append(
                    {
                        "storm_id": sid,
                        "name": tr["name"].iloc[0],
                        "year": int(tr["year"].iloc[0]),
                        "time": (
                            pd.Timestamp(time[i]) + pd.Timedelta(hours=float(t * hours))
                        ).round("min"),
                        "lat": lat[i] + t * (lat[i + 1] - lat[i]),
                        "lon": lon[i] + t * (lon[i + 1] - lon[i]),
                        "wind": float(w),
                        "heading_deg": float(bearing_deg(lon[i], lat[i], lon[i + 1], lat[i + 1])),
                        "speed_kmh": dist / hours if hours > 0 else float("nan"),
                    }
                )
    return _one_per_storm(pd.DataFrame(rows), "geometric")


FLAG_GAP_YEARS = (1966, 1984)
SYNOPTIC_HOURS = (0, 6, 12, 18)


def synoptic_fixes(tracks: pd.DataFrame) -> pd.DataFrame:
    """Only the fixes at 00, 06, 12 and 18 UTC on the hour.

    In the 1966-1984 flag gap the tracks have (almost) no off-synoptic fixes, and the "L" fixes
    of other years are off-synoptic, so the geometric definition applied to the synoptic fixes of
    the flagged years measures how it performs in the gap: it must interpolate the wind between
    6-hourly fixes instead of crossing the coast at a fix that carries the landfall wind.
    """
    t = tracks["time"]
    return tracks[t.dt.hour.isin(SYNOPTIC_HOURS) & (t.dt.minute == 0)]


def hybrid_landfalls(
    tracks: pd.DataFrame, region: Region, gap: tuple[int, int] = FLAG_GAP_YEARS
) -> pd.DataFrame:
    """Record definition outside ``gap``; geometric definition for years inside ``gap``.

    In ``AL.rda`` the "L" record identifier is almost absent for 1966-1984 (six "L" records in
    19 years, none in Florida, against 36 to 85 per five-year period in 1945-1964 and 1985-2019;
    see results/hazard/l_flags_by_period.csv), so the record definition alone would undercount
    those years.
    """
    in_gap = tracks["year"].between(gap[0], gap[1])
    rec = landfalls_from_records(tracks[~in_gap], region)
    geo = landfalls_geometric(tracks[in_gap], region)
    out = pd.concat([rec, geo], ignore_index=True) if len(geo) else rec
    return out.sort_values("time", kind="stable").reset_index(drop=True)


def annual_counts(landfalls: pd.DataFrame, first_year: int, last_year: int) -> pd.Series:
    """Number of qualifying storms per year for every year in [first_year, last_year] (zeros kept).

    Only rows with ``first_year <= year <= last_year`` are read.
    """
    years = np.arange(first_year, last_year + 1)
    inwin = landfalls[(landfalls["year"] >= first_year) & (landfalls["year"] <= last_year)]
    return inwin.groupby("year").size().reindex(years, fill_value=0).rename("count")


SAFFIR_SIMPSON_KT = (64.0, 83.0, 96.0, 113.0, 137.0)


def category(wind_kt: np.ndarray) -> np.ndarray:
    """Saffir-Simpson category 1-5 from 1-min wind in kt (0 below 64 kt).

    Category lower bounds: 64, 83, 96, 113, 137 kt (NHC Saffir-Simpson Hurricane Wind Scale).
    """
    return np.searchsorted(
        np.asarray(SAFFIR_SIMPSON_KT), np.asarray(wind_kt, dtype=float), side="right"
    )
