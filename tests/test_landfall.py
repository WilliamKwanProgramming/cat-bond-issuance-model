import numpy as np
import pandas as pd
import pytest

from catbond.hazard.landfall import (
    annual_counts,
    bearing_deg,
    category,
    hybrid_landfalls,
    landfalls_from_records,
    landfalls_geometric,
    near_misses,
    synoptic_fixes,
)


def track(sid, year, pts, start="09-01 00:00", step_h=6.0):
    """pts: list of (lon, lat, wind, record)."""
    t0 = pd.Timestamp(f"{year}-{start}")
    return pd.DataFrame(
        {
            "storm_id": sid,
            "name": sid,
            "year": year,
            "time": [t0 + pd.Timedelta(hours=step_h * i) for i in range(len(pts))],
            "record": [p[3] for p in pts],
            "status": "HU",
            "lat": [p[1] for p in pts],
            "lon": [p[0] for p in pts],
            "wind": [float(p[2]) for p in pts],
            "pressure": np.nan,
        }
    )


@pytest.fixture()
def toy_tracks():
    # The toy region is the box lon [-82, -80], lat [26, 27]; its east edge lon = -80 is coast.
    a = track("A", 1950, [(-79.0, 26.5, 90, ""), (-80.0, 26.5, 85, "L"), (-81.0, 26.5, 60, "")])
    b = track("B", 1950, [(-79.0, 26.6, 60, ""), (-80.0, 26.6, 60, "L"), (-81.0, 26.6, 40, "")])
    c = track("C", 1951, [(-75.0, 26.5, 90, ""), (-76.0, 26.5, 90, "L")])  # far from coast
    # D: two landfalls on the toy coast (east side, then south side after exit); the second
    # is stronger and is the one kept
    d = track(
        "D",
        1952,
        [
            (-79.0, 26.2, 70, ""),
            (-80.0, 26.2, 70, "L"),
            (-80.5, 25.5, 70, ""),
            (-80.5, 26.0, 100, "L"),
            (-80.5, 26.5, 80, ""),
        ],
    )
    return pd.concat([a, b, c, d], ignore_index=True)


def test_bearing():
    assert bearing_deg(0.0, 0.0, 0.0, 1.0) == pytest.approx(0.0)
    assert bearing_deg(0.0, 0.0, 1.0, 0.0) == pytest.approx(90.0)
    assert bearing_deg(0.0, 0.0, -1.0, 0.0) == pytest.approx(270.0)


def test_record_definition(square_region, toy_tracks):
    lf = landfalls_from_records(toy_tracks, square_region)
    assert lf["storm_id"].tolist() == ["A", "D"]
    a = lf.iloc[0]
    assert a["wind"] == 85 and a["lon"] == -80.0
    # motion from (-79, 26.5) to (-81, 26.5) in 12 h: due west, about 2 * 99.5 km / 12 h
    assert a["heading_deg"] == pytest.approx(270.0, abs=0.5)
    assert a["speed_kmh"] == pytest.approx(2 * 111.195 * np.cos(np.deg2rad(26.5)) / 12, rel=0.01)
    d = lf.iloc[1]
    assert d["wind"] == 100 and d["n_fl_landfalls"] == 2 and d["lat"] == 26.0


def test_record_definition_threshold_is_inclusive(square_region, toy_tracks):
    t = toy_tracks.copy()
    t.loc[(t["storm_id"] == "B") & (t["record"] == "L"), "wind"] = 64.0
    assert "B" in set(landfalls_from_records(t, square_region)["storm_id"])


def test_geometric_definition(square_region, toy_tracks):
    lf = landfalls_geometric(toy_tracks.assign(record=""), square_region)
    assert lf["storm_id"].tolist() == ["A", "D"]
    a = lf.iloc[0]
    assert a["lon"] == pytest.approx(-80.0) and a["wind"] == pytest.approx(85.0)
    # D: east-coast crossing at 70 kt, exit through the south coast is not a landfall, and the
    # re-entry at lat 26 has the interpolated wind 100 kt
    d = lf.iloc[1]
    assert d["wind"] == pytest.approx(100.0) and d["n_fl_landfalls"] == 2


def test_geometric_interpolates_wind_at_crossing(square_region):
    # crossing at t = 0.5 between 70 kt and 56 kt fixes: 63 kt, below the threshold
    t = track("E", 1960, [(-79.5, 26.5, 70, ""), (-80.5, 26.5, 56, "")])
    assert landfalls_geometric(t, square_region).empty
    t2 = track("E", 1960, [(-79.5, 26.5, 72, ""), (-80.5, 26.5, 56, "")])  # 64 kt at crossing
    assert len(landfalls_geometric(t2, square_region)) == 1


def test_annual_counts_keep_zero_years():
    lf = pd.DataFrame({"year": [1901, 1901, 1903, 1990]})
    c = annual_counts(lf, 1900, 1904)
    assert c.index.tolist() == [1900, 1901, 1902, 1903, 1904]
    assert c.tolist() == [0, 2, 0, 1, 0]


def test_hybrid_uses_geometry_only_inside_gap(square_region, toy_tracks):
    t = toy_tracks.assign(record="")  # no flags at all
    assert hybrid_landfalls(t, square_region, gap=(1900, 1949)).empty
    assert hybrid_landfalls(t, square_region, gap=(1950, 1952))["storm_id"].tolist() == ["A", "D"]


def test_real_florida_landfalls(al_tracks, florida):
    rec = landfalls_from_records(al_tracks[al_tracks["year"] >= 1900], florida)
    geo = landfalls_geometric(al_tracks[al_tracks["year"] >= 1900], florida)
    gap = rec["year"].between(1966, 1984)
    assert not gap.any()  # no Florida "L" records in the flag gap
    # in years with flags the geometric definition finds exactly the same storms
    geo_flagged = set(geo.loc[~geo["year"].between(1966, 1984), "storm_id"])
    assert geo_flagged == set(rec["storm_id"]) and len(geo_flagged) == 62
    hyb = hybrid_landfalls(al_tracks[al_tracks["year"] >= 1900], florida)
    assert len(hyb) == 67
    names = set(zip(hyb["name"], hyb["year"], strict=True))
    assert {("ANDREW", 1992), ("MICHAEL", 2018), ("ELOISE", 1975), ("IRMA", 2017)} <= names
    assert hyb.loc[hyb["name"] == "MICHAEL", "wind"].item() == 140
    assert ("IVAN", 2004) not in names  # Alabama landfall


def test_real_near_misses_are_not_landfalls(al_tracks, florida):
    tr = al_tracks[al_tracks["year"].between(1900, 2019)]
    hyb = hybrid_landfalls(tr, florida)
    nm = near_misses(tr, hyb, florida, max_km=50.0)
    names = set(zip(nm["name"], nm["year"], strict=True))
    # Ivan came ashore 3 km from Alabama's coast; Matthew passed 42 km off Cape Canaveral
    assert {("IVAN", 2004), ("MATTHEW", 2016)} <= names
    assert not set(nm["storm_id"]) & set(hyb["storm_id"])
    assert (nm["wind"] >= 64).all() and (nm["distance_florida_km"] <= 50).all()
    assert len(near_misses(tr, hyb, florida, max_km=100.0)) > len(nm)
    ivan = nm[nm["name"] == "IVAN"].iloc[0]
    assert ivan["distance_other_coast_km"] < ivan["distance_florida_km"]


def test_saffir_simpson_category():
    v = np.array([50, 63.9, 64, 82, 83, 95, 96, 112, 113, 136, 137, 180])
    assert category(v).tolist() == [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5]


def test_synoptic_fixes_lose_the_landfall_wind(square_region):
    # 90 kt at an off-synoptic landfall fix on the coast (lon -80), 100 kt offshore before and
    # 30 kt inland after: with every fix the crossing carries 90 kt; with the 6-hourly fixes only
    # (as in the 1966-1984 flag gap) the interpolated wind is 65 kt, and with 20 kt inland it is
    # below hurricane strength, so the landfall is lost.
    t0 = pd.Timestamp("1990-09-01 00:00")
    times = [t0, t0 + pd.Timedelta(hours=3), t0 + pd.Timedelta(hours=6)]

    def make(inland_wind):
        return pd.DataFrame(
            {
                "storm_id": "S",
                "name": "S",
                "year": 1990,
                "time": times,
                "record": ["", "L", ""],
                "status": "HU",
                "lat": [26.5, 26.5, 26.5],
                "lon": [-79.0, -80.0, -81.0],
                "wind": [100.0, 90.0, inland_wind],
                "pressure": np.nan,
            }
        )

    full = make(30.0)
    syn = synoptic_fixes(full)
    assert list(syn["time"]) == [times[0], times[2]]
    assert landfalls_geometric(full, square_region)["wind"].iloc[0] == pytest.approx(90.0)
    assert landfalls_geometric(syn, square_region)["wind"].iloc[0] == pytest.approx(65.0)
    assert landfalls_geometric(synoptic_fixes(make(20.0)), square_region).empty


def test_near_misses_are_hurricanes_close_to_the_coast_that_are_not_counted(square_region):
    counted = track(
        "A", 1950, [(-79.0, 26.5, 90, ""), (-80.0, 26.5, 85, "L"), (-81.0, 26.5, 60, "")]
    )
    # E: passes 0.3 degrees of longitude (about 30 km) off the coast at 70 kt, no landfall
    e = track("E", 1951, [(-79.7, 25.0, 60, ""), (-79.7, 26.5, 70, ""), (-79.7, 28.0, 65, "")])
    weak = track("F", 1952, [(-79.7, 26.5, 50, "")])  # tropical storm strength: not a hurricane
    far = track("G", 1953, [(-77.0, 26.5, 100, "")])  # about 300 km offshore
    # H: 60 kt at 30 km (not a hurricane), then 90 kt at about 80 km
    near_weak_fix = track("H", 1954, [(-79.7, 26.5, 60, ""), (-79.2, 26.5, 90, "")])
    tr = pd.concat([counted, e, weak, far, near_weak_fix], ignore_index=True)
    lf = landfalls_from_records(tr, square_region)
    assert lf["storm_id"].tolist() == ["A"]
    nm = near_misses(tr, lf, square_region, max_km=50.0)
    assert nm["storm_id"].tolist() == ["E"]
    r = nm.iloc[0]
    assert r["wind"] == 70 and r["lat"] == 26.5
    km_per_degree = 111.195 * np.cos(np.deg2rad(26.5))
    assert r["distance_florida_km"] == pytest.approx(0.3 * km_per_degree, rel=0.01)
    # a wider radius also finds the 90 kt fix of H (its 60 kt fix is not a hurricane)
    wide = near_misses(tr, lf, square_region, max_km=100.0)
    assert sorted(wide["storm_id"]) == ["E", "H"]
    assert wide.set_index("storm_id").loc["H", "wind"] == 90
    # a counted storm is never a near miss, and an empty input gives an empty table
    assert "A" not in wide["storm_id"].tolist()
    assert near_misses(tr.iloc[:0], lf, square_region).empty
