import numpy as np
import pandas as pd

from catbond.hazard.data import STANDARD_COLUMNS, load_hurdat2_text, load_storms_csv

HURDAT2_SNIPPET = """\
AL011992,             ANDREW,      3,
19920823, 1800,  , HU, 25.4N,  77.7W, 135,  941,  110,   90,   80,  110,
19920824, 0840, L, HU, 25.5N,  80.2W, 145,  937,  110,   90,   80,  110,
19920824, 1200,  , HU, 25.6N,  81.4W, 120, -999, -999, -999, -999, -999,
AL022000,            UNNAMED,      1,
20000701, 0000,  , TD, 10.0S, 170.0E,-999,  1010, -999, -999, -999, -999,
"""


def test_hurdat2_text_parser(tmp_path):
    p = tmp_path / "h.txt"
    p.write_text(HURDAT2_SNIPPET)
    df = load_hurdat2_text(p)
    assert list(df.columns) == STANDARD_COLUMNS
    assert len(df) == 4
    a = df[df["storm_id"] == "AL011992"].reset_index(drop=True)
    assert a.loc[1, "record"] == "L"
    assert a.loc[1, "time"] == pd.Timestamp("1992-08-24 08:40")
    assert a.loc[1, "lat"] == 25.5 and a.loc[1, "lon"] == -80.2
    assert a.loc[1, "wind"] == 145 and np.isnan(a.loc[2, "pressure"])
    b = df[df["storm_id"] == "AL022000"].iloc[0]
    assert b["lat"] == -10.0 and b["lon"] == 170.0 and np.isnan(b["wind"]) and b["year"] == 2000


def test_storms_csv_loader_keys_and_gaps(tmp_path):
    rows = [
        ("Amy", 1975, 6, 27, 0, 27.5, -79.0, "tropical depression", 25, 1013),
        ("Amy", 1975, 6, 27, 6, 28.5, -79.0, "tropical storm", 35, 1010),
        ("Amy", 1975, 9, 1, 0, 20.0, -60.0, "hurricane", 70, 990),  # > 72 h later: new storm
        ("Bob", 1975, 7, 1, 17, 30.0, -85.0, "hurricane", 80, 980),
        ("Bob", 1975, 7, 1, 17, 30.1, -85.1, "hurricane", 75, 982),  # same hour: off-synoptic
    ]
    df = pd.DataFrame(
        rows,
        columns=[
            "name",
            "year",
            "month",
            "day",
            "hour",
            "lat",
            "long",
            "status",
            "wind",
            "pressure",
        ],
    )
    df["category"] = np.nan
    df["tropicalstorm_force_diameter"] = np.nan
    df["hurricane_force_diameter"] = np.nan
    p = tmp_path / "s.csv"
    df.to_csv(p, index=False)
    out = load_storms_csv(p)
    assert sorted(out["storm_id"].unique()) == ["1975-Amy", "1975-Amy-2", "1975-Bob"]
    assert (out["record"] == "").all()
    bob = out[out["storm_id"] == "1975-Bob"]
    assert (bob["status"] == "HU").all()
    assert bob["time"].tolist() == [
        pd.Timestamp("1975-07-01 17:00"),
        pd.Timestamp("1975-07-01 17:30"),
    ]


def test_real_al_file(al_tracks):
    assert len(al_tracks) == 51840 - 14  # first storm has no key in this file
    assert al_tracks["year"].min() == 1851 and al_tracks["year"].max() == 2019
    andrew = al_tracks[(al_tracks["name"] == "ANDREW") & (al_tracks["year"] == 1992)]
    lf = andrew[andrew["record"] == "L"]
    fl = lf[(lf["lat"] == 25.5) & (lf["lon"] == -80.2)]  # Florida landfall (Bahamas, LA too)
    assert len(fl) == 1 and fl["wind"].item() == 145
    assert al_tracks["lon"].dropna().between(-180, 180).all()


def test_real_storms_file(storms_tracks):
    assert storms_tracks["year"].min() == 1975 and storms_tracks["year"].max() == 2024
    ian = storms_tracks[storms_tracks["storm_id"] == "2022-Ian"]
    assert ian["wind"].max() == 140
