"""Hypothetical Florida homeowners portfolio: total insured value spread over county centroids.

THE PORTFOLIO IS HYPOTHETICAL. The total insured value (TIV) is a stated round number and is
allocated to Florida's 67 counties in proportion to 2020 Census households (occupied housing
units) as tabulated by the University of Florida BEBR (``data/fl_county_households_2020.csv``).
Each county's value sits at the area-weighted centroid of its polygon. A real sponsor's book is
concentrated differently (by company, coastal distance, construction, coverage type and value);
households also exclude seasonal and vacant homes, which are a large share of the housing stock
in some coastal counties.
"""

from __future__ import annotations

from importlib import resources

import numpy as np
import pandas as pd

from catbond.geo import Region, ring_area_centroid

DEFAULT_TIV_USD = 100e9


def household_counts() -> pd.DataFrame:
    """The packaged county household table (columns ``county``, ``households_2020``)."""
    with resources.files("catbond").joinpath("data/fl_county_households_2020.csv").open() as fh:
        return pd.read_csv(fh, comment="#")


def county_exposure(region: Region, tiv_usd: float = DEFAULT_TIV_USD) -> pd.DataFrame:
    """County table: fips, county, lon, lat (centroid), households_2020, share, tiv_usd.

    Raises if any county in the region has no household count or vice versa.
    """
    rows = []
    for fips, name, rings in region.counties:
        lon, lat = ring_area_centroid(rings)
        rows.append((fips, name, lon, lat))
    geo = pd.DataFrame(rows, columns=["fips", "county", "lon", "lat"])
    hh = household_counts()
    df = geo.merge(hh, on="county", how="outer", indicator=True)
    if (df["_merge"] != "both").any():
        bad = df.loc[df["_merge"] != "both", "county"].tolist()
        raise ValueError(f"counties without a match between geometry and households: {bad}")
    df = df.drop(columns="_merge").sort_values("fips").reset_index(drop=True)
    df["share"] = df["households_2020"] / df["households_2020"].sum()
    df["tiv_usd"] = tiv_usd * df["share"]
    assert np.isclose(df["tiv_usd"].sum(), tiv_usd)
    return df
