"""Shared fixtures. Tests that need the downloaded public data skip when it is absent."""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd
import pytest

from catbond.geo import Region, load_region
from catbond.hazard.intensity import IntensityModel
from catbond.hazard.model import HazardModel

RAW = pathlib.Path(__file__).resolve().parents[1] / "data" / "raw"
NEED = "public data not downloaded; run `python3 scripts/fetch_data.py` first"


def _need(name: str) -> pathlib.Path:
    p = RAW / name
    if not p.exists():
        pytest.skip(f"{name}: {NEED}")
    return p


@pytest.fixture(scope="session")
def florida() -> Region:
    return load_region(_need("us_counties.geojson"))


@pytest.fixture(scope="session")
def al_tracks() -> pd.DataFrame:
    from catbond.hazard.data import load_hurdat_rda

    return load_hurdat_rda(_need("AL.rda"))


@pytest.fixture(scope="session")
def storms_tracks() -> pd.DataFrame:
    from catbond.hazard.data import load_storms_csv

    return load_storms_csv(_need("storms.csv"))


def write_geojson(path: pathlib.Path, features: list[tuple[str, str, list[list[float]]]]) -> None:
    """Write a GeoJSON of single-ring polygon features (id, name, ring)."""
    path.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "id": fid,
                        "properties": {"NAME": name},
                        "geometry": {"type": "Polygon", "coordinates": [ring]},
                    }
                    for fid, name, ring in features
                ],
            }
        )
    )


@pytest.fixture()
def square_region(tmp_path: pathlib.Path) -> Region:
    """Toy 'state 12': two counties forming the box lon [-82, -80], lat [26, 27].

    A 'state 13' county shares the northern edge of the western county, so that edge is a land
    border and not coast. Every other outer edge is coast.
    """
    west = [[-82, 26], [-81, 26], [-81, 27], [-82, 27], [-82, 26]]
    east = [[-81, 26], [-80, 26], [-80, 27], [-81, 27], [-81, 26]]
    north = [[-82, 27], [-81, 27], [-81, 28], [-82, 28], [-82, 27]]
    p = tmp_path / "toy.geojson"
    write_geojson(p, [("12001", "West", west), ("12003", "East", east), ("13001", "North", north)])
    return load_region(p)


def toy_hazard(
    lon: float = -80.0,
    lat: float = 26.5,
    heading: float = 270.0,
    speed: float = 20.0,
    freq: float = 0.6,
    shape: float = -0.3,
    scale: float = 35.0,
) -> HazardModel:
    """A hazard model with one-point track pool and a GPD intensity (no data needed)."""
    return HazardModel(
        first_year=1900,
        last_year=2019,
        freq_mean=freq,
        nb_size=None,
        intensity=IntensityModel("gpd", shape=shape, scale=scale),
        pool_lon=np.array([lon]),
        pool_lat=np.array([lat]),
        pool_heading=np.array([heading]),
        pool_speed=np.array([speed]),
        pos_sigma_km=0.0,
        heading_sigma_deg=0.0,
        speed_log_sigma=0.0,
    )
