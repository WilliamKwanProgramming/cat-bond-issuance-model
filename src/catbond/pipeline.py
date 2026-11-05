"""Pre-specified analysis choices and the data loading shared by the scripts in ``scripts/``.

Every choice here was fixed before any loss, layer or out-of-sample result was looked at; see
docs/model.md ("Pre-specified choices").
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass

import numpy as np
import pandas as pd

from catbond.exposure import DEFAULT_TIV_USD, county_exposure
from catbond.geo import Region, load_region
from catbond.hazard.data import load_hurdat_rda, load_storms_csv
from catbond.hazard.landfall import hybrid_landfalls, landfalls_geometric
from catbond.manifest import repo_root

CALIBRATION = (1900, 2019)
ALT_WINDOW = (1950, 2019)
EXTENDED = (1900, 2024)  # 2020-2024 added from storms.csv with the geometric definition
OOS_FIT = (1900, 1979)
OOS_TEST = (1980, 2019)

BASE_SEED = 1
BASE_N_YEARS = 100_000
TIV_USD = DEFAULT_TIV_USD
RP_BOOTSTRAP = 200
TERM_YEARS = 3

# layer grid, as fractions of TIV
GRID_ATTACHMENTS = np.round(np.arange(0.0025, 0.04001, 0.0025), 6)
GRID_LIMITS = np.array([0.005, 0.01, 0.02, 0.03])
# reference layer: attachment at the base-case 1-in-50 OEP loss, exhaustion at the 1-in-100 OEP
# loss, each rounded to the nearest 0.05 percent of TIV
REF_ATTACH_RP, REF_EXHAUST_RP, REF_ROUND = 50, 100, 0.0005


@dataclass(frozen=True)
class Inputs:
    """Loaded public data and derived tables."""

    region: Region
    tracks: pd.DataFrame  # AL.rda, 1851-2019
    storms: pd.DataFrame  # storms.csv, 1975-2024
    landfalls: pd.DataFrame  # hybrid definition, all years of AL.rda
    landfalls_extended: pd.DataFrame  # hybrid to 2019 plus geometric from storms.csv 2020-2024
    exposure: pd.DataFrame


def raw_dir() -> pathlib.Path:
    """``data/raw`` under the repository root."""
    return repo_root() / "data" / "raw"


def load_inputs(raw: pathlib.Path | None = None) -> Inputs:
    """Load the downloaded files and build the landfall and exposure tables."""
    raw = raw or raw_dir()
    region = load_region(raw / "us_counties.geojson")
    tracks = load_hurdat_rda(raw / "AL.rda")
    storms = load_storms_csv(raw / "storms.csv")
    lf = hybrid_landfalls(tracks[tracks["year"] >= CALIBRATION[0]], region)
    recent = landfalls_geometric(storms[storms["year"] > CALIBRATION[1]], region)
    ext = pd.concat([lf, recent], ignore_index=True)
    return Inputs(region, tracks, storms, lf, ext, county_exposure(region, TIV_USD))


def reference_layer(occurrence_rp_losses: dict[int, float]) -> tuple[float, float]:
    """(A, E) in USD from the base-case OEP return-period losses (see REF_* above)."""

    def rnd(x: float) -> float:
        return float(np.round(x / TIV_USD / REF_ROUND) * REF_ROUND * TIV_USD)

    return rnd(occurrence_rp_losses[REF_ATTACH_RP]), rnd(occurrence_rp_losses[REF_EXHAUST_RP])
