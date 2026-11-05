"""Fitted stochastic hazard model and event sampling.

An event is a hurricane landfall in Florida. Its attributes are drawn as follows:

* annual count: Poisson or negative binomial (``frequency.compare_poisson_negbin`` chooses);
* landfall position, heading and translation speed: one historical landfall is drawn at random
  (so the three keep their joint, coast-dependent distribution) and jittered: the position by a
  2-D Gaussian with ``pos_sigma_km`` per axis and then moved to the nearest point of the Florida
  coastline; the heading by a Gaussian with ``heading_sigma_deg``; the speed by a log-normal
  factor exp(N(0, ``speed_log_sigma``^2));
* intensity: independent draw from the fitted intensity model (``intensity``).

Only landfalls with ``first_year <= year <= last_year`` are read by ``fit_hazard``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from catbond.geo import local_xy, snap_to_edges
from catbond.hazard.frequency import compare_poisson_negbin
from catbond.hazard.intensity import IntensityModel, fit_intensity
from catbond.hazard.landfall import annual_counts

POS_SIGMA_KM = 25.0
HEADING_SIGMA_DEG = 10.0
SPEED_LOG_SIGMA = 0.2


@dataclass(frozen=True)
class HazardModel:
    """Parameters of the event generator (see module docstring)."""

    first_year: int
    last_year: int
    freq_mean: float
    nb_size: float | None
    intensity: IntensityModel
    pool_lon: np.ndarray
    pool_lat: np.ndarray
    pool_heading: np.ndarray
    pool_speed: np.ndarray
    pos_sigma_km: float = POS_SIGMA_KM
    heading_sigma_deg: float = HEADING_SIGMA_DEG
    speed_log_sigma: float = SPEED_LOG_SIGMA

    def scaled(self, freq_multiplier: float) -> HazardModel:
        """Same model with the annual mean count multiplied by ``freq_multiplier``."""
        return replace(self, freq_mean=self.freq_mean * freq_multiplier)


def fit_hazard(
    landfalls: pd.DataFrame,
    first_year: int,
    last_year: int,
    intensity_kind: str = "gpd",
    frequency: str = "auto",
) -> HazardModel:
    """Fit frequency, intensity and the track-parameter pool on landfalls in the window.

    ``frequency``: "auto" (NB only if the boundary likelihood-ratio test rejects Poisson at 5%),
    "poisson" or "negbin".
    """
    win = landfalls[(landfalls["year"] >= first_year) & (landfalls["year"] <= last_year)]
    counts = annual_counts(win, first_year, last_year).to_numpy()
    cmp = compare_poisson_negbin(counts)
    choice = cmp.chosen if frequency == "auto" else frequency
    nb_size = cmp.nb_size if choice == "negbin" else None
    pool = win.dropna(subset=["heading_deg", "speed_kmh"])
    return HazardModel(
        first_year=first_year,
        last_year=last_year,
        freq_mean=float(counts.mean()),
        nb_size=nb_size,
        intensity=fit_intensity(win["wind"].to_numpy(), intensity_kind),
        pool_lon=pool["lon"].to_numpy(dtype=float),
        pool_lat=pool["lat"].to_numpy(dtype=float),
        pool_heading=pool["heading_deg"].to_numpy(dtype=float),
        pool_speed=pool["speed_kmh"].to_numpy(dtype=float),
    )


def sample_event_parameters(
    model: HazardModel, coast_edges: np.ndarray, rng: np.random.Generator, n: int
) -> pd.DataFrame:
    """Draw ``n`` events' landfall lon/lat, heading (deg), speed (km/h) and wind (kt)."""
    idx = rng.integers(0, model.pool_lon.size, size=n)
    lon0, lat0 = model.pool_lon[idx], model.pool_lat[idx]
    dx, dy = model.pos_sigma_km * rng.standard_normal((2, n))
    # invert local_xy about the historical point to move it by (dx, dy) km
    kx, ky = local_xy(lon0 + 1.0, lat0 + 1.0, lon0, lat0)
    lon_j, lat_j = snap_to_edges(lon0 + dx / kx, lat0 + dy / ky, coast_edges)
    heading = (model.pool_heading[idx] + model.heading_sigma_deg * rng.standard_normal(n)) % 360
    speed = model.pool_speed[idx] * np.exp(model.speed_log_sigma * rng.standard_normal(n))
    wind = model.intensity.sample(rng, n)
    return pd.DataFrame(
        {"lon": lon_j, "lat": lat_j, "heading_deg": heading, "speed_kmh": speed, "wind_kt": wind}
    )
