"""Stochastic event set: simulated years of Florida hurricane landfalls and portfolio losses.

THIS IS A TEACHING-GRADE MODEL, NOT A VENDOR CATASTROPHE MODEL (RMS, Verisk/AIR, CoreLogic). Its
losses are not comparable to the modelled expected loss quoted in a cat bond offering.

For each simulated year y = 0..N-1 the number of events is drawn from the frequency model; each
event gets landfall parameters (``hazard.model.sample_event_parameters``), a radius of maximum
wind and Holland B (``hazard.windfield``), peak winds at the county centroids, a mean damage ratio
(``vulnerability.emanuel_damage``), an insured loss ratio after policy terms, and an event loss
equal to the sum over counties of loss ratio times county TIV. Results are deterministic given
the seed: separate random streams (numpy SeedSequence.spawn) drive the counts and the event
attributes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from catbond.hazard.frequency import sample_counts
from catbond.hazard.model import HazardModel, sample_event_parameters
from catbond.hazard.windfield import (
    holland_b_willoughby_rahn,
    peak_site_winds,
    rmw_willoughby_rahn,
)
from catbond.vulnerability import (
    EMANUEL_V_HALF_KT,
    EMANUEL_V_THRESH_KT,
    emanuel_damage,
    insured_loss_ratio,
)

EVENT_COLUMNS = [
    "year",
    "event_id",
    "lon",
    "lat",
    "wind_kt",
    "heading_deg",
    "speed_kmh",
    "rmw_km",
    "holland_b",
    "loss_usd",
]


@dataclass(frozen=True)
class LossSettings:
    """Vulnerability, policy and wind-field settings (defaults are the base case)."""

    v_thresh_kt: float = EMANUEL_V_THRESH_KT
    v_half_kt: float = EMANUEL_V_HALF_KT
    deductible: float = 0.02
    limit: float = 1.0
    rmw_scale: float = 1.0
    decay: bool = True


def event_losses(
    params: pd.DataFrame, exposure: pd.DataFrame, settings: LossSettings, chunk: int = 2000
) -> pd.DataFrame:
    """Add rmw_km, holland_b and loss_usd to a table of event parameters.

    ``params`` needs lon, lat, heading_deg, speed_kmh, wind_kt; ``exposure`` needs lon, lat,
    tiv_usd (one row per site).
    """
    out = params.copy()
    rmw = settings.rmw_scale * rmw_willoughby_rahn(out["wind_kt"].to_numpy(), out["lat"].to_numpy())
    b = holland_b_willoughby_rahn(out["wind_kt"].to_numpy(), out["lat"].to_numpy())
    site_lon = exposure["lon"].to_numpy(dtype=float)
    site_lat = exposure["lat"].to_numpy(dtype=float)
    tiv = exposure["tiv_usd"].to_numpy(dtype=float)
    loss = np.empty(len(out))
    cols = [
        out[c].to_numpy(dtype=float) for c in ("lon", "lat", "heading_deg", "speed_kmh", "wind_kt")
    ]
    for i0 in range(0, len(out), chunk):
        sl = slice(i0, i0 + chunk)
        w = peak_site_winds(
            cols[0][sl],
            cols[1][sl],
            cols[2][sl],
            cols[3][sl],
            cols[4][sl],
            rmw[sl],
            b[sl],
            site_lon,
            site_lat,
            decay=settings.decay,
        )
        dmg = emanuel_damage(w, settings.v_thresh_kt, settings.v_half_kt)
        loss[sl] = insured_loss_ratio(dmg, settings.deductible, settings.limit) @ tiv
    out["rmw_km"] = rmw
    out["holland_b"] = b
    out["loss_usd"] = loss
    return out


@dataclass(frozen=True)
class EventSet:
    """Simulated events (``EVENT_COLUMNS``) over ``n_years`` years, with the generating seed."""

    events: pd.DataFrame
    n_years: int
    seed: int


def simulate_event_set(
    model: HazardModel,
    coast_edges: np.ndarray,
    exposure: pd.DataFrame,
    n_years: int,
    seed: int,
    settings: LossSettings | None = None,
) -> EventSet:
    """Simulate ``n_years`` years of events and their portfolio losses (see module docstring)."""
    settings = settings or LossSettings()
    ss_counts, ss_events = np.random.SeedSequence(seed).spawn(2)
    counts = sample_counts(
        np.random.default_rng(ss_counts), n_years, model.freq_mean, model.nb_size
    )
    year = np.repeat(np.arange(n_years), counts)
    params = sample_event_parameters(
        model, coast_edges, np.random.default_rng(ss_events), int(counts.sum())
    )
    ev = event_losses(params, exposure, settings)
    ev.insert(0, "year", year)
    ev.insert(1, "event_id", np.arange(len(ev)))
    return EventSet(ev[EVENT_COLUMNS], n_years, seed)
