"""Parameter uncertainty: bootstrap the historical years, refit the hazard model, re-simulate.

One bootstrap replicate draws n = last_year - first_year + 1 calendar years with replacement from
the calibration window and keeps every landfall of each drawn year (years, not events, are the
resampling unit, so years without landfalls are drawn too and the count of landfalls varies
between replicates). The drawn years are relabelled first_year, first_year + 1, ... and the hazard
model is refitted with the same frequency family and intensity model as the base case.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from catbond.hazard.model import HazardModel, fit_hazard


def resample_years(
    landfalls: pd.DataFrame, first_year: int, last_year: int, rng: np.random.Generator
) -> pd.DataFrame:
    """One bootstrap replicate of the landfall table (calendar years drawn with replacement)."""
    years = np.arange(first_year, last_year + 1)
    pick = years[rng.integers(0, years.size, size=years.size)]
    by_year = {y: g for y, g in landfalls.groupby("year")}
    parts = []
    for i, y in enumerate(pick):
        g = by_year.get(y)
        if g is not None:
            parts.append(g.assign(year=first_year + i))
    if not parts:
        return landfalls.iloc[0:0]
    return pd.concat(parts, ignore_index=True)


def bootstrap_hazard_models(
    landfalls: pd.DataFrame,
    base: HazardModel,
    n_boot: int,
    seed: int,
    intensity_kind: str = "gpd",
) -> list[HazardModel]:
    """``n_boot`` hazard models refitted on bootstrap replicates of the base calibration window."""
    rng = np.random.default_rng(seed)
    family = "poisson" if base.nb_size is None else "negbin"
    models = []
    for _ in range(n_boot):
        rep = resample_years(landfalls, base.first_year, base.last_year, rng)
        models.append(fit_hazard(rep, base.first_year, base.last_year, intensity_kind, family))
    return models
