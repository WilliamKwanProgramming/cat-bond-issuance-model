"""Out-of-sample check of the hazard frequency-intensity model: landfall counts by category.

Fit on ``fit_years`` only, predict the number of Florida hurricane landfalls by Saffir-Simpson
category in ``test_years``, and compare with the observed counts.

Under the fitted Poisson model the count of category k in n_t test years is
Poisson(n_t lambda p_k), with p_k the fitted intensity model's probability of category k. Two
predictive intervals are reported: the plug-in Poisson interval, and a parameter-bootstrap
interval (fit years resampled with replacement, lambda and the intensity model refitted, then a
Poisson count drawn), which also carries parameter uncertainty. For the observed count the exact
(Garwood) confidence interval of its Poisson mean is reported. ``p_two_sided`` is
2 min(P(X <= obs), P(X >= obs)) under the bootstrap predictive distribution, capped at 1.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from catbond.hazard.frequency import poisson_mean_ci
from catbond.hazard.intensity import IntensityModel, fit_intensity
from catbond.hazard.landfall import annual_counts

GROUPS: dict[str, tuple[float, float]] = {
    "Cat 1": (64.0, 83.0),
    "Cat 2": (83.0, 96.0),
    "Cat 3": (96.0, 113.0),
    "Cat 4": (113.0, 137.0),
    "Cat 5": (137.0, np.inf),
    "Cat 3-5 (major)": (96.0, np.inf),
    "All hurricanes": (64.0, np.inf),
}


def group_probability(model: IntensityModel, lo: float, hi: float) -> float:
    """P(lo <= V < hi) under the truncated intensity model."""
    sf_lo = float(model.sf_truncated(np.array(lo)))
    sf_hi = 0.0 if np.isinf(hi) else float(model.sf_truncated(np.array(hi)))
    return sf_lo - sf_hi


def out_of_sample_check(
    landfalls: pd.DataFrame,
    fit_years: tuple[int, int],
    test_years: tuple[int, int],
    intensity_kind: str = "gpd",
    level: float = 0.90,
    n_boot: int = 2000,
    seed: int = 0,
) -> pd.DataFrame:
    """Predicted against observed counts by category (see module docstring).

    The fitted quantities read only rows with year in ``fit_years``.
    """
    f0, f1 = fit_years
    t0, t1 = test_years
    n_test = t1 - t0 + 1
    fit = landfalls[(landfalls["year"] >= f0) & (landfalls["year"] <= f1)]
    test = landfalls[(landfalls["year"] >= t0) & (landfalls["year"] <= t1)]
    counts = annual_counts(fit, f0, f1).to_numpy()
    lam = counts.mean()
    model = fit_intensity(fit["wind"].to_numpy(), intensity_kind)

    rng = np.random.default_rng(seed)
    years = np.arange(f0, f1 + 1)
    winds_by_year = [fit.loc[fit["year"] == y, "wind"].to_numpy(dtype=float) for y in years]
    boot_lam = np.empty(n_boot)
    boot_models = []
    for b in range(n_boot):
        pick = rng.integers(0, years.size, size=years.size)
        boot_lam[b] = counts[pick].mean()
        winds = np.concatenate([winds_by_year[i] for i in pick])
        boot_models.append(fit_intensity(winds, intensity_kind) if winds.size >= 5 else model)

    a = (1 - level) / 2
    rows = []
    for name, (lo, hi) in GROUPS.items():
        p = group_probability(model, lo, hi)
        mean = n_test * lam * p
        obs = int(((test["wind"] >= lo) & (test["wind"] < hi)).sum())
        p_boot = np.array([group_probability(m, lo, hi) for m in boot_models])
        draws = rng.poisson(n_test * boot_lam * p_boot)
        ci_lo, ci_hi = poisson_mean_ci(obs, 1.0, level=0.95)
        rows.append(
            {
                "group": name,
                "predicted_mean": mean,
                "plugin_lo": float(stats.poisson.ppf(a, mean)),
                "plugin_hi": float(stats.poisson.ppf(1 - a, mean)),
                "boot_lo": float(np.quantile(draws, a)),
                "boot_hi": float(np.quantile(draws, 1 - a)),
                "observed": obs,
                "observed_ci95_lo": ci_lo,
                "observed_ci95_hi": ci_hi,
                "p_two_sided": float(
                    min(1.0, 2 * min((draws <= obs).mean(), (draws >= obs).mean()))
                ),
            }
        )
    out = pd.DataFrame(rows)
    out.attrs["n_test_years"] = n_test
    out.attrs["lambda_fit"] = float(lam)
    return out
