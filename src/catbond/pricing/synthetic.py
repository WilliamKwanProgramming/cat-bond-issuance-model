"""Synthetic deal tables with a known generating relation, to test the spread regression.

SYNTHETIC DATA. Nothing here describes real cat bonds. The relation is

    log(spread_pct) = b0 + b1 log(EL_pct) + b_year (issue_year - 2000)
                      + b_param 1[trigger = "parametric"] + u_sponsor + e,

with EL log-uniform on [0.5, 8] percent, issue years uniform on 2010-2024, 30 percent of
tranches parametric, sponsor effects u ~ N(0, sponsor_sd^2) shared by the tranches of one sponsor
and e ~ N(0, s_i^2) with s_i = sigma exp(hetero (log EL_i - log 2)) (hetero = 0: constant
variance). Columns follow the deals interface (``fit.DEAL_COLUMNS``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from catbond.pricing.fit import DEAL_COLUMNS, YEAR_ORIGIN

TRUE_COEF = {"const": 1.0, "log_el": 0.65, "issue_year": -0.01, "trigger[parametric]": 0.2}


def synthetic_deals(
    n: int,
    rng: np.random.Generator,
    sigma: float = 0.25,
    n_sponsors: int = 25,
    sponsor_sd: float = 0.0,
    hetero: float = 0.0,
    coef: dict[str, float] | None = None,
) -> pd.DataFrame:
    """``n`` synthetic tranches drawn from the relation in the module docstring."""
    c = coef or TRUE_COEF
    el = np.exp(rng.uniform(np.log(0.5), np.log(8.0), size=n))
    year = rng.integers(2010, 2025, size=n)
    param = rng.uniform(size=n) < 0.3
    sponsor = rng.integers(0, n_sponsors, size=n)
    u = sponsor_sd * rng.standard_normal(n_sponsors)
    log_s = (
        c["const"]
        + c["log_el"] * np.log(el)
        + c["issue_year"] * (year - YEAR_ORIGIN)
        + c["trigger[parametric]"] * param
        + u[sponsor]
        + sigma * np.exp(hetero * (np.log(el) - np.log(2.0))) * rng.standard_normal(n)
    )
    df = pd.DataFrame(
        {
            "deal_name": [f"Synthetic Re {i}" for i in range(n)],
            "tranche": "A",
            "sponsor": [f"Sponsor {j}" for j in sponsor],
            "issue_year": year,
            "issue_month": rng.integers(1, 13, size=n),
            "size_usd_m": np.round(np.exp(rng.uniform(np.log(50), np.log(500), size=n)), 1),
            "peril": "US named storm",
            "region": "Florida",
            "trigger": np.where(param, "parametric", "indemnity"),
            "expected_loss_pct": el,
            "attachment_prob_pct": np.nan,
            "exhaustion_prob_pct": np.nan,
            "spread_pct": np.exp(log_s),
            "spread_basis": "final",
            "rating": "",
            "source_url": "",
            "access_date": "",
            "notes": "synthetic",
        }
    )
    return df[DEAL_COLUMNS]
