"""Comparable-deal screen: the k tranches nearest to a target layer by expected loss and year.

Distance of tranche i to the target (EL_0, year_0):

    d_i = sqrt( ((ln EL_i - ln EL_0) / s_EL)^2 + ((year_i - year_0) / s_year)^2 ),

with defaults s_EL = 0.5 (a factor e^0.5 = 1.65 in EL counts as one unit) and s_year = 2 (two
issue years count as one unit). Ties are broken by the order of the input rows. Rows without a
positive EL, a positive spread or an issue year are excluded; the counts are in
``DataFrame.attrs``. The screen shows what similar public deals paid; it does not adjust for
sponsor, peril, region, trigger or the modelling firm behind each EL.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from catbond.pricing.multiple import spread_multiple


def comparable_deals(
    deals: pd.DataFrame,
    el_pct: float,
    year: int,
    k: int = 5,
    s_el: float = 0.5,
    s_year: float = 2.0,
) -> pd.DataFrame:
    """The ``k`` nearest usable tranches, sorted by distance, with their spread multiple."""
    el = pd.to_numeric(deals["expected_loss_pct"], errors="coerce")
    sp = pd.to_numeric(deals["spread_pct"], errors="coerce")
    yr = pd.to_numeric(deals["issue_year"], errors="coerce")
    usable = (el > 0) & (sp > 0) & yr.notna()
    df = deals[usable].copy()
    z_el = (np.log(el[usable]) - np.log(el_pct)) / s_el
    z_year = (yr[usable] - year) / s_year
    df["distance"] = np.sqrt(z_el**2 + z_year**2).astype(float)
    df["multiple"] = spread_multiple(sp[usable], el[usable])
    out = df.sort_values("distance", kind="stable").head(k).reset_index(drop=True)
    out.attrs["n_input"] = len(deals)
    out.attrs["n_usable"] = int(usable.sum())
    return out
