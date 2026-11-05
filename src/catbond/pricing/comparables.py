"""Comparable-deal screen: the k tranches nearest to a target layer.

Distance of tranche i to the target (EL_0, size_0, peril_0, region_0):

    d_i = sqrt( ((ln EL_i - ln EL_0) / s_EL)^2 + ((ln size_i - ln size_0) / s_size)^2 )
          + p_peril 1[peril_i != peril_0] + p_region 1[region_i != region_0],

with defaults s_EL = 0.5 (a factor e^0.5 = 1.65 in EL counts as one unit), s_size = 1.0,
p_peril = 2.0, p_region = 1.0 (a different peril weighs more than a different region). Peril and
region are compared case-insensitively after trimming. A missing size contributes one unit to the
first term; a missing peril or region counts as different. Rows without a positive EL or spread
are excluded; the counts are in ``DataFrame.attrs``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from catbond.pricing.multiple import spread_multiple


def _norm(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower().where(s.notna(), None)


def comparable_deals(
    deals: pd.DataFrame,
    el_pct: float,
    size_usd_m: float,
    peril: str,
    region: str,
    k: int = 5,
    s_el: float = 0.5,
    s_size: float = 1.0,
    p_peril: float = 2.0,
    p_region: float = 1.0,
) -> pd.DataFrame:
    """The ``k`` nearest usable tranches, sorted by distance, with their spread multiple."""
    el = pd.to_numeric(deals["expected_loss_pct"], errors="coerce")
    sp = pd.to_numeric(deals["spread_pct"], errors="coerce")
    usable = (el > 0) & (sp > 0)
    df = deals[usable].copy()
    el, sp = el[usable], sp[usable]
    size = pd.to_numeric(df["size_usd_m"], errors="coerce")
    z_el = (np.log(el) - np.log(el_pct)) / s_el
    z_size = ((np.log(size) - np.log(size_usd_m)) / s_size).where(size > 0, 1.0).fillna(1.0)
    d = np.sqrt(z_el**2 + z_size**2)
    d = d + p_peril * (_norm(df["peril"]) != peril.strip().lower()).fillna(True)
    d = d + p_region * (_norm(df["region"]) != region.strip().lower()).fillna(True)
    df["distance"] = d.astype(float)
    df["multiple"] = spread_multiple(sp, el)
    out = df.sort_values("distance", kind="stable").head(k).reset_index(drop=True)
    out.attrs["n_input"] = len(deals)
    out.attrs["n_usable"] = int(usable.sum())
    return out
