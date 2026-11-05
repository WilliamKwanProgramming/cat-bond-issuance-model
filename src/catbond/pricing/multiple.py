"""Spread multiple: spread divided by expected loss (both in percent per annum of principal)."""

from __future__ import annotations

import numpy as np


def spread_multiple(spread_pct: np.ndarray, expected_loss_pct: np.ndarray) -> np.ndarray:
    """spread / EL; NaN where EL is missing or not positive."""
    s = np.asarray(spread_pct, dtype=float)
    el = np.asarray(expected_loss_pct, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(el > 0, s / el, np.nan)
