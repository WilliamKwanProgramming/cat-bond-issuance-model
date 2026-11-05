"""Hypothetical transaction: choice of a three-layer stack and the regression spread mapping.

Layer choice (docs/pricing_protocol.md): from a grid of layers (attachment a, limit l, EL in
percent of limit), choose three adjacent layers, bottom to top [a1, a1 + l1], [a2, a2 + l2],
[a3, a3 + l3] with a2 = a1 + l1 and a3 = a2 + l2, every (a, l) a grid row, that minimize

    sum_k (log(EL_k / target_k))^2,  targets (4, 2, 1) percent bottom to top;

ties go to the lowest a1 (then the smallest l1, l2, l3).

Spread mapping: median spread exp(x0'b) and the classical prediction interval of a fitted
``SpreadFit`` at a given EL (and issue year for the year fixed effects).
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from catbond.pricing.fit import SpreadFit

TARGETS_PCT = (4.0, 2.0, 1.0)
ROUND = 6  # decimals used to match grid attachments in percent of TIV


def choose_stack(grid: pd.DataFrame, targets: tuple[float, ...] = TARGETS_PCT) -> pd.DataFrame:
    """Rows of ``grid`` (columns attachment_pct_tiv, limit_pct_tiv, el_pct) forming the stack.

    Returned bottom to top, with columns ``layer`` (1 = bottom), ``target_el_pct`` and the
    criterion value in ``attrs['criterion']``.
    """
    g = grid.assign(
        _a=grid["attachment_pct_tiv"].round(ROUND), _l=grid["limit_pct_tiv"].round(ROUND)
    )
    if g.duplicated(["_a", "_l"]).any():
        raise ValueError("grid has duplicate (attachment, limit) rows")
    lookup = {(a, lim): i for i, (a, lim) in enumerate(zip(g["_a"], g["_l"], strict=True))}
    el = g["el_pct"].to_numpy(dtype=float)
    limits = sorted(g["_l"].unique())
    best: tuple[float, float, tuple[float, ...], list[int]] | None = None
    for a1 in sorted(g["_a"].unique()):
        for lims in itertools.product(limits, repeat=len(targets)):
            idx, a = [], a1
            for lim in lims:
                i = lookup.get((round(a, ROUND), lim))
                if i is None:
                    break
                idx.append(i)
                a = a + lim
            if len(idx) != len(targets):
                continue
            if np.any(el[idx] <= 0):
                continue
            crit = float(np.sum(np.log(el[idx] / np.asarray(targets)) ** 2))
            key = (crit, a1, lims, idx)
            if best is None or key[:3] < best[:3]:
                best = key
    if best is None:
        raise ValueError("no stack of adjacent grid layers exists")
    out = grid.iloc[best[3]].reset_index(drop=True).copy()
    out.insert(0, "layer", np.arange(1, len(targets) + 1))
    out.insert(1, "target_el_pct", targets)
    out.attrs["criterion"] = best[0]
    return out


def spread_at(fit: SpreadFit, el_pct: float, year: int, level: float = 0.90) -> dict[str, float]:
    """Median spread, prediction interval and multiple of ``fit`` at one EL (percent)."""
    row = pd.DataFrame(
        {
            "expected_loss_pct": [el_pct],
            "issue_year": [year],
            "trigger": ["indemnity"],
            "size_usd_m": [np.nan],
        }
    )
    p = fit.predict(row, level=level).iloc[0]
    return {
        "spread_median_pct": float(p["spread_median_pct"]),
        "pi_lo_pct": float(p["pi_lo_pct"]),
        "pi_hi_pct": float(p["pi_hi_pct"]),
        "multiple": float(p["multiple"]),
    }
