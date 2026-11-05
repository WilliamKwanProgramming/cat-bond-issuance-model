"""Per-occurrence cat bond layer analytics.

A layer [A, E] (attachment A < exhaustion E, in USD) pays, for an event with portfolio loss L,

    event layer loss  l = min(max(L - A, 0), E - A).

Aggregation within a year (stated convention): every event in the year can erode the principal,
and the annual principal loss is capped at the limit, P_y = min(sum over events of l, E - A).

Reported per layer (market conventions; see docs/model.md for sources):

* attachment probability AP = P(at least one event in a year has L >= A);
* exhaustion probability EP = P(P_y >= E - A), i.e. the whole principal is lost in the year
  (this counts exhaustion by several partial events; the occurrence-only version
  P(max L >= E) is also returned as ``ep_occurrence``);
* expected loss EL = E[P_y] / (E - A), as a fraction of the limit;
* for a term of T years, P(any principal loss over the term) = fraction of non-overlapping
  T-year blocks of simulated years with at least one year with P_y > 0.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


def event_layer_loss(loss: np.ndarray, attachment: float, exhaustion: float) -> np.ndarray:
    """min(max(L - A, 0), E - A) for each event loss L."""
    return np.minimum(
        np.maximum(np.asarray(loss, dtype=float) - attachment, 0.0), exhaustion - attachment
    )


def event_layer_expected_loss(
    loss: np.ndarray, n_years: int, attachment: float, exhaustion: float
) -> float:
    """Event-layer EL without the annual cap: sum of event layer losses / (n_years (E - A)).

    This is the quantity the Excel layer calculator computes with SUMPRODUCT. It equals the
    capped EL of ``layer_metrics`` when no simulated year has more principal loss than the limit,
    and is otherwise larger.
    """
    lay = event_layer_loss(loss, attachment, exhaustion)
    return float(lay.sum() / (n_years * (exhaustion - attachment)))


def annual_principal_loss(
    year: np.ndarray, loss: np.ndarray, n_years: int, attachment: float, exhaustion: float
) -> np.ndarray:
    """P_y = min(sum of event layer losses in year y, E - A) for y = 0..n_years-1."""
    lay = event_layer_loss(loss, attachment, exhaustion)
    total = np.bincount(np.asarray(year, dtype=np.int64), weights=lay, minlength=n_years)
    return np.minimum(total, exhaustion - attachment)


@dataclass(frozen=True)
class LayerMetrics:
    """Annual metrics of one layer (probabilities and EL as fractions)."""

    attachment_usd: float
    exhaustion_usd: float
    attachment_prob: float
    exhaustion_prob: float
    ep_occurrence: float
    expected_loss: float
    term_years: int
    term_prob_any_loss: float
    n_years: int


def layer_metrics(
    year: np.ndarray,
    loss: np.ndarray,
    n_years: int,
    attachment: float,
    exhaustion: float,
    term_years: int = 3,
) -> LayerMetrics:
    """Compute ``LayerMetrics`` from an event table (year index 0..n_years-1, event loss)."""
    if not exhaustion > attachment >= 0:
        raise ValueError("need 0 <= attachment < exhaustion")
    year = np.asarray(year, dtype=np.int64)
    loss = np.asarray(loss, dtype=float)
    limit = exhaustion - attachment
    occ = np.zeros(n_years)
    np.maximum.at(occ, year, loss)
    principal = annual_principal_loss(year, loss, n_years, attachment, exhaustion)
    nb = n_years // term_years
    hit = (principal[: nb * term_years] > 0).reshape(nb, term_years).any(axis=1)
    return LayerMetrics(
        attachment_usd=float(attachment),
        exhaustion_usd=float(exhaustion),
        attachment_prob=float(np.mean(occ >= attachment)),
        exhaustion_prob=float(np.mean(principal >= limit)),
        ep_occurrence=float(np.mean(occ >= exhaustion)),
        expected_loss=float(principal.mean() / limit),
        term_years=term_years,
        term_prob_any_loss=float(hit.mean()) if nb else float("nan"),
        n_years=n_years,
    )


def layer_table(
    year: np.ndarray,
    loss: np.ndarray,
    n_years: int,
    attachments: np.ndarray,
    limits: np.ndarray,
    term_years: int = 3,
) -> pd.DataFrame:
    """Metrics for every (attachment, limit) pair in the grid, one row each (tidy)."""
    rows = []
    for a in np.asarray(attachments, dtype=float):
        for lim in np.asarray(limits, dtype=float):
            m = layer_metrics(year, loss, n_years, a, a + lim, term_years)
            rows.append(asdict(m) | {"limit_usd": float(lim)})
    return pd.DataFrame(rows)
