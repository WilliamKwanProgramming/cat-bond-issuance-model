"""Leave-one-group-out prediction errors of the log-spread regression.

For every group g (a note series or a sponsor): fit the model on the rows outside g and predict
the median spread exp(x_i'b_(-g)) of each row i in g. A row whose categorical level (an issue
year or a trigger type) does not occur among the training rows cannot be predicted; it is
flagged, not dropped silently. Errors are reported on the log scale,
e_i = log(spread_i) - x_i'b_(-g), and in percentage points of spread,
spread_i - exp(x_i'b_(-g)). With one row per group this is ordinary leave-one-out, and the log
errors equal e_i / (1 - h_ii) of the full fit (tested).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from catbond.pricing.fit import CATEGORICAL_COVARIATES, cluster_labels, fit_log_spread


def _level(row: pd.Series, cov: str) -> str:
    if cov == "year_fe":
        return str(int(row["issue_year"]))
    return str(row[cov]).strip()


def leave_group_out(rows: pd.DataFrame, covariates: Sequence[str], group: str) -> pd.DataFrame:
    """One output row per input row: group, observed spread, prediction and errors.

    ``rows`` must already be the regression sample (no missing values in the model columns).
    """
    rows = rows.reset_index(drop=True)
    codes = cluster_labels(rows, group)
    out = []
    for g in np.unique(codes):
        test = rows[codes == g]
        train = rows[codes != g]
        fit = fit_log_spread(train, covariates)
        for i, r in test.iterrows():
            ok = all(
                _level(r, c) in fit.levels[c] for c in covariates if c in CATEGORICAL_COVARIATES
            )
            spread = float(r["spread_pct"])
            rec = {
                "row": i,
                "group": str(r[group]),
                "spread_pct": spread,
                "predictable": ok,
                "pred_spread_pct": np.nan,
                "err_log": np.nan,
                "err_pp": np.nan,
            }
            if ok:
                pred = float(fit.predict(test.loc[[i]])["spread_median_pct"].iloc[0])
                rec |= {
                    "pred_spread_pct": pred,
                    "err_log": float(np.log(spread) - np.log(pred)),
                    "err_pp": spread - pred,
                }
            out.append(rec)
    return pd.DataFrame(out).sort_values("row", kind="stable").reset_index(drop=True)


def summarize_errors(errors: pd.DataFrame) -> dict[str, float]:
    """RMSE and mean absolute error (log scale and percentage points) over predictable rows."""
    ok = errors[errors["predictable"]]
    el, ep = ok["err_log"].to_numpy(), ok["err_pp"].to_numpy()
    return {
        "n_rows": len(errors),
        "n_predicted": len(ok),
        "n_unpredictable": int((~errors["predictable"]).sum()),
        "rmse_log": float(np.sqrt(np.mean(el**2))),
        "mae_log": float(np.mean(np.abs(el))),
        "rmse_pp": float(np.sqrt(np.mean(ep**2))),
        "mae_pp": float(np.mean(np.abs(ep))),
        "median_abs_pp": float(np.median(np.abs(ep))),
    }
