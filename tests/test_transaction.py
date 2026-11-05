import numpy as np
import pandas as pd
import pytest

from catbond.pricing.fit import fit_log_spread
from catbond.pricing.synthetic import synthetic_deals
from catbond.transaction import choose_stack, spread_at


def _grid(el_fn):
    rows = []
    for a in np.round(np.arange(0.25, 4.001, 0.25), 6):
        for lim in (0.5, 1.0, 2.0, 3.0):
            rows.append({"attachment_pct_tiv": a, "limit_pct_tiv": lim, "el_pct": el_fn(a, lim)})
    return pd.DataFrame(rows)


def test_choose_stack_picks_adjacent_layers_closest_to_targets():
    # EL falls with attachment; the exact targets 4, 2, 1 are hit by [0.5, 1.0], [1.0, 2.0],
    # [2.0, 2.5] (EL depends on attachment only here)
    el = {0.5: 4.0, 1.0: 2.0, 2.0: 1.0}
    g = _grid(lambda a, lim: el.get(a, 10.0 / (1 + 10 * a)) if lim == 0.5 or a != 1.0 else 2.0)
    s = choose_stack(g)
    assert s["layer"].tolist() == [1, 2, 3]
    a = s["attachment_pct_tiv"].to_numpy()
    lim = s["limit_pct_tiv"].to_numpy()
    np.testing.assert_allclose(a[1:], a[:-1] + lim[:-1])  # adjacent
    assert s.attrs["criterion"] == pytest.approx(0.0)
    assert a[0] == 0.5 and a[1] == 1.0 and a[2] == 2.0 and lim[1] == 1.0
    assert s["target_el_pct"].tolist() == [4.0, 2.0, 1.0]


def test_choose_stack_tie_goes_to_lowest_attachment_and_criterion_is_log_ratio():
    g = _grid(lambda a, lim: 2.0)  # every stack has the same criterion
    s = choose_stack(g)
    assert s["attachment_pct_tiv"].iloc[0] == 0.25 and s["limit_pct_tiv"].iloc[0] == 0.5
    assert s.attrs["criterion"] == pytest.approx(np.log(0.5) ** 2 + np.log(2.0) ** 2)


def test_choose_stack_rejects_bad_grids():
    g = _grid(lambda a, lim: 1.0)
    with pytest.raises(ValueError, match="duplicate"):
        choose_stack(pd.concat([g, g.iloc[[0]]]))
    with pytest.raises(ValueError, match="no stack"):
        choose_stack(g[g["attachment_pct_tiv"] < 0.5])


def test_spread_at_matches_predict():
    rng = np.random.default_rng(21)
    d = synthetic_deals(60, rng)
    f = fit_log_spread(d, ("year_fe",))
    r = spread_at(f, 2.0, 2020)
    x = np.zeros(len(f.coef))
    x[0], x[1] = 1.0, np.log(2.0)
    x[f.names.index("year[2020]")] = 1.0
    assert r["spread_median_pct"] == pytest.approx(np.exp(x @ f.coef))
    assert r["pi_lo_pct"] < r["spread_median_pct"] < r["pi_hi_pct"]
    assert r["multiple"] == pytest.approx(r["spread_median_pct"] / 2.0)
