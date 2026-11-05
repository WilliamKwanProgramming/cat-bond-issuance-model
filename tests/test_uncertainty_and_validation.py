import numpy as np
import pandas as pd
import pytest
from conftest import toy_hazard

from catbond.hazard.landfall import annual_counts
from catbond.hazard.model import fit_hazard
from catbond.hazard.validation import group_probability, out_of_sample_check
from catbond.uncertainty import bootstrap_hazard_models, resample_years


def synthetic_landfalls(seed=0, first=1900, last=2019, lam=0.55):
    rng = np.random.default_rng(seed)
    years = np.arange(first, last + 1)
    counts = rng.poisson(lam, years.size)
    y = np.repeat(years, counts)
    n = y.size
    return pd.DataFrame(
        {
            "storm_id": [f"S{i}" for i in range(n)],
            "name": "X",
            "year": y,
            "time": pd.to_datetime([f"{yy}-09-01" for yy in y]),
            "lat": rng.uniform(25, 30.5, n),
            "lon": rng.uniform(-87, -80, n),
            "wind": 64 + rng.gamma(2.0, 15.0, n),
            "heading_deg": rng.uniform(0, 360, n),
            "speed_kmh": rng.uniform(5, 40, n),
            "n_fl_landfalls": 1,
            "definition": "record",
        }
    )


def test_resample_years_resamples_years_not_events():
    lf = synthetic_landfalls()
    rng = np.random.default_rng(1)
    reps = [resample_years(lf, 1900, 2019, rng) for _ in range(300)]
    n_events = np.array([len(r) for r in reps])
    # the number of events varies between replicates (resampling events would keep it fixed)
    assert n_events.std() > 0
    # bootstrap variance of the mean count matches var(counts) / n_years
    counts = annual_counts(lf, 1900, 2019).to_numpy()
    lam_boot = n_events / 120
    assert lam_boot.var() == pytest.approx(counts.var() / 120, rel=0.25)
    # every replicate year label lies in the window, and a drawn empty year stays empty
    assert all(r["year"].between(1900, 2019).all() for r in reps)
    assert any((annual_counts(r, 1900, 2019) == 0).sum() > 0 for r in reps)


def test_bootstrap_models_vary_and_keep_family():
    lf = synthetic_landfalls(seed=2)
    base = fit_hazard(lf, 1900, 2019, frequency="poisson")
    models = bootstrap_hazard_models(lf, base, n_boot=30, seed=3)
    assert len(models) == 30 and all(m.nb_size is None for m in models)
    lams = np.array([m.freq_mean for m in models])
    assert lams.std() > 0 and abs(lams.mean() - base.freq_mean) < 0.1
    assert len({m.intensity.scale for m in models}) > 1


def test_fit_hazard_reads_only_the_window():
    lf = synthetic_landfalls(seed=4)
    a = fit_hazard(lf, 1900, 1979)
    later = lf["year"] > 1979
    mutated = lf.copy()
    mutated.loc[later, "wind"] = 170.0
    mutated.loc[later, "lat"] = 0.0
    dropped = lf[~later]
    extra = pd.concat([lf, lf[later].assign(storm_id=lambda d: d["storm_id"] + "b")])
    for other in (mutated, dropped, extra):
        b = fit_hazard(other, 1900, 1979)
        assert b.freq_mean == a.freq_mean and b.nb_size == a.nb_size
        assert b.intensity == a.intensity
        np.testing.assert_array_equal(b.pool_lat, a.pool_lat)


def test_out_of_sample_prediction_unchanged_by_later_data():
    lf = synthetic_landfalls(seed=5)
    kw = dict(fit_years=(1900, 1979), test_years=(1980, 2019), n_boot=200, seed=1)
    a = out_of_sample_check(lf, **kw)
    later = lf["year"] > 1979
    mutated = lf.copy()
    mutated.loc[later, "wind"] = 120.0
    b = out_of_sample_check(mutated, **kw)
    pred = ["predicted_mean", "plugin_lo", "plugin_hi"]
    pd.testing.assert_frame_equal(a[pred], b[pred])
    assert a.attrs["lambda_fit"] == b.attrs["lambda_fit"]
    assert b.loc[b["group"] == "Cat 4", "observed"].item() == int(
        later.sum()
    )  # 120 kt is category 4: the observed side did change


def test_out_of_sample_closed_form_mean():
    lf = synthetic_landfalls(seed=6)
    out = out_of_sample_check(lf, (1900, 1979), (1980, 2019), n_boot=100)
    fit = lf[lf["year"] <= 1979]
    lam = len(fit) / 80
    model = fit_hazard(lf, 1900, 1979).intensity
    row = out.set_index("group").loc["Cat 3"]
    assert row["predicted_mean"] == pytest.approx(40 * lam * group_probability(model, 96, 113))
    allh = out.set_index("group").loc["All hurricanes"]
    assert allh["predicted_mean"] == pytest.approx(40 * lam)
    assert allh["observed"] == int(((lf["year"] >= 1980) & (lf["year"] <= 2019)).sum())
    cats = out[out["group"].str.match(r"Cat \d$")]
    assert cats["predicted_mean"].sum() == pytest.approx(40 * lam)


def test_group_probabilities_sum_to_one():
    m = toy_hazard().intensity
    p = [
        group_probability(m, lo, hi)
        for lo, hi in [(64, 83), (83, 96), (96, 113), (113, 137), (137, np.inf)]
    ]
    assert sum(p) == pytest.approx(1.0) and min(p) >= 0
