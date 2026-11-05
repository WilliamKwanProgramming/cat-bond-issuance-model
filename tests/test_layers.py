import numpy as np
import pytest
from scipy import integrate, stats

from catbond.layers import (
    annual_principal_loss,
    event_layer_expected_loss,
    event_layer_loss,
    layer_metrics,
    layer_table,
)

LAM, MU, N = 0.6, 1.0, 200_000


@pytest.fixture(scope="module")
def poisson_exponential():
    rng = np.random.default_rng(99)
    counts = rng.poisson(LAM, N)
    year = np.repeat(np.arange(N), counts)
    return year, rng.exponential(MU, year.size)


def test_event_layer_loss_by_hand():
    got = event_layer_loss(np.array([0.0, 5.0, 6.0, 8.0, 20.0]), 5.0, 10.0)
    assert got.tolist() == [0.0, 0.0, 1.0, 3.0, 5.0]


def test_metrics_by_hand():
    # 5 years; layer [5, 10]; year 0: one event exactly at A; year 1: events 7 and 9
    # (2 + 4 = 6 capped at the limit 5); year 3: event 8; years 2, 4: nothing
    year = np.array([0, 1, 1, 3])
    loss = np.array([5.0, 7.0, 9.0, 8.0])
    m = layer_metrics(year, loss, 5, 5.0, 10.0, term_years=1)
    assert m.attachment_prob == pytest.approx(3 / 5)  # an event exactly at A counts (>=)
    assert m.exhaustion_prob == pytest.approx(1 / 5)  # year 1 exhausted by two partial events
    assert m.ep_occurrence == 0.0
    # principal losses 0, 5, 0, 3, 0 -> EL = 8 / 5 / 5
    assert m.expected_loss == pytest.approx(8 / 25)
    np.testing.assert_allclose(annual_principal_loss(year, loss, 5, 5.0, 10.0), [0, 5, 0, 3, 0])
    assert m.term_prob_any_loss == pytest.approx(2 / 5)


def test_rejects_bad_layer():
    with pytest.raises(ValueError):
        layer_metrics(np.array([0]), np.array([1.0]), 1, 5.0, 5.0)


def test_layer_el_matches_closed_form(poisson_exponential):
    year, loss = poisson_exponential
    a, e = 1.0, 2.5
    # expected sum of event layer losses per year: lambda mu e^{-A/mu} (1 - e^{-(E-A)/mu})
    uncapped = LAM * MU * np.exp(-a / MU) * (1 - np.exp(-(e - a) / MU))
    sim_uncapped = event_layer_loss(loss, a, e).sum() / N
    se = event_layer_loss(loss, a, e).std() * np.sqrt(LAM / N) * 1.5
    assert sim_uncapped == pytest.approx(uncapped, abs=4 * se)
    m = layer_metrics(year, loss, N, a, e)
    # Exact capped annual EL. Layer events arrive as Poisson(lambda q), q = e^{-A/mu}, and each
    # layer loss is min(X, W) with X ~ Exp(mu) (memoryless), W = E - A. For s < W the annual
    # sum S exceeds s iff the uncapped gamma sum does, so
    #   P(S > s) = sum_k Pois(k; lambda q) Gamma(k, mu).sf(s),  EL = int_0^W P(S > s) ds / W,
    #   EP = P(S >= W) = sum_k Pois(k; lambda q) Gamma(k, mu).sf(W).
    w, rate = e - a, LAM * np.exp(-a / MU)
    k = np.arange(1, 40)

    def sf(x):
        return float((stats.poisson.pmf(k, rate) * stats.gamma.sf(x, k, scale=MU)).sum())

    el_exact = integrate.quad(sf, 0.0, w)[0] / w
    principal = annual_principal_loss(year, loss, N, a, e)
    se_el = principal.std() / np.sqrt(N) / w
    assert m.expected_loss == pytest.approx(el_exact, abs=4 * se_el)
    assert el_exact * w < 0.97 * uncapped  # the cap matters at these parameters
    ep = sf(w)
    assert m.exhaustion_prob == pytest.approx(ep, abs=4 * np.sqrt(ep * (1 - ep) / N))
    # AP = 1 - exp(-lambda P(L >= A)), EP_occ = 1 - exp(-lambda P(L >= E))
    ap = 1 - np.exp(-LAM * np.exp(-a / MU))
    assert m.attachment_prob == pytest.approx(ap, abs=4 * np.sqrt(ap * (1 - ap) / N))
    epo = 1 - np.exp(-LAM * np.exp(-e / MU))
    assert m.ep_occurrence == pytest.approx(epo, abs=4 * np.sqrt(epo * (1 - epo) / N))


def test_term_probability(poisson_exponential):
    year, loss = poisson_exponential
    m = layer_metrics(year, loss, N, 2.0, 3.0, term_years=3)
    expected = 1 - (1 - m.attachment_prob) ** 3
    nb = N // 3
    assert m.term_prob_any_loss == pytest.approx(expected, abs=4 * np.sqrt(expected / nb))


def test_invariants_on_grid(poisson_exponential):
    year, loss = poisson_exponential
    t = layer_table(
        year,
        loss,
        N,
        attachments=np.array([0.5, 1.0, 1.5, 2.0, 3.0]),
        limits=np.array([0.5, 1.0, 2.0]),
    )
    assert (t["attachment_prob"] >= t["exhaustion_prob"]).all()
    assert (t["expected_loss"] <= t["attachment_prob"] + 1e-12).all()
    assert (t["expected_loss"] >= t["exhaustion_prob"] - 1e-12).all()
    for _, g in t.groupby("limit_usd"):
        g = g.sort_values("attachment_usd")
        assert np.all(np.diff(g["expected_loss"]) <= 1e-12)
        assert np.all(np.diff(g["attachment_prob"]) <= 1e-12)


def test_stacked_layers(poisson_exponential):
    year, loss = poisson_exponential
    edges = [0.5, 1.0, 2.0, 4.0]
    # event level: stacked layer losses add up exactly
    parts = sum(event_layer_loss(loss, a, e) for a, e in zip(edges[:-1], edges[1:], strict=True))
    np.testing.assert_allclose(parts, event_layer_loss(loss, edges[0], edges[-1]))
    # annual level with at most one event per year: stacked ELs (in USD) add up exactly
    first = np.r_[True, np.diff(year) > 0]
    y1, l1 = year[first], loss[first]
    usd = [
        layer_metrics(y1, l1, N, a, e).expected_loss * (e - a)
        for a, e in zip(edges[:-1], edges[1:], strict=True)
    ]
    full = layer_metrics(y1, l1, N, edges[0], edges[-1]).expected_loss * (edges[-1] - edges[0])
    assert sum(usd) == pytest.approx(full, rel=1e-12)
    # with several events per year the per-layer annual caps bind earlier: stacked <= full
    usd_all = [
        layer_metrics(year, loss, N, a, e).expected_loss * (e - a)
        for a, e in zip(edges[:-1], edges[1:], strict=True)
    ]
    full_all = layer_metrics(year, loss, N, edges[0], edges[-1]).expected_loss * (
        edges[-1] - edges[0]
    )
    assert sum(usd_all) <= full_all + 1e-12


def test_event_layer_expected_loss_hand_computed_and_cap():
    year = np.array([0, 0, 2])
    loss = np.array([15.0, 30.0, 12.0])
    # layer [10, 20]: event layer losses 5, 10, 2 -> 17 / (4 years * 10)
    assert event_layer_expected_loss(loss, 4, 10.0, 20.0) == pytest.approx(17 / 40)
    # the capped EL limits year 0 to 10: (10 + 2) / 40
    assert layer_metrics(year, loss, 4, 10.0, 20.0).expected_loss == pytest.approx(12 / 40)
