"""Recalculate the Excel layer pricer with LibreOffice and compare its formulas with Python.

Skipped where ``soffice`` is not installed (for example in CI); run locally after
``python3 scripts/make_workbook.py``. Tolerance: relative 1e-9 (both sides are double
precision computations on the same event values stored in the workbook).
"""

import pathlib
import re
import shutil
import subprocess

import numpy as np
import pandas as pd
import pytest

from catbond.layers import event_layer_expected_loss, layer_metrics
from catbond.pricing.fit import fit_log_spread
from catbond.pricing.sample import regression_sample
from catbond.transaction import spread_at

ROOT = pathlib.Path(__file__).resolve().parents[1]
BOOK = ROOT / "deliverables" / "catbond_layer_pricer.xlsx"
DEALS = ROOT / "data" / "deals" / "cat_bond_deals.csv"
LAYERS = ROOT / "results" / "transaction" / "layers.csv"
ERRORS = ("#DIV/0!", "#N/A", "#NAME?", "#NULL!", "#NUM!", "#REF!", "#VALUE!", "Err:")
REL = 1e-9

pytestmark = pytest.mark.skipif(
    shutil.which("soffice") is None, reason="LibreOffice (soffice) not found"
)


@pytest.fixture(scope="module")
def recalculated(tmp_path_factory):
    openpyxl = pytest.importorskip("openpyxl")
    if not BOOK.exists():
        pytest.skip("workbook not built (python3 scripts/make_workbook.py)")
    tmp = tmp_path_factory.mktemp("book")
    src = tmp / "in" / BOOK.name
    src.parent.mkdir()
    wb = openpyxl.load_workbook(BOOK)
    wb["Inputs"]["B5"] = 700.0  # a user layer that is not one of the check layers
    wb["Inputs"]["B6"] = 800.0
    wb.save(src)
    subprocess.run(
        ["soffice", "--headless", "--calc", "--convert-to", "xlsx", "--outdir", str(tmp), str(src)],
        check=True,
        capture_output=True,
        timeout=300,
    )
    return openpyxl.load_workbook(tmp / BOOK.name, data_only=True)


def _column(ws, col: int, first: int = 6) -> np.ndarray:
    vals = []
    for (v,) in ws.iter_rows(min_row=first, min_col=col, max_col=col, values_only=True):
        if v is None:
            break
        vals.append(float(v))
    return np.array(vals)


def test_no_error_cells(recalculated):
    bad = []
    for ws in recalculated.worksheets:
        for row in ws.iter_rows():
            for c in row:
                if isinstance(c.value, str) and c.value.startswith(ERRORS):
                    bad.append(f"{ws.title}!{c.coordinate}={c.value}")
    assert not bad, bad[:10]


def test_first_sheet_says_illustrative(recalculated):
    ws = recalculated.worksheets[0]
    assert "ILLUSTRATIVE AND TEACHING-GRADE" in ws["A2"].value
    assert "not an offering" in ws["A2"].value


def test_layer_formulas_match_python(recalculated):
    lc = recalculated["Loss curve"]
    loss = _column(lc, 2)
    year = _column(lc, 1).astype(int)
    annual_max = _column(lc, 5)
    n_years = int(recalculated["Inputs"]["B8"].value)
    calc = recalculated["Layer calculator"]
    layers = [(calc.cell(r, 1).value, calc.cell(r, 2).value, r) for r in range(27, 32)]
    layers.append((700.0, 800.0, None))
    assert len(layers) >= 6
    for a, lim, r in layers:
        e = a + lim
        ap = np.sum(annual_max >= a) / n_years
        ep = np.sum(annual_max >= e) / n_years
        el = event_layer_expected_loss(loss, n_years, a, e)
        got = (
            (calc.cell(r, 3).value, calc.cell(r, 4).value, calc.cell(r, 5).value)
            if r
            else (calc["B6"].value, calc["B7"].value, calc["B23"].value)
        )
        np.testing.assert_allclose(got, (ap, ep, el), rtol=REL, atol=1e-15)
        if r:  # the Python columns written by make_workbook.py agree as well
            np.testing.assert_allclose(
                [calc.cell(r, c).value for c in (6, 7, 8)], (ap, ep, el), rtol=REL
            )
        else:  # the user's layer: B8 is the EL with the annual cap (B23, tested above, without)
            capped = layer_metrics(year, loss, n_years, a, e).expected_loss
            assert el > capped * (1 + 1e-4)  # the cap matters for this layer, or this tests nothing
            np.testing.assert_allclose(calc["B8"].value, capped, rtol=REL, atol=1e-15)


def test_default_layer_is_the_middle_layer_of_the_layer_table():
    """The workbook as shipped (cached values) agrees with results/transaction/layers.csv."""
    openpyxl = pytest.importorskip("openpyxl")
    if not BOOK.exists():
        pytest.skip("workbook not built (python3 scripts/make_workbook.py)")
    calc = openpyxl.load_workbook(BOOK, data_only=True)["Layer calculator"]
    if calc["B8"].value is None:
        pytest.skip("workbook not recalculated (built with --no-recalc)")
    middle = pd.read_csv(LAYERS).set_index("layer").loc["Layer 2 (middle)"]
    assert (calc["B3"].value, calc["B4"].value) == (
        middle["attachment_usd_m"],
        middle["limit_usd_m"],
    )
    # the table prints el_pct and the percentages to 6 significant digits
    np.testing.assert_allclose(100 * calc["B8"].value, middle["el_pct"], rtol=1e-5)
    np.testing.assert_allclose(100 * calc["B6"].value, middle["ap_pct"], rtol=1e-3)
    np.testing.assert_allclose(100 * calc["B7"].value, middle["ep_occurrence_pct"], rtol=1e-3)
    assert calc["B23"].value > calc["B8"].value  # the uncapped EL is the larger


def test_annual_cap_helper_columns_cover_each_year_exactly():
    """Each annual formula sums the event rows of its own year, and the years partition them."""
    openpyxl = pytest.importorskip("openpyxl")
    if not BOOK.exists():
        pytest.skip("workbook not built (python3 scripts/make_workbook.py)")
    wb = openpyxl.load_workbook(BOOK)
    lc = wb["Loss curve"]
    n_events = sum(
        1
        for (v,) in lc.iter_rows(min_row=6, min_col=2, max_col=2, values_only=True)
        if v is not None
    )
    ev_year = [lc.cell(6 + i, 1).value for i in range(n_events)]
    assert ev_year == sorted(ev_year)
    for i in (0, n_events // 2, n_events - 1):
        want = f"=MIN(MAX(B{6 + i}-Attachment_USDm,0),Limit_USDm)"
        assert lc.cell(6 + i, 13).value == want
    assert lc.cell(6 + n_events, 13).value is None  # no stray formula below the last event
    covered = 0
    k = 0
    while (yr := lc.cell(6 + k, 14).value) is not None:
        f = lc.cell(6 + k, 15).value
        m = re.fullmatch(r"=MIN\(SUM\(M(\d+):M(\d+)\),Limit_USDm\)", f)
        assert m, f
        first, last = int(m[1]), int(m[2])
        assert first == 6 + covered  # the ranges follow one another without gaps or overlap
        assert {ev_year[r - 6] for r in range(first, last + 1)} == {yr}
        covered = last - 6 + 1
        k += 1
    assert covered == n_events
    assert k == sum(
        1
        for (v,) in lc.iter_rows(min_row=6, min_col=5, max_col=5, values_only=True)
        if v is not None
    )
    assert wb.defined_names["AnnualCapped"].attr_text == f"'Loss curve'!$O$6:$O${5 + k}"


def test_spread_formulas_match_python(recalculated):
    calc = recalculated["Layer calculator"]
    deals = regression_sample(pd.read_csv(DEALS))
    el_pct = calc["B9"].value
    m1 = spread_at(fit_log_spread(deals), el_pct, 2025)
    m2 = spread_at(fit_log_spread(deals, ("year_fe",)), el_pct, 2025)
    got = [calc[c].value for c in ("B12", "B14", "B15", "B18", "B20", "B21")]
    want = [
        m1["spread_median_pct"],
        m1["pi_lo_pct"],
        m1["pi_hi_pct"],
        m2["spread_median_pct"],
        m2["pi_lo_pct"],
        m2["pi_hi_pct"],
    ]
    np.testing.assert_allclose(got, want, rtol=1e-7)


def test_formula_text_on_a_small_event_set(tmp_path):
    """The formula builders, evaluated by LibreOffice on 40 random events, equal Python."""
    openpyxl = pytest.importorskip("openpyxl")
    from openpyxl.workbook.defined_name import DefinedName

    from catbond.workbook import (
        annual_capped_formula,
        ap_formula,
        capped_el_formula,
        ep_formula,
        event_el_formula,
        layer_loss_formula,
    )

    rng = np.random.default_rng(31)
    n_years = 25
    year = np.sort(rng.integers(0, n_years, 40))
    loss = np.round(rng.lognormal(3.0, 1.0, 40), 3)
    occ = np.zeros(n_years)
    np.maximum.at(occ, year, loss)
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Data"
    for i, v in enumerate(loss, start=1):
        ws.cell(i, 1, float(v))
    nz = occ[occ > 0]
    for i, v in enumerate(nz, start=1):
        ws.cell(i, 2, float(v))
    ws["D1"] = n_years
    for name, ref in (
        ("EventLoss", f"Data!$A$1:$A${len(loss)}"),
        ("AnnualMax", f"Data!$B$1:$B${len(nz)}"),
        ("N_years", "Data!$D$1"),
    ):
        wb.defined_names[name] = DefinedName(name, attr_text=ref)
    q = np.quantile(loss, [0.2, 0.5, 0.8, 0.95])
    layers = [
        (float(q[0]), float(q[1] - q[0])),
        (float(q[1]), float(q[3] - q[1])),
        (float(q[2]), 1000.0),
        (float(loss[3]), 5.0),
    ]
    # the events of a year are in adjacent rows (sorted by year): one row range per year
    starts = np.flatnonzero(np.r_[True, year[1:] != year[:-1]])
    ends = np.r_[starts[1:], year.size] - 1
    for r, (a, lim) in enumerate(layers, start=1):
        ws.cell(r, 6, a)
        ws.cell(r, 7, lim)
        ws.cell(r, 8, ap_formula(f"F{r}"))
        ws.cell(r, 9, ep_formula(f"F{r}", f"G{r}"))
        ws.cell(r, 10, event_el_formula(f"F{r}", f"G{r}"))
        # helper columns of layer r: event layer loss (column 20 + 2 (r - 1)) and annual layer
        # loss (the next column), as on the 'Loss curve' sheet of the workbook
        ev_col, yr_col = 20 + 2 * (r - 1), 21 + 2 * (r - 1)
        ev_letter = openpyxl.utils.get_column_letter(ev_col)
        for i in range(len(loss)):
            ws.cell(1 + i, ev_col, layer_loss_formula(f"A{1 + i}", f"$F${r}", f"$G${r}"))
        for k, (s, e) in enumerate(zip(starts, ends, strict=True), start=1):
            ws.cell(
                k,
                yr_col,
                annual_capped_formula(f"{ev_letter}{1 + s}", f"{ev_letter}{1 + e}", f"$G${r}"),
            )
        yr_letter = openpyxl.utils.get_column_letter(yr_col)
        ws.cell(r, 11, capped_el_formula(f"G{r}", f"${yr_letter}$1:${yr_letter}${starts.size}"))
    src = tmp_path / "in" / "small.xlsx"
    src.parent.mkdir()
    wb.save(src)
    subprocess.run(
        [
            "soffice",
            "--headless",
            "--calc",
            "--convert-to",
            "xlsx",
            "--outdir",
            str(tmp_path),
            str(src),
        ],
        check=True,
        capture_output=True,
        timeout=300,
    )
    out = openpyxl.load_workbook(tmp_path / "small.xlsx", data_only=True)["Data"]
    for r, (a, lim) in enumerate(layers, start=1):
        m = layer_metrics(year, loss, n_years, a, a + lim)
        el = event_layer_expected_loss(loss, n_years, a, a + lim)
        got = [out.cell(r, c).value for c in (8, 9, 10, 11)]
        want = (m.attachment_prob, m.ep_occurrence, el, m.expected_loss)
        np.testing.assert_allclose(got, want, rtol=REL)
    # the annual cap is active in at least one layer, so the capped formula is tested
    assert any(
        event_layer_expected_loss(loss, n_years, a, a + lim)
        > layer_metrics(year, loss, n_years, a, a + lim).expected_loss * (1 + 1e-6)
        for a, lim in layers
    )
