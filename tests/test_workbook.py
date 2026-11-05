"""Recalculate the Excel layer pricer with LibreOffice and compare its formulas with Python.

Skipped where ``soffice`` is not installed (for example in CI); run locally after
``python3 scripts/make_workbook.py``. Tolerance: relative 1e-9 (both sides are double
precision computations on the same event values stored in the workbook).
"""

import pathlib
import shutil
import subprocess

import numpy as np
import pandas as pd
import pytest

from catbond.layers import event_layer_expected_loss
from catbond.pricing.fit import fit_log_spread
from catbond.pricing.sample import regression_sample
from catbond.transaction import spread_at

ROOT = pathlib.Path(__file__).resolve().parents[1]
BOOK = ROOT / "deliverables" / "catbond_layer_pricer.xlsx"
DEALS = ROOT / "data" / "deals" / "cat_bond_deals.csv"
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
            else (calc["B6"].value, calc["B7"].value, calc["B8"].value)
        )
        np.testing.assert_allclose(got, (ap, ep, el), rtol=REL, atol=1e-15)
        if r:  # the Python columns written by make_workbook.py agree as well
            np.testing.assert_allclose(
                [calc.cell(r, c).value for c in (6, 7, 8)], (ap, ep, el), rtol=REL
            )


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

    from catbond.layers import layer_metrics
    from catbond.workbook import ap_formula, ep_formula, event_el_formula

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
    for r, (a, lim) in enumerate(layers, start=1):
        ws.cell(r, 6, a)
        ws.cell(r, 7, lim)
        ws.cell(r, 8, ap_formula(f"F{r}"))
        ws.cell(r, 9, ep_formula(f"F{r}", f"G{r}"))
        ws.cell(r, 10, event_el_formula(f"F{r}", f"G{r}"))
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
        got = [out.cell(r, c).value for c in (8, 9, 10)]
        np.testing.assert_allclose(got, (m.attachment_prob, m.ep_occurrence, el), rtol=REL)
