#!/usr/bin/env python3
"""Build deliverables/catbond_layer_pricer.xlsx: an Excel layer calculator with live formulas.

ILLUSTRATIVE, TEACHING-GRADE, HYPOTHETICAL SPONSOR. Sheets:

* Inputs: named input cells (attachment, limit) and fixed parameters;
* Loss curve: the base-case OEP/AEP return-period table, and the simulated event losses and
  annual maximum losses (values) of the 100,000-year event set (seed 1), years with a loss only;
* Layer calculator: live formulas. For the user's layer, attachment probability
  COUNTIF(annual max >= A) / N, occurrence exhaustion probability COUNTIF(annual max >= E) / N,
  event-layer expected loss SUMPRODUCT(min(max(L - A, 0), E - A)) / (N (E - A)) (no annual cap,
  so it can exceed the capped EL of the Python layer table), and the M1/M2 median spread with a
  90% prediction interval from the coefficients on the Pricing sheet. A check table repeats the
  formulas for five fixed layers next to the Python values (catbond.layers);
* Deals: the public deal table with source URLs and the regression-sample flags;
* Pricing: M1 and M2 coefficients (main sample), the parts of (X'X)^-1 the interval needs, the
  slope intervals, and a chart of the deals with the M1 and M2 (2025) median curves;
* Notes: assumptions, limits and sources.

The file is recalculated with LibreOffice when ``soffice`` is available, so that cached values
are stored; tests/test_workbook.py recalculates a copy and compares with Python.

Needs results/eventset/event_table.csv.gz (regenerate with scripts/run_eventset.py),
results/eventset/rp_losses.csv, results/transaction/layers.csv, results/pricing/*.

Usage: python3 scripts/make_workbook.py [--no-recalc]
"""

from __future__ import annotations

import argparse
import pathlib
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.chart import Reference, ScatterChart, Series
from openpyxl.comments import Comment
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.workbook.defined_name import DefinedName

from catbond.layers import event_layer_expected_loss, layer_metrics
from catbond.manifest import repo_root
from catbond.pipeline import TIV_USD
from catbond.pricing.fit import fit_log_spread
from catbond.pricing.sample import regression_sample

R = repo_root() / "results"
OUT = repo_root() / "deliverables" / "catbond_layer_pricer.xlsx"
DEALS = repo_root() / "data" / "deals" / "cat_bond_deals.csv"
YEAR = 2025
FONT = "Arial"
BLUE = Font(name=FONT, color="0000FF")
BOLD = Font(name=FONT, bold=True)
TITLE = Font(name=FONT, bold=True, size=14)
WARN = Font(name=FONT, bold=True, color="C00000")
GREEN = Font(name=FONT, color="008000")
PLAIN = Font(name=FONT)
INPUT_FILL = PatternFill("solid", fgColor="FFFF00")
DISCLAIMER = (
    "ILLUSTRATIVE AND TEACHING-GRADE. Hypothetical sponsor and portfolio; not an offering, a "
    "quote or investment advice. The loss model is not a vendor catastrophe model and its "
    "expected losses are not comparable to those in a cat bond offering circular."
)
CHECK_LAYERS_USD_M = [  # (attachment, limit) in USD million
    (500.0, 500.0),
    (1000.0, 500.0),
    (1500.0, 500.0),
    (1200.0, 600.0),
    (250.0, 2000.0),
]
MIN_ROW = 6  # first data row of the event and annual-max columns on "Loss curve"


def _name(wb: Workbook, name: str, ref: str) -> None:
    wb.defined_names[name] = DefinedName(name, attr_text=ref)


def _style_all(wb: Workbook) -> None:
    for ws in wb.worksheets:
        for row in ws.iter_rows():
            for c in row:
                if c.value is not None and (c.font is None or c.font.name != FONT):
                    f = c.font
                    c.font = Font(
                        name=FONT, bold=f.bold, color=f.color, size=f.size, italic=f.italic
                    )


def build(path: pathlib.Path) -> None:
    ev = pd.read_csv(R / "eventset" / "event_table.csv.gz")
    manifest = (R / "eventset" / "MANIFEST.txt").read_text()
    n_years = int(next(ln.split(":")[1] for ln in manifest.splitlines() if ln.startswith("n_y")))
    seed = next(ln.split(":")[1].strip() for ln in manifest.splitlines() if ln.startswith("seed"))
    loss_m = ev["loss_usd"].to_numpy() / 1e6
    pos = loss_m > 0
    ev_year, ev_loss = ev["year"].to_numpy()[pos], loss_m[pos]
    occ = np.zeros(n_years)
    np.maximum.at(occ, ev["year"].to_numpy(), loss_m)
    yrs = np.flatnonzero(occ > 0)

    deals_all = pd.read_csv(DEALS)
    sample = pd.read_csv(R / "pricing" / "sample.csv")
    deals = regression_sample(deals_all)
    fits = {
        "M1": fit_log_spread(deals, (), cluster="deal_name"),
        "M2": fit_log_spread(deals, ("year_fe",), cluster="deal_name"),
    }
    coefs = pd.read_csv(R / "pricing" / "coefficients.csv")
    main_name = coefs["sample"].iloc[0]
    for m, f in fits.items():  # the workbook must use the committed regression results
        ref = coefs[(coefs["sample"] == main_name) & (coefs["model"] == m)]["coef"].to_numpy()
        if not np.allclose(ref, f.coef, rtol=1e-5):
            raise RuntimeError(f"{m} coefficients differ from results/pricing/coefficients.csv")
    slopes = pd.read_csv(R / "pricing" / "slope_intervals.csv")
    layers = pd.read_csv(R / "transaction" / "layers.csv")
    rp = pd.read_csv(R / "eventset" / "rp_losses.csv")

    wb = Workbook()
    # ---------------------------------------------------------------- Inputs
    ws = wb.active
    ws.title = "Inputs"
    ws["A1"] = "Cat bond layer pricer (Florida hurricane, hypothetical sponsor)"
    ws["A1"].font = TITLE
    ws["A2"] = DISCLAIMER
    ws["A2"].font = WARN
    ws["A2"].alignment = Alignment(wrap_text=True, vertical="top")
    ws.merge_cells("A2:D2")
    ws.row_dimensions[2].height = 60
    ws["A4"] = "Inputs (blue on yellow: edit these two cells)"
    ws["A4"].font = BOLD
    rows = [
        ("Attachment (USD m)", 1000.0, "Attachment_USDm", True, "per-occurrence attachment"),
        (
            "Limit (USD m)",
            500.0,
            "Limit_USDm",
            True,
            "layer limit; exhaustion = attachment + limit",
        ),
        ("Total insured value (USD m)", TIV_USD / 1e6, "TIV_USDm", False, "hypothetical"),
        ("Simulated years N", n_years, "N_years", False, f"event set seed {seed}"),
        ("Prediction interval level", 0.90, "PI_level", False, "two-sided"),
        ("Issue year used for M2", YEAR, "M2_year", False, "fixed: the year effect is 2025's"),
    ]
    for i, (label, val, name, editable, note) in enumerate(rows, start=5):
        ws.cell(i, 1, label)
        c = ws.cell(i, 2, val)
        c.font = BLUE if editable else PLAIN
        if editable:
            c.fill = INPUT_FILL
        ws.cell(i, 3, note)
        _name(wb, name, f"Inputs!$B${i}")
    ws["B5"].number_format = ws["B6"].number_format = "#,##0"
    ws["B7"].number_format = ws["B8"].number_format = "#,##0"
    ws["B9"].number_format = "0%"
    ws["A12"] = "Exhaustion (USD m)"
    ws["B12"] = "=Attachment_USDm+Limit_USDm"
    ws["B12"].number_format = "#,##0"
    ws["A13"] = "Attachment (% of TIV)"
    ws["B13"] = "=Attachment_USDm/TIV_USDm"
    ws["A14"] = "Exhaustion (% of TIV)"
    ws["B14"] = "=B12/TIV_USDm"
    ws["B13"].number_format = ws["B14"].number_format = "0.00%"
    ws["A16"] = "Example: the three layers of the hypothetical transaction (USD m)"
    ws["A16"].font = BOLD
    ws.append(["Layer", "Attachment", "Limit", "EL (capped, Python) %"])
    for r in layers.itertuples():
        ws.append([r.layer, r.attachment_usd_m, r.limit_usd_m, round(r.el_pct, 4)])
    ws["A22"] = "Results are on the 'Layer calculator' sheet; sources and limits on 'Notes'."
    for col, w in zip("ABCD", (34, 16, 48, 22), strict=True):
        ws.column_dimensions[col].width = w

    # ---------------------------------------------------------------- Loss curve
    lc = wb.create_sheet("Loss curve")
    lc["A1"] = (
        f"Simulated event set: N = {n_years:,} years, seed {seed}, base case "
        "(hazard fitted to HURDAT2 1900-2019; HYPOTHETICAL USD 100 bn portfolio). "
        "Values, not formulas."
    )
    lc["A1"].font = BOLD
    lc["A2"] = (
        f"Columns A-B: the {pos.sum():,} events with a loss. Columns D-E: the annual maximum "
        f"event loss of the {yrs.size:,} years with a loss (the other years have no loss)."
    )
    lc["A3"] = "From results/eventset/event_table.csv.gz (scripts/run_eventset.py)."
    for c, h in zip(
        "ABDE",
        ("year index", "event loss (USD m)", "year index", "annual max (USD m)"),
        strict=True,
    ):
        lc[f"{c}{MIN_ROW - 1}"] = h
        lc[f"{c}{MIN_ROW - 1}"].font = BOLD
    for i, (y, v) in enumerate(zip(ev_year, ev_loss, strict=True)):
        lc.cell(MIN_ROW + i, 1, int(y))
        lc.cell(MIN_ROW + i, 2, float(v))
    for i, y in enumerate(yrs):
        lc.cell(MIN_ROW + i, 4, int(y))
        lc.cell(MIN_ROW + i, 5, float(occ[y]))
    last_ev, last_yr = MIN_ROW + pos.sum() - 1, MIN_ROW + yrs.size - 1
    _name(wb, "EventLoss", f"'Loss curve'!$B${MIN_ROW}:$B${last_ev}")
    _name(wb, "AnnualMax", f"'Loss curve'!$E${MIN_ROW}:$E${last_yr}")
    lc["G5"] = "Return-period losses, base case (results/eventset/rp_losses.csv)"
    lc["G5"].font = BOLD
    for j, h in enumerate(
        ["curve", "return period (years)", "loss (% TIV)", "SE (% TIV)", "loss (USD m)"]
    ):
        lc.cell(6, 7 + j, h).font = BOLD
    for i, r in enumerate(rp.itertuples(), start=7):
        lc.cell(i, 7, r.curve)
        lc.cell(i, 8, "" if pd.isna(r.return_period) else int(r.return_period))
        lc.cell(i, 9, round(r.loss_pct_tiv, 4))
        lc.cell(i, 10, round(r.se_pct_tiv, 4))
        lc.cell(i, 11, f"=I{i}/100*TIV_USDm").number_format = "#,##0"
    for col, w in zip("ABCDEFGHIJK", (11, 18, 3, 11, 18, 3, 8, 20, 13, 11, 13), strict=True):
        lc.column_dimensions[col].width = w

    # ---------------------------------------------------------------- Pricing
    pr = wb.create_sheet("Pricing")
    pr["A1"] = (
        "Spread regression on 56 public layers (REAL data, Artemis pages), main sample. "
        "Regression output, not a quote."
    )
    pr["A1"].font = BOLD
    pr["A2"] = "log(spread %) = a + b log(EL %) [+ year effect]; median spread = exp(fit)."
    pr["A4"], pr["B4"], pr["C4"] = "Parameter", "M1", "M2 (year 2025)"
    for c in ("A4", "B4", "C4"):
        pr[c].font = BOLD
    yi = fits["M2"].names.index(f"year[{YEAR}]")
    params = [
        ("a (const)", "a", 0, 0),
        ("b (log EL)", "b", 1, 1),
        (f"year[{YEAR}] effect", "y", None, yi),
    ]
    rown = 5
    for label, key, i1, i2 in params:
        pr.cell(rown, 1, label)
        if i1 is not None:
            pr.cell(rown, 2, float(fits["M1"].coef[i1]))
            _name(wb, f"M1_{key}", f"Pricing!$B${rown}")
        pr.cell(rown, 3, float(fits["M2"].coef[i2]))
        _name(wb, f"M2_{key}", f"Pricing!$C${rown}")
        rown += 1
    pr.cell(rown, 1, "residual SD (log scale)")
    pr.cell(rown, 2, fits["M1"].sigma)
    pr.cell(rown, 3, fits["M2"].sigma)
    _name(wb, "M1_sigma", f"Pricing!$B${rown}")
    _name(wb, "M2_sigma", f"Pricing!$C${rown}")
    rown += 1
    pr.cell(rown, 1, "residual degrees of freedom n - k")
    pr.cell(rown, 2, fits["M1"].n - len(fits["M1"].coef))
    pr.cell(rown, 3, fits["M2"].n - len(fits["M2"].coef))
    _name(wb, "M1_df", f"Pricing!$B${rown}")
    _name(wb, "M2_df", f"Pricing!$C${rown}")
    rown += 2
    pr.cell(rown, 1, "(X'X)^-1 entries for the prediction interval").font = BOLD
    rown += 1
    idx = {"M1": {"a": 0, "b": 1}, "M2": {"a": 0, "b": 1, "y": yi}}
    for m, keys in idx.items():
        v = fits[m].xtx_inv
        for k1 in keys:
            for k2 in keys:
                if list(keys).index(k2) < list(keys).index(k1):
                    continue
                pr.cell(rown, 1, f"{m} V[{k1},{k2}]")
                pr.cell(rown, 2, float(v[keys[k1], keys[k2]]))
                _name(wb, f"{m}_V{k1}{k2}", f"Pricing!$B${rown}")
                rown += 1
    rown += 1
    pr.cell(rown, 1, "Slope b, 95% intervals (results/pricing/slope_intervals.csv)").font = BOLD
    rown += 1
    for j, h in enumerate(["model", "method", "b", "lo", "hi"]):
        pr.cell(rown, 1 + j, h).font = BOLD
    rown += 1
    for r in slopes[slopes["sample"] == main_name].itertuples():
        for j, v in enumerate([r.model, r.method, r.b, r.lo, r.hi]):
            pr.cell(rown, 1 + j, round(v, 4) if isinstance(v, float) else v)
        rown += 1
    # chart data: deals (values) and fitted median curves (formulas)
    c0 = 8
    pr.cell(4, c0, "Deal EL %").font = BOLD
    pr.cell(4, c0 + 1, "Deal spread %").font = BOLD
    for i, r in enumerate(deals.itertuples(), start=5):
        pr.cell(i, c0, float(r.expected_loss_pct))
        pr.cell(i, c0 + 1, float(r.spread_pct))
    last_d = 4 + len(deals)
    pr.cell(4, c0 + 3, "EL grid %").font = BOLD
    pr.cell(4, c0 + 4, "M1 median %").font = BOLD
    pr.cell(4, c0 + 5, "M2 (2025) median %").font = BOLD
    grid = np.round(np.linspace(0.5, 5.0, 19), 3)
    for i, x in enumerate(grid, start=5):
        pr.cell(i, c0 + 3, float(x))
        pr.cell(i, c0 + 4, f"=EXP(M1_a+M1_b*LN(K{i}))")
        pr.cell(i, c0 + 5, f"=EXP(M2_a+M2_y+M2_b*LN(K{i}))")
    last_g = 4 + len(grid)
    ch = ScatterChart()
    ch.title = "Public layers (real data) and fitted median spread"
    ch.x_axis.title = "Expected loss (% of principal)"
    ch.y_axis.title = "Spread (%)"
    ch.style = 13
    s1 = Series(
        Reference(pr, min_col=c0 + 1, min_row=5, max_row=last_d),
        Reference(pr, min_col=c0, min_row=5, max_row=last_d),
        title="Public layers (56)",
    )
    s1.marker.symbol = "circle"
    s1.graphicalProperties.line.noFill = True
    ch.series.append(s1)
    for col, title in ((c0 + 4, "M1 median"), (c0 + 5, "M2 median, 2025")):
        s = Series(
            Reference(pr, min_col=col, min_row=5, max_row=last_g),
            Reference(pr, min_col=c0 + 3, min_row=5, max_row=last_g),
            title=title,
        )
        s.marker.symbol = "none"
        ch.series.append(s)
    ch.x_axis.delete = False
    ch.y_axis.delete = False
    ch.height, ch.width = 10, 18
    pr.add_chart(ch, "H26")
    pr.column_dimensions["A"].width = 34
    pr.column_dimensions["B"].width = 44
    for col in "CDEFHIKLM":
        pr.column_dimensions[col].width = 14

    # ---------------------------------------------------------------- Layer calculator
    lcalc = wb.create_sheet("Layer calculator", 1)
    lcalc["A1"] = "Layer calculator (live formulas). Illustrative, teaching-grade; not a quote."
    lcalc["A1"].font = BOLD
    cells = [
        ("Attachment A (USD m)", "=Attachment_USDm", "#,##0"),
        ("Limit E - A (USD m)", "=Limit_USDm", "#,##0"),
        ("Exhaustion E (USD m)", "=B3+B4", "#,##0"),
        (
            "Attachment probability (annual max >= A)",
            '=COUNTIF(AnnualMax,">="&B3)/N_years',
            "0.000%",
        ),
        (
            "Exhaustion probability, occurrence (annual max >= E)",
            '=COUNTIF(AnnualMax,">="&B5)/N_years',
            "0.000%",
        ),
        (
            "Event-layer expected loss (% of limit, no annual cap)",
            "=SUMPRODUCT((EventLoss>B3)*((EventLoss<B5)*(EventLoss-B3)+(EventLoss>=B5)*B4))"
            "/(N_years*B4)",
            "0.000%",
        ),
        ("EL in percent (regression input)", "=100*B8", "0.000"),
        ("log EL", "=IF(B9>0,LN(B9),NA())", "0.0000"),
        ("t quantile", "=TINV(1-PI_level,M1_df)", "0.0000"),
        ("M1 median spread %", "=EXP(M1_a+M1_b*B10)", "0.00"),
        ("M1 prediction SE (log)", "=M1_sigma*SQRT(1+M1_Vaa+2*M1_Vab*B10+M1_Vbb*B10^2)", "0.0000"),
        ("M1 interval low %", "=EXP(M1_a+M1_b*B10-B11*B13)", "0.00"),
        ("M1 interval high %", "=EXP(M1_a+M1_b*B10+B11*B13)", "0.00"),
        ("M1 multiple (spread / EL)", "=B12/B9", "0.00"),
        ("t quantile (M2)", "=TINV(1-PI_level,M2_df)", "0.0000"),
        ("M2 median spread % (2025)", "=EXP(M2_a+M2_y+M2_b*B10)", "0.00"),
        (
            "M2 prediction SE (log)",
            "=M2_sigma*SQRT(1+M2_Vaa+M2_Vbb*B10^2+M2_Vyy+2*M2_Vab*B10+2*M2_Vay+2*M2_Vby*B10)",
            "0.0000",
        ),
        ("M2 interval low %", "=EXP(M2_a+M2_y+M2_b*B10-B17*B19)", "0.00"),
        ("M2 interval high %", "=EXP(M2_a+M2_y+M2_b*B10+B17*B19)", "0.00"),
        ("M2 multiple (spread / EL)", "=B18/B9", "0.00"),
    ]
    for i, (label, f, fmt) in enumerate(cells, start=3):
        lcalc.cell(i, 1, label)
        c = lcalc.cell(i, 2, f)
        c.number_format = fmt
        if f.startswith("=Attachment") or f.startswith("=Limit"):
            c.font = GREEN
    lcalc["C8"] = (
        "Uncapped: every event pays its own layer loss. The Python layer table caps the "
        "annual principal loss at the limit, so its EL is lower when a year has two large events."
    )
    lcalc["C12"] = "Spread range = regression output (90% prediction interval), not a quote."
    lcalc["B8"].comment = Comment(
        "SUMPRODUCT of min(max(L - A, 0), E - A) over the events on 'Loss curve', divided by "
        "N years times the limit.",
        "make_workbook.py",
    )
    r0 = 26
    lcalc.cell(
        r0 - 1, 1, "Check: five fixed layers, formulas (left) against Python (right)"
    ).font = BOLD
    heads = [
        "Attachment (USD m)",
        "Limit (USD m)",
        "AP (formula)",
        "EP occ. (formula)",
        "Event EL (formula)",
        "AP (Python)",
        "EP occ. (Python)",
        "Event EL (Python)",
        "Capped EL (Python)",
        "max abs difference",
    ]
    for j, h in enumerate(heads):
        lcalc.cell(r0, 1 + j, h).font = BOLD
    for i, (a, lim) in enumerate(CHECK_LAYERS_USD_M, start=r0 + 1):
        e = a + lim
        lcalc.cell(i, 1, a)
        lcalc.cell(i, 2, lim)
        lcalc.cell(i, 3, f'=COUNTIF(AnnualMax,">="&A{i})/N_years')
        lcalc.cell(i, 4, f'=COUNTIF(AnnualMax,">="&(A{i}+B{i}))/N_years')
        lcalc.cell(
            i,
            5,
            f"=SUMPRODUCT((EventLoss>A{i})*((EventLoss<A{i}+B{i})*(EventLoss-A{i})"
            f"+(EventLoss>=A{i}+B{i})*B{i}))/(N_years*B{i})",
        )
        m = layer_metrics(ev["year"], loss_m, n_years, a, e)
        lcalc.cell(i, 6, m.attachment_prob)
        lcalc.cell(i, 7, m.ep_occurrence)
        lcalc.cell(i, 8, event_layer_expected_loss(loss_m, n_years, a, e))
        lcalc.cell(i, 9, m.expected_loss)
        lcalc.cell(i, 10, f"=MAX(ABS(C{i}-F{i}),ABS(D{i}-G{i}),ABS(E{i}-H{i}))")
        for col in range(3, 10):
            lcalc.cell(i, col).number_format = "0.0000%"
        lcalc.cell(i, 10).number_format = "0.0E+00"
    lcalc.column_dimensions["A"].width = 52
    lcalc.column_dimensions["B"].width = 16
    for col in "CDEFGHIJ":
        lcalc.column_dimensions[col].width = 18

    # ---------------------------------------------------------------- Deals
    dl = wb.create_sheet("Deals")
    dl["A1"] = (
        "Public cat bond tranches (REAL data from Artemis deal pages, accessed 2026-11-05; "
        "see docs/deal_sources.md). 'in main sample' = one of the 56 distinct layers."
    )
    dl["A1"].font = BOLD
    cols = [
        "deal_name",
        "tranche",
        "sponsor",
        "issue_year",
        "size_usd_m",
        "trigger",
        "expected_loss_pct",
        "attachment_prob_pct",
        "spread_pct",
        "spread_basis",
        "rating",
        "source_url",
    ]
    flags = ["duplicate", "guidance", "discount", "launch_el"]
    head = cols + flags + ["in main sample"]
    for j, h in enumerate(head):
        dl.cell(3, 1 + j, h).font = BOLD
    main_col = [c for c in sample.columns if c.startswith("in: main")][0]
    for i, (r, s) in enumerate(
        zip(deals_all.itertuples(index=False), sample.itertuples(index=False), strict=True),
        start=4,
    ):
        rd = r._asdict()
        vals = [rd[c] for c in cols] + [bool(getattr(s, f)) for f in flags]
        vals.append(bool(sample.iloc[i - 4][main_col]))
        for j, v in enumerate(vals):
            dl.cell(i, 1 + j, None if (isinstance(v, float) and np.isnan(v)) else v)
    for j, w in enumerate([36, 12, 40, 10, 10, 18, 10, 10, 10, 10, 8, 60, 10, 10, 10, 10, 14]):
        dl.column_dimensions[chr(65 + j)].width = w

    # ---------------------------------------------------------------- Notes
    nt = wb.create_sheet("Notes")
    notes = [
        DISCLAIMER,
        "",
        "What the sheets compute",
        "Loss curve: values from the base-case event set (100,000 simulated years, seed 1). "
        "Hazard: Florida hurricane landfalls fitted to HURDAT2 1900-2019 (Poisson frequency, "
        "GPD intensity); Holland wind profile; Emanuel (2011) damage function; USD 100 bn "
        "hypothetical portfolio spread over county centroids by 2020 Census households.",
        "Layer calculator: per-occurrence layer [A, E]. AP = share of years whose largest event "
        "loss is at least A; EP (occurrence) = share of years whose largest event loss is at "
        "least E; event-layer EL = mean annual sum of min(max(L - A, 0), E - A) over events, "
        "divided by the limit, with no annual cap.",
        "Pricing: M1 log(spread) = a + b log(EL); M2 adds issue-year effects (2025 used). "
        "Fitted on the 56 distinct public layers; classical 90% prediction interval on the log "
        "scale. The spread range is a regression output, not a quote.",
        "",
        "Limits",
        "The loss model is teaching-grade: no surge, rain, terrain or building inventory, "
        "straight tracks, centroid exposure, losses not calibrated to any insured loss. "
        "Model-form choices move layer EL by factors of 2 to 7 (see results/transaction/"
        "sensitivity.csv).",
        "Deal ELs come from each deal's own vendor model at the offering stage; this model's EL "
        "is on a different scale, so putting it through the regression mixes two scales.",
        "The regression has year effects that absorb the market cycle; out-of-sample errors "
        "are about 3 percentage points of spread (results/pricing/cv.csv).",
        "",
        "Sources",
        "HURDAT2 (NOAA NHC; Landsea and Franklin 2013), via the timtrice/HURDAT GitHub mirror.",
        "Deal terms: Artemis deal directory pages (URLs on the Deals sheet).",
        "Households: UF BEBR table of 2020 Census households by county.",
        "Code, results and method: the cat-bond-issuance-model repository (README.md, "
        "docs/model.md, docs/pricing_protocol.md, docs/findings.md).",
        "Generated by scripts/make_workbook.py; recalculated with LibreOffice.",
    ]
    for i, t in enumerate(notes, start=1):
        c = nt.cell(i, 1, t)
        c.alignment = Alignment(wrap_text=True, vertical="top")
        if t in ("What the sheets compute", "Limits", "Sources"):
            c.font = BOLD
    nt["A1"].font = WARN
    nt.column_dimensions["A"].width = 120
    _style_all(wb)
    path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(path)


def recalc(path: pathlib.Path) -> None:
    """Recalculate ``path`` in place with LibreOffice headless (stores cached values)."""
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(
            ["soffice", "--headless", "--calc", "--convert-to", "xlsx", "--outdir", tmp, str(path)],
            check=True,
            capture_output=True,
            timeout=600,
        )
        shutil.move(str(pathlib.Path(tmp) / path.name), path)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--no-recalc", action="store_true")
    args = ap.parse_args()
    build(OUT)
    if not args.no_recalc and shutil.which("soffice"):
        recalc(OUT)
    print(f"wrote {OUT.relative_to(repo_root())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
