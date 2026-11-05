"""Excel formula text for the layer calculator (scripts/make_workbook.py).

The formulas refer to the defined names ``EventLoss`` (event losses, USD m, events with a
loss), ``AnnualMax`` (largest event loss of each year with a loss) and ``N_years`` (simulated
years, including years without a loss). For a layer with attachment A and limit L (cell
references or expressions), E = A + L:

* attachment probability: COUNTIF(AnnualMax, ">=" & A) / N_years;
* occurrence exhaustion probability: COUNTIF(AnnualMax, ">=" & E) / N_years;
* event-layer EL (no annual cap):
  SUMPRODUCT((L_i > A) ((L_i < E)(L_i - A) + (L_i >= E) L)) / (N_years L),
  which is sum_i min(max(L_i - A, 0), L) / (N L), as in
  ``catbond.layers.event_layer_expected_loss``.

Only Excel-2007 functions are used, so LibreOffice evaluates them.
"""

from __future__ import annotations


def ap_formula(a: str) -> str:
    """Attachment probability for attachment ``a``."""
    return f'=COUNTIF(AnnualMax,">="&{a})/N_years'


def ep_formula(a: str, lim: str) -> str:
    """Occurrence exhaustion probability for attachment ``a`` and limit ``lim``."""
    return f'=COUNTIF(AnnualMax,">="&({a}+{lim}))/N_years'


def event_el_formula(a: str, lim: str) -> str:
    """Event-layer expected loss (fraction of the limit) for attachment ``a`` and limit ``lim``."""
    e = f"({a}+{lim})"
    return (
        f"=SUMPRODUCT((EventLoss>{a})*((EventLoss<{e})*(EventLoss-{a})"
        f"+(EventLoss>={e})*{lim}))/(N_years*{lim})"
    )
