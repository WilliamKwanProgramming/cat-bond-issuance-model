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
  ``catbond.layers.event_layer_expected_loss``;
* expected loss with the annual cap, as in ``catbond.layers.layer_metrics``: the helper
  column of ``layer_loss_formula`` holds min(max(L_i - A, 0), L) for each event, the helper
  column of ``annual_capped_formula`` holds, for each year with a loss, the sum of that year's
  event layer losses capped at L (events of a year are in adjacent rows), and
  ``capped_el_formula`` is SUM(annual column) / (N_years L).

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


def layer_loss_formula(loss_cell: str, a: str, lim: str) -> str:
    """Layer loss of one event (USD m): min(max(L - A, 0), limit), L in cell ``loss_cell``."""
    return f"=MIN(MAX({loss_cell}-{a},0),{lim})"


def annual_capped_formula(first: str, last: str, lim: str) -> str:
    """Annual layer loss of one year: the event layer losses in ``first``:``last``, capped."""
    return f"=MIN(SUM({first}:{last}),{lim})"


def capped_el_formula(lim: str, annual: str = "AnnualCapped") -> str:
    """Expected loss as a fraction of the limit, with the annual cap (as ``layer_metrics``).

    ``annual`` is the range or defined name holding one capped annual layer loss per year with
    a loss; years without a loss contribute nothing to the sum.
    """
    return f"=SUM({annual})/(N_years*{lim})"
