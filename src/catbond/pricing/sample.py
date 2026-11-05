"""Regression sample and flagged rows of the public deal table (docs/pricing_protocol.md).

The 58 rows of data/deals/cat_bond_deals.csv are 56 distinct layers: Integrity Re III 2025-1
Classes A1/A2 and B1/B2 duplicate each other (same expected loss, attachment probability and
spread). The first-listed class of each pair (A1, B1) is kept. Flagged rows, removed in the
pre-specified sensitivity samples:

* ``guidance``: the price is the last guidance, not a confirmed final price (spread_basis);
* ``discount``: discount or zero-coupon notes whose price is 100 minus the issue price;
* ``launch_el``: the expected loss was stated for the launch plan of a deal that was then
  restructured (Alamo Re 2021-1 and 2023-1).

Every listed row must occur exactly once, so a changed table fails loudly instead of silently
changing the sample.
"""

from __future__ import annotations

import pandas as pd

DUPLICATES = {  # dropped row -> the kept row it duplicates
    ("Integrity Re III Ltd. (Series 2025-1)", "Class A2"): (
        "Integrity Re III Ltd. (Series 2025-1)",
        "Class A1",
    ),
    ("Integrity Re III Ltd. (Series 2025-1)", "Class B2"): (
        "Integrity Re III Ltd. (Series 2025-1)",
        "Class B1",
    ),
}
DISCOUNT_ROWS = (
    ("Everglades Re II Ltd. (Series 2023-1)", "Class A"),
    ("Everglades Re II Ltd. (Series 2023-2)", "Class A"),
    ("Gateway Re Ltd. (Series 2025-1)", "Class AA"),
    ("Gateway Re Ltd. (Series 2025-1)", "Class C1"),
)
LAUNCH_EL_ROWS = (
    ("Alamo Re Ltd. (Series 2021-1)", "Class A"),
    ("Alamo Re Ltd. (Series 2023-1)", "Class A"),
)
N_GUIDANCE = 3
LAYER_FIELDS = ("expected_loss_pct", "attachment_prob_pct", "spread_pct")

# pre-specified samples: name -> flags whose rows are removed from the 56 distinct layers
SAMPLES = {
    "main (56 distinct layers)": (),
    "without guidance prices": ("guidance",),
    "without discount notes": ("discount",),
    "without launch-plan EL (Alamo Re 2021-1, 2023-1)": ("launch_el",),
    "without all flagged rows": ("guidance", "discount", "launch_el"),
}


def _key(df: pd.DataFrame) -> pd.Series:
    return pd.Series(list(zip(df["deal_name"], df["tranche"], strict=True)), index=df.index)


def _locate(df: pd.DataFrame, keys) -> pd.Series:
    k = _key(df)
    mask = k.isin(list(keys))
    for key in keys:
        if int((k == key).sum()) != 1:
            raise ValueError(f"row {key} occurs {int((k == key).sum())} times, expected once")
    return mask


def flag_rows(deals: pd.DataFrame) -> pd.DataFrame:
    """Copy of ``deals`` with boolean columns duplicate, guidance, discount and launch_el."""
    df = deals.copy()
    dup = _locate(df, DUPLICATES)
    k = _key(df)
    for drop, keep in DUPLICATES.items():
        a = df[k == drop].iloc[0]
        b = df[k == keep].iloc[0]
        for f in LAYER_FIELDS:
            same = (pd.isna(a[f]) and pd.isna(b[f])) or a[f] == b[f]
            if not same:
                raise ValueError(f"{drop} and {keep} differ in {f}; not a duplicate layer")
    df["duplicate"] = dup
    df["guidance"] = df["spread_basis"].astype(str).str.strip() == "guidance"
    if int(df["guidance"].sum()) != N_GUIDANCE:
        raise ValueError(f"expected {N_GUIDANCE} guidance rows, found {int(df['guidance'].sum())}")
    df["discount"] = _locate(df, DISCOUNT_ROWS)
    df["launch_el"] = _locate(df, LAUNCH_EL_ROWS)
    return df


def regression_sample(deals: pd.DataFrame, drop_flags: tuple[str, ...] = ()) -> pd.DataFrame:
    """The distinct layers without the rows carrying any of ``drop_flags``."""
    df = flag_rows(deals)
    keep = ~df["duplicate"]
    for f in drop_flags:
        keep &= ~df[f]
    return df[keep].reset_index(drop=True)
