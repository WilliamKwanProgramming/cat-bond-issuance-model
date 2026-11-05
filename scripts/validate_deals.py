"""Validate the hand-collected catastrophe bond tranche table.

Usage (from the repository root)::

    python scripts/validate_deals.py [path/to/cat_bond_deals.csv]

Prints one line per violation and exits with status 1 if there is any violation, 0 otherwise.
The checks are structural and logical only; they cannot tell whether a value matches its
source page (that is what the re-verification recorded in docs/deal_sources.md is for).
"""

from __future__ import annotations

import csv
import datetime as dt
import sys
from pathlib import Path

DEFAULT_PATH = Path(__file__).resolve().parent.parent / "data" / "deals" / "cat_bond_deals.csv"

COLUMNS: tuple[str, ...] = (
    "deal_name",
    "tranche",
    "sponsor",
    "issue_year",
    "issue_month",
    "size_usd_m",
    "peril",
    "region",
    "trigger",
    "expected_loss_pct",
    "attachment_prob_pct",
    "exhaustion_prob_pct",
    "spread_pct",
    "spread_basis",
    "rating",
    "source_url",
    "access_date",
    "notes",
)
REQUIRED: tuple[str, ...] = (
    "deal_name",
    "tranche",
    "sponsor",
    "issue_year",
    "issue_month",
    "peril",
    "region",
    "trigger",
    "source_url",
    "access_date",
)
PERCENT_COLUMNS: tuple[str, ...] = (
    "expected_loss_pct",
    "attachment_prob_pct",
    "exhaustion_prob_pct",
    "spread_pct",
)
TRIGGERS: frozenset[str] = frozenset(
    {"indemnity", "industry loss index", "parametric", "modelled loss"}
)
SPREAD_BASES: frozenset[str] = frozenset({"final", "guidance"})
MIN_YEAR = 1997
MAX_SPREAD_PCT = 50.0
FIRST_ACCESS_DATE = dt.date(2026, 11, 5)


def _number(text: str) -> float | None:
    """Return ``text`` as a float, ``None`` if it is empty; raise ValueError if malformed."""
    text = text.strip()
    if text == "":
        return None
    return float(text)


def check_row(row: dict[str, str], line: int, today: dt.date) -> list[str]:
    """Return the violations found in one data row.

    ``line`` is the 1-based line number in the file (the header is line 1). Rules: required
    fields non-empty; issue year in [1997, today.year] and month in [1, 12]; size > 0;
    every percentage in (0, 100) and the spread at most 50; ordering
    exhaustion_prob <= expected_loss <= attachment_prob and expected_loss <= spread
    (a layer's expected loss lies between its exhaustion and attachment probabilities, and a
    bond is not issued below its expected loss); spread_basis is ``final`` or ``guidance``
    exactly when a spread is given; source_url is an https URL; access_date is an ISO date
    between 2026-11-05 and ``today``.
    """
    errors: list[str] = []

    def bad(message: str) -> None:
        errors.append(f"line {line}: {message}")

    for col in REQUIRED:
        if row.get(col, "").strip() == "":
            bad(f"{col} is empty")

    for col, low, high in (("issue_year", MIN_YEAR, today.year), ("issue_month", 1, 12)):
        text = row.get(col, "").strip()
        if text and (not text.isdigit() or not low <= int(text) <= high):
            bad(f"{col}={text!r} is not an integer in [{low}, {high}]")

    values: dict[str, float | None] = {}
    for col in ("size_usd_m", *PERCENT_COLUMNS):
        try:
            values[col] = _number(row.get(col, ""))
        except ValueError:
            bad(f"{col}={row.get(col)!r} is not a number")
            values[col] = None

    size = values["size_usd_m"]
    if size is not None and size <= 0:
        bad(f"size_usd_m={size} is not positive")
    for col in PERCENT_COLUMNS:
        value = values[col]
        if value is not None and not 0 < value < 100:
            bad(f"{col}={value} is outside (0, 100)")

    el = values["expected_loss_pct"]
    attach = values["attachment_prob_pct"]
    exhaust = values["exhaustion_prob_pct"]
    spread = values["spread_pct"]
    if spread is not None and spread > MAX_SPREAD_PCT:
        bad(f"spread_pct={spread} exceeds the plausibility bound {MAX_SPREAD_PCT}")
    if el is not None and spread is not None and spread < el:
        bad(f"spread_pct={spread} is below expected_loss_pct={el}")
    if el is not None and attach is not None and el > attach:
        bad(f"expected_loss_pct={el} exceeds attachment_prob_pct={attach}")
    if el is not None and exhaust is not None and exhaust > el:
        bad(f"exhaustion_prob_pct={exhaust} exceeds expected_loss_pct={el}")

    basis = row.get("spread_basis", "").strip()
    if spread is not None and basis not in SPREAD_BASES:
        bad(f"spread_basis={basis!r} must be one of {sorted(SPREAD_BASES)} when spread is given")
    if spread is None and basis:
        bad("spread_basis is set but spread_pct is empty")

    trigger = row.get("trigger", "").strip()
    if trigger and trigger not in TRIGGERS:
        bad(f"trigger={trigger!r} is not one of {sorted(TRIGGERS)}")

    url = row.get("source_url", "").strip()
    if url and not url.startswith("https://"):
        bad(f"source_url={url!r} is not an https URL")

    access = row.get("access_date", "").strip()
    if access:
        try:
            day = dt.date.fromisoformat(access)
        except ValueError:
            bad(f"access_date={access!r} is not an ISO date (YYYY-MM-DD)")
        else:
            if len(access) != 10 or not FIRST_ACCESS_DATE <= day <= today:
                bad(f"access_date={access} is not between {FIRST_ACCESS_DATE} and {today}")
    return errors


def validate(path: Path, today: dt.date | None = None) -> list[str]:
    """Return every violation in the CSV at ``path`` (an empty list means the file is valid).

    Checks the UTF-8 encoding, the exact header (column names and order), the number of fields
    per line, the per-row rules of :func:`check_row`, and duplicate (deal_name, tranche) keys.
    """
    today = today or dt.date.today()
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        return [f"{path}: not valid UTF-8 ({exc})"]
    reader = csv.reader(text.splitlines())
    header = next(reader, None)
    if header is None:
        return [f"{path}: file is empty"]
    if tuple(header) != COLUMNS:
        return [f"line 1: header {header} does not match the schema {list(COLUMNS)}"]

    errors: list[str] = []
    seen: dict[tuple[str, str], int] = {}
    n_rows = 0
    for line, fields in enumerate(reader, start=2):
        if len(fields) != len(COLUMNS):
            errors.append(f"line {line}: {len(fields)} fields, expected {len(COLUMNS)}")
            continue
        n_rows += 1
        row = dict(zip(COLUMNS, fields, strict=True))
        errors.extend(check_row(row, line, today))
        key = (row["deal_name"].strip(), row["tranche"].strip())
        if key in seen:
            errors.append(f"line {line}: duplicate of line {seen[key]} for {key}")
        else:
            seen[key] = line
    if n_rows == 0:
        errors.append(f"{path}: no data rows")
    return errors


def main(argv: list[str]) -> int:
    """Validate the file named in ``argv`` (default: the repository table); return exit code."""
    path = Path(argv[1]) if len(argv) > 1 else DEFAULT_PATH
    errors = validate(path)
    for message in errors:
        print(message)
    if errors:
        print(f"{path.name}: {len(errors)} violation(s)")
        return 1
    print(f"{path.name}: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
