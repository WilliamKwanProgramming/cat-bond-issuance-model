"""Tests for data/deals/cat_bond_deals.csv and for its validator scripts/validate_deals.py.

The first tests run the validator on the shipped table. The rest check the checker: each one
plants a single known defect in an otherwise valid copy of the table and asserts that the
validator reports it, so a validator that silently accepts bad data fails the suite.
"""

from __future__ import annotations

import csv
import datetime as dt
import importlib.util
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "validate_deals.py"
TABLE = ROOT / "data" / "deals" / "cat_bond_deals.csv"
TODAY = dt.date(2026, 11, 5)


def _load_validator() -> ModuleType:
    spec = importlib.util.spec_from_file_location("validate_deals", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


vd = _load_validator()


def _rows() -> list[dict[str, str]]:
    with TABLE.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write(path: Path, rows: list[dict[str, str]], header: tuple[str, ...] = vd.COLUMNS) -> Path:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(header), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_shipped_table_is_valid() -> None:
    assert vd.validate(TABLE) == []


def test_command_line_exit_codes(tmp_path: Path) -> None:
    ok = subprocess.run([sys.executable, str(SCRIPT), str(TABLE)], capture_output=True, text=True)
    assert ok.returncode == 0, ok.stdout
    rows = _rows()
    rows[0]["source_url"] = ""
    bad_path = _write(tmp_path / "bad.csv", rows)
    bad = subprocess.run(
        [sys.executable, str(SCRIPT), str(bad_path)], capture_output=True, text=True
    )
    assert bad.returncode == 1
    assert "source_url is empty" in bad.stdout


def test_every_row_has_a_source_and_a_number_pair() -> None:
    rows = _rows()
    assert all(r["source_url"].startswith("https://") for r in rows)
    assert all(r["expected_loss_pct"] and r["spread_pct"] for r in rows)


def test_valid_copy_passes(tmp_path: Path) -> None:
    assert vd.validate(_write(tmp_path / "copy.csv", _rows()), TODAY) == []


@pytest.mark.parametrize(
    ("column", "value", "expected"),
    [
        ("expected_loss_pct", "150", "outside (0, 100)"),
        ("expected_loss_pct", "-1", "outside (0, 100)"),
        ("expected_loss_pct", "1,5", "not a number"),
        ("spread_pct", "0.5", "below expected_loss_pct"),
        ("spread_pct", "75", "plausibility bound"),
        ("attachment_prob_pct", "0.01", "exceeds attachment_prob_pct"),
        ("exhaustion_prob_pct", "99", "exceeds expected_loss_pct"),
        ("size_usd_m", "0", "not positive"),
        ("issue_year", "2031", "issue_year"),
        ("issue_month", "13", "issue_month"),
        ("issue_month", "May", "issue_month"),
        ("spread_basis", "mid", "spread_basis"),
        ("spread_basis", "", "spread_basis"),
        ("trigger", "Indemnity", "trigger"),
        ("source_url", "", "source_url is empty"),
        ("source_url", "http://example.org/", "not an https URL"),
        ("access_date", "", "access_date is empty"),
        ("access_date", "05/11/2026", "not an ISO date"),
        ("access_date", "2026-11-31", "not an ISO date"),
        ("access_date", "2026-10-01", "is not between"),
        ("access_date", "2027-01-01", "is not between"),
        ("deal_name", " ", "deal_name is empty"),
    ],
)
def test_planted_defect_is_reported(tmp_path: Path, column: str, value: str, expected: str) -> None:
    rows = _rows()
    target = next(r for r in rows if r["attachment_prob_pct"] and r["spread_basis"] == "final")
    target[column] = value
    errors = vd.validate(_write(tmp_path / "bad.csv", rows), TODAY)
    assert any(expected in e for e in errors), errors


def test_basis_without_spread_is_reported(tmp_path: Path) -> None:
    rows = _rows()
    rows[0]["spread_pct"] = ""
    errors = vd.validate(_write(tmp_path / "bad.csv", rows), TODAY)
    assert any("spread_basis is set but spread_pct is empty" in e for e in errors), errors


def test_duplicate_tranche_is_reported(tmp_path: Path) -> None:
    rows = _rows()
    rows.append(dict(rows[3]))
    errors = vd.validate(_write(tmp_path / "dup.csv", rows), TODAY)
    assert any("duplicate of line 5" in e for e in errors), errors


def test_wrong_header_is_reported(tmp_path: Path) -> None:
    swapped = list(vd.COLUMNS)
    swapped[9], swapped[12] = swapped[12], swapped[9]
    path = _write(tmp_path / "hdr.csv", _rows(), header=tuple(swapped))
    errors = vd.validate(path, TODAY)
    assert len(errors) == 1 and "header" in errors[0]


def test_short_line_and_empty_file_are_reported(tmp_path: Path) -> None:
    path = _write(tmp_path / "short.csv", _rows())
    with path.open("a", encoding="utf-8") as handle:
        handle.write("only,three,fields\n")
    assert any("3 fields" in e for e in vd.validate(path, TODAY))
    empty = tmp_path / "empty.csv"
    empty.write_text("", encoding="utf-8")
    assert vd.validate(empty, TODAY) == [f"{empty}: file is empty"]
    header_only = _write(tmp_path / "header_only.csv", [])
    assert any("no data rows" in e for e in vd.validate(header_only, TODAY))


def test_non_utf8_is_reported(tmp_path: Path) -> None:
    path = tmp_path / "latin1.csv"
    path.write_bytes(",".join(vd.COLUMNS).encode() + b"\n\xe9\n")
    assert "not valid UTF-8" in vd.validate(path, TODAY)[0]


def test_gateway_re_2025_class_aaa_region_excludes_florida() -> None:
    # Artemis deal page (accessed 2026-11-05): the Class AAA notes cover named storm "across the
    # states of Alabama, Louisiana, Mississippi, New York, North Carolina, South Carolina and
    # Texas"; Florida is listed for Classes AA, A, C1 and C2 only.
    row = next(
        r
        for r in _rows()
        if r["deal_name"] == "Gateway Re Ltd. (Series 2025-1)" and r["tranche"] == "Class AAA"
    )
    assert "FL" not in row["region"] and "Florida" not in row["region"]
