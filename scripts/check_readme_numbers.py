#!/usr/bin/env python3
"""Check that the headline numbers in README.md and docs/findings.md are in the results files.

Each entry of CHECKS names a results file, a text that must occur in that file (a table row of
results/tables.md or a value in a CSV) and a text that must occur in the document. When a result
is regenerated and its number changes, the first test fails; when the document is edited by hand
and no longer quotes the number, the second fails. An optional fifth element is the number of
times the file text must occur. The script also fails when a relative link in README.md points
to a file that does not exist, and when a generated table pasted into README.md differs from the
same table in results/tables.md.

Usage: python3 scripts/check_readme_numbers.py      (exit status 1 on any problem)
Standard library only.
"""

# ruff: noqa: E501  (CHECKS quotes whole table rows; they are kept on one line to be greppable)
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
T = "results/tables.md"
README = "README.md"
FINDINGS = "docs/findings.md"

CHECKS = [
    # slope b, main sample (wild cluster bootstrap by series and by sponsor, HC3)
    (T, "| M1 | wild cluster bootstrap series (WCR, Rademacher) | 0.553 [0.094, 0.966] |",
     README, "0.553 [0.094, 0.966]"),
    (T, "| M2 | wild cluster bootstrap series (WCR, Rademacher) | 0.466 [0.115, 0.830] |",
     README, "0.466 [0.115, 0.830]"),
    (T, "| M2 | wild cluster bootstrap series (WCR, Rademacher) | 0.466 [0.115, 0.830] |",
     FINDINGS, "| M2: + issue-year effects | 0.466 | [0.115, 0.830] | [0.146, 0.871] | [0.128, 0.805] |"),
    (T, "| M1 | wild cluster bootstrap sponsor (WCR, Rademacher) | 0.553 [0.167, 1.155] | 0.1573 |",
     FINDINGS, "| M1: log EL only | 0.553 | [0.094, 0.966] | [0.167, 1.155] | [0.179, 0.926] |"),
    (T, "| M1 | wild cluster bootstrap sponsor (WCR, Rademacher) | 0.553 [0.167, 1.155] | 0.1573 |",
     FINDINGS, "(p = 0.1573)"),
    (T, "| M3 | HC3, t(n-k) | 0.492 [0.141, 0.844] |", FINDINGS, "| 0.492 | not computed"),
    (T, "| M3 | CR1 series, t(G-1) | 0.492 [0.174, 0.810] |", FINDINGS, "[0.174, 0.810]"),
    # pre-specified rule in every sample
    (T, "| main (56 distinct layers) | 0.553 [0.094, 0.966] | 0.466 [0.115, 0.830] | yes |",
     FINDINGS, "| main (56) | [0.094, 0.966] | [0.115, 0.830] | yes |"),
    (T, "| without guidance prices | 0.619 [0.169, 1.016] | 0.491 [0.112, 0.896] | no |",
     FINDINGS, "| without 3 guidance prices (53) | [0.169, 1.016] | [0.112, 0.896] | no |"),
    (T, "| without discount notes | 0.635 [0.174, 1.033] | 0.510 [0.163, 0.877] | no |",
     FINDINGS, "| without 4 discount notes (52) | [0.174, 1.033] | [0.163, 0.877] | no |"),
    (T, "| 0.556 [0.094, 0.972] | 0.510 [0.171, 0.881] | yes |",
     FINDINGS, "| without 2 launch-plan EL rows (54) | [0.094, 0.972] | [0.171, 0.881] | yes |"),
    (T, "| without all flagged rows | 0.703 [0.262, 1.084] | 0.547 [0.160, 0.944] | no |",
     FINDINGS, "| without all 8 flagged rows (48) | [0.262, 1.084] | [0.160, 0.944] | no |"),
    # fit statistics and out-of-sample errors
    (T, "| main (56 distinct layers) | M1 | 56 | 2 | 37 | 14 | 0.380 | 0.208 |", README, "0.208 for M1"),
    (T, "| main (56 distinct layers) | M2 | 56 | 10 | 37 | 14 | 0.258 | 0.691 |", README, "0.691 with year"),
    (T, "| main (56 distinct layers) | M3 | 56 | 12 | 37 | 14 | 0.253 | 0.714 |", FINDINGS, "0.253 for M3"),
    (T, "| main (56 distinct layers) | M2 | series | 56 | 56 | 0 | 0.322 | 0.268 | 3.02 | 2.14 | 1.45 |",
     README, "RMSE of 3.02 percentage points (0.322 on the log scale)"),
    (T, "| main (56 distinct layers) | M2 | sponsor | 56 | 56 | 0 | 0.367 | 0.303 | 3.20 |",
     README, "3.20 points"),
    (T, "| main (56 distinct layers) | M1 | series | 56 | 56 | 0 | 0.398 | 0.322 | 3.33 |",
     FINDINGS, "M1: 0.398 and 3.33 points"),
    (T, "| 2018 | 4 | 1.62 | 3.88 | 2.36 |", FINDINGS, "2.36\n(2018)"),
    (T, "| 2024 | 9 | 1.51 | 10.50 | 6.67 |", FINDINGS, "6.67 (2024)"),
    (T, "| 2025 | 23 | 1.79 | 8.00 | 4.85 |", FINDINGS, "4.85 in 2025"),
    # hypothetical transaction
    (T, "| Layer 1 (bottom) | 0.50 - 1.00 | 500 - 1,000 | 5.67 | 2.70 | 3.90 (0.06) | 2.19 - 5.34 | 16.1 |",
     FINDINGS, "| bottom, USD 500-1,000 million | 5.67 | 2.70 | 3.90 | 2.19 - 5.34 |"),
    (T, "| Layer 2 (middle) | 1.00 - 1.50 | 1,000 - 1,500 | 2.69 | 1.45 | 1.98 (0.04) | 0.97 - 2.85 | 7.8 |",
     FINDINGS, "| middle, USD 1,000-1,500 million | 2.69 | 1.45 | 1.98 | 0.97 - 2.85 |"),
    (T, "| Layer 3 (top) | 1.50 - 2.00 | 1,500 - 2,000 | 1.44 | 0.82 | 1.09 (0.03) | 0.44 - 1.59 | 4.3 |",
     FINDINGS, "| top, USD 1,500-2,000 million | 1.44 | 0.82 | 1.09 | 0.44 - 1.59 |"),
    (T, "| Layer 2 (middle) | M2 | EL point | 1.98 | 8.79 [5.65, 13.68] | 4.45 |",
     FINDINGS, "| middle | 7.88 [4.14, 15.00] | 8.79 [5.65, 13.68] | 6.32 - 10.42 |"),
    (T, "| Layer 1 (bottom) | M2 | EL point | 3.90 | 12.07 [7.61, 19.16] | 3.09 |",
     FINDINGS, "| bottom | 11.47 [5.84, 22.54] | 12.07 [7.61, 19.16] | 9.22 - 13.97 |"),
    (T, "| Layer 3 (top) | M2 | EL point | 1.09 | 6.67 [4.25, 10.47] | 6.10 |",
     FINDINGS, "| top | 5.68 [2.96, 10.89] | 6.67 [4.25, 10.47] | 4.37 - 7.93 |"),
    (T, "| Layer 2 (middle) | V_half 110 kt (Emanuel 2011 sensitivity) | True | 10.32 |",
     README, "middle-layer EL of 10.32%"),
    (T, "| Layer 2 (middle) | window 1950-2019 | True | 1.04 | 5.52 | 6.51 | 0.52 | -2.36 | -2.28 |",
     FINDINGS, "| calibration window 1950-2019 | 2.33 | 1.04 | 0.43 | -2.28 |"),
    (T, "| Layer 1 (bottom) | window 1950-2019 | True | 2.33 | 8.62 | 9.49 | 0.60 |", README,
     "0.39 to 0.60 times"),
    (T, "| Layer 3 (top) | window 1950-2019 | True | 0.43 | 3.39 | 4.31 | 0.39 |", README,
     "0.39 to 0.60 times"),
    # hazard and losses
    (T, "| All hurricanes | 25.0 | 17 | 33 | 15 | 36 | 17 |", README, "25.0 hurricanes for 1980-2019 against 17 observed (p = 0.21)"),
    (T, "| OEP | 100 | 1.814 | 0.027 |", README, "1.814% of TIV"),
    # mutation check
    ("results/mutation_check.txt", "37 of 37 mutants caught", README, "37 of 37 mutants caught"),
]


def broken_links(readme: str) -> list[str]:
    """Relative links in the README whose target file or directory does not exist."""
    missing = []
    for target in re.findall(r"\]\(([^)\s]+)\)", readme):
        if target.startswith(("http://", "https://", "#", "mailto:")):
            continue
        if not (ROOT / target.split("#", 1)[0]).exists():
            missing.append(target)
    return missing


def pasted_tables_differ(readme: str, tables: str) -> list[str]:
    """Generated tables (blocks that end with 'Source: `results/...`') that differ from tables.md."""
    bad = []
    for block in re.findall(r"#### (.*?)\n\n(\|.*?\n)\nSource: `results/", readme, flags=re.S):
        title, body = block
        if f"### {title}\n\n{body}" not in tables:
            bad.append(title)
    return bad


def main() -> int:
    texts: dict[str, str] = {}

    def text(path: str) -> str | None:
        if path not in texts:
            try:
                texts[path] = (ROOT / path).read_text(encoding="utf-8")
            except FileNotFoundError:
                return None
        return texts[path]

    problems = 0
    readme = text(README) or ""
    for target in broken_links(readme):
        print(f"README.md links to {target!r}, which does not exist")
        problems += 1
    for title in pasted_tables_differ(readme, text(T) or ""):
        print(f"README.md table {title!r} differs from {T}")
        problems += 1
    for check in CHECKS:
        path, in_file, doc, in_doc = check[:4]
        expected = check[4] if len(check) > 4 else None
        src = text(path)
        if src is None:
            print(f"{path}: file not found")
            problems += 1
            continue
        found = src.count(in_file)
        if found == 0 or (expected is not None and found != expected):
            print(f"{path}: {in_file!r} occurs {found} times")
            problems += 1
        if in_doc not in (text(doc) or ""):
            print(f"{doc} does not contain {in_doc!r} (checked against {path})")
            problems += 1
    print(f"{len(CHECKS)} checks, {problems} problems")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
