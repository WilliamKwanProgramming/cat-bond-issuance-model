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
MODEL = "docs/model.md"

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
    (T, "| main (56 distinct layers) | M1 | 56 | 2 | 37 | 14 | 0.380 | 0.208 |", FINDINGS, "0.208 for M1"),
    (T, "| main (56 distinct layers) | M2 | 56 | 10 | 37 | 14 | 0.258 | 0.691 |", FINDINGS, "0.691 for M2"),
    # post hoc diagnostics (results/pricing_posthoc, tables labelled POST HOC in results/tables.md)
    (T, "| issue-year effects only | 56 | 9 | 0.562 | 0.487 |", README, "issue-year effects alone 0.562, adjusted 0.487"),
    (T, "| log EL only (M1) | 56 | 2 | 0.208 | 0.193 |", README, "log EL alone 0.208, adjusted 0.193"),
    (T, "| log EL and issue-year effects (M2) | 56 | 10 | 0.691 | 0.630 |", README, "both\n0.691"),
    (T, "| issue-year effects only | 56 | 9 | 0.562 | 0.487 |", FINDINGS, "alone give 0.562 (adjusted 0.487)"),
    (T, "| log EL only (M1) | 56 | 2 | 0.208 | 0.193 |", FINDINGS, "log EL alone 0.208 (adjusted 0.193)"),
    (T, "| M1 | wild cluster bootstrap sponsor (WCR, Rademacher) | 0.553 [0.167, 1.155] | 0.1573 |", README, "[0.167, 1.155] (p = 0.157)"),
    (T, "| Florida only | 1 | 20 | 21 |", README, "Only 21 of the 56 layers list Florida alone"),
    (T, "| no Florida listed | 13 | 6 | 19 |", README, "19 list\nno Florida"),
    (T, "| all | 15 | 41 | 56 |", README, "15 cover perils besides named storm"),
    (T, "| Florida only | 1 | 20 | 21 |", FINDINGS, "layers 21 list only Florida"),
    (T, "| Florida and other states | 0 | 15 | 15 |", FINDINGS, "15 name Florida among other states"),
    (T, "| no Florida listed | 13 | 6 | 19 |", FINDINGS, "19 list no Florida (Texas 9,"),
    (T, "| M2 + region group | 56 (37) | 0.501 [0.251, 0.751] |", FINDINGS, "leaves b at 0.50 (CR1 series interval [0.251,"),
    (T, "region[no Florida listed] -0.423 (0.065)", FINDINGS, "0.42 log points (about 35%) below the layers"),
    (T, "region[no Florida listed] -0.423 (0.065)", FINDINGS, "(CR1 se 0.065)"),
    (T, "| M2 + multi-peril indicator | 56 (37) | 0.508 [0.209, 0.808] | [0.190, 0.827] | peril[named storm only] 0.356 (0.067) |", FINDINGS, "gives b = 0.51, and its coefficient (named-storm-only layers 0.36 above) is -0.02"),
    (T, "peril[named storm only] -0.023 (0.071)", FINDINGS, "is -0.02 once the"),
    (T, "| M1, Florida-only layers | 21 (14) | 0.333 [-0.159, 0.826] |", FINDINGS, "M1 gives b = 0.33 [-0.16, 0.83]"),
    (T, "| Layer 1 (bottom) | M2 + region group (post hoc) | 3.90 | 13.27 [9.84, 17.89] | 3.40 |", FINDINGS, "13.27, 9.44 and 7.02 against 12.07, 8.79 and"),
    (T, "| Layer 2 (middle) | M2 + region group (post hoc) | 1.98 | 9.44 [7.10, 12.55] | 4.77 |", README, "middle layer 9.44 against 8.79"),
    (T, "| Layer 3 (top) | M2 + region group (post hoc) | 1.09 | 7.02 [5.26, 9.37] | 6.42 |", FINDINGS, "13.27, 9.44 and 7.02 against 12.07, 8.79 and"),
    (T, "| M1 | 0.553 | 1 | 0.1200 | 0.232 |", FINDINGS, "an error SD of 0.23 in log EL"),
    (T, "| M2 | 0.466 | 1 | 0.1046 | 0.236 |", FINDINGS, "the estimated 0.55 (M1) and 0.47 (M2)"),
    (T, "| M1 | 0.553 | 1 | 0.1200 | 0.232 |", README, "an error of SD 0.23 in log EL"),
    (T, "| 0.20 | 1.000 | 0.999 | 0.675 | 0.679 |", FINDINGS, "mean slope 1.00 at every error SD, against 0.68 for"),
    (T, "| 0.25 | 0.996 | 0.996 | 0.486 | 0.494 |", FINDINGS, "mean slope 1.00 at every error SD, against 0.68 for"),
    # near-miss hurricanes (results/hazard/near_miss_hurricanes.csv)
    (T, "(7 storms, against 67 counted landfalls;", README, "the 67 hurricane-strength landfalls"),
    (T, "(7 storms, against 67 counted landfalls;", README, "7 hurricanes that came within 50 km"),
    (T, "| IVAN | 2004 | 30.2 | -87.9 | 105 | L | 38 | 3 |", README, "Ivan 2004 and Matthew 2016 at 105 kt"),
    (T, "| MATTHEW | 2016 | 28.9 | -80.3 | 105 |  | 42 | 230 |", README, "Ivan 2004 and Matthew 2016 at 105 kt"),
    (T, "| AGNES | 1972 | 29.6 | -85.6 | 65 |  | 24 | 200 |", README, "Agnes 1972"),
    (T, "(7 storms, against 67 counted landfalls;", FINDINGS, "Ivan 2004,\n  Matthew 2016 and five other hurricanes within 50 km"),
    (T, "| 1900-2019 (primary) | 120 | 67 | 0.558 | 0.433 | 0.709 | 0.652 | 0.102 |", FINDINGS, "(dispersion test p = 0.102)"),
    (T, "| 1950-2019 | 70 | 32 | 0.457 | 0.313 | 0.645 | 0.658 | 0.010 |", FINDINGS, "(p = 0.010) and selects a negative binomial"),
    (T, "| AAL |  | 0.101 | 0.001 |", FINDINGS, "(0.101% of TIV)"),
    # sensitivity: frequency ratios and Monte Carlo errors quoted in findings.md
    (T, "| Layer 1 (bottom) | frequency x 0.8 | True | 3.15 | 10.19 | 10.92 | 0.81 |", FINDINGS, "(x 0.8 gives 0.81 to 0.83 of base"),
    (T, "| Layer 2 (middle) | frequency x 0.8 | True | 1.64 | 7.09 | 8.05 | 0.83 |", FINDINGS, "(x 0.8 gives 0.81 to 0.83 of base"),
    (T, "| Layer 3 (top) | frequency x 1.2 | True | 1.25 | 6.12 | 7.10 | 1.14 |", FINDINGS, "x 1.2 gives 1.14"),
    (T, "| Layer 1 (bottom) | frequency x 1.2 | True | 4.57 | 12.51 | 12.99 | 1.17 |", FINDINGS, "to 1.17), the calibration window"),
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
    # sensitivity table of docs/findings.md, every cell against results/tables.md
    (T, "| Layer 1 (bottom) | base | True | 3.90 |", FINDINGS, "| base | 3.90 | 1.98 | 1.09 | 0.00 |"),
    (T, "| Layer 2 (middle) | base | True | 1.98 | 7.88 | 8.79 | 1.00 | 0.00 | 0.00 |", FINDINGS, "| base | 3.90 | 1.98 | 1.09 | 0.00 |"),
    (T, "| Layer 3 (top) | base | True | 1.09 |", FINDINGS, "| base | 3.90 | 1.98 | 1.09 | 0.00 |"),
    (T, "| Layer 1 (bottom) | frequency x 0.8 | True | 3.15 |", FINDINGS, "| frequency x 0.8 | 3.15 | 1.64 | 0.91 | -0.75 |"),
    (T, "| Layer 2 (middle) | frequency x 0.8 | True | 1.64 | 7.09 | 8.05 | 0.83 | -0.78 | -0.75 |", FINDINGS, "| frequency x 0.8 | 3.15 | 1.64 | 0.91 | -0.75 |"),
    (T, "| Layer 3 (top) | frequency x 0.8 | True | 0.91 |", FINDINGS, "| frequency x 0.8 | 3.15 | 1.64 | 0.91 | -0.75 |"),
    (T, "| Layer 1 (bottom) | frequency x 1.2 | True | 4.57 |", FINDINGS, "| frequency x 1.2 | 4.57 | 2.29 | 1.25 | 0.63 |"),
    (T, "| Layer 2 (middle) | frequency x 1.2 | True | 2.29 | 8.55 | 9.42 | 1.16 | 0.67 | 0.63 |", FINDINGS, "| frequency x 1.2 | 4.57 | 2.29 | 1.25 | 0.63 |"),
    (T, "| Layer 3 (top) | frequency x 1.2 | True | 1.25 |", FINDINGS, "| frequency x 1.2 | 4.57 | 2.29 | 1.25 | 0.63 |"),
    (T, "| Layer 1 (bottom) | window 1950-2019 | True | 2.33 |", FINDINGS, "| calibration window 1950-2019 | 2.33 | 1.04 | 0.43 | -2.28 |"),
    (T, "| Layer 2 (middle) | window 1950-2019 | True | 1.04 | 5.52 | 6.51 | 0.52 | -2.36 | -2.28 |", FINDINGS, "| calibration window 1950-2019 | 2.33 | 1.04 | 0.43 | -2.28 |"),
    (T, "| Layer 3 (top) | window 1950-2019 | True | 0.43 |", FINDINGS, "| calibration window 1950-2019 | 2.33 | 1.04 | 0.43 | -2.28 |"),
    (T, "| Layer 1 (bottom) | V_half 145.2 kt (CLIMADA default) | True | 4.67 |", FINDINGS, "| V_half 145.2 kt | 4.67 | 2.51 | 1.44 | 1.03 |"),
    (T, "| Layer 2 (middle) | V_half 145.2 kt (CLIMADA default) | True | 2.51 | 8.98 | 9.82 | 1.27 | 1.10 | 1.03 |", FINDINGS, "| V_half 145.2 kt | 4.67 | 2.51 | 1.44 | 1.03 |"),
    (T, "| Layer 3 (top) | V_half 145.2 kt (CLIMADA default) | True | 1.44 |", FINDINGS, "| V_half 145.2 kt | 4.67 | 2.51 | 1.44 | 1.03 |"),
    (T, "| Layer 1 (bottom) | V_thresh 40 kt | True | 6.11 |", FINDINGS, "| V_thresh 40 kt | 6.11 | 3.24 | 1.83 | 2.27 |"),
    (T, "| Layer 2 (middle) | V_thresh 40 kt | True | 3.24 | 10.34 | 11.06 | 1.64 | 2.47 | 2.27 |", FINDINGS, "| V_thresh 40 kt | 6.11 | 3.24 | 1.83 | 2.27 |"),
    (T, "| Layer 3 (top) | V_thresh 40 kt | True | 1.83 |", FINDINGS, "| V_thresh 40 kt | 6.11 | 3.24 | 1.83 | 2.27 |"),
    (T, "| Layer 1 (bottom) | V_half 110 kt (Emanuel 2011 sensitivity) | True | 13.98 |", FINDINGS, "| V_half 110 kt | 13.98 | 10.32 | 7.95 | 10.21 |"),
    (T, "| Layer 2 (middle) | V_half 110 kt (Emanuel 2011 sensitivity) | True | 10.32 | 19.63 | 19.00 | 5.22 | 11.75 | 10.21 |", FINDINGS, "| V_half 110 kt | 13.98 | 10.32 | 7.95 | 10.21 |"),
    (T, "| Layer 3 (top) | V_half 110 kt (Emanuel 2011 sensitivity) | True | 7.95 |", FINDINGS, "| V_half 110 kt | 13.98 | 10.32 | 7.95 | 10.21 |"),
    # landfall definitions: geometric rule on 6-hourly fixes only
    (T, "| record vs geometric on synoptic fixes only, AL.rda, years with L flags | 62 | 58 | 58 | 6.3 | 21.9 | mean wind difference (geometric - record) -4.0 kt;",
     MODEL, "it finds 58 of the 62, with landfall winds\n4.0 kt lower on average (up to 21.9 kt)"),
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
