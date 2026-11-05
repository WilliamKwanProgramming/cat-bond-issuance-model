#!/usr/bin/env python3
"""Mutation check for the catbond test suite: do the tests notice deliberate bugs?

Copies src/ and tests/ into a temporary directory (data/ is linked, not copied), then applies ONE
deliberate bug ("mutant") at a time to the copy and runs pytest on it with the copied package
first on the import path. A mutant is "caught" if at least one test fails; pytest stops at the
first failure (-x), so the test named is the first that caught it. The working tree is never
modified. tests/test_deals_data.py (the deal-table validator, maintained separately) is not
part of the run. Exits with status 1 if a mutant survives or the unmutated baseline fails.

Usage: python3 tests/mutation_check.py [--out results/mutation_check.txt]
"""

from __future__ import annotations

import argparse
import datetime
import os
import pathlib
import platform
import shlex
import shutil
import subprocess
import sys
import tempfile
import time

REPO = pathlib.Path(__file__).resolve().parents[1]
S = "src/catbond/"

# (name, [(file, text that must occur exactly once, replacement), ...])
MUTANTS = [
    (
        "M01 annual counts: calibration window one year short (Poisson mean off by one year)",
        [
            (
                S + "hazard/landfall.py",
                "years = np.arange(first_year, last_year + 1)",
                "years = np.arange(first_year, last_year)",
            )
        ],
    ),
    (
        "M02 dispersion test: n degrees of freedom instead of n - 1",
        [
            (
                S + "hazard/frequency.py",
                "d, n - 1, float(stats.chi2.sf(d, n - 1))",
                "d, n, float(stats.chi2.sf(d, n))",
            )
        ],
    ),
    (
        "M03 negative binomial sampling: gamma scale mean instead of mean / k",
        [
            (
                S + "hazard/frequency.py",
                "rng.gamma(nb_size, mean / nb_size, size=n_years)",
                "rng.gamma(nb_size, mean, size=n_years)",
            )
        ],
    ),
    (
        # An earlier M04 (profile log-likelihood shifted by the constant -n) survived because it
        # is equivalent: a constant shift leaves the maximiser unchanged. Replaced by this one.
        "M04 GPD profile likelihood: log-likelihood uses log(sigma) with sigma = xi * theta",
        [
            (
                S + "hazard/intensity.py",
                "    sigma = xi / theta\n    return -n * np.log(sigma)",
                "    sigma = xi * theta\n    return -n * np.log(sigma)",
            )
        ],
    ),
    (
        "M05 Holland profile: square root dropped",
        [
            (
                S + "hazard/windfield.py",
                "return np.sqrt(x * np.exp(1.0 - x))",
                "return x * np.exp(1.0 - x)",
            )
        ],
    ),
    (
        "M06 Willoughby-Rahn RMW: knots used where the fit expects m/s",
        [
            (
                S + "hazard/windfield.py",
                "v_ms = np.asarray(vmax_kt, dtype=float) / KT_PER_MS\n    return 46.29",
                "v_ms = np.asarray(vmax_kt, dtype=float)\n    return 46.29",
            )
        ],
    ),
    (
        "M07 inland decay not applied",
        [
            (
                S + "hazard/windfield.py",
                "vm = kaplan_demaria(e(vmax_kt), s / v, decay=decay)",
                "vm = kaplan_demaria(e(vmax_kt), s / v, decay=False)",
            )
        ],
    ),
    (
        "M08 Kaplan-DeMaria: landfall reduction factor R_f dropped",
        [(S + "hazard/windfield.py", "(KD_R * v0 - KD_VB_KT)", "(v0 - KD_VB_KT)")],
    ),
    (
        "M09 translation asymmetry: left and right of track swapped",
        [(S + "hazard/windfield.py", "c = x * uy - y * ux", "c = y * ux - x * uy")],
    ),
    (
        "M10 damage function: wrong sign of the threshold",
        [
            (
                S + "vulnerability.py",
                "np.asarray(wind_kt, dtype=float) - v_thresh_kt",
                "np.asarray(wind_kt, dtype=float) + v_thresh_kt",
            )
        ],
    ),
    (
        "M11 damage function fed m/s instead of knots",
        [
            (
                S + "eventset.py",
                "dmg = emanuel_damage(w, settings.v_thresh_kt",
                "dmg = emanuel_damage(w * 0.514444, settings.v_thresh_kt",
            )
        ],
    ),
    (
        "M12 deductible applied twice",
        [
            (
                S + "vulnerability.py",
                "np.asarray(damage) - deductible",
                "np.asarray(damage) - 2 * deductible",
            )
        ],
    ),
    (
        "M13 layer EL divided by E instead of E - A",
        [
            (
                S + "layers.py",
                "expected_loss=float(principal.mean() / limit)",
                "expected_loss=float(principal.mean() / exhaustion)",
            )
        ],
    ),
    (
        "M14 attachment probability with > instead of >=",
        [
            (
                S + "layers.py",
                "attachment_prob=float(np.mean(occ >= attachment))",
                "attachment_prob=float(np.mean(occ > attachment))",
            )
        ],
    ),
    (
        "M15 annual principal loss: sum without the cap at the limit",
        [(S + "layers.py", "return np.minimum(total, exhaustion - attachment)", "return total")],
    ),
    (
        "M16 return period: quantile 1 - 1/(T + 1) instead of 1 - 1/T",
        [
            (
                S + "losses.py",
                "q = 1.0 - 1.0 / np.asarray(periods, dtype=float)",
                "q = 1.0 - 1.0 / (np.asarray(periods, dtype=float) + 1)",
            )
        ],
    ),
    (
        "M17 exceedance probability counts ties as exceeding",
        [(S + "losses.py", 'side="right") / a.size', 'side="left") / a.size')],
    ),
    (
        "M18 parameter bootstrap resamples events instead of years",
        [
            (
                S + "uncertainty.py",
                "    years = np.arange(first_year, last_year + 1)\n",
                "    idx = rng.integers(0, len(landfalls), size=len(landfalls))\n"
                "    return landfalls.iloc[idx].reset_index(drop=True)\n"
                "    years = np.arange(first_year, last_year + 1)\n",
            )
        ],
    ),
    (
        "M19 look-ahead: hazard fit reads years after the window",
        [
            (
                S + "hazard/model.py",
                'win = landfalls[(landfalls["year"] >= first_year) & '
                '(landfalls["year"] <= last_year)]',
                'win = landfalls[landfalls["year"] >= first_year]',
            )
        ],
    ),
    (
        "M20 look-ahead: out-of-sample fit uses the test years too",
        [
            (
                S + "hazard/validation.py",
                'fit = landfalls[(landfalls["year"] >= f0) & (landfalls["year"] <= f1)]',
                'fit = landfalls[(landfalls["year"] >= f0) & (landfalls["year"] <= t1)]',
            )
        ],
    ),
    (
        "M21 geometric landfall: exits from land counted as landfalls",
        [
            (
                S + "hazard/landfall.py",
                "                if not state:\n                    continue",
                "                if state:\n                    continue",
            )
        ],
    ),
    (
        "M22 spread regression on EL instead of log EL",
        [
            (
                S + "pricing/fit.py",
                'np.log(df["expected_loss_pct"].to_numpy(dtype=float))',
                'df["expected_loss_pct"].to_numpy(dtype=float)',
            )
        ],
    ),
    (
        "M23 HC3 replaced by HC0",
        [
            (
                S + "pricing/fit.py",
                "meat = (x * loo[:, None] ** 2).T @ x",
                "meat = (x * e[:, None] ** 2).T @ x",
            )
        ],
    ),
    (
        "M24 cluster SE without small-sample correction",
        [
            (
                S + "pricing/fit.py",
                "c = g / (g - 1) * (n - 1) / (n - k) if g > 1 else np.nan",
                "c = 1.0",
            )
        ],
    ),
    (
        "M25 prediction interval omits the new observation's error",
        [
            (
                S + "pricing/fit.py",
                'self.sigma * np.sqrt(1.0 + np.einsum("ij,jk,ik->i"',
                'self.sigma * np.sqrt(np.einsum("ij,jk,ik->i"',
            )
        ],
    ),
    (
        "M26 wild bootstrap weights: 0/1 instead of Rademacher +1/-1",
        [
            (
                S + "pricing/wild.py",
                "return 2.0 * rng.integers(0, 2, size=(n_boot, n_clusters)) - 1.0",
                "return 1.0 * rng.integers(0, 2, size=(n_boot, n_clusters))",
            )
        ],
    ),
    (
        "M27 wild bootstrap: t* centred at the estimate instead of the null value",
        [
            (
                S + "pricing/wild.py",
                "tstar = (bstar[self.j] - beta0) / self._se(estar)",
                "tstar = (bstar[self.j] - self.estimate) / self._se(estar)",
            )
        ],
    ),
    (
        "M28 wild bootstrap: CR1 small-sample factor dropped G/(G-1)",
        [
            (
                S + "pricing/wild.py",
                "self.c = self.n_clusters / (self.n_clusters - 1) * (n - 1) / (n - k)",
                "self.c = (n - 1) / (n - k)",
            )
        ],
    ),
    (
        "M29 year fixed effects: last issue-year dummy dropped",
        [
            (
                S + "pricing/fit.py",
                "for lev in levels[cov][1:]:",
                "for lev in levels[cov][1:-1]:",
            )
        ],
    ),
    (
        "M30 clustering: the cluster argument is ignored (always sponsor)",
        [
            (
                S + "pricing/fit.py",
                "groups = cluster_labels(df, cluster)",
                'groups = cluster_labels(df, "sponsor")',
            )
        ],
    ),
    (
        "M31 clustering: all blank labels pooled into one cluster",
        [
            (
                S + "pricing/fit.py",
                'lab = lab.where(~_blank(df[column]), "__row" + row)',
                'lab = lab.where(~_blank(df[column]), "__blank")',
            )
        ],
    ),
    (
        "M32 leave-group-out: held-out group stays in the training rows",
        [
            (
                S + "pricing/validation.py",
                "train = rows[codes != g]",
                "train = rows",
            )
        ],
    ),
    (
        "M33 leave-group-out: error in points has the wrong sign",
        [
            (
                S + "pricing/validation.py",
                '"err_pp": spread - pred,',
                '"err_pp": pred - spread,',
            )
        ],
    ),
    (
        "M34 workbook formula: exhaustion probability counted at the attachment",
        [
            (
                S + "workbook.py",
                "return f'=COUNTIF(AnnualMax,\">=\"&({a}+{lim}))/N_years'",
                "return f'=COUNTIF(AnnualMax,\">=\"&{a})/N_years'",
            )
        ],
    ),
    (
        "M35 workbook formula: event layer loss capped at E instead of the limit",
        [
            (
                S + "workbook.py",
                'f"+(EventLoss>={e})*{lim}))/(N_years*{lim})"',
                'f"+(EventLoss>={e})*{e}))/(N_years*{lim})"',
            )
        ],
    ),
    (
        "M36 transaction stack: next layer attaches half a limit up (layers overlap)",
        [
            (
                S + "transaction.py",
                "a = a + lim",
                "a = a + lim / 2",
            )
        ],
    ),
    (
        "M37 deal sample: duplicate layers not removed",
        [
            (
                S + "pricing/sample.py",
                'df["duplicate"] = dup',
                'df["duplicate"] = dup & False',
            )
        ],
    ),
    (
        "M38 workbook formula: annual layer loss not capped at the limit",
        [
            (
                S + "workbook.py",
                'return f"=MIN(SUM({first}:{last}),{lim})"',
                'return f"=SUM({first}:{last})"',
            )
        ],
    ),
    (
        "M39 workbook formula: capped EL divided by the number of years only",
        [
            (
                S + "workbook.py",
                'return f"=SUM({annual})/(N_years*{lim})"',
                'return f"=SUM({annual})/N_years"',
            )
        ],
    ),
]


def run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def pytest_run(work: pathlib.Path, timeout: float):
    env = dict(
        os.environ,
        PYTHONPATH=str(work / "src"),
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
    )
    t0 = time.monotonic()
    try:
        p = run(
            [
                sys.executable,
                "-m",
                "pytest",
                str(work / "tests"),
                "-q",
                "-x",
                "-rf",
                "--tb=no",
                "-p",
                "no:cacheprovider",
                "--ignore",
                str(work / "tests" / "test_deals_data.py"),
            ],
            env=env,
            timeout=timeout,
            cwd=work,
        )
    except subprocess.TimeoutExpired:
        return None, ["(timeout)"], time.monotonic() - t0, "timeout"
    failed = [
        ln.split(" - ")[0].replace("FAILED ", "").split("tests/")[-1]
        for ln in p.stdout.splitlines()
        if ln.startswith("FAILED")
    ]
    summary = p.stdout.strip().splitlines()[-1] if p.stdout.strip() else p.stderr[-500:]
    return p.returncode, failed, time.monotonic() - t0, summary


def main() -> int:
    ap = argparse.ArgumentParser(description="Mutation check for the catbond test suite.")
    ap.add_argument("--out", type=pathlib.Path, default=None)
    ap.add_argument("--timeout", type=float, default=900.0, help="seconds per pytest run")
    args = ap.parse_args()
    lines: list[str] = []

    def log(s: str = "") -> None:
        print(s, flush=True)
        lines.append(s)

    commit = run(["git", "-C", str(REPO), "rev-parse", "HEAD"]).stdout.strip() or "unknown"
    dirty = run(["git", "-C", str(REPO), "status", "--porcelain", "--", "src", "tests"]).stdout
    import numpy

    cmd = [
        sys.argv[0]
        if not pathlib.Path(sys.argv[0]).is_absolute()
        else str(pathlib.Path(sys.argv[0]).relative_to(REPO)),
        *sys.argv[1:],
    ]
    log("# Generated by tests/mutation_check.py; do not edit by hand.")
    log("# command: python3 " + " ".join(shlex.quote(a) for a in cmd))
    log(f"# git:     {commit} ({'uncommitted changes in src/ or tests/' if dirty else 'clean'})")
    log(f"# date:    {datetime.datetime.now(datetime.UTC):%Y-%m-%dT%H:%M:%SZ}")
    log(f"# python:  {platform.python_version()}; numpy {numpy.__version__}")
    log("# pytest -x: the test named is the first failing one, not the only one.")
    log("")

    survived = 0
    with tempfile.TemporaryDirectory(prefix="catbond_mutation_") as tmp:
        work = pathlib.Path(tmp)
        ignore = shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache", "*.egg-info")
        shutil.copytree(REPO / "src", work / "src", ignore=ignore)
        shutil.copytree(REPO / "tests", work / "tests", ignore=ignore)
        if (REPO / "data").exists():
            (work / "data").symlink_to(REPO / "data", target_is_directory=True)
        probe = run(
            [sys.executable, "-c", "import catbond; print(catbond.__file__)"],
            env=dict(os.environ, PYTHONPATH=str(work / "src")),
            cwd=work,
        )
        if not probe.stdout.strip().startswith(str(work)):
            log(f"the copied package is not the one imported: {probe.stdout}{probe.stderr}")
            return 1
        rc, failed, secs, summary = pytest_run(work, args.timeout)
        log(f"baseline (no mutant): {summary} [{secs:.0f} s]")
        if rc != 0:
            log("baseline must pass; aborting")
            return 1
        log("")
        originals: dict[str, str] = {}
        for name, edits in MUTANTS:
            touched: dict[str, str] = {}
            try:
                for rel, old, new in edits:
                    path = work / rel
                    originals.setdefault(rel, path.read_text())
                    text = touched.get(rel, originals[rel])
                    if text.count(old) != 1:
                        log(f"{name}: SETUP ERROR, pattern found {text.count(old)} times in {rel}")
                        return 1
                    touched[rel] = text.replace(old, new)
                for rel, text in touched.items():
                    (work / rel).write_text(text)
                rc, failed, secs, summary = pytest_run(work, args.timeout)
                caught = rc != 0
                survived += not caught
                log(f"{name}: {'caught' if caught else 'SURVIVED'} [{secs:.0f} s]")
                for f in failed[:1]:
                    log(f"    first failure: {f}")
            finally:
                for rel in touched:
                    (work / rel).write_text(originals[rel])
    log("")
    log(f"{len(MUTANTS) - survived} of {len(MUTANTS)} mutants caught.")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text("\n".join(lines) + "\n")
        print(f"wrote {args.out}")
    return 1 if survived else 0


if __name__ == "__main__":
    sys.exit(main())
