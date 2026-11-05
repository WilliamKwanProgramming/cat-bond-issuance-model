"""Write a MANIFEST.txt next to every set of results: command, seed, commit, environment."""

from __future__ import annotations

import datetime
import pathlib
import platform
import shlex
import subprocess
import sys

import numpy as np


def repo_root() -> pathlib.Path:
    """Repository root (two levels above this package's ``src`` directory)."""
    return pathlib.Path(__file__).resolve().parents[2]


def _git(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(repo_root()), *args], capture_output=True, text=True, check=False
        ).stdout.strip()
    except OSError:
        return ""


def relative_command(argv: list[str]) -> str:
    """The command line with the script path made relative to the repository root."""
    root = repo_root()
    parts = []
    for a in argv:
        p = pathlib.Path(a)
        if p.is_absolute() and root in p.parents:
            a = str(p.relative_to(root))
        parts.append(shlex.quote(a))
    return "python3 " + " ".join(parts)


def write_manifest(
    out_dir: pathlib.Path, seed: int | None, notes: dict[str, str] | None = None
) -> None:
    """Write ``out_dir/MANIFEST.txt`` with the command, seed, commit and versions."""
    out_dir.mkdir(parents=True, exist_ok=True)
    commit = _git("rev-parse", "HEAD") or "unknown"
    dirty = _git("status", "--porcelain", "--", "src", "scripts")
    lines = [
        f"command: {relative_command(sys.argv)}",
        f"seed:    {seed if seed is not None else 'none (deterministic)'}",
        f"git:     {commit} ({'uncommitted changes in src/ or scripts/' if dirty else 'clean'})",
        f"date:    {datetime.datetime.now(datetime.UTC):%Y-%m-%dT%H:%M:%SZ}",
        f"python:  {platform.python_version()}; numpy {np.__version__}",
        f"machine: {platform.machine()} {platform.system()}",
    ]
    for k, v in (notes or {}).items():
        lines.append(f"{k}: {v}")
    (out_dir / "MANIFEST.txt").write_text("\n".join(lines) + "\n")
