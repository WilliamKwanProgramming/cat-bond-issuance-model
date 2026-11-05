#!/usr/bin/env python3
"""Download the public HURDAT2 hurricane-track files used by this repository into data/raw/.

Two sources are supported:

* ``mirror`` (default, the only route exercised while building this repository): copies of the
  NOAA National Hurricane Center HURDAT2 best-track data (public domain) kept in MIT-licensed
  GitHub repositories. Every file is pinned by SHA-256 and the download fails on a mismatch.
* ``nhc``: the canonical text file from https://www.nhc.noaa.gov/data/hurdat/. Its content changes
  with every annual reanalysis, so it is not pinned; the script prints its SHA-256 instead.
  ``catbond.hazard.data.load_hurdat2_text`` reads it. This route was not exercised in the sandbox
  where the repository was built (the NHC site was not reachable from it). The county boundary
  file always comes from its GitHub copy.

Usage: python3 scripts/fetch_data.py [--source mirror|nhc] [--dest data/raw]
"""

from __future__ import annotations

import argparse
import hashlib
import pathlib
import sys
import urllib.request

REPO = pathlib.Path(__file__).resolve().parents[1]

# name -> (url, sha256). Checksums verified on 2026-11-05.
MIRROR_FILES: dict[str, tuple[str, str]] = {
    "AL.rda": (
        "https://raw.githubusercontent.com/timtrice/HURDAT/master/data/AL.rda",
        "08457d2ab2e936c276e11a87916c0b9c1b043cb01fe6cef735ca3a1c1ea62b75",
    ),
    "storms.csv": (
        "https://raw.githubusercontent.com/tidyverse/dplyr/main/data-raw/storms.csv",
        "ee46bd48b49ebd586da86192678de412fb84dfc6534d250863b8fc8f4ff16ebd",
    ),
    # US county boundaries with Census GEO_ID / CENSUSAREA fields, as GeoJSON (plotly/datasets).
    # Used for the Florida coastline, the landfall test and the county centroids.
    "us_counties.geojson": (
        "https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json",
        "e540149b7525e71ee6b6cab6dea2a95205f11e0c3e7374d27a7c9c47ea96e8c0",
    ),
}

# Latest Atlantic HURDAT2 file listed at https://www.nhc.noaa.gov/data/hurdat/ on 2026-11-05.
NHC_URL = "https://www.nhc.noaa.gov/data/hurdat/hurdat2-1851-2025-092326.txt"


def sha256_of(path: pathlib.Path) -> str:
    """Return the hex SHA-256 digest of a file."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: pathlib.Path) -> None:
    """Download ``url`` to ``dest`` through a temporary file."""
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url, timeout=120) as resp, tmp.open("wb") as out:
        out.write(resp.read())
    tmp.replace(dest)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--source", choices=["mirror", "nhc"], default="mirror")
    ap.add_argument("--dest", type=pathlib.Path, default=REPO / "data" / "raw")
    args = ap.parse_args()
    args.dest.mkdir(parents=True, exist_ok=True)

    wanted = dict(MIRROR_FILES)
    if args.source == "nhc":
        target = args.dest / pathlib.Path(NHC_URL).name
        download(NHC_URL, target)
        print(f"{target.name}: sha256 {sha256_of(target)} (not pinned)")
        wanted = {"us_counties.geojson": MIRROR_FILES["us_counties.geojson"]}

    status = 0
    for name, (url, expected) in wanted.items():
        target = args.dest / name
        if target.exists() and sha256_of(target) == expected:
            print(f"{name}: present, checksum ok")
            continue
        download(url, target)
        got = sha256_of(target)
        if got != expected:
            print(f"{name}: CHECKSUM MISMATCH (got {got}, expected {expected})", file=sys.stderr)
            target.unlink()
            status = 1
        else:
            print(f"{name}: downloaded, checksum ok")
    return status


if __name__ == "__main__":
    sys.exit(main())
