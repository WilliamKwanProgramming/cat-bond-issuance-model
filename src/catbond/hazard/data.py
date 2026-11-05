"""Loaders for Atlantic best-track data (HURDAT2) in three formats, to one standard table.

Standard columns: ``storm_id`` (str), ``name`` (str), ``year`` (int), ``time``
(datetime64), ``record`` (HURDAT2 record identifier, "" if none or not available), ``status``
(two-letter HURDAT2 code), ``lat`` (deg N), ``lon`` (deg E, negative west), ``wind`` (maximum
1-minute sustained 10-m wind, kt, NaN if missing), ``pressure`` (hPa, NaN if missing).

Formats:

* ``AL.rda`` (1851-2019, R data file in the timtrice/HURDAT GitHub repository): has the record
  identifier, so landfall records ("L") are available.
* ``storms.csv`` (1975-2024, the dplyr ``storms`` source file): no record identifier and no storm
  number; storms are identified by (name, year). Time is truncated to the hour.
* The NHC HURDAT2 text file (Landsea and Franklin 2013 format). Supported for use outside the
  sandbox; not exercised on the real file here.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd

STANDARD_COLUMNS = [
    "storm_id",
    "name",
    "year",
    "time",
    "record",
    "status",
    "lat",
    "lon",
    "wind",
    "pressure",
]

_STORMS_STATUS = {
    "tropical depression": "TD",
    "tropical storm": "TS",
    "hurricane": "HU",
    "extratropical": "EX",
    "subtropical depression": "SD",
    "subtropical storm": "SS",
    "other low": "LO",
    "tropical wave": "WV",
    "disturbance": "DB",
}


def _finish(df: pd.DataFrame) -> pd.DataFrame:
    df = df[STANDARD_COLUMNS].copy()
    df["lon"] = df["lon"].where(df["lon"].abs() <= 180.0)
    df["wind"] = df["wind"].where(df["wind"] >= 0)
    df["pressure"] = df["pressure"].where(df["pressure"] > 0)
    return df.sort_values(["storm_id", "time"], kind="stable").reset_index(drop=True)


def load_hurdat_rda(path: pathlib.Path) -> pd.DataFrame:
    """Read ``AL.rda`` (timtrice/HURDAT) into the standard table.

    In this file the first storm (AL011851, 14 rows) has an empty ``Key``; rows without a key are
    dropped. Longitudes outside [-180, 180] (one obvious entry error) are set to NaN.
    """
    import pyreadr

    raw = next(iter(pyreadr.read_r(str(path)).values()))
    raw = raw[raw["Key"].notna()]
    df = pd.DataFrame(
        {
            "storm_id": raw["Key"].astype(str),
            "name": raw["Name"].astype(str),
            "time": pd.to_datetime(raw["DateTime"]),
            "record": raw["Record"].fillna("").astype(str).str.strip(),
            "status": raw["Status"].astype(str).str.strip(),
            "lat": pd.to_numeric(raw["Lat"], errors="coerce"),
            "lon": pd.to_numeric(raw["Lon"], errors="coerce"),
            "wind": pd.to_numeric(raw["Wind"], errors="coerce").astype(float),
            "pressure": pd.to_numeric(raw["Pressure"], errors="coerce").astype(float),
        }
    )
    df["year"] = df["storm_id"].str[-4:].astype(int)
    return _finish(df)


def load_storms_csv(path: pathlib.Path) -> pd.DataFrame:
    """Read the dplyr ``storms.csv`` source (1975-2024) into the standard table.

    Storms are keyed by (name, year); a new storm is started when consecutive rows of the same
    key are more than 72 hours apart. ``record`` is empty (not in this file). Times are truncated
    to the hour in this file; when two rows of a storm share a timestamp, the later one is moved
    30 minutes later.
    """
    raw = pd.read_csv(path)
    time = pd.to_datetime(
        dict(year=raw["year"], month=raw["month"], day=raw["day"], hour=raw["hour"])
    )
    df = pd.DataFrame(
        {
            "name": raw["name"].astype(str),
            "year": raw["year"].astype(int),
            "time": time,
            "record": "",
            "status": raw["status"].map(_STORMS_STATUS).fillna("??"),
            "lat": raw["lat"].astype(float),
            "lon": raw["long"].astype(float),
            "wind": raw["wind"].astype(float),
            "pressure": raw["pressure"].astype(float),
        }
    )
    df = df.sort_values(["year", "name", "time"], kind="stable")
    key = df["year"].astype(str) + "-" + df["name"]
    gap = df["time"].diff() > pd.Timedelta(hours=72)
    new = (key != key.shift()) | gap
    seq = new.groupby(key).cumsum()
    df["storm_id"] = key + np.where(seq > 1, "-" + seq.astype(str), "")
    # Off-synoptic fixes (often landfalls) lose their minutes in this file and can then share
    # the hour of a synoptic fix; the later row of such a pair is placed 30 minutes later.
    dup = df.duplicated(subset=["storm_id", "time"], keep="first")
    df.loc[dup, "time"] = df.loc[dup, "time"] + pd.Timedelta(minutes=30)
    return _finish(df)


def load_hurdat2_text(path: pathlib.Path) -> pd.DataFrame:
    """Parse the NHC HURDAT2 text format (header line per storm, then one line per fix).

    Header: ``AL092017, IRMA, 66,``. Data line fields: date YYYYMMDD, time HHMM, record identifier,
    status, latitude (e.g. ``28.0N``), longitude (``94.8W``), max wind (kt), min pressure (hPa),
    then wind radii (ignored). -999 means missing.
    """
    rows = []
    sid = name = ""
    for line in pathlib.Path(path).read_text().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            continue
        if parts[0][:2].isalpha() and len(parts[0]) == 8:
            sid, name = parts[0], parts[1]
            continue
        lat = float(parts[4][:-1]) * (1 if parts[4][-1] == "N" else -1)
        lon = float(parts[5][:-1]) * (-1 if parts[5][-1] == "W" else 1)
        rows.append(
            (
                sid,
                name,
                pd.Timestamp(f"{parts[0]} {parts[1]}"),
                parts[2],
                parts[3],
                lat,
                lon,
                float(parts[6]),
                float(parts[7]),
            )
        )
    df = pd.DataFrame(
        rows,
        columns=["storm_id", "name", "time", "record", "status", "lat", "lon", "wind", "pressure"],
    )
    df.loc[df["wind"] <= -99, "wind"] = np.nan
    df.loc[df["pressure"] <= -99, "pressure"] = np.nan
    df["year"] = df["storm_id"].str[-4:].astype(int)
    return _finish(df)
