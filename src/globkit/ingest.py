from __future__ import annotations

import glob
import os
from collections.abc import Iterable

import pandas as pd


def load_data_files(patterns: Iterable[str]) -> dict[str, pd.DataFrame]:
    """Load heterogenous data files matched by patterns (csv/xlsx/json).

    Returns a dict mapping each input pattern to a single concatenated
    DataFrame of all files that matched that pattern.
    """
    data: dict[str, pd.DataFrame] = {}
    for pattern in patterns:
        files = glob.glob(pattern)
        frames: list[pd.DataFrame] = []
        for fp in files:
            ext = fp.rsplit(".", 1)[-1].lower()
            if ext == "csv":
                frames.append(pd.read_csv(fp))
            elif ext in ("xlsx", "xls"):
                frames.append(pd.read_excel(fp))
            elif ext == "json":
                frames.append(pd.read_json(fp))
        if frames:
            data[pattern] = pd.concat(frames, ignore_index=True)
    return data


def load_csv_dataset(pattern: str, **pandas_kwargs) -> pd.DataFrame:
    """Bulk-load CSV files into one DataFrame."""
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")
    dfs = [pd.read_csv(f, **pandas_kwargs) for f in files]
    return pd.concat(dfs, ignore_index=True)


def load_time_series_data(
    pattern: str,
    date_column: str = "timestamp",
) -> pd.DataFrame:
    """Load and chronologically sort time-series data from multiple files."""
    files = sorted(glob.glob(pattern))
    dfs = []
    for f in files:
        df = pd.read_csv(f)
        if date_column in df.columns:
            df[date_column] = pd.to_datetime(df[date_column])
        df["source_file"] = os.path.basename(f)
        dfs.append(df)
    out = pd.concat(dfs, ignore_index=True)
    return out.sort_values(date_column) if date_column in out.columns else out
