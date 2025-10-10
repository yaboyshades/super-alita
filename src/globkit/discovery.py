from __future__ import annotations

import glob
import os
from datetime import datetime

import pandas as pd


def analyze_file_collection(
    pattern: str, recursive: bool = True
) -> pd.DataFrame:
    """Collect basic stats for matching files: size, mtime, extension, and depth."""
    files = glob.glob(pattern, recursive=recursive)
    rows = []
    for p in files:
        try:
            st = os.stat(p)
        except FileNotFoundError:
            continue
        rows.append(
            {
                "path": p,
                "size": st.st_size,
                "modified": datetime.fromtimestamp(st.st_mtime),
                "extension": os.path.splitext(p)[1],
                "depth": p.count(os.sep),
            }
        )
    return pd.DataFrame(rows)


def discover_data_files(base_path: str, patterns=None) -> dict[str, list[str]]:
    """Return {pattern: [files]} for several patterns under base_path recursively."""
    if patterns is None:
        patterns = ["*.csv", "*.json", "*.xlsx", "*.parquet"]
    out = {}
    for pat in patterns:
        full = f"{base_path}/**/{pat}"
        out[pat] = glob.glob(full, recursive=True)
    return out


def optimize_file_loading(file_list: list[str], batch_size: int = 100):
    """Simple batching helper (hook your processing inside the loop)."""
    total = len(file_list)
    for i in range(0, total, batch_size):
        yield file_list[i : i + batch_size]
