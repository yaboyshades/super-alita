"""DK-Glob: Pattern-based file & data access utilities (CMA v5.3)."""

from .discovery import (
    analyze_file_collection,
    discover_data_files,
    optimize_file_loading,
)
from .ingest import load_csv_dataset, load_data_files, load_time_series_data
from .ml import cross_validate_multiple_datasets, prepare_ml_dataset
from .patterns import iglob_iter, list_files

__all__ = [
    "list_files",
    "iglob_iter",
    "analyze_file_collection",
    "discover_data_files",
    "optimize_file_loading",
    "load_data_files",
    "load_csv_dataset",
    "load_time_series_data",
    "prepare_ml_dataset",
    "cross_validate_multiple_datasets",
]
