# DK-Glob (Pattern-Based File & Data Access)
Purpose: Canonical utilities for file discovery, ingestion, and dataset prep using Python `glob`.
## When to use
- Bulk data ingestion, time-series loading, dataset prep for ML, and reproducible pipelines.
## API Surface
- patterns: `list_files()`, `iglob_iter()`
- discovery: `analyze_file_collection()`, `discover_data_files()`, `optimize_file_loading()`
- ingest: `load_data_files()`, `load_csv_dataset()`, `load_time_series_data()`
- ml: `prepare_ml_dataset()`, `cross_validate_multiple_datasets()`
## Pitfalls & Notes
- Use `recursive=True` with `**` patterns.
- Prefer absolute or repo-relative patterns in CI.
- Document patterns in specs; add source_file column for lineage.
