# Mangle-Style Code Reasoning — Minimal Scaffold

Treat a code repo as facts in SQLite, then run a few Datalog-like rules (as SQL) to produce explainable findings.

## Sample included
- `a.py` <-> `b.py` form a **file cycle**
- `a.compute()` is **complex** and **untested**, and is **called** (indegree >= 1) → flags as a **hot_path**

## Quickstart
```bash
python scripts/ingest_code.py sample_repo facts.sqlite
python scripts/run_engine.py facts.sqlite report.json
```

## Outputs
- `facts.sqlite` — tables: `file, symbol, complexity, imports, calls, dep, defines_test, tests_targets`
- `report.json`   — per-rule results and counts

## Rules implemented
- `untested_function` — complexity >= 0.3 and not targeted by any test
- `orphan_complex`    — complexity >= 0.6 and zero in/out degree
- `cycle`             — file-level import cycle
- `hot_path`          — complexity >= 0.5, untested, indegree >= 1
- `reinvention_json`  — heuristic: JSON-like function name without `import json`

Extend ingestion to add coverage facts, architectural boundaries, ownership, etc.
