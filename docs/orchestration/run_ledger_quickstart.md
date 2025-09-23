# Run Ledger Quickstart

The canonical orchestrator can mirror each run into an append-only NDJSON ledger.
This guide shows how to enable the feature, inspect the ledger, and diff legacy vs canonical streams.

## 1. Enable the ledger in your shell

```bash
export RUN_LEDGER_ENABLED=true
export RUN_LEDGER_PATH="data/run_ledger.ndjson"
# optional: disable canonical event mirroring to keep ledger only
export CANONICAL_EVENTS_ENABLED=false
```

## 2. Execute a canonical run

```bash
python -m src.main --prompt "Outline the telemetry aggregation upgrade"
```

When the run completes, entries are appended to `data/run_ledger.ndjson` with canonical envelopes.

## 3. Inspect the latest entries

```bash
tail -n 20 "$RUN_LEDGER_PATH" | jq '.'
```

Each entry includes `kind`, `sequence`, `data`, and reliability metadata in `meta.reliability`.

## 4. Compare legacy vs canonical output

Use the helper script introduced with the P0 work:

```bash
python scripts/compare_legacy_canonical.py \
  --legacy artifacts/legacy_stream.ndjson \
  --canonical "$RUN_LEDGER_PATH"
```

The script reports sequence mismatches, missing termination events, and reliability deltas.

## 5. Reset the ledger (optional)

```bash
rm "$RUN_LEDGER_PATH"
```

A fresh file is created automatically on the next run with restrictive file permissions (0600 on POSIX systems).
