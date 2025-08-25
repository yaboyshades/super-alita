# Diagnostics

For quick checks without pytest:

```
PYTHONPATH=src ./scripts/run_diagnostics.py
```

What it verifies:

- Deterministic ID invariants
- Memory store & lineage traversal
- Registry register/invoke
- Atomizer end-to-end with provenance
