# Quality Gauntlet Agent Instructions

This package owns the closed-loop quality verification workflow. Follow these
rules when modifying files underneath `src/quality_gauntlet/`:

1. **Do not block the runtime.** Long-running external scans must be executed
   asynchronously and cancellable. Use cooperative async helpers and enforce
   timeouts.
2. **All subprocess execution must keep `shell=False`** and sanitize arguments.
   Prefer the helper utilities provided in this package (see `tools/`).
3. **Outputs must be machine parseable.** Return Pydantic models from every
   public API so callers can serialize to JSON without post-processing.
4. **No direct network calls in tests.** Tests should monkeypatch command
   runners and provide fixture data. Ship fake adapters in `tests/runtime`.
5. **Document thresholds.** Changes to scoring or thresholds must update
   `config.py` defaults and reference Article VII in the constitution.
6. **Telemetry first.** When adding new orchestration phases, extend the
   refinement history objects to capture structured telemetry for each
   iteration.

## Quickstart

```python
from pathlib import Path
from src.quality_gauntlet import (
    GauntletConfig,
    QualityGauntletOrchestrator,
)
from src.sdd.models import SDDTask

config = GauntletConfig(max_iterations=2)
# Provide task + injected agents/tools (see orchestrator module for details)
```
