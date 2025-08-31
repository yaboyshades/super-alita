# Snippet Library

A quick, editor-agnostic set of copy-paste snippets aligned with this repo’s patterns.

## CLI & Servers
- Install deps: `uv pip install -r requirements.txt -c constraints.txt` (or `make deps`).
- Runtime server: `python -m src.main`.
- FastAPI dev: `make run` (serves `app:app` on port `8080`).
- Tests: `pytest -q`, filter: `pytest -k "expr"`, marker: `pytest -m integration_redis`.

## Subprocess (proc.py)
```python
from src.core.proc import run

# Safe subprocess (no shell=True)
code, out, err = run(["python", "-V"], timeout=10)
if code != 0:
    raise RuntimeError(err)
print(out.strip())
```

## YAML Utils
```python
from src.core.yaml_utils import load_yaml_file, dump_yaml_file

cfg = load_yaml_file("config/example.yaml")
cfg["enabled"] = True
dump_yaml_file("config/example.out.yaml", cfg)
```

## Sandbox Execution
```python
from src.sandbox.exec_sandbox import execute_snippet

code = "result = sum(range(5))"  # no raw eval/exec elsewhere
res = execute_snippet(code, globals_in={}, locals_in={})
assert res["result"] == 10
```

## FastAPI Endpoint (app.py)
```python
from fastapi import APIRouter
router = APIRouter()

@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}

# In app.py: app.include_router(router)
```

## Pytest Patterns
```python
# tests/test_example.py
import pytest

@pytest.mark.unit
def test_sum():
    assert sum([1, 2, 3]) == 6

@pytest.mark.integration_redis
def test_redis_marker():
    pytest.skip("requires redis test env")
```

## Typing & Style
```python
from collections.abc import Iterable

def average(nums: Iterable[float]) -> float:
    xs = list(nums)
    return sum(xs) / len(xs) if xs else 0.0
```

## Env & Modes
```bash
# SUPER_ALITA_MODE: shadow | act | batch
export SUPER_ALITA_MODE=shadow
python -m src.main
```
