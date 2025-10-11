"""CLI helper to run consolidation job once."""
from __future__ import annotations

from src.controller.consolidate import run_consolidation


if __name__ == "__main__":
    result = run_consolidation()
    print(result)
