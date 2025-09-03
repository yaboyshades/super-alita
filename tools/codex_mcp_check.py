from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


async def run_check(repo_root: Path, telemetry_file: Path) -> dict[str, Any]:
    # Set env BEFORE import so wrapper picks it up
    os.environ.setdefault("SUPER_ALITA_TELEMETRY_FILE", str(telemetry_file))
    ensure_parent(telemetry_file)

    # Import the MCP wrapper and call one tool
    import importlib

    # Ensure repo root on sys.path so we can import the wrapper from project root
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    try:
        m = importlib.import_module("mcp_server_wrapper")
    except ModuleNotFoundError:
        # Fallback to loading by absolute file path
        import importlib.util

        wrapper_path = repo_root / "mcp_server_wrapper.py"
        spec = importlib.util.spec_from_file_location(
            "mcp_server_wrapper", wrapper_path
        )
        if spec is None or spec.loader is None:  # type: ignore[truthy-bool]
            raise
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)  # type: ignore[arg-type]
    # Sanity: the decorated tools should be available
    missing = await m.find_missing_docstrings_tool(
        root=str(repo_root), include_tests=False
    )

    # Check telemetry output was written with success
    events = []
    if telemetry_file.exists():
        for line in telemetry_file.read_text(encoding="utf-8").splitlines():
            try:
                events.append(json.loads(line))
            except Exception:
                continue

    ok = any(
        e.get("type") == "AbilitySucceeded"
        and e.get("tool") == "find_missing_docstrings"
        for e in events
    )
    return {
        "ok": ok,
        "telemetry_path": str(telemetry_file),
        "events": len(events),
        "sample": next(
            (
                e
                for e in reversed(events[-10:])
                if e.get("type")
                in {"AbilityCalled", "AbilitySucceeded", "AbilityFailed"}
            ),
            {},
        ),
        "missing_docstrings_result": (
            missing if isinstance(missing, dict) else {"_type": type(missing).__name__}
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Validate MCP wrapper loads and tools execute with telemetry"
    )
    ap.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[1]))
    ap.add_argument(
        "--telemetry",
        default=str(
            Path(__file__).resolve().parents[1] / "logs/mcp_check_telemetry.jsonl"
        ),
    )
    args = ap.parse_args()

    result = asyncio.run(run_check(Path(args.repo_root), Path(args.telemetry)))
    print(json.dumps(result, indent=2))
    return 0 if result.get("ok") else 2


if __name__ == "__main__":
    raise SystemExit(main())
