"""Smoke test for the github_integration_spec ability.

Runs in-process using FastAPI's TestClient; no external server required.
"""

from __future__ import annotations

import os
import sys
from contextlib import suppress

from fastapi.testclient import TestClient


def main() -> int:
    # Ensure repo root on sys.path so `app` import works in all launch contexts
    repo_root = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(repo_root, os.pardir))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    # Ensure .env is loaded before importing app (for GITHUB_TOKEN, etc.)
    with suppress(Exception):
        from src.core.env import ensure_env_loaded  # type: ignore

        ensure_env_loaded(silent=True)

    # Lazy import to respect sys.path mutation
    from app import app  # noqa: WPS433,E402

    client = TestClient(app)

    payload = {
        "q": "fastapi middleware",
        "language": "python",
        "per_page": 5,
        "max_candidates": 3,
    }

    print(
        "- Using GITHUB_TOKEN:",
        "SET" if os.getenv("GITHUB_TOKEN") else "MISSING",
    )

    resp = client.post(
        "/ability/execute/github_integration_spec",
        json={"args": payload},
    )
    print("- Status:", resp.status_code)
    if resp.status_code != 200:
        print("- Error body:")
        print(resp.text)
        return 1

    body = resp.json()
    print("- Keys:", sorted(list(body.keys())))
    if not body.get("ok"):
        print("- Ability error:", body)
        return 1

    result = body.get("result") or {}
    if not result.get("ok"):
        print("- Result error:", result)
        return 1

    spec = result.get("integration_spec") or {}
    cands = result.get("candidates") or []
    print(f"- Candidates: {len(cands)}")
    if cands:
        first = cands[0]
        print(
            f"  1. {first.get('repo')}/{first.get('path')}  "
            f"license={first.get('license_spdx')}"
        )
    print("- Spec keys:", sorted(list(spec.keys())))
    print("\nSuccess: Integration spec ability responded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
