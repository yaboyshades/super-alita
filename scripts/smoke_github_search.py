"""
Quick smoke test for the built-in GitHub search abilities.

Runs in-process using FastAPI's TestClient, so no server needs to be running.

Prerequisites:
- Copy .env.example to .env and set GITHUB_TOKEN to a valid PAT
    (optional but recommended).
- Activate your virtualenv and install dependencies.
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
    from app import app  # noqa: WPS433

    client = TestClient(app)

    # Build a realistic query that finds small, useful snippets
    payload = {
        "q": "dataclass parse json file language:python",
        "per_page": 5,
        "page": 1,
    }

    print(
        "- Using GITHUB_TOKEN:",
        "SET" if os.getenv("GITHUB_TOKEN") else "MISSING",
    )

    resp = client.post(
        "/ability/execute/github_search_code",
        json={"args": payload},
    )
    print("- Status:", resp.status_code)
    if resp.status_code != 200:
        print("- Error body:")
        print(resp.text)
        return 1

    data = resp.json()
    print("- Keys:", sorted(list(data.keys())))

    if not data.get("ok"):
        print("- Ability error:", data)
        return 1

    result = data.get("result") or {}
    items = result.get("items") or []
    total = result.get("total_count")
    print(f"- Found {len(items)} results (total_count={total})")

    for i, it in enumerate(items[:3], 1):
        repo = it.get("repo")
        path = it.get("path")
        score = it.get("score")
        print(f"  {i}. {repo}/{path}  score={score}")
        print(f"     {it.get('html_url')}")

    print("\nSuccess: GitHub code search ability is reachable and responding.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
