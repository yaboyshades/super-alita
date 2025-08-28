#!/usr/bin/env python3
"""Simple Ollama smoke test.

Sends a single non-streaming chat request to the local Ollama server and
verifies that a recognizable token appears in the response. Exits non-zero
on failure so it can be used in CI or Make targets.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from urllib import error, request


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--host", default=os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
    )
    ap.add_argument("--model", default=os.getenv("OLLAMA_MODEL", "llama3.1:8b"))
    ap.add_argument(
        "--prompt",
        default="Respond with the exact word: pong",
    )
    ap.add_argument(
        "--expect",
        default="pong",
        help="Substring expected in response (case-insensitive)",
    )
    args = ap.parse_args()

    url = args.host.rstrip("/") + "/api/chat"
    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": args.prompt}],
        "stream": False,
    }
    data = json.dumps(payload).encode("utf-8")
    req = request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with request.urlopen(req, timeout=30) as resp:  # nosec - local call
            raw = resp.read()
    except error.HTTPError as e:
        sys.stderr.write(f"[ollama-smoke] HTTP {e.code}: {e.read()[:200]!r}\n")
        return 2
    except Exception as e:
        sys.stderr.write(f"[ollama-smoke] request failed: {e}\n")
        return 3

    try:
        obj = json.loads(raw.decode("utf-8"))
    except Exception:
        sys.stderr.write("[ollama-smoke] invalid JSON response\n")
        return 4

    content = ""
    if isinstance(obj, dict):
        msg = obj.get("message") or {}
        if isinstance(msg, dict):
            content = (msg.get("content") or "").strip()

    ok = args.expect.lower() in content.lower()
    print(f"[ollama-smoke] model={args.model} ok={ok} content={content[:120]!r}")
    return 0 if ok else 5


if __name__ == "__main__":
    raise SystemExit(main())
