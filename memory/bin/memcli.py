#!/usr/bin/env python3
import sys

import requests


def main() -> None:
    q = " ".join(sys.argv[1:]).strip()
    if not q:
        print("# MEMORIES: (none)")
        return
    try:
        r = requests.post(
            "http://127.0.0.1:8765/search", json={"q": q, "k": 5}, timeout=2
        )
        hits = r.json().get("hits", [])
    except Exception:
        print("# MEMORIES: (daemon offline)")
        return
    if not hits:
        print("# MEMORIES: (no hits)")
        return
    print("# MEMORIES (context recall)")
    for i, h in enumerate(hits, 1):
        text = h["text"].replace("\n", " ").strip()
        if len(text) > 220:
            text = f"{text[:220]}…"
        print(f"{i}. **{h['path']}#{h['off']}** — {text}")


if __name__ == "__main__":
    main()
