#!/usr/bin/env python
import argparse
import json
import pathlib
import random
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=5000)
    args = ap.parse_args()

    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for i in range(args.limit):
        data = {
            "episode_id": f"ep_{int(time.time())}_{i}",
            "steps": random.randint(3, 12),
            "goal": random.choice(["summarize docs", "find bug", "plan refactor"]),
            "reward": random.uniform(-0.1, 1.0),
            "trace": ["step reasoning ...", "tool result ..."]
        }
        (out_dir / f"{data['episode_id']}.json").write_text(json.dumps(data))
    print(f"Generated {args.limit} synthetic episodes in {out_dir} (replace with real fetch).")

if __name__ == "__main__":
    main()