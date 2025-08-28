#!/usr/bin/env python
import argparse
import json
import pathlib
import random
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--since-hours", type=int, default=6)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    # Placeholder: Replace with API / DB fetch.
    now = int(time.time())
    records = []
    for i in range(200):
        records.append({
            "timestamp": now - random.randint(0, args.since_hours * 3600),
            "arm_id": random.choice(["cot", "concise", "retrieval_hybrid", "retrieval_vector"]),
            "reward": random.uniform(-0.2, 1.0),
            "components": {"quality": random.uniform(0, 1), "latency": random.uniform(0, 1), "cost": random.uniform(0, 1)}
        })

    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(records, indent=2))
    print(f"Wrote simulated rewards to {out_path}")

if __name__ == "__main__":
    main()