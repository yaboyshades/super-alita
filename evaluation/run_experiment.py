#!/usr/bin/env python
import argparse
import json
import pathlib
import random
import time

import yaml


def run_trial(case, arm):
    time.sleep(0.01)
    return {
        "case_id": case["id"],
        "arm": arm["name"],
        "reward": random.uniform(0, 1),
        "latency_ms": random.randint(50, 800),
        "cost_tokens": random.randint(100, 1500)
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    spec = yaml.safe_load(open(args.spec))
    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    cases = spec["cases"]
    arms = spec["arms"]
    results = []
    for case in cases:
        for arm in arms:
            results.append(run_trial(case, arm))
    (out_dir / "results.json").write_text(json.dumps(results, indent=2))
    print(f"Experiment complete: {len(results)} rows")

if __name__ == "__main__":
    main()