#!/usr/bin/env python
import argparse, json, pathlib, statistics as stats
from collections import defaultdict

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    in_dir = pathlib.Path(args.input)
    results_file = in_dir / "results.json"
    results = json.loads(results_file.read_text())

    by_arm = defaultdict(list)
    for r in results:
        by_arm[r["arm"]].append(r)

    summary_lines = ["# Experiment Summary", ""]
    for arm, rows in by_arm.items():
        rewards = [r["reward"] for r in rows]
        lat = [r["latency_ms"] for r in rows]
        summary_lines.append(f"## Arm: {arm}")
        summary_lines.append(f"- Trials: {len(rows)}")
        summary_lines.append(f"- Reward mean: {sum(rewards)/len(rewards):.3f}")
        summary_lines.append(f"- Reward p90: {sorted(rewards)[int(0.9*len(rewards))-1]:.3f}")
        summary_lines.append(f"- Latency mean (ms): {sum(lat)/len(lat):.1f}")
        summary_lines.append(f"- Latency p90 (ms): {sorted(lat)[int(0.9*len(lat))-1]}")
        summary_lines.append("")
    pathlib.Path(args.out).write_text("\n".join(summary_lines))
    print(f"Wrote experiment summary to {args.out}")

if __name__ == "__main__":
    main()