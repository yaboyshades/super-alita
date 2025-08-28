#!/usr/bin/env python
import argparse
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rewards", required=True)
    ap.add_argument("--strategies", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rewards = json.loads(open(args.rewards).read())
    strategies = json.loads(open(args.strategies).read())

    # Aggregate rewards by arm
    buckets = {}
    for r in rewards:
        arm = r["arm_id"]
        buckets.setdefault(arm, []).append(r["reward"])

    for strat in strategies:
        arm_id = strat.get("arm_id")
        if arm_id in buckets:
            arr = buckets[arm_id]
            mean = sum(arr) / len(arr)
            # Simple smoothing & scaling to adjust base_weight
            strat["base_weight"] = round((0.5 * strat.get("base_weight", 1.0) + 0.5 * (mean + 0.5)), 4)
            strat["stats"] = {"samples": len(arr), "recent_mean_reward": mean}
        else:
            strat.setdefault("stats", {})["samples"] = strat["stats"].get("samples", 0) if "stats" in strat else 0

    with open(args.out, "w") as f:
        json.dump(strategies, f, indent=2)
    print(f"Updated strategies -> {args.out}")

if __name__ == "__main__":
    main()