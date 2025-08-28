#!/usr/bin/env python
import argparse, json, pathlib, random, datetime as dt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--window-days", type=int, default=1)
    args = ap.parse_args()

    # Placeholder: replace with actual usage API calls.
    items = []
    for i in range(25):
        items.append({
            "timestamp": (dt.datetime.utcnow() - dt.timedelta(minutes=30*i)).isoformat() + "Z",
            "model": random.choice(["fast-model", "quality-model"]),
            "tokens_in": random.randint(50, 1200),
            "tokens_out": random.randint(20, 900),
            "cost_usd": round(random.uniform(0.0003, 0.08), 4)
        })
    total_cost = sum(x["cost_usd"] for x in items)
    md = ["# Daily Cost Report",
          f"Window: last {args.window_days} day(s)",
          "",
          f"Total sessions: {len(items)}",
          f"Total cost (USD): {total_cost:.4f}",
          "",
          "## Breakdown (sample)",
          "| Time | Model | In | Out | Cost |",
          "|------|-------|----|-----|------|"]
    for x in items[:50]:
        md.append(f"| {x['timestamp']} | {x['model']} | {x['tokens_in']} | {x['tokens_out']} | {x['cost_usd']:.4f} |")
    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(md))
    print(f"Wrote cost report to {out_path}")

if __name__ == "__main__":
    main()