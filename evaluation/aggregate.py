#!/usr/bin/env python
import json
import pathlib
import statistics as stats
import sys


def main():
    if len(sys.argv) < 3:
        print("Usage: aggregate.py <results_dir> <out_file>")
        sys.exit(1)
    results_dir = pathlib.Path(sys.argv[1])
    out_file = pathlib.Path(sys.argv[2])
    records = []
    for f in results_dir.glob("*.json"):
        try:
            records.extend(json.loads(f.read_text()))
        except Exception as e:
            print(f"Skipping {f}: {e}", file=sys.stderr)
    if not records:
        print("No records found")
        sys.exit(1)
    grouped = {}
    for r in records:
        key = (r["model"], r["prompt_style"])
        grouped.setdefault(key, []).append(r)
    summary = []
    for (model, style), rows in grouped.items():
        lat = [r["latency_sec"] for r in rows]
        q = [r["quality_score"] for r in rows if r.get("quality_score") is not None]
        summary.append({
            "model": model,
            "prompt_style": style,
            "cases": len(rows),
            "latency_mean": sum(lat)/len(lat),
            "latency_p95": stats.quantiles(lat, n=20)[18] if len(lat) > 1 else lat[0],
            "quality_mean": sum(q)/len(q) if q else None
        })
    out_file.write_text(json.dumps(summary, indent=2))
    print(f"Aggregated summary -> {out_file}")

if __name__ == "__main__":
    main()