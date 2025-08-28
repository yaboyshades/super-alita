#!/usr/bin/env python
import argparse, json, pathlib, hashlib

def simple_summarize(content: list[dict]) -> str:
    # Placeholder summarization logic
    goals = {}
    for c in content:
        goals[c["goal"]] = goals.get(c["goal"], 0) + 1
    lines = ["Distilled Summary:", ""]
    for g, count in goals.items():
        lines.append(f"- Goal '{g}' occurred {count} times.")
    return "\n".join(lines)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--max-chunk-tokens", type=int, default=2000)
    args = ap.parse_args()

    in_dir = pathlib.Path(args.input)
    out_dir = pathlib.Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    episodes = []
    for f in in_dir.glob("*.json"):
        try:
            episodes.append(json.loads(f.read_text()))
        except Exception:
            pass

    summary_text = simple_summarize(episodes)
    digest = hashlib.sha256(summary_text.encode()).hexdigest()[:12]
    out_file = out_dir / f"distilled_{digest}.md"
    out_file.write_text(summary_text)
    latest_link = out_dir / "LATEST.md"
    latest_link.write_text(summary_text)
    print(f"Wrote distilled summary to {out_file} and LATEST.md")

if __name__ == "__main__":
    main()