"""
Ollama GPU/CPU split tuner: tries multiple gpu_layers values and reports speed.

Usage examples:
  python tools/ollama_gpu_cpu_tuner.py \
    --model llama3:8b --layers 8 12 16 20 24 28 32 --num-thread 12 \
    --prompt "Summarize: The quick brown fox jumps over the lazy dog."

Notes:
- Ensure Ollama is running locally (http://localhost:11434) with the requested model pulled.
- gpu_layers: layers offloaded to GPU; remaining layers run on CPU.
- num_thread: threads for CPU-bound layers; use your physical core count.
"""

from __future__ import annotations

import argparse
import json
import time
from typing import Any, Iterable

import requests


def generate_once(
    host: str,
    model: str,
    prompt: str,
    gpu_layers: int,
    num_thread: int | None,
) -> dict[str, Any]:
    url = f"{host.rstrip('/')}/api/generate"
    options: dict[str, Any] = {"num_gpu": int(gpu_layers)}
    if num_thread is not None:
        options["num_thread"] = int(num_thread)

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": True,
        "options": options,
    }

    started = time.perf_counter()
    total_chars = 0
    eval_count = None
    eval_duration_ns = None
    with requests.post(url, json=payload, stream=True, timeout=900) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line:
                continue
            msg = json.loads(line.decode("utf-8"))
            if "response" in msg and msg["response"]:
                chunk = msg["response"]
                total_chars += len(chunk)
            if msg.get("done"):
                eval_count = msg.get("eval_count")
                eval_duration_ns = msg.get("eval_duration")
                break
    elapsed = time.perf_counter() - started

    tokens_per_s = None
    if eval_count and eval_duration_ns:
        # Ollama reports durations in nanoseconds.
        seconds = float(eval_duration_ns) / 1e9
        if seconds > 0:
            tokens_per_s = float(eval_count) / seconds

    return {
        "gpu_layers": gpu_layers,
        "chars": total_chars,
        "elapsed_s": elapsed,
        "eval_tokens": eval_count,
        "eval_duration_ns": eval_duration_ns,
        "tokens_per_s": tokens_per_s,
    }


def parse_layers(specs: Iterable[str]) -> list[int]:
    vals: list[int] = []
    for s in specs:
        if "-" in s:
            a, b = s.split("-", 1)
            start, end = int(a), int(b)
            step = 2 if end - start > 10 else 1
            vals.extend(range(start, end + 1, step))
        else:
            vals.append(int(s))
    # deduplicate + sort
    return sorted(set(vals))


def main() -> None:
    ap = argparse.ArgumentParser(description="Tune Ollama gpu_layers for CPU+GPU split")
    ap.add_argument("--model", required=True, help="Model name (e.g., llama3:8b)")
    ap.add_argument(
        "--layers",
        nargs="+",
        required=True,
        help="List or ranges of gpu_layers (e.g., 8 12 16 or 8-32)",
    )
    ap.add_argument("--num-thread", type=int, default=None, help="CPU threads for CPU layers")
    ap.add_argument("--prompt", default="Hello!", help="Short prompt to measure")
    ap.add_argument("--host", default="http://localhost:11434", help="Ollama host URL")
    args = ap.parse_args()

    layers = parse_layers(args.layers)
    print(f"Tuning model={args.model} layers={layers} num_thread={args.num_thread}")
    results = []
    for L in layers:
        print(f"\n==> Testing gpu_layers={L} ...", flush=True)
        try:
            res = generate_once(args.host, args.model, args.prompt, L, args.num_thread)
        except Exception as e:  # noqa: BLE001
            print(f"gpu_layers={L} ERROR: {e}")
            continue
        results.append(res)
        tps = res.get("tokens_per_s")
        tps_str = f"{tps:.2f} tok/s" if tps else "n/a"
        print(
            f"gpu_layers={res['gpu_layers']} elapsed={res['elapsed_s']:.2f}s "
            f"tokens={res.get('eval_tokens')} speed={tps_str}"
        )

    if not results:
        print("No successful runs. Ensure Ollama is running and the model is available.")
        return

    # Sort by tokens_per_s then elapsed
    results.sort(key=lambda r: (-(r.get("tokens_per_s") or 0.0), r["elapsed_s"]))
    best = results[0]
    print("\nBest configuration:")
    tps = best.get("tokens_per_s")
    tps_str = f"{tps:.2f} tok/s" if tps else "n/a"
    print(
        f"gpu_layers={best['gpu_layers']} elapsed={best['elapsed_s']:.2f}s "
        f"tokens={best.get('eval_tokens')} speed={tps_str}"
    )


if __name__ == "__main__":
    main()
