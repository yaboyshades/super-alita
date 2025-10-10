"""
Configure and test Ollama CPU+GPU split via options.

Usage examples:
  python tools/ollama_cpu_gpu_split_example.py \
    --model llama3:8b --gpu-layers 20 --num-thread 12 --prompt "Hello!"

Notes:
- gpu_layers: number of transformer layers offloaded to GPU. Remaining layers run on CPU.
- num_thread: CPU threads used by CPU-bound layers.
- Ensure Ollama is running locally (default: http://localhost:11434).
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

import requests


def generate(
    model: str,
    prompt: str,
    num_gpu: int | None,
    num_thread: int | None,
    host: str = "http://localhost:11434",
) -> str:
    url = f"{host.rstrip('/')}/api/generate"
    options: dict[str, Any] = {}
    if num_gpu is not None:
        options["num_gpu"] = int(num_gpu)
    if num_thread is not None:
        options["num_thread"] = int(num_thread)

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": True,
        "options": options,
    }

    with requests.post(url, json=payload, stream=True, timeout=600) as r:
        r.raise_for_status()
        chunks: list[str] = []
        for line in r.iter_lines():
            if not line:
                continue
            data = json.loads(line.decode("utf-8"))
            chunk = data.get("response", "")
            if chunk:
                sys.stdout.write(chunk)
                sys.stdout.flush()
                chunks.append(chunk)
        # final newline
        print()
        return "".join(chunks)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ollama CPU+GPU split test")
    parser.add_argument(
        "--model", required=True, help="Model name (e.g., llama3:8b or custom)"
    )
    parser.add_argument("--prompt", default="Say hi!", help="Prompt text")
    parser.add_argument(
        "--gpu-layers",
        type=int,
        default=None,
        help="Layers offloaded to GPU (mapped to num_gpu)",
    )
    parser.add_argument(
        "--num-thread",
        type=int,
        default=None,
        help="CPU threads for CPU-bound layers",
    )
    parser.add_argument(
        "--host", default="http://localhost:11434", help="Ollama host URL"
    )
    args = parser.parse_args()

    generate(
        model=args.model,
        prompt=args.prompt,
        num_gpu=args.gpu_layers,
        num_thread=args.num_thread,
        host=args.host,
    )


if __name__ == "__main__":
    main()
