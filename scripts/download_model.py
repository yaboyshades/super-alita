#!/usr/bin/env python3
"""Download a Hugging Face model snapshot into the local models/ directory.

Usage:
  python scripts/download_model.py --model gpt2 --revision main

Dependencies:
  pip install huggingface_hub

Environment variables (optional):
  HF_TOKEN: Hugging Face token if the repo is gated

The script places the model under: models/<model-id>
"""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from huggingface_hub import snapshot_download
except ImportError:  # pragma: no cover
    print("[ERROR] huggingface_hub not installed.")
    print("[HINT] Run: pip install huggingface_hub")
    print("[HINT] Or install project with: pip install -e .")
    raise SystemExit(1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model",
        required=True,
        help="Model repo id (e.g. gpt2 or TheBloke/MyModel-GPTQ)",
    )
    ap.add_argument("--revision", default=None, help="Optional revision / commit / tag")
    ap.add_argument("--cache_dir", default=None, help="Optional custom cache directory")
    args = ap.parse_args()

    target_root = Path("models") / args.model.replace("/", "--")
    target_root.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Downloading {args.model} -> {target_root}")
    snapshot_download(
        repo_id=args.model,
        revision=args.revision,
        local_dir=str(target_root),
        local_dir_use_symlinks=False,
        cache_dir=args.cache_dir,
        resume_download=True,
    )
    print("[DONE] Model download complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
