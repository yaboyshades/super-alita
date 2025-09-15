#!/usr/bin/env python3
"""Render validation messages provided via CLI arguments to JSON.

Usage:
    python scripts/validation_messages_to_json.py status::Message ...

Each positional argument is treated as ``<status>::<message>``.  When the
status prefix is omitted the message inherits the default status.  The script is
primarily used by Bash validation harnesses that gather human-friendly strings
in arrays and need to ship JSON to downstream tooling.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(
        description="Convert validation messages to JSON for pipelines",
    )
    parser.add_argument(
        "entries",
        nargs="*",
        help=(
            "Validation messages captured from Bash arrays. Entries use "
            "'<status>::<message>' format; status defaults to 'info'."
        ),
    )
    parser.add_argument(
        "--default-status",
        default="info",
        help="Status applied when an entry omits the status prefix",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="Indentation level for the JSON payload",
    )
    return parser.parse_args()


def main() -> int:
    """Entry point."""

    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from src.utils.validation_output import to_json

    payload = to_json(
        args.entries, default_status=args.default_status, indent=args.indent
    )
    sys.stdout.write(payload + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
