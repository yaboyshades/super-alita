#!/usr/bin/env python3
"""
Thin CLI to access native DeepCode tools for VS Code integrations.

Usage:
  python tools/deepcode_cli.py request --task-kind web_scraper --requirements "..." [--repo-path .]
  python tools/deepcode_cli.py latest
  python tools/deepcode_cli.py apply [--path file1 --path file2 ...]
  python tools/deepcode_cli.py analyze-file --file path/to/file.py

Prints JSON to stdout.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import Any

from src.deepcode import analyze_current_file as dc_analyze_current_file
from src.plugins.native_deepcode_plugin import NativeDeepCodePlugin

_plugin: NativeDeepCodePlugin | None = None


async def _ensure_plugin() -> NativeDeepCodePlugin:
    global _plugin
    if _plugin is None:
        _plugin = NativeDeepCodePlugin()
        # Safe to setup with no event bus/store for direct tool use
        await _plugin.setup(event_bus=None, store=None, config={})
        await _plugin.start()
    return _plugin


async def cmd_request(args: argparse.Namespace) -> dict[str, Any]:
    plugin = await _ensure_plugin()
    payload = {
        "task_kind": args.task_kind,
        "requirements": args.requirements,
        "repo_path": args.repo_path,
    }
    return await plugin.invoke_tool("deepcode_request", payload)


async def cmd_latest(_args: argparse.Namespace) -> dict[str, Any]:
    plugin = await _ensure_plugin()
    return await plugin.invoke_tool("deepcode_latest", {})


async def cmd_apply(args: argparse.Namespace) -> dict[str, Any]:
    plugin = await _ensure_plugin()
    payload: dict[str, Any] = {}
    if args.path:
        payload["paths"] = args.path
    return await plugin.invoke_tool("deepcode_apply", payload)


async def cmd_analyze_file(args: argparse.Namespace) -> dict[str, Any]:
    file_path = args.file
    return await dc_analyze_current_file(file_path)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="deepcode_cli")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_req = sub.add_parser("request")
    p_req.add_argument("--task-kind", required=True)
    p_req.add_argument("--requirements", required=True)
    p_req.add_argument("--repo-path", default=str(Path.cwd()))

    sub.add_parser("latest")

    p_apply = sub.add_parser("apply")
    p_apply.add_argument("--path", action="append")

    p_an = sub.add_parser("analyze-file")
    p_an.add_argument("--file", required=True)

    return p


async def _main_async(argv: list[str]) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.cmd == "request":
        out = await cmd_request(args)
    elif args.cmd == "latest":
        out = await cmd_latest(args)
    elif args.cmd == "apply":
        out = await cmd_apply(args)
    elif args.cmd == "analyze-file":
        out = await cmd_analyze_file(args)
    else:  # pragma: no cover - guarded by argparse
        raise SystemExit(2)

    print(json.dumps(out, ensure_ascii=False))
    return 0


def main() -> int:
    return asyncio.run(_main_async(os.sys.argv[1:]))


if __name__ == "__main__":
    raise SystemExit(main())
