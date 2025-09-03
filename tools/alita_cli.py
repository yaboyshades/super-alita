#!/usr/bin/env python3
"""Super-Alita minimal CLI for local interaction.

Usage:
  python -m tools.alita_cli query "What is the capital of France?" --stream

Env vars:
  ALITA_BASE_URL       Base URL (default: http://127.0.0.1:8080)
  ALITA_API_KEY        API key (optional unless server requires it)
  ALITA_API_HEADER     Header name (default: Authorization)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import requests

CONFIG_PATH = Path(
    os.getenv("ALITA_CONFIG_FILE", Path.home() / ".alita" / "config.json")
)


def load_config() -> dict[str, Any]:
    try:
        if CONFIG_PATH.exists():
            return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}


def save_config(data: dict[str, Any]) -> None:
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _headers() -> dict[str, str]:
    headers: dict[str, str] = {"Content-Type": "application/json"}
    cfg = load_config()
    key = os.getenv("ALITA_API_KEY", cfg.get("api_key", "")).strip()
    header_name = os.getenv("ALITA_API_HEADER", cfg.get("api_header", "Authorization"))
    if key:
        # Use Bearer convention
        headers[header_name] = (
            f"Bearer {key}" if header_name.lower() == "authorization" else key
        )
    return headers


def stream_sse(url: str, json_body: dict[str, Any]) -> Iterator[dict[str, Any]]:
    with requests.post(url, json=json_body, headers=_headers(), stream=True) as r:
        r.raise_for_status()
        current_event: str | None = None
        for line in r.iter_lines(decode_unicode=True):
            if not line:
                continue
            if isinstance(line, bytes):
                line = line.decode("utf-8", errors="ignore")
            if line.startswith(":"):
                # heartbeat
                continue
            if line.startswith("event:"):
                current_event = line.split(":", 1)[1].strip() or None
                continue
            if line.startswith("id:"):
                # ignore; id is embedded in data payload too
                continue
            if line.startswith("data:"):
                payload = line[5:].strip()
                try:
                    obj = json.loads(payload)
                    if isinstance(obj, dict) and current_event:
                        obj.setdefault("__event", current_event)
                    yield obj
                except Exception:
                    d = {"raw": payload}
                    if current_event:
                        d["__event"] = current_event
                    yield d


def cmd_query(args: argparse.Namespace) -> int:
    cfg = load_config()
    base = os.getenv(
        "ALITA_BASE_URL", cfg.get("base_url", "http://127.0.0.1:8080")
    ).rstrip("/")
    url = f"{base}/api/v1/query"
    body = {
        "prompt": args.prompt,
        "mode": args.mode,
        "session": args.session,
        "stream": args.stream,
        "max_tokens": args.max_tokens,
    }

    try:
        if args.stream:
            printed_header = False
            for payload in stream_sse(url, body):
                if isinstance(payload, dict):
                    ev = payload.get("__event")
                    # On start frame, show rate-limit if available
                    if not printed_header and ev == "start" and args.show_metadata:
                        rl = payload.get("rate_limit") or {}
                        if rl:
                            print(
                                f"[rate-limit] limit={rl.get('limit')} remaining={rl.get('remaining')} reset_in={rl.get('reset_in')}s"
                            )
                        printed_header = True
                    # Content frames
                    if "content" in payload:
                        print(payload["content"], end="", flush=True)
                    # Done marker from server
                    if payload.get("type") == "done":
                        print()
                else:
                    # Fallback print raw
                    print(payload)
            return 0
        else:
            r = requests.post(url, json=body, headers=_headers(), timeout=120)
            r.raise_for_status()
            data = r.json()
            if args.output == "json":
                print(json.dumps(data, ensure_ascii=False))
            elif args.output == "rich":
                try:
                    from rich.console import Console
                    from rich.table import Table

                    console = Console()
                    table = Table(title="Super‑Alita Response")
                    table.add_column("Field", style="cyan")
                    table.add_column("Value", style="magenta")
                    table.add_row("Answer", data.get("answer", ""))
                    model = data.get("model") or {}
                    if args.show_metadata:
                        table.add_row("Session", data.get("session", ""))
                        table.add_row("Mode", data.get("mode", ""))
                        table.add_row("Model", model.get("model", "unknown"))
                        table.add_row("Provider", model.get("provider", "unknown"))
                        rl = data.get("rate_limit") or {}
                        if rl:
                            table.add_row("RL Limit", str(rl.get("limit", "")))
                            table.add_row("RL Remaining", str(rl.get("remaining", "")))
                            table.add_row("RL Reset In", str(rl.get("reset_in", "")))
                    console.print(table)
                except Exception:
                    print(data.get("answer", ""))
            else:
                print(data.get("answer", ""))
            return 0
    except requests.HTTPError as e:
        sys.stderr.write(
            f"HTTP error: {e}\n{getattr(e, 'response', None) and e.response.text}\n"
        )
        return 2
    except Exception as e:  # pragma: no cover - CLI convenience
        sys.stderr.write(f"Error: {e}\n")
        return 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="alita", description="Super-Alita CLI")
    sub = ap.add_subparsers(dest="cmd", required=True)

    q = sub.add_parser("query", help="Send a prompt to Super-Alita")
    q.add_argument("prompt", help="Your prompt text")
    q.add_argument(
        "--mode",
        default="hybrid",
        choices=["hybrid", "neural", "symbolic"],
        help="Reasoning mode (advisory)",
    )
    q.add_argument("--session", default=None, help="Session ID for context")
    q.add_argument("--stream", action="store_true", help="Stream response via SSE")
    q.add_argument("--max-tokens", type=int, default=None, help="Token cap (advisory)")
    q.add_argument(
        "--output",
        choices=["plain", "json", "rich"],
        default="plain",
        help="Output format",
    )
    q.add_argument(
        "--show-metadata",
        action="store_true",
        help="Include model and session info in output",
    )
    q.set_defaults(func=cmd_query)

    c = sub.add_parser("configure", help="Write CLI configuration file")
    c.add_argument("--base-url", required=False, help="Default base URL")
    c.add_argument("--api-key", required=False, help="Default API key")
    c.add_argument(
        "--api-header",
        required=False,
        default=None,
        help="Header name for API key (default Authorization)",
    )

    def cmd_configure(args: argparse.Namespace) -> int:
        cfg = load_config()
        if args.base_url:
            cfg["base_url"] = args.base_url
        if args.api_key:
            cfg["api_key"] = args.api_key
        if args.api_header is not None:
            cfg["api_header"] = args.api_header
        save_config(cfg)
        print(f"Saved config to {CONFIG_PATH}")
        return 0

    c.set_defaults(func=cmd_configure)

    # --- Keys subcommands ---
    k = sub.add_parser("keys", help="Manage API keys")
    ksub = k.add_subparsers(dest="action", required=True)

    kc = ksub.add_parser("create", help="Create a new API key (admin or open reg)")
    kc.add_argument("owner", help="Owner email or id")
    kc.add_argument("--ttl-hours", type=int, default=None, help="Key TTL in hours")
    kc.add_argument(
        "--admin-key",
        default=None,
        help="Admin key; defaults to config 'admin_key' or env ALITA_ADMIN_KEY",
    )

    def cmd_keys_create(args: argparse.Namespace) -> int:
        cfg = load_config()
        base = os.getenv(
            "ALITA_BASE_URL", cfg.get("base_url", "http://127.0.0.1:8080")
        ).rstrip("/")
        url = f"{base}/api/v1/auth/keys"
        headers = _headers()
        admin = (
            args.admin_key
            or cfg.get("admin_key")
            or os.getenv("ALITA_ADMIN_KEY", "").strip()
        )
        if admin:
            headers[
                os.getenv("ALITA_API_HEADER", cfg.get("api_header", "Authorization"))
            ] = f"Bearer {admin}"
        r = requests.post(
            url,
            json={"owner": args.owner, "ttl_hours": args.ttl_hours},
            headers=headers,
            timeout=30,
        )
        r.raise_for_status()
        print(r.text)
        return 0

    kc.set_defaults(func=cmd_keys_create)

    kr = ksub.add_parser("rotate", help="Rotate current API key")

    def cmd_keys_rotate(args: argparse.Namespace) -> int:
        cfg = load_config()
        base = os.getenv(
            "ALITA_BASE_URL", cfg.get("base_url", "http://127.0.0.1:8080")
        ).rstrip("/")
        url = f"{base}/api/v1/auth/keys/rotate"
        r = requests.post(url, headers=_headers(), timeout=30)
        r.raise_for_status()
        print(r.text)
        return 0

    kr.set_defaults(func=cmd_keys_rotate)

    kv = ksub.add_parser("revoke", help="Revoke an API key (admin)")
    kv.add_argument("--key", default=None)
    kv.add_argument("--key-id", default=None)
    kv.add_argument("--admin-key", default=None)

    def cmd_keys_revoke(args: argparse.Namespace) -> int:
        cfg = load_config()
        base = os.getenv(
            "ALITA_BASE_URL", cfg.get("base_url", "http://127.0.0.1:8080")
        ).rstrip("/")
        url = f"{base}/api/v1/auth/keys/revoke"
        headers = _headers()
        admin = (
            args.admin_key
            or cfg.get("admin_key")
            or os.getenv("ALITA_ADMIN_KEY", "").strip()
        )
        if admin:
            headers[
                os.getenv("ALITA_API_HEADER", cfg.get("api_header", "Authorization"))
            ] = f"Bearer {admin}"
        payload: dict[str, Any] = {}
        if args.key:
            payload["key"] = args.key
        if args.key_id:
            payload["key_id"] = args.key_id
        r = requests.post(url, json=payload, headers=headers, timeout=30)
        r.raise_for_status()
        print(r.text)
        return 0

    kv.set_defaults(func=cmd_keys_revoke)

    km = ksub.add_parser("me", help="Show info for current key")

    def cmd_keys_me(args: argparse.Namespace) -> int:
        cfg = load_config()
        base = os.getenv(
            "ALITA_BASE_URL", cfg.get("base_url", "http://127.0.0.1:8080")
        ).rstrip("/")
        url = f"{base}/api/v1/auth/keys/me"
        r = requests.get(url, headers=_headers(), timeout=30)
        r.raise_for_status()
        print(r.text)
        return 0

    km.set_defaults(func=cmd_keys_me)

    kl = ksub.add_parser("list", help="List keys (admin)")
    kl.add_argument("--admin-key", default=None)

    def cmd_keys_list(args: argparse.Namespace) -> int:
        cfg = load_config()
        base = os.getenv(
            "ALITA_BASE_URL", cfg.get("base_url", "http://127.0.0.1:8080")
        ).rstrip("/")
        url = f"{base}/api/v1/auth/keys"
        headers = _headers()
        admin = (
            args.admin_key
            or cfg.get("admin_key")
            or os.getenv("ALITA_ADMIN_KEY", "").strip()
        )
        if admin:
            headers[
                os.getenv("ALITA_API_HEADER", cfg.get("api_header", "Authorization"))
            ] = f"Bearer {admin}"
        r = requests.get(url, headers=headers, timeout=30)
        r.raise_for_status()
        print(r.text)
        return 0

    kl.set_defaults(func=cmd_keys_list)

    args = ap.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
