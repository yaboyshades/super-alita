#!/usr/bin/env python3
from __future__ import annotations

import glob
import os
import re
from pathlib import Path

from ruamel.yaml import YAML

try:
    # type: ignore
    from models import ChatModeConfig, EnvironmentSettings, deep_merge
except Exception:  # pragma: no cover
    # type: ignore
    from scripts.models import ChatModeConfig, EnvironmentSettings, deep_merge

yaml = YAML(typ="safe")
settings = EnvironmentSettings()

FRONT_RE = re.compile(r"^---\s*\n(.*?)\n---\s*(?:\n|$)", re.S)


def load_config(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    data = None
    if path.suffix.lower() in {".yml", ".yaml"}:
        data = yaml.load(text) or {}
    else:
        # .md: extract YAML front matter; if none, try whole file as YAML
        m = FRONT_RE.search(text)
        if m is None:
            loaded = yaml.load(text)
            if isinstance(loaded, dict):
                data = loaded
                data.setdefault("instructions", "")
            else:
                data = {"instructions": text.strip()}
        else:
            front = m.group(1)
            body = text[m.end():]
            data = yaml.load(front) or {}
            if isinstance(data, dict) and "instructions" not in data:
                data["instructions"] = body.strip()
    if not isinstance(data, dict):
        raise ValueError("Top-level YAML must be a mapping")
    return data


def is_mode_file(path: Path) -> bool:
    name = path.name.lower()
    # We only strictly validate files that appear to use the new
    # chatmode schema to avoid breaking legacy chatmode markdown
    # with extensive prose.
    return (
        name.endswith(".chatmode.yml")
        or name.endswith(".chatmode.yaml")
        or name.endswith(".chatmode.md")
    )


def main() -> None:
    env = os.getenv("DEPLOY_ENV", "dev").lower()
    override = settings.for_env(env)
    errors = 0

    paths: list[str] = []
    paths += glob.glob(".github/chatmodes/*.*")
    paths += glob.glob("copilot/modes/*.*")

    seen: set[Path] = set()
    for p in sorted({Path(p) for p in paths}):
        if not p.exists() or p in seen:
            continue
        seen.add(p)
        if not is_mode_file(p):
            # Skip validation of non-schema files in these directories
            continue
        try:
            data = load_config(p)
            merged = deep_merge(data, override)
            # Validate required core fields to catch typos/unknown keys
            ChatModeConfig.model_validate(merged)
            print(f"✔ {p}")
        except Exception as e:  # noqa: BLE001
            print(f"✖ {p}: {e}")
            errors += 1
    if errors:
        raise SystemExit(f"{errors} invalid mode file(s) found")


if __name__ == "__main__":
    main()
