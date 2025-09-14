#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path


def patch(path: Path, schema_path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    pretty = json.dumps(schema, indent=2, ensure_ascii=False)
    new = text.replace("{{SCHEMA_JSON}}", pretty)
    path.write_text(new, encoding="utf-8")
    print(f"Patched schema into {path}")


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    patch(
        root / ".github/chatmodes/cma-architect.chatmode.yaml",
        root / ".github/chatmodes/schema.json",
    )
