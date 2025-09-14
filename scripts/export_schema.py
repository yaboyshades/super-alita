#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _get_schema() -> dict:
    # Local import to avoid path/lint ordering issues
    from scripts.models import ChatModeConfig

    return ChatModeConfig.model_json_schema()


def main() -> None:
    schema = _get_schema()
    out = ROOT / ".github/chatmodes/schema.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(schema, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
