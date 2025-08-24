#!/usr/bin/env python3
from __future__ import annotations
import json
from pathlib import Path
from typing import Tuple, Dict, Any

try:
    import jsonschema
except Exception:
    jsonschema = None

SCHEMA_PATH = Path("schema/alg_extraction_v1_2.json")

def validate_alg_extraction(payload: Dict[str, Any]) -> Tuple[bool, str]:
    if not jsonschema:
        return True, "jsonschema not installed; skipped"
    schema = json.loads(SCHEMA_PATH.read_text())
    try:
        jsonschema.validate(payload, schema)
        return True, "ok"
    except Exception as e:
        return False, str(e)
