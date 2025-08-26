from __future__ import annotations

import hashlib
import json
import re

_WS = re.compile(r"\s+")
_PUNC = re.compile(r"[^\w\s]", flags=re.UNICODE)


def normalize_text(text: str) -> str:
    """Lowercase, trim, collapse spaces, remove punctuation."""
    if not isinstance(text, str):
        text = str(text)
    t = text.lower().strip()
    t = _PUNC.sub(" ", t)
    t = _WS.sub(" ", t)
    return t.strip()


def blake2b_hexdigest(text: str) -> str:
    return hashlib.blake2b(text.encode("utf-8"), digest_size=32).hexdigest()


def sha256_json(obj) -> str:
    """Stable SHA256 over canonical JSON."""
    data = json.dumps(obj, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(data.encode("utf-8")).hexdigest()
