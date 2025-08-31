from __future__ import annotations

import os
from pathlib import Path

from src.security.api_key_store import APIKeyStore


def test_store_add_verify_rotate_revoke(tmp_path: Path) -> None:
    store_path = tmp_path / "keys.json"
    os.environ["ALITA_API_KEY_STORE"] = str(store_path)
    store = APIKeyStore.from_env()

    # Create with TTL
    result = store.add("owner@example.com", {"role": "test"}, ttl_hours=1)
    key = result["api_key"]
    assert store.verify(key) is True

    # Rotate
    rotated = store.rotate(key)
    assert rotated is not None
    assert store.verify(key) is False  # old key revoked
    new_key = rotated["api_key"]
    assert store.verify(new_key) is True

    # Revoke new
    ok = store.revoke(new_key)
    assert ok is True
    assert store.verify(new_key) is False


def test_store_expired_key(tmp_path: Path) -> None:
    store_path = tmp_path / "keys.json"
    os.environ["ALITA_API_KEY_STORE"] = str(store_path)
    store = APIKeyStore.from_env()

    # TTL 0 => immediately expired
    result = store.add("expired@example.com", ttl_hours=0)
    k = result["api_key"]
    assert store.verify(k) is False

