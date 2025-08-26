from __future__ import annotations

import pytest

from src.core.yaml_utils import safe_dump, safe_load


def test_safe_roundtrip_simple_map():
    data = {"a": 1, "b": ["x", "y"]}
    y = safe_dump(data)
    got = safe_load(y)
    assert got == data


def test_safe_blocks_python_tags():
    malicious = "!!python/object/apply:os.system ['echo HACKED']"
    with pytest.raises(Exception):
        _ = safe_load(malicious)
