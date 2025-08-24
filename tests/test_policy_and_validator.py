import asyncio, os, json
import pytest
from src.prompt.policy_constants import build_header, PROMPT_VERSION, SCHEMA_VERSION
from src.utils.schema_validator import validate_alg_extraction
from src.utils.telemetry import telemetry_footer


def test_policy_header_includes_versions():
    hdr = build_header("algorithm_extraction")
    assert PROMPT_VERSION in hdr
    assert SCHEMA_VERSION in hdr
    assert "fallback_policy" in hdr
    assert "unknown_field_policy" in hdr


def test_schema_validation_ok_minimal():
    payload = {
        "alg_extraction_v1": {
            "algorithms": [], "components": [], "hyperparameters": [],
            "equations": [], "implementation_plan": [], "sources": []
        },
        "validation_summary": {"missing_required_fields": [], "unknown_fields": []},
        "telemetry": {"prompt_version": PROMPT_VERSION, "schema_version": SCHEMA_VERSION, "retrieval_mode":"segmented", "retrieval_rounds":1, "segments_used":0}
    }
    ok, msg = validate_alg_extraction(payload)
    assert ok, msg


def test_telem_footer():
    t = telemetry_footer(2, 5, {"algorithms":1})
    assert t["telemetry"]["retrieval_rounds"] == 2
    assert t["telemetry"]["segments_used"] == 5
