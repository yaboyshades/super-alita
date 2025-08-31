from src.plugins.oak_core.failure_learning import (
    propose_resilience_patches,
)


def test_propose_resilience_patches_from_summary() -> None:
    summary = {
        "total_events": 5,
        "clusters": {
            "toolX|timeout waiting for": {
                "tool": "toolX",
                "count": 4,
                "last_seen": 1,
                "example": {},
            },
            "toolY|random error": {
                "tool": "toolY",
                "count": 2,
                "last_seen": 1,
                "example": {},
            },
        },
    }
    props = propose_resilience_patches(summary)
    assert any("health-check" in p for p in props)
