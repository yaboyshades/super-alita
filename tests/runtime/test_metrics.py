"""Verify Prometheus metrics export for runtime endpoints."""

import asyncio
import re

from fastapi.testclient import TestClient

from src.main import create_app


def _metric_value(text: str, name: str, labels: dict[str, str]) -> float:
    label_str = ",".join(f'{k}="{v}"' for k, v in labels.items())
    pattern = rf"{name}{{{label_str}}} (\d+(?:\.\d+)?)"
    match = re.search(pattern, text)
    assert match, f"metric {name} with labels {labels} not found"
    return float(match.group(1))


def test_metrics_endpoint_tracks_requests_and_events() -> None:
    app = create_app()
    client = TestClient(app)

    # Trigger request metrics
    client.post("/v1/chat/stream", json={"message": "hi", "session_id": "m"})
    client.get("/tools/catalog")

    # Emit event bus metric
    async def _emit() -> None:
        await app.state.event_bus.emit({"type": "unit_event", "source": "test"})

    asyncio.run(_emit())

    resp = client.get("/metrics")
    assert resp.status_code == 200
    text = resp.text

    assert (
        _metric_value(
            text,
            "request_total",
            {"endpoint": "/v1/chat/stream", "method": "POST", "status": "200"},
        )
        >= 1
    )
    assert (
        _metric_value(
            text,
            "request_total",
            {"endpoint": "/tools/catalog", "method": "GET", "status": "200"},
        )
        >= 1
    )
    assert (
        _metric_value(
            text,
            "event_bus_messages_total",
            {"event_type": "unit_event", "source": "test", "status": "success"},
        )
        >= 1
    )

