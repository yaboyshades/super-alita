import asyncio

import pytest

from src.orchestration.reliability_manager import (
    ReliabilityConfig,
    ReliabilityManager,
)


@pytest.mark.asyncio
async def test_transient_retry_success() -> None:
    attempts: dict[str, int] = {"count": 0}

    async def flaky_fn() -> str:
        attempts["count"] += 1
        if attempts["count"] < 3:
            # Contains keyword 'timeout' to classify as transient
            raise RuntimeError("timeout during upstream call")
        return "ok"

    rm = ReliabilityManager(
        ReliabilityConfig(
            max_retries=5,
            base_backoff_ms=5,
            enable_jitter=False,
            circuit_break_threshold=10,
        )
    )
    result = await rm.execute_with_retries("flaky", flaky_fn, timeout_s=1)
    assert result["status"] == "success"
    assert result["attempts"] == 3
    snap = rm.snapshot()["flaky"]
    assert snap["failures"] == 2
    assert snap["successes"] >= 1


@pytest.mark.asyncio
async def test_permanent_failure_exhausts_retries() -> None:
    async def always_fail() -> None:
        # No transient marker => permanent classification
        raise ValueError("logic violation condition")

    rm = ReliabilityManager(
        ReliabilityConfig(
            max_retries=2,
            base_backoff_ms=1,
            enable_jitter=False,
            circuit_break_threshold=5,
        )
    )
    result = await rm.execute_with_retries(
        "perm_fail", always_fail, timeout_s=1
    )
    assert result["status"] == "failed"
    # Should not retry permanent errors -> single attempt
    assert result["attempts"] == 1
    assert result["classified_error"] == "permanent"
    snap = rm.snapshot()["perm_fail"]
    assert snap["failures"] == 1
    assert snap["successes"] == 0


@pytest.mark.asyncio
async def test_circuit_breaker_trips() -> None:
    async def always_fail() -> None:
        raise RuntimeError("boom")

    cfg = ReliabilityConfig(
        max_retries=0,
        base_backoff_ms=1,
        enable_jitter=False,
        circuit_break_threshold=3,
        circuit_reset_after_s=1,
    )
    rm = ReliabilityManager(cfg)
    for _ in range(3):
        r = await rm.execute_with_retries("cb", always_fail, timeout_s=1)
        assert r["status"] == "failed"
    snap = rm.snapshot()["cb"]
    assert snap["circuit_open_until"] is not None
    # Should skip while open
    r_open = await rm.execute_with_retries("cb", always_fail, timeout_s=1)
    assert r_open["status"] == "skipped"
    assert r_open.get("reason") == "circuit_open"
    await asyncio.sleep(cfg.circuit_reset_after_s + 0.1)
    # After reset it should attempt again (and fail)
    r_after = await rm.execute_with_retries("cb", always_fail, timeout_s=1)
    assert r_after["status"] == "failed"


@pytest.mark.asyncio
async def test_permanent_error_no_retries() -> None:
    attempts = {"count": 0}

    class PermanentError(RuntimeError):
        """Custom permanent error."""

    async def fail_permanent() -> None:
        attempts["count"] += 1
        raise PermanentError("hard violation")

    cfg = ReliabilityConfig(
        max_retries=5,
        base_backoff_ms=10,
        enable_jitter=False,
        circuit_break_threshold=50,
    )
    rm = ReliabilityManager(cfg)
    result = await rm.execute_with_retries(
        "perm_once", fail_permanent, timeout_s=1
    )
    assert result["status"] == "failed"
    assert result["classified_error"] == "permanent"
    assert result["attempts"] == 1
    assert attempts["count"] == 1


@pytest.mark.asyncio
async def test_transient_retries_exhausted() -> None:
    attempts = {"count": 0}

    async def fail_transient() -> None:
        attempts["count"] += 1
        raise RuntimeError("connection reset by peer")

    cfg = ReliabilityConfig(
        max_retries=3,
        base_backoff_ms=1,
        enable_jitter=False,
        circuit_break_threshold=10,
    )
    rm = ReliabilityManager(cfg)
    result = await rm.execute_with_retries(
        "transient", fail_transient, timeout_s=1
    )
    assert result["status"] == "failed"
    assert result["attempts"] == 4  # 1 + 3 retries
    assert result["classified_error"] == "transient"
