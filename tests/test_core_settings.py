import importlib

import pytest

import src.core.settings as settings


def test_compute_retry_schedule_basic() -> None:
    result = settings.compute_retry_schedule(
        3, 0.5, multiplier=2.0, jitter_ratio=0.0
    )
    assert result == pytest.approx([0.5, 1.0, 2.0])


def test_compute_retry_schedule_validates_inputs() -> None:
    with pytest.raises(ValueError):
        settings.compute_retry_schedule(1, 0.0)
    with pytest.raises(ValueError):
        settings.compute_retry_schedule(1, 0.5, multiplier=0.0)
    with pytest.raises(ValueError):
        settings.compute_retry_schedule(1, 0.5, jitter_ratio=-0.1)


def test_retry_schedule_respects_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM_RETRIES", "2")
    monkeypatch.setenv("LLM_RETRY_BASE_DELAY_SEC", "0.25")
    monkeypatch.setenv("LLM_RETRY_MULTIPLIER", "1.5")
    monkeypatch.setenv("LLM_RETRY_JITTER_RATIO", "0.2")

    module = importlib.reload(settings)
    try:
        assert pytest.approx([0.3, 0.45]) == module.LLM_RETRY_SCHEDULE
    finally:
        monkeypatch.delenv("LLM_RETRIES", raising=False)
        monkeypatch.delenv("LLM_RETRY_BASE_DELAY_SEC", raising=False)
        monkeypatch.delenv("LLM_RETRY_MULTIPLIER", raising=False)
        monkeypatch.delenv("LLM_RETRY_JITTER_RATIO", raising=False)
        importlib.reload(settings)
