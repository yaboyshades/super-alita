import time
from src.core.utils import (
    normalize_text,
    blake2b_hexdigest,
    sha256_json,
    redact_prompt_and_context,
    CircuitBreaker,
    CooldownLRU,
)


def test_normalize_and_hash():
    n = normalize_text(" Hello,  WORLD!! ")
    assert n == "hello world"
    h1 = blake2b_hexdigest(n)
    h2 = blake2b_hexdigest("hello world")
    assert h1 == h2
    sj = sha256_json({"b": 1, "a": 2})
    sj2 = sha256_json({"a": 2, "b": 1})
    assert sj == sj2


def test_redaction_masks_sensitive():
    p = "contact me at user@example.com; api_key=SECRET123SECRET123"
    red_p, red_c, report = redact_prompt_and_context(p, {"token": "abcd" * 10})
    assert "@" not in red_p
    assert "api_key" in red_p
    assert "…" in red_p
    assert "…" in str(red_c)
    assert "prompt_redactions" in report


def test_circuit_breaker_opens_and_recovers():
    cb = CircuitBreaker(failure_threshold=2, recovery_timeout=0.2)
    assert cb.allowed() is True
    cb.on_failure()
    assert cb.allowed() is True
    cb.on_failure()
    assert cb.allowed() is False
    time.sleep(0.25)
    assert cb.allowed() is True
    cb.on_success()
    assert cb.allowed() is True


def test_cooldown_lru():
    cd = CooldownLRU(ttl_seconds=0.1)
    assert cd.hit("k") is False
    assert cd.hit("k") is True
    time.sleep(0.15)
    assert cd.hit("k") is False
