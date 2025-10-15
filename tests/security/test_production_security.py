from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from src.security.production_security import (
    AuthContext,
    DistributedRateLimiter,
    ProductionSecurityManager,
    RequestValidator,
)


class FakePipeline:
    def __init__(self, store: dict[str, int]) -> None:
        self.store = store
        self.commands: list[tuple[str, str, int | None]] = []

    def incr(self, key: str) -> "FakePipeline":
        self.commands.append(("incr", key, None))
        return self

    def expire(self, key: str, ttl: int) -> "FakePipeline":
        self.commands.append(("expire", key, ttl))
        return self

    async def execute(self) -> list[int | bool]:
        results: list[int | bool] = []
        for command, key, value in self.commands:
            if command == "incr":
                self.store[key] = self.store.get(key, 0) + 1
                results.append(self.store[key])
            elif command == "expire":
                results.append(True)
        self.commands.clear()
        return results


class FakeRedis:
    def __init__(self) -> None:
        self.store: dict[str, int] = {}

    def pipeline(self) -> FakePipeline:
        return FakePipeline(self.store)


class FakeAuditLogger:
    def __init__(self) -> None:
        self.records: list[dict[str, object]] = []

    def info(self, message: str, *, extra: dict[str, object]) -> None:
        self.records.append({"message": message, "extra": extra})


@pytest.mark.asyncio()
async def test_rate_limiter_allows_within_limits() -> None:
    redis_client = FakeRedis()
    limiter = DistributedRateLimiter(redis_client)  # type: ignore[arg-type]
    result = await limiter.check_rate_limit("user:test", "standard")
    assert result.allowed is True
    assert result.limit == 1_000


def test_request_validator_detects_injection() -> None:
    validator = RequestValidator()
    schema = {
        "properties": {
            "input": {
                "type": "string",
                "check_sql_injection": True,
                "check_xss": True,
                "sanitize": True,
            }
        }
    }
    payload = {"input": "SELECT * FROM users; --"}
    result = validator.validate_json_schema(payload, schema)
    assert result.valid is False
    assert any("SQL injection" in error for error in result.errors)
    assert result.sanitized_data["input"].startswith("SELECT")


@pytest.mark.asyncio()
async def test_security_manager_authenticates_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    manager = ProductionSecurityManager(FakeRedis(), FakeAuditLogger())  # type: ignore[arg-type]
    user_payload = {
        "user_id": "user_1",
        "api_key_id": "key_1",
        "roles": ["developer"],
        "permissions": ["read"],
        "rate_limit_tier": "basic",
    }
    monkeypatch.setattr(
        manager,
        "_get_user_by_api_key",
        AsyncMock(return_value=user_payload),
    )
    request = {"x-api-key": "abcde12345"}
    auth = await manager.authenticate_request(request)
    assert isinstance(auth, AuthContext)
    assert auth.user_id == "user_1"


@pytest.mark.asyncio()
async def test_security_manager_audit_scrubs_pii(monkeypatch: pytest.MonkeyPatch) -> None:
    audit_logger = FakeAuditLogger()
    manager = ProductionSecurityManager(FakeRedis(), audit_logger)  # type: ignore[arg-type]
    spy = AsyncMock()
    monkeypatch.setattr(manager, "_alert_suspicious_activity", spy)
    context = AuthContext("user", "key", ["role"], ["perm"], "standard", 0)
    await manager.audit_action(
        context,
        "authentication_error",
        {"email": "user@example.com", "detail": "invalid"},
    )
    assert audit_logger.records
    redacted = audit_logger.records[0]["extra"]["details"]
    assert redacted["email"] == "***REDACTED***"
    spy.assert_awaited()
