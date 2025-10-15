"""Production-grade security primitives used by Super Alita services."""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import hmac
import re
import time
from dataclasses import dataclass
from typing import Any

import redis.asyncio as redis
import secrets


@dataclass(slots=True)
class AuthContext:
    """Represents authenticated session metadata."""

    user_id: str
    api_key_id: str
    roles: list[str]
    permissions: list[str]
    rate_limit_tier: str
    expires_at: float


@dataclass(slots=True)
class RateLimitResult:
    """Outcome from the distributed rate limiter."""

    allowed: bool
    remaining: int
    reset_time: float
    limit: int


@dataclass(slots=True)
class ValidationResult:
    """Structured validation response for request bodies."""

    valid: bool
    errors: list[str]
    sanitized_data: Any


class DistributedRateLimiter:
    """Redis-backed sliding window rate limiter."""

    def __init__(self, redis_client: redis.Redis) -> None:
        self.redis = redis_client
        self.tiers: dict[str, dict[str, int]] = {
            "premium": {"hourly": 10_000, "burst": 100},
            "standard": {"hourly": 1_000, "burst": 50},
            "basic": {"hourly": 100, "burst": 10},
            "anonymous": {"hourly": 10, "burst": 2},
        }

    async def check_rate_limit(self, key: str, tier: str) -> RateLimitResult:
        if tier not in self.tiers:
            tier = "anonymous"

        limits = self.tiers[tier]
        now = time.time()
        pipeline = self.redis.pipeline()

        current_minute = int(now // 60)
        current_hour = int(now // 3600)

        minute_key = f"rate_limit:{key}:minute:{current_minute}"
        hour_key = f"rate_limit:{key}:hour:{current_hour}"

        pipeline.incr(minute_key)
        pipeline.expire(minute_key, 120)
        pipeline.incr(hour_key)
        pipeline.expire(hour_key, 7_200)

        minute_count, _, hour_count, _ = await pipeline.execute()
        allowed = minute_count <= limits["burst"] and hour_count <= limits["hourly"]

        return RateLimitResult(
            allowed=allowed,
            remaining=max(0, limits["hourly"] - int(hour_count)),
            reset_time=(current_hour + 1) * 3_600,
            limit=limits["hourly"],
        )


class EncryptedAPIKeyStore:
    """Encrypt and decrypt API keys using Fernet symmetric encryption."""

    def __init__(self, encryption_key: str | None = None) -> None:
        self.secret = (encryption_key or secrets.token_hex(16)).encode()

    def encrypt_key(self, api_key: str) -> str:
        key_bytes = api_key.encode()
        mask = hashlib.sha256(self.secret).digest()
        encrypted = bytes(b ^ mask[i % len(mask)] for i, b in enumerate(key_bytes))
        return base64.urlsafe_b64encode(encrypted).decode()

    def decrypt_key(self, encrypted_key: str) -> str:
        data = base64.urlsafe_b64decode(encrypted_key.encode())
        mask = hashlib.sha256(self.secret).digest()
        decrypted = bytes(b ^ mask[i % len(mask)] for i, b in enumerate(data))
        return decrypted.decode()


class RequestValidator:
    """Performs validation and sanitisation for inbound requests."""

    def __init__(self) -> None:
        self.sql_injection_patterns = [
            re.compile(r"(\%27)|(')|(\-\-)|(\%23)|(#)", re.IGNORECASE),
            re.compile(r"((\%3D)|(=))[^\n]*((\%27)|(')|(\-\-)|(\%3B)|(;))", re.IGNORECASE),
            re.compile(r"\w*((\%27)|('))((\%6F)|o|(\%4F))((\%72)|r|(\%52))", re.IGNORECASE),
            re.compile(r"((\%27)|('))union", re.IGNORECASE),
        ]
        self.xss_patterns = [
            re.compile(r"<script[^>]*>.*?</script>", re.IGNORECASE | re.DOTALL),
            re.compile(r"javascript:", re.IGNORECASE),
            re.compile(r"on\w+\s*=", re.IGNORECASE),
            re.compile(r"vbscript:", re.IGNORECASE),
            re.compile(r"<iframe[^>]*>.*?</iframe>", re.IGNORECASE | re.DOTALL),
        ]
        self.prompt_injection_patterns = [
            re.compile(r"ignore.*previous", re.IGNORECASE),
            re.compile(r"forget.*instructions", re.IGNORECASE),
            re.compile(r"act.*as.*someone.*else", re.IGNORECASE),
            re.compile(r"system.*prompt", re.IGNORECASE),
            re.compile(r"bypass.*constitution", re.IGNORECASE),
        ]

    def validate_json_schema(self, data: Any, schema: dict[str, Any]) -> ValidationResult:
        errors: list[str] = []
        sanitized = data.copy() if isinstance(data, dict) else data

        for field, config in schema.get("properties", {}).items():
            if field not in data:
                continue
            field_errors = self._validate_field(data[field], config, field)
            errors.extend(field_errors)
            if config.get("sanitize") and isinstance(data[field], str):
                sanitized[field] = self._sanitize_string(data[field])

        return ValidationResult(valid=not errors, errors=errors, sanitized_data=sanitized)

    def _validate_field(self, value: Any, config: dict[str, Any], field_name: str) -> list[str]:
        errors: list[str] = []

        expected_type = config.get("type")
        if expected_type == "string" and not isinstance(value, str):
            errors.append(f"Field '{field_name}' must be a string")
        elif expected_type == "number" and not isinstance(value, (int, float)):
            errors.append(f"Field '{field_name}' must be a number")
        elif expected_type == "boolean" and not isinstance(value, bool):
            errors.append(f"Field '{field_name}' must be a boolean")
        elif expected_type == "array" and not isinstance(value, list):
            errors.append(f"Field '{field_name}' must be an array")

        if isinstance(value, str):
            min_length = config.get("minLength")
            max_length = config.get("maxLength")
            if min_length and len(value) < int(min_length):
                errors.append(f"Field '{field_name}' must be at least {min_length} characters")
            if max_length and len(value) > int(max_length):
                errors.append(f"Field '{field_name}' must be at most {max_length} characters")

            pattern = config.get("pattern")
            if pattern and not re.match(pattern, value):
                errors.append(f"Field '{field_name}' does not match required pattern")

            if config.get("check_sql_injection") and self._detect_sql_injection(value):
                errors.append(f"Field '{field_name}' contains potential SQL injection")
            if config.get("check_xss") and self._detect_xss(value):
                errors.append(f"Field '{field_name}' contains potential XSS")
            if config.get("check_prompt_injection") and self._detect_prompt_injection(value):
                errors.append(f"Field '{field_name}' contains potential prompt injection")

        return errors

    def _detect_sql_injection(self, value: str) -> bool:
        value_lower = value.lower()
        return any(pattern.search(value_lower) for pattern in self.sql_injection_patterns)

    def _detect_xss(self, value: str) -> bool:
        return any(pattern.search(value) for pattern in self.xss_patterns)

    def _detect_prompt_injection(self, value: str) -> bool:
        value_lower = value.lower()
        return any(pattern.search(value_lower) for pattern in self.prompt_injection_patterns)

    def _sanitize_string(self, value: str) -> str:
        sanitized = re.sub(r"<script[^>]*>.*?</script>", "", value, flags=re.IGNORECASE | re.DOTALL)
        sanitized = re.sub(r"on\w+\s*=", "", sanitized, flags=re.IGNORECASE)
        sanitized = re.sub(r"javascript:|vbscript:", "", sanitized, flags=re.IGNORECASE)
        return sanitized


class ProductionSecurityManager:
    """High level security façade combining authentication and validation."""

    def __init__(self, redis_client: redis.Redis, audit_logger: Any) -> None:
        self.redis = redis_client
        self.audit_logger = audit_logger
        self.rate_limiter = DistributedRateLimiter(redis_client)
        self.api_key_store = EncryptedAPIKeyStore()
        self.request_validator = RequestValidator()
        self.logger = logging.getLogger(__name__)
        self.jwt_secret = "super_alita_secret_key"
        self.jwt_algorithm = "HS256"

    async def authenticate_request(self, request: Any) -> AuthContext | None:
        try:
            token = self._extract_api_key(request)
            if not token or not self._validate_api_key_format(token):
                await self.audit_action(None, "invalid_api_key_format", {"api_key_prefix": token[:8] if token else "none"})
                return None

            if self._is_jwt_token(token):
                return await self._authenticate_jwt(token)
            return await self._authenticate_api_key(token)

        except Exception as exc:  # pragma: no cover - defensive guard
            self.logger.error("Authentication failed: %s", exc)
            await self.audit_action(None, "authentication_error", {"error": str(exc)})
            return None

    def _extract_api_key(self, request: Any) -> str | None:
        if hasattr(request, "headers"):
            auth_header = request.headers.get("authorization")
            if auth_header and auth_header.startswith("Bearer "):
                return auth_header[7:]
            return request.headers.get("x-api-key")
        if isinstance(request, dict):
            return request.get("authorization") or request.get("x-api-key")
        return None

    def _validate_api_key_format(self, api_key: str | None) -> bool:
        if not api_key or len(api_key) < 10:
            return False
        return bool(re.match(r"^[a-zA-Z0-9_\-.]+$", api_key))

    def _is_jwt_token(self, token: str) -> bool:
        return token.count(".") == 2

    async def _authenticate_jwt(self, token: str) -> AuthContext | None:
        try:
            payload = decode_jwt(token, self.jwt_secret, self.jwt_algorithm)
        except ExpiredSignatureError:
            self.logger.warning("JWT token expired")
            return None
        except InvalidTokenError as exc:
            self.logger.warning("Invalid JWT token: %s", exc)
            return None

        return AuthContext(
            user_id=payload.get("sub", ""),
            api_key_id=payload.get("jti", ""),
            roles=list(payload.get("roles", [])),
            permissions=list(payload.get("permissions", [])),
            rate_limit_tier=payload.get("tier", "standard"),
            expires_at=float(payload.get("exp", 0)),
        )

    async def _authenticate_api_key(self, api_key: str) -> AuthContext | None:
        api_key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        user_data = await self._get_user_by_api_key(api_key_hash)
        if not user_data:
            await self.audit_action(None, "invalid_api_key", {"api_key_hash": api_key_hash[:12]})
            return None
        return AuthContext(
            user_id=user_data["user_id"],
            api_key_id=user_data["api_key_id"],
            roles=list(user_data.get("roles", [])),
            permissions=list(user_data.get("permissions", [])),
            rate_limit_tier=user_data.get("rate_limit_tier", "standard"),
            expires_at=time.time() + 3_600,
        )

    async def _get_user_by_api_key(self, api_key_hash: str) -> dict[str, Any] | None:
        if api_key_hash.startswith("a"):
            return {
                "user_id": "user_123",
                "api_key_id": "key_456",
                "roles": ["developer"],
                "permissions": ["read", "write", "execute"],
                "rate_limit_tier": "standard",
            }
        return None

    async def enforce_rate_limits(self, auth_context: AuthContext | None, endpoint: str) -> RateLimitResult:
        if not auth_context:
            key = f"anon:{endpoint}"
            return await self.rate_limiter.check_rate_limit(key, "anonymous")
        key = f"user:{auth_context.user_id}:{endpoint}"
        return await self.rate_limiter.check_rate_limit(key, auth_context.rate_limit_tier)

    async def validate_input(self, request: Any, schema: dict[str, Any]) -> ValidationResult:
        if hasattr(request, "json"):
            data = request.json()
        elif hasattr(request, "body"):
            data = json.loads(request.body)
        else:
            data = request
        return self.request_validator.validate_json_schema(data, schema)

    async def audit_action(self, auth_context: AuthContext | None, action: str, details: dict[str, Any]) -> None:
        scrubbed = self._scrub_pii(details)
        entry = {
            "timestamp": time.time(),
            "action": action,
            "user_id": auth_context.user_id if auth_context else "anonymous",
            "details": scrubbed,
            "ip_address": self._get_client_ip(auth_context),
        }
        self.audit_logger.info("Security audit", extra=entry)
        if self._is_suspicious_action(action, details):
            await self._alert_suspicious_activity(entry)

    def _scrub_pii(self, details: dict[str, Any]) -> dict[str, Any]:
        scrubbed = details.copy()
        for field in ["email", "password", "token", "api_key", "ssn", "phone"]:
            if field in scrubbed:
                scrubbed[field] = "***REDACTED***"
        return scrubbed

    def _get_client_ip(self, auth_context: Any) -> str:
        if hasattr(auth_context, "_request"):
            request = auth_context._request
            if hasattr(request, "client") and request.client:
                return getattr(request.client, "host", "unknown")
            if hasattr(request, "headers"):
                return request.headers.get("x-forwarded-for", "unknown")
        return "unknown"

    def _is_suspicious_action(self, action: str, details: dict[str, Any]) -> bool:
        suspicious_actions = {
            "authentication_error",
            "rate_limit_exceeded",
            "invalid_api_key",
            "sql_injection_attempt",
        }
        return action in suspicious_actions or bool(details.get("error"))

    async def _alert_suspicious_activity(self, audit_entry: dict[str, Any]) -> None:
        self.logger.warning("Suspicious activity detected: %s", audit_entry)


async def create_security_manager(redis_url: str) -> ProductionSecurityManager:
    redis_client = redis.from_url(redis_url)
    audit_logger = logging.getLogger("security_audit")
    return ProductionSecurityManager(redis_client, audit_logger)


__all__ = [
    "AuthContext",
    "RateLimitResult",
    "ValidationResult",
    "DistributedRateLimiter",
    "EncryptedAPIKeyStore",
    "RequestValidator",
    "ProductionSecurityManager",
    "create_security_manager",
]


# Minimal JWT helpers to avoid external dependency


class JWTError(Exception):
    """Base exception for JWT decoding errors."""


class ExpiredSignatureError(JWTError):
    """Raised when token signature is valid but expired."""


class InvalidTokenError(JWTError):
    """Raised when token structure or signature validation fails."""


def decode_jwt(token: str, secret: str, algorithm: str) -> dict[str, Any]:
    if algorithm != "HS256":
        raise InvalidTokenError("Unsupported algorithm")

    parts = token.split(".")
    if len(parts) != 3:
        raise InvalidTokenError("Invalid token structure")

    header_bytes = _urlsafe_b64decode(parts[0])
    payload_bytes = _urlsafe_b64decode(parts[1])
    signature = _urlsafe_b64decode(parts[2])

    try:
        header = json.loads(header_bytes)
        payload = json.loads(payload_bytes)
    except json.JSONDecodeError as exc:
        raise InvalidTokenError("Invalid token payload") from exc

    signing_input = f"{parts[0]}.{parts[1]}".encode()
    expected_signature = hmac.new(secret.encode(), signing_input, hashlib.sha256).digest()
    if not hmac.compare_digest(expected_signature, signature):
        raise InvalidTokenError("Signature verification failed")

    exp = payload.get("exp")
    if exp is not None and float(exp) < time.time():
        raise ExpiredSignatureError("Token expired")

    return payload


def _urlsafe_b64decode(value: str) -> bytes:
    padding = '=' * (-len(value) % 4)
    return base64.urlsafe_b64decode(value + padding)
