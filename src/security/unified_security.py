"""
Unified Security System - Authentication, authorization, and audit logging
Provides comprehensive security for the Super Alita system
"""

import json
import logging
import re
import secrets
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

import bcrypt
import jwt
from cryptography.fernet import Fernet


@dataclass
class SecurityConfig:
    """Configuration for security components"""

    jwt_secret: str = ""
    jwt_algorithm: str = "HS256"
    jwt_expiry_hours: int = 24
    bcrypt_rounds: int = 12
    rate_limit_requests: int = 100
    rate_limit_window: int = 60
    encryption_key: str | None = None
    audit_retention_days: int = 90

    def __post_init__(self):
        if not self.jwt_secret:
            self.jwt_secret = secrets.token_urlsafe(32)
        if not self.encryption_key:
            self.encryption_key = Fernet.generate_key().decode()


@dataclass
class User:
    """User model for authentication"""

    username: str
    email: str
    password_hash: str
    roles: list[str] = field(default_factory=list)
    permissions: list[str] = field(default_factory=list)
    is_active: bool = True
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    last_login: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "username": self.username,
            "email": self.email,
            "roles": self.roles,
            "permissions": self.permissions,
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat(),
            "last_login": (self.last_login.isoformat() if self.last_login else None),
            "metadata": self.metadata,
        }


@dataclass
class AuditLogEntry:
    """Audit log entry for security events"""

    timestamp: datetime
    user_id: str
    action: str
    resource: str
    result: str  # success, failure, blocked
    ip_address: str
    user_agent: str
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "user_id": self.user_id,
            "action": self.action,
            "resource": self.resource,
            "result": self.result,
            "ip_address": self.ip_address,
            "user_agent": self.user_agent,
            "details": self.details,
        }


class PasswordManager:
    """Secure password hashing and validation"""

    def __init__(self, rounds: int = 12):
        self.rounds = rounds

    def hash_password(self, password: str) -> str:
        """Hash password using bcrypt"""
        salt = bcrypt.gensalt(rounds=self.rounds)
        password_hash = bcrypt.hashpw(password.encode("utf-8"), salt)
        return password_hash.decode("utf-8")

    def verify_password(self, password: str, password_hash: str) -> bool:
        """Verify password against hash"""
        try:
            return bcrypt.checkpw(
                password.encode("utf-8"), password_hash.encode("utf-8")
            )
        except Exception:
            return False

    def validate_password_strength(self, password: str) -> dict[str, Any]:
        """Validate password strength"""
        issues = []
        score = 0

        if len(password) < 8:
            issues.append("Password must be at least 8 characters long")
        else:
            score += 1

        if len(password) >= 12:
            score += 1

        if re.search(r"[a-z]", password):
            score += 1
        else:
            issues.append("Password must contain lowercase letters")

        if re.search(r"[A-Z]", password):
            score += 1
        else:
            issues.append("Password must contain uppercase letters")

        if re.search(r"\d", password):
            score += 1
        else:
            issues.append("Password must contain numbers")

        if re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
            score += 1
        else:
            issues.append("Password must contain special characters")

        strength_levels = [
            "Very Weak",
            "Weak",
            "Fair",
            "Good",
            "Strong",
            "Very Strong",
        ]
        strength = strength_levels[min(score, 5)]

        return {
            "score": score,
            "max_score": 6,
            "strength": strength,
            "is_valid": not issues,
            "issues": issues,
        }


class TokenManager:
    """JWT token management for authentication"""

    def __init__(self, config: SecurityConfig):
        self.config = config
        self.blacklisted_tokens: set[str] = set()

    def generate_token(self, user: User) -> str:
        """Generate JWT token for user"""
        payload = {
            "user_id": user.username,
            "email": user.email,
            "roles": user.roles,
            "permissions": user.permissions,
            "iat": datetime.now(UTC),
            "exp": datetime.now(UTC) + timedelta(hours=self.config.jwt_expiry_hours),
            "jti": secrets.token_urlsafe(16),  # Unique token ID for blacklisting
        }

        return jwt.encode(
            payload,
            self.config.jwt_secret,
            algorithm=self.config.jwt_algorithm,
        )

    def verify_token(self, token: str) -> dict[str, Any] | None:
        """Verify and decode JWT token"""
        if token in self.blacklisted_tokens:
            return None

        try:
            payload = jwt.decode(
                token,
                self.config.jwt_secret,
                algorithms=[self.config.jwt_algorithm],
            )

            # Check if token is expired
            exp = payload.get("exp")
            if exp and datetime.fromtimestamp(exp, UTC) < datetime.now(UTC):
                return None

            return payload

        except jwt.InvalidTokenError:
            return None

    def blacklist_token(self, token: str) -> bool:
        """Add token to blacklist"""
        try:
            payload = jwt.decode(
                token,
                self.config.jwt_secret,
                algorithms=[self.config.jwt_algorithm],
                options={"verify_exp": False},  # Allow expired tokens for blacklisting
            )

            jti = payload.get("jti")
            if jti:
                self.blacklisted_tokens.add(jti)
                return True

        except jwt.InvalidTokenError:
            pass

        return False

    def refresh_token(self, token: str) -> str | None:
        """Refresh an existing token"""
        payload = self.verify_token(token)
        if not payload:
            return None

        # Create new token with extended expiry
        new_payload = payload.copy()
        new_payload["iat"] = datetime.now(UTC)
        new_payload["exp"] = datetime.now(UTC) + timedelta(
            hours=self.config.jwt_expiry_hours
        )
        new_payload["jti"] = secrets.token_urlsafe(16)

        # Blacklist old token
        old_jti = payload.get("jti")
        if old_jti:
            self.blacklisted_tokens.add(old_jti)

        return jwt.encode(
            new_payload,
            self.config.jwt_secret,
            algorithm=self.config.jwt_algorithm,
        )


class RateLimiter:
    """Rate limiting for API endpoints"""

    def __init__(self, redis_client=None):
        self.redis_client = redis_client
        self.local_cache: dict[str, dict[str, Any]] = {}

    async def is_allowed(
        self, identifier: str, limit: int, window: int
    ) -> tuple[bool, dict[str, Any]]:
        """
        Check if request is allowed under rate limit

        Args:
            identifier: Unique identifier (IP, user ID, etc.)
            limit: Maximum requests allowed
            window: Time window in seconds

        Returns:
            Tuple of (is_allowed, info_dict)
        """
        current_time = time.time()
        window_start = current_time - window

        if self.redis_client:
            return await self._redis_rate_limit(identifier, limit, window, current_time)
        else:
            return self._local_rate_limit(
                identifier, limit, window, current_time, window_start
            )

    def _local_rate_limit(
        self,
        identifier: str,
        limit: int,
        window: int,
        current_time: float,
        window_start: float,
    ) -> tuple[bool, dict[str, Any]]:
        """Local in-memory rate limiting"""
        if identifier not in self.local_cache:
            self.local_cache[identifier] = {
                "requests": [],
                "blocked_until": 0,
            }

        cache_entry = self.local_cache[identifier]

        # Check if still blocked
        if current_time < cache_entry["blocked_until"]:
            return False, {
                "allowed": False,
                "limit": limit,
                "remaining": 0,
                "reset_time": cache_entry["blocked_until"],
                "retry_after": cache_entry["blocked_until"] - current_time,
            }

        # Clean old requests
        cache_entry["requests"] = [
            req_time for req_time in cache_entry["requests"] if req_time > window_start
        ]

        # Check if limit exceeded
        if len(cache_entry["requests"]) >= limit:
            # Block for remaining window time
            cache_entry["blocked_until"] = current_time + window
            return False, {
                "allowed": False,
                "limit": limit,
                "remaining": 0,
                "reset_time": cache_entry["blocked_until"],
                "retry_after": window,
            }

        # Allow request
        cache_entry["requests"].append(current_time)
        remaining = limit - len(cache_entry["requests"])

        return True, {
            "allowed": True,
            "limit": limit,
            "remaining": remaining,
            "reset_time": current_time + window,
            "retry_after": 0,
        }

    async def _redis_rate_limit(
        self, identifier: str, limit: int, window: int, current_time: float
    ) -> tuple[bool, dict[str, Any]]:
        """Redis-based distributed rate limiting"""
        key = f"rate_limit:{identifier}"

        try:
            # Use Redis sliding window algorithm
            pipe = self.redis_client.pipeline()
            pipe.zremrangebyscore(key, 0, current_time - window)
            pipe.zcard(key)
            pipe.zadd(key, {str(current_time): current_time})
            pipe.expire(key, window)

            results = await pipe.execute()
            request_count = results[1]

            if request_count >= limit:
                return False, {
                    "allowed": False,
                    "limit": limit,
                    "remaining": 0,
                    "reset_time": current_time + window,
                    "retry_after": window,
                }

            return True, {
                "allowed": True,
                "limit": limit,
                "remaining": limit - request_count - 1,
                "reset_time": current_time + window,
                "retry_after": 0,
            }

        except Exception as e:
            logging.error(f"Redis rate limiting error: {e}")
            # Fallback to local rate limiting
            return self._local_rate_limit(
                identifier, limit, window, current_time, current_time - window
            )


class InputValidator:
    """Input validation and sanitization"""

    @staticmethod
    def validate_email(email: str) -> bool:
        """Validate email format"""
        pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
        return bool(re.match(pattern, email))

    @staticmethod
    def validate_username(username: str) -> bool:
        """Validate username format"""
        # Allow alphanumeric, underscore, hyphen, 3-32 characters
        pattern = r"^[a-zA-Z0-9_-]{3,32}$"
        return bool(re.match(pattern, username))

    @staticmethod
    def sanitize_string(text: str, max_length: int = 1000) -> str:
        """Sanitize string input"""
        if not isinstance(text, str):
            return ""

        # Remove null bytes and control characters
        sanitized = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", text)

        # Limit length
        return sanitized[:max_length]

    @staticmethod
    def validate_json(json_str: str) -> tuple[bool, Any]:
        """Validate and parse JSON"""
        try:
            data = json.loads(json_str)
            return True, data
        except (json.JSONDecodeError, TypeError):
            return False, None

    @staticmethod
    def validate_sql_injection(text: str) -> bool:
        """Check for potential SQL injection patterns"""
        sql_patterns = [
            r"(\b(select|insert|update|delete|drop|create|alter|exec|execute)\b)",
            r"(\b(union|or|and)\s+\d+\s*=\s*\d+)",
            r"(--|#|/\*|\*/)",
            r"(\bxp_cmdshell\b)",
        ]

        text_lower = text.lower()
        for pattern in sql_patterns:
            if re.search(pattern, text_lower):
                return False

        return True


class EncryptionManager:
    """Data encryption and decryption"""

    def __init__(self, key: str):
        self.fernet = Fernet(key.encode())

    def encrypt(self, data: str) -> str:
        """Encrypt string data"""
        encrypted = self.fernet.encrypt(data.encode("utf-8"))
        return encrypted.decode("utf-8")

    def decrypt(self, encrypted_data: str) -> str:
        """Decrypt string data"""
        try:
            decrypted = self.fernet.decrypt(encrypted_data.encode("utf-8"))
            return decrypted.decode("utf-8")
        except Exception:
            raise ValueError("Failed to decrypt data")

    def encrypt_dict(self, data: dict[str, Any]) -> str:
        """Encrypt dictionary as JSON"""
        json_str = json.dumps(data)
        return self.encrypt(json_str)

    def decrypt_dict(self, encrypted_data: str) -> dict[str, Any]:
        """Decrypt dictionary from JSON"""
        json_str = self.decrypt(encrypted_data)
        return json.loads(json_str)


class AuditLogger:
    """Security audit logging"""

    def __init__(self, log_file: str = "security_audit.log"):
        self.log_file = log_file
        self.logger = logging.getLogger("security_audit")

        # Configure file handler
        handler = logging.FileHandler(log_file)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

        # In-memory buffer for recent logs
        self.recent_logs: list[AuditLogEntry] = []
        self.max_recent_logs = 1000

    def log_event(
        self,
        user_id: str,
        action: str,
        resource: str,
        result: str,
        ip_address: str = "",
        user_agent: str = "",
        **details,
    ):
        """Log security event"""
        entry = AuditLogEntry(
            timestamp=datetime.now(UTC),
            user_id=user_id,
            action=action,
            resource=resource,
            result=result,
            ip_address=ip_address,
            user_agent=user_agent,
            details=details,
        )

        # Log to file
        self.logger.info(json.dumps(entry.to_dict()))

        # Add to recent logs
        self.recent_logs.append(entry)
        if len(self.recent_logs) > self.max_recent_logs:
            self.recent_logs = self.recent_logs[-self.max_recent_logs // 2 :]

    def get_recent_logs(
        self,
        limit: int = 100,
        user_id: str | None = None,
        action: str | None = None,
    ) -> list[dict[str, Any]]:
        """Get recent audit logs with optional filtering"""
        logs = self.recent_logs

        if user_id:
            logs = [log for log in logs if log.user_id == user_id]

        if action:
            logs = [log for log in logs if log.action == action]

        # Sort by timestamp (newest first)
        logs.sort(key=lambda x: x.timestamp, reverse=True)

        return [log.to_dict() for log in logs[:limit]]


class UnifiedSecurity:
    """
    Unified security system combining all security components
    """

    def __init__(self, config: SecurityConfig = None, redis_client=None):
        self.config = config or SecurityConfig()

        # Initialize components
        self.password_manager = PasswordManager(self.config.bcrypt_rounds)
        self.token_manager = TokenManager(self.config)
        self.rate_limiter = RateLimiter(redis_client)
        self.input_validator = InputValidator()
        self.encryption_manager = EncryptionManager(self.config.encryption_key)
        self.audit_logger = AuditLogger()

        # User store (in production, use a proper database)
        self.users: dict[str, User] = {}

    async def register_user(
        self,
        username: str,
        email: str,
        password: str,
        roles: list[str] = None,
        ip_address: str = "",
        user_agent: str = "",
    ) -> dict[str, Any]:
        """Register a new user"""

        # Validate inputs
        if not self.input_validator.validate_username(username):
            self.audit_logger.log_event(
                user_id=username,
                action="register",
                resource="user",
                result="failure",
                ip_address=ip_address,
                user_agent=user_agent,
                error="Invalid username format",
            )
            return {"success": False, "error": "Invalid username format"}

        if not self.input_validator.validate_email(email):
            self.audit_logger.log_event(
                user_id=username,
                action="register",
                resource="user",
                result="failure",
                ip_address=ip_address,
                user_agent=user_agent,
                error="Invalid email format",
            )
            return {"success": False, "error": "Invalid email format"}

        # Check password strength
        password_validation = self.password_manager.validate_password_strength(password)
        if not password_validation["is_valid"]:
            self.audit_logger.log_event(
                user_id=username,
                action="register",
                resource="user",
                result="failure",
                ip_address=ip_address,
                user_agent=user_agent,
                error="Weak password",
                password_issues=password_validation["issues"],
            )
            return {
                "success": False,
                "error": "Password does not meet requirements",
                "password_validation": password_validation,
            }

        # Check if user exists
        if username in self.users:
            self.audit_logger.log_event(
                user_id=username,
                action="register",
                resource="user",
                result="failure",
                ip_address=ip_address,
                user_agent=user_agent,
                error="User already exists",
            )
            return {"success": False, "error": "User already exists"}

        # Create user
        password_hash = self.password_manager.hash_password(password)
        user = User(
            username=username,
            email=email,
            password_hash=password_hash,
            roles=roles or ["user"],
            permissions=self._get_default_permissions(roles or ["user"]),
        )

        self.users[username] = user

        self.audit_logger.log_event(
            user_id=username,
            action="register",
            resource="user",
            result="success",
            ip_address=ip_address,
            user_agent=user_agent,
            roles=user.roles,
        )

        return {
            "success": True,
            "user": user.to_dict(),
            "password_strength": password_validation["strength"],
        }

    async def authenticate_user(
        self,
        username: str,
        password: str,
        ip_address: str = "",
        user_agent: str = "",
    ) -> dict[str, Any]:
        """Authenticate user and return token"""

        # Rate limiting
        rate_limit_result = await self.rate_limiter.is_allowed(
            f"login:{ip_address}", 10, 300  # 10 attempts  # per 5 minutes
        )

        if not rate_limit_result[0]:
            self.audit_logger.log_event(
                user_id=username,
                action="login",
                resource="user",
                result="blocked",
                ip_address=ip_address,
                user_agent=user_agent,
                error="Rate limit exceeded",
            )
            return {
                "success": False,
                "error": "Too many login attempts",
                "rate_limit": rate_limit_result[1],
            }

        # Check if user exists
        user = self.users.get(username)
        if not user:
            self.audit_logger.log_event(
                user_id=username,
                action="login",
                resource="user",
                result="failure",
                ip_address=ip_address,
                user_agent=user_agent,
                error="User not found",
            )
            return {"success": False, "error": "Invalid credentials"}

        # Check if user is active
        if not user.is_active:
            self.audit_logger.log_event(
                user_id=username,
                action="login",
                resource="user",
                result="failure",
                ip_address=ip_address,
                user_agent=user_agent,
                error="User inactive",
            )
            return {"success": False, "error": "Account is inactive"}

        # Verify password
        if not self.password_manager.verify_password(password, user.password_hash):
            self.audit_logger.log_event(
                user_id=username,
                action="login",
                resource="user",
                result="failure",
                ip_address=ip_address,
                user_agent=user_agent,
                error="Invalid password",
            )
            return {"success": False, "error": "Invalid credentials"}

        # Generate token
        token = self.token_manager.generate_token(user)

        # Update last login
        user.last_login = datetime.now(UTC)

        self.audit_logger.log_event(
            user_id=username,
            action="login",
            resource="user",
            result="success",
            ip_address=ip_address,
            user_agent=user_agent,
            roles=user.roles,
        )

        return {
            "success": True,
            "token": token,
            "user": user.to_dict(),
            "expires_at": (
                datetime.now(UTC) + timedelta(hours=self.config.jwt_expiry_hours)
            ).isoformat(),
        }

    async def authorize_request(
        self,
        token: str,
        required_permission: str,
        ip_address: str = "",
        user_agent: str = "",
    ) -> dict[str, Any]:
        """Authorize request with token and permission check"""

        # Verify token
        payload = self.token_manager.verify_token(token)
        if not payload:
            self.audit_logger.log_event(
                user_id="unknown",
                action="authorize",
                resource=required_permission,
                result="failure",
                ip_address=ip_address,
                user_agent=user_agent,
                error="Invalid token",
            )
            return {"success": False, "error": "Invalid or expired token"}

        user_id = payload.get("user_id")
        permissions = payload.get("permissions", [])

        # Check permission
        if required_permission not in permissions and "admin" not in payload.get(
            "roles", []
        ):
            self.audit_logger.log_event(
                user_id=user_id,
                action="authorize",
                resource=required_permission,
                result="failure",
                ip_address=ip_address,
                user_agent=user_agent,
                error="Insufficient permissions",
            )
            return {"success": False, "error": "Insufficient permissions"}

        self.audit_logger.log_event(
            user_id=user_id,
            action="authorize",
            resource=required_permission,
            result="success",
            ip_address=ip_address,
            user_agent=user_agent,
        )

        return {
            "success": True,
            "user_id": user_id,
            "permissions": permissions,
            "roles": payload.get("roles", []),
        }

    def _get_default_permissions(self, roles: list[str]) -> list[str]:
        """Get default permissions for roles"""
        permissions = set()

        role_permissions = {
            "user": ["read_own_data", "write_own_data"],
            "moderator": ["read_all_data", "moderate_content"],
            "admin": [
                "read_all_data",
                "write_all_data",
                "manage_users",
                "system_admin",
            ],
        }

        for role in roles:
            permissions.update(role_permissions.get(role, []))

        return list(permissions)

    def get_security_stats(self) -> dict[str, Any]:
        """Get comprehensive security statistics"""
        return {
            "total_users": len(self.users),
            "active_users": sum(1 for user in self.users.values() if user.is_active),
            "blacklisted_tokens": len(self.token_manager.blacklisted_tokens),
            "recent_audit_logs": len(self.audit_logger.recent_logs),
            "config": {
                "jwt_expiry_hours": self.config.jwt_expiry_hours,
                "rate_limit_requests": self.config.rate_limit_requests,
                "rate_limit_window": self.config.rate_limit_window,
                "audit_retention_days": self.config.audit_retention_days,
            },
        }


# Export main classes
__all__ = [
    "UnifiedSecurity",
    "SecurityConfig",
    "User",
    "AuditLogEntry",
    "PasswordManager",
    "TokenManager",
    "RateLimiter",
    "InputValidator",
    "EncryptionManager",
    "AuditLogger",
]
