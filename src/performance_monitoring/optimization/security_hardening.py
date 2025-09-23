"""
Security Hardening Module

Provides comprehensive security features:
- Encryption (at rest and in transit)
- Authentication and authorization
- Audit logging and security monitoring
- Input validation and sanitization
- Security policy enforcement
- Threat detection and response
"""

import asyncio
import hashlib
import hmac
import secrets
import time
import json
import logging
from typing import Dict, Any, List, Optional, Set, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import jwt
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import re
from collections import defaultdict, deque

logger = logging.getLogger(__name__)

class SecurityLevel(Enum):
    """Security clearance levels"""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    SECRET = "secret"
    TOP_SECRET = "top_secret"

class AuthenticationMethod(Enum):
    """Authentication methods"""
    PASSWORD = "password"
    TOKEN = "token"
    CERTIFICATE = "certificate"
    MULTI_FACTOR = "multi_factor"
    BIOMETRIC = "biometric"

class AuditEventType(Enum):
    """Types of audit events"""
    LOGIN = "login"
    LOGOUT = "logout"
    ACCESS_GRANTED = "access_granted"
    ACCESS_DENIED = "access_denied"
    DATA_ACCESS = "data_access"
    DATA_MODIFICATION = "data_modification"
    SYSTEM_CHANGE = "system_change"
    SECURITY_VIOLATION = "security_violation"
    ANOMALY_DETECTED = "anomaly_detected"

@dataclass
class SecurityPolicy:
    """Security policy configuration"""
    min_password_length: int = 12
    require_special_chars: bool = True
    require_numbers: bool = True
    require_uppercase: bool = True
    password_expiry_days: int = 90
    max_login_attempts: int = 5
    session_timeout_minutes: int = 60
    require_mfa: bool = False
    allowed_ip_ranges: List[str] = field(default_factory=list)
    encryption_algorithm: str = "AES-256"
    hash_algorithm: str = "SHA-256"

@dataclass
class User:
    """User account representation"""
    user_id: str
    username: str
    email: str
    password_hash: str
    salt: str
    roles: Set[str] = field(default_factory=set)
    permissions: Set[str] = field(default_factory=set)
    security_level: SecurityLevel = SecurityLevel.PUBLIC
    created_at: datetime = field(default_factory=datetime.now)
    last_login: Optional[datetime] = None
    failed_login_attempts: int = 0
    account_locked: bool = False
    password_changed_at: datetime = field(default_factory=datetime.now)
    mfa_enabled: bool = False
    mfa_secret: Optional[str] = None

@dataclass
class AuditEvent:
    """Audit log event"""
    event_id: str
    timestamp: datetime
    event_type: AuditEventType
    user_id: Optional[str]
    source_ip: str
    user_agent: str
    resource: str
    action: str
    result: str
    details: Dict[str, Any] = field(default_factory=dict)
    risk_score: float = 0.0

class EncryptionManager:
    """Handles encryption operations"""
    
    def __init__(self, master_key: Optional[bytes] = None):
        if master_key is None:
            master_key = Fernet.generate_key()
        
        self.fernet = Fernet(master_key)
        
        # Generate RSA key pair for asymmetric encryption
        self.private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048
        )
        self.public_key = self.private_key.public_key()
    
    def encrypt_data(self, data: str) -> str:
        """Encrypt data using symmetric encryption"""
        try:
            encrypted = self.fernet.encrypt(data.encode())
            return base64.b64encode(encrypted).decode()
        except Exception as e:
            logger.error(f"Encryption failed: {e}")
            raise
    
    def decrypt_data(self, encrypted_data: str) -> str:
        """Decrypt data using symmetric encryption"""
        try:
            encrypted_bytes = base64.b64decode(encrypted_data.encode())
            decrypted = self.fernet.decrypt(encrypted_bytes)
            return decrypted.decode()
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            raise
    
    def encrypt_large_data(self, data: bytes) -> bytes:
        """Encrypt large data using asymmetric encryption"""
        try:
            encrypted = self.public_key.encrypt(
                data,
                padding.OAEP(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None
                )
            )
            return encrypted
        except Exception as e:
            logger.error(f"Large data encryption failed: {e}")
            raise
    
    def decrypt_large_data(self, encrypted_data: bytes) -> bytes:
        """Decrypt large data using asymmetric encryption"""
        try:
            decrypted = self.private_key.decrypt(
                encrypted_data,
                padding.OAEP(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None
                )
            )
            return decrypted
        except Exception as e:
            logger.error(f"Large data decryption failed: {e}")
            raise
    
    def hash_password(self, password: str, salt: Optional[bytes] = None) -> Tuple[str, str]:
        """Hash password with salt"""
        if salt is None:
            salt = secrets.token_bytes(32)
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        
        password_hash = base64.b64encode(kdf.derive(password.encode())).decode()
        salt_b64 = base64.b64encode(salt).decode()
        
        return password_hash, salt_b64
    
    def verify_password(self, password: str, stored_hash: str, salt: str) -> bool:
        """Verify password against stored hash"""
        try:
            salt_bytes = base64.b64decode(salt.encode())
            computed_hash, _ = self.hash_password(password, salt_bytes)
            return hmac.compare_digest(computed_hash, stored_hash)
        except Exception as e:
            logger.error(f"Password verification failed: {e}")
            return False

class AuthenticationManager:
    """Handles user authentication and authorization"""
    
    def __init__(self, encryption_manager: EncryptionManager, policy: SecurityPolicy):
        self.encryption_manager = encryption_manager
        self.policy = policy
        self.users: Dict[str, User] = {}
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        self.jwt_secret = secrets.token_urlsafe(32)
        
        # Rate limiting
        self.login_attempts: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
    
    async def register_user(self, username: str, email: str, password: str,
                          roles: Set[str] = None, security_level: SecurityLevel = SecurityLevel.PUBLIC) -> User:
        """Register new user"""
        if not self._validate_password(password):
            raise ValueError("Password does not meet security requirements")
        
        if username in self.users:
            raise ValueError("Username already exists")
        
        user_id = secrets.token_urlsafe(16)
        password_hash, salt = self.encryption_manager.hash_password(password)
        
        user = User(
            user_id=user_id,
            username=username,
            email=email,
            password_hash=password_hash,
            salt=salt,
            roles=roles or set(),
            security_level=security_level
        )
        
        self.users[username] = user
        logger.info(f"User registered: {username}")
        return user
    
    async def authenticate_user(self, username: str, password: str,
                              source_ip: str = "unknown") -> Optional[str]:
        """Authenticate user and return session token"""
        # Rate limiting check
        now = time.time()
        attempts = self.login_attempts[source_ip]
        
        # Remove old attempts (older than 1 hour)
        while attempts and attempts[0] < now - 3600:
            attempts.popleft()
        
        if len(attempts) >= self.policy.max_login_attempts:
            logger.warning(f"Rate limit exceeded for IP: {source_ip}")
            return None
        
        # Check if user exists
        if username not in self.users:
            attempts.append(now)
            logger.warning(f"Authentication failed - unknown user: {username}")
            return None
        
        user = self.users[username]
        
        # Check if account is locked
        if user.account_locked:
            attempts.append(now)
            logger.warning(f"Authentication failed - account locked: {username}")
            return None
        
        # Verify password
        if not self.encryption_manager.verify_password(password, user.password_hash, user.salt):
            user.failed_login_attempts += 1
            attempts.append(now)
            
            # Lock account if too many failed attempts
            if user.failed_login_attempts >= self.policy.max_login_attempts:
                user.account_locked = True
                logger.warning(f"Account locked due to failed attempts: {username}")
            
            logger.warning(f"Authentication failed - invalid password: {username}")
            return None
        
        # Successful authentication
        user.failed_login_attempts = 0
        user.last_login = datetime.now()
        
        # Create session token
        session_token = self._create_session_token(user)
        
        logger.info(f"User authenticated: {username}")
        return session_token
    
    def _create_session_token(self, user: User) -> str:
        """Create JWT session token"""
        payload = {
            'user_id': user.user_id,
            'username': user.username,
            'roles': list(user.roles),
            'security_level': user.security_level.value,
            'iat': time.time(),
            'exp': time.time() + (self.policy.session_timeout_minutes * 60)
        }
        
        token = jwt.encode(payload, self.jwt_secret, algorithm='HS256')
        
        # Store active session
        self.active_sessions[token] = {
            'user_id': user.user_id,
            'created_at': datetime.now(),
            'last_activity': datetime.now()
        }
        
        return token
    
    async def validate_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Validate session token"""
        try:
            # Decode token
            payload = jwt.decode(token, self.jwt_secret, algorithms=['HS256'])
            
            # Check if session is active
            if token not in self.active_sessions:
                return None
            
            # Update last activity
            self.active_sessions[token]['last_activity'] = datetime.now()
            
            return payload
            
        except jwt.ExpiredSignatureError:
            logger.info("Token expired")
            if token in self.active_sessions:
                del self.active_sessions[token]
            return None
        except jwt.InvalidTokenError:
            logger.warning("Invalid token")
            return None
    
    async def logout_user(self, token: str):
        """Logout user and invalidate session"""
        if token in self.active_sessions:
            del self.active_sessions[token]
            logger.info("User logged out")
    
    def _validate_password(self, password: str) -> bool:
        """Validate password against security policy"""
        if len(password) < self.policy.min_password_length:
            return False
        
        if self.policy.require_uppercase and not re.search(r'[A-Z]', password):
            return False
        
        if self.policy.require_numbers and not re.search(r'\d', password):
            return False
        
        if self.policy.require_special_chars and not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
            return False
        
        return True
    
    def check_permission(self, user_id: str, required_permission: str) -> bool:
        """Check if user has required permission"""
        # Find user by user_id
        user = None
        for u in self.users.values():
            if u.user_id == user_id:
                user = u
                break
        
        if not user:
            return False
        
        return required_permission in user.permissions

class AuditLogger:
    """Comprehensive audit logging system"""
    
    def __init__(self, encryption_manager: EncryptionManager):
        self.encryption_manager = encryption_manager
        self.audit_events: deque = deque(maxlen=10000)
        self.anomaly_patterns: Dict[str, List[str]] = {
            'suspicious_ips': [],
            'failed_login_patterns': [],
            'unusual_access_times': [],
            'bulk_data_access': []
        }
        
        # Background monitoring
        self._monitoring_task: Optional[asyncio.Task] = None
        self._running = False
    
    async def start_monitoring(self):
        """Start background anomaly monitoring"""
        self._running = True
        self._monitoring_task = asyncio.create_task(self._anomaly_detection_loop())
        logger.info("Audit monitoring started")
    
    async def stop_monitoring(self):
        """Stop background monitoring"""
        self._running = False
        if self._monitoring_task:
            self._monitoring_task.cancel()
        logger.info("Audit monitoring stopped")
    
    async def log_event(self, event_type: AuditEventType, user_id: Optional[str],
                       source_ip: str, user_agent: str, resource: str,
                       action: str, result: str, details: Dict[str, Any] = None):
        """Log audit event"""
        event = AuditEvent(
            event_id=secrets.token_urlsafe(16),
            timestamp=datetime.now(),
            event_type=event_type,
            user_id=user_id,
            source_ip=source_ip,
            user_agent=user_agent,
            resource=resource,
            action=action,
            result=result,
            details=details or {},
            risk_score=self._calculate_risk_score(event_type, source_ip, details)
        )
        
        self.audit_events.append(event)
        
        # Log to external audit log if configured
        await self._persist_audit_event(event)
        
        # Check for immediate threats
        if event.risk_score > 8.0:
            await self._handle_high_risk_event(event)
    
    def _calculate_risk_score(self, event_type: AuditEventType, source_ip: str,
                             details: Dict[str, Any]) -> float:
        """Calculate risk score for event"""
        base_score = {
            AuditEventType.LOGIN: 2.0,
            AuditEventType.LOGOUT: 1.0,
            AuditEventType.ACCESS_GRANTED: 3.0,
            AuditEventType.ACCESS_DENIED: 6.0,
            AuditEventType.DATA_ACCESS: 4.0,
            AuditEventType.DATA_MODIFICATION: 7.0,
            AuditEventType.SYSTEM_CHANGE: 8.0,
            AuditEventType.SECURITY_VIOLATION: 9.0,
            AuditEventType.ANOMALY_DETECTED: 8.0
        }.get(event_type, 5.0)
        
        # Adjust based on factors
        multiplier = 1.0
        
        # Check for suspicious IP patterns
        if self._is_suspicious_ip(source_ip):
            multiplier *= 1.5
        
        # Check for unusual timing
        hour = datetime.now().hour
        if hour < 6 or hour > 22:  # Outside business hours
            multiplier *= 1.2
        
        # Check details for concerning patterns
        if details:
            if details.get('bulk_operation', False):
                multiplier *= 1.3
            if details.get('admin_operation', False):
                multiplier *= 1.4
            if details.get('external_access', False):
                multiplier *= 1.3
        
        return min(base_score * multiplier, 10.0)
    
    def _is_suspicious_ip(self, ip: str) -> bool:
        """Check if IP is suspicious"""
        # Simple heuristic - in real implementation would check threat intelligence
        return ip in self.anomaly_patterns['suspicious_ips']
    
    async def _persist_audit_event(self, event: AuditEvent):
        """Persist audit event to secure storage"""
        try:
            # Encrypt sensitive audit data
            event_data = {
                'event_id': event.event_id,
                'timestamp': event.timestamp.isoformat(),
                'event_type': event.event_type.value,
                'user_id': event.user_id,
                'source_ip': event.source_ip,
                'user_agent': event.user_agent,
                'resource': event.resource,
                'action': event.action,
                'result': event.result,
                'details': event.details,
                'risk_score': event.risk_score
            }
            
            encrypted_data = self.encryption_manager.encrypt_data(json.dumps(event_data))
            
            # In real implementation, would write to secure audit database
            logger.debug(f"Audit event persisted: {event.event_id}")
            
        except Exception as e:
            logger.error(f"Failed to persist audit event: {e}")
    
    async def _handle_high_risk_event(self, event: AuditEvent):
        """Handle high-risk security events"""
        logger.critical(f"HIGH RISK EVENT DETECTED: {event.event_type.value} "
                       f"from {event.source_ip} (Risk: {event.risk_score})")
        
        # In real implementation, would trigger alerts, block IPs, etc.
        # For now, just add to suspicious patterns
        if event.source_ip not in self.anomaly_patterns['suspicious_ips']:
            self.anomaly_patterns['suspicious_ips'].append(event.source_ip)
    
    async def _anomaly_detection_loop(self):
        """Background anomaly detection"""
        while self._running:
            try:
                await asyncio.sleep(300)  # Check every 5 minutes
                
                # Analyze recent events for patterns
                recent_events = [e for e in self.audit_events 
                               if (datetime.now() - e.timestamp).total_seconds() < 3600]
                
                await self._detect_anomalies(recent_events)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Anomaly detection error: {e}")
    
    async def _detect_anomalies(self, events: List[AuditEvent]):
        """Detect security anomalies in events"""
        # Group events by IP
        ip_events = defaultdict(list)
        for event in events:
            ip_events[event.source_ip].append(event)
        
        # Look for suspicious patterns
        for ip, ip_event_list in ip_events.items():
            # Multiple failed logins
            failed_logins = [e for e in ip_event_list 
                           if e.event_type == AuditEventType.ACCESS_DENIED]
            
            if len(failed_logins) > 10:
                await self.log_event(
                    AuditEventType.ANOMALY_DETECTED,
                    None, ip, "system", "anomaly_detection",
                    "multiple_failed_logins", "detected",
                    {"failed_count": len(failed_logins)}
                )
            
            # Bulk data access
            data_access = [e for e in ip_event_list 
                          if e.event_type == AuditEventType.DATA_ACCESS]
            
            if len(data_access) > 50:
                await self.log_event(
                    AuditEventType.ANOMALY_DETECTED,
                    None, ip, "system", "anomaly_detection",
                    "bulk_data_access", "detected",
                    {"access_count": len(data_access)}
                )
    
    def get_security_summary(self) -> Dict[str, Any]:
        """Get security summary statistics"""
        recent_events = [e for e in self.audit_events 
                        if (datetime.now() - e.timestamp).total_seconds() < 3600]
        
        event_counts = defaultdict(int)
        high_risk_events = 0
        unique_ips = set()
        
        for event in recent_events:
            event_counts[event.event_type.value] += 1
            unique_ips.add(event.source_ip)
            if event.risk_score > 7.0:
                high_risk_events += 1
        
        return {
            'total_events_last_hour': len(recent_events),
            'event_type_counts': dict(event_counts),
            'high_risk_events': high_risk_events,
            'unique_ips': len(unique_ips),
            'suspicious_ips': len(self.anomaly_patterns['suspicious_ips']),
            'avg_risk_score': sum(e.risk_score for e in recent_events) / len(recent_events) if recent_events else 0
        }

class SecurityHardening:
    """Main security hardening coordinator"""
    
    def __init__(self, policy: Optional[SecurityPolicy] = None):
        self.policy = policy or SecurityPolicy()
        self.encryption_manager = EncryptionManager()
        self.auth_manager = AuthenticationManager(self.encryption_manager, self.policy)
        self.audit_logger = AuditLogger(self.encryption_manager)
        
        # Security monitoring
        self._running = False
        self._security_tasks: List[asyncio.Task] = []
    
    async def start(self):
        """Start security hardening services"""
        self._running = True
        
        # Start audit monitoring
        await self.audit_logger.start_monitoring()
        
        # Start session cleanup
        self._security_tasks.append(
            asyncio.create_task(self._session_cleanup_loop())
        )
        
        logger.info("Security hardening started")
    
    async def stop(self):
        """Stop security hardening services"""
        self._running = False
        
        # Stop audit monitoring
        await self.audit_logger.stop_monitoring()
        
        # Cancel security tasks
        for task in self._security_tasks:
            task.cancel()
        
        logger.info("Security hardening stopped")
    
    async def authenticate_request(self, token: str, required_permission: str = None,
                                 source_ip: str = "unknown", user_agent: str = "unknown") -> Optional[Dict[str, Any]]:
        """Authenticate and authorize request"""
        # Validate token
        payload = await self.auth_manager.validate_token(token)
        if not payload:
            await self.audit_logger.log_event(
                AuditEventType.ACCESS_DENIED, None, source_ip, user_agent,
                "authentication", "validate_token", "failed"
            )
            return None
        
        # Check permission if required
        if required_permission:
            if not self.auth_manager.check_permission(payload['user_id'], required_permission):
                await self.audit_logger.log_event(
                    AuditEventType.ACCESS_DENIED, payload['user_id'], source_ip, user_agent,
                    "authorization", "check_permission", "failed",
                    {"required_permission": required_permission}
                )
                return None
        
        # Log successful access
        await self.audit_logger.log_event(
            AuditEventType.ACCESS_GRANTED, payload['user_id'], source_ip, user_agent,
            "authentication", "validate_request", "success"
        )
        
        return payload
    
    def encrypt_sensitive_data(self, data: str) -> str:
        """Encrypt sensitive data"""
        return self.encryption_manager.encrypt_data(data)
    
    def decrypt_sensitive_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive data"""
        return self.encryption_manager.decrypt_data(encrypted_data)
    
    async def log_data_access(self, user_id: str, resource: str, action: str,
                            source_ip: str = "unknown", details: Dict[str, Any] = None):
        """Log data access for audit trail"""
        await self.audit_logger.log_event(
            AuditEventType.DATA_ACCESS, user_id, source_ip, "system",
            resource, action, "success", details
        )
    
    async def _session_cleanup_loop(self):
        """Clean up expired sessions"""
        while self._running:
            try:
                await asyncio.sleep(300)  # Check every 5 minutes
                
                now = datetime.now()
                expired_sessions = []
                
                for token, session_info in self.auth_manager.active_sessions.items():
                    last_activity = session_info['last_activity']
                    if (now - last_activity).total_seconds() > (self.policy.session_timeout_minutes * 60):
                        expired_sessions.append(token)
                
                # Remove expired sessions
                for token in expired_sessions:
                    del self.auth_manager.active_sessions[token]
                    logger.debug(f"Cleaned up expired session")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Session cleanup error: {e}")
    
    def get_security_status(self) -> Dict[str, Any]:
        """Get comprehensive security status"""
        return {
            'policy': {
                'min_password_length': self.policy.min_password_length,
                'require_mfa': self.policy.require_mfa,
                'session_timeout_minutes': self.policy.session_timeout_minutes,
                'max_login_attempts': self.policy.max_login_attempts
            },
            'users': {
                'total_users': len(self.auth_manager.users),
                'locked_accounts': len([u for u in self.auth_manager.users.values() if u.account_locked]),
                'active_sessions': len(self.auth_manager.active_sessions)
            },
            'audit': self.audit_logger.get_security_summary()
        }