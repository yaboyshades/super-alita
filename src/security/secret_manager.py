#!/usr/bin/env python3
"""
Secret Manager - Secure runtime secret resolution for Super Alita

This module provides a unified interface for accessing secrets across different environments:
1. Development: .env files and environment variables
2. Production: HashiCorp Vault, AWS SSM Parameter Store, Azure Key Vault
3. Testing: In-memory store for isolated testing

Design Philosophy:
- Environment-agnostic interface with `get_secret(name)` 
- Automatic backend detection based on environment
- Secure caching with TTL and encryption at rest
- Audit logging for compliance
- Graceful fallback between backends
"""

import json
import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from cryptography.fernet import Fernet

logger = logging.getLogger(__name__)


class SecretBackend(Enum):
    """Supported secret backends"""
    ENV = "env"  # Environment variables and .env files
    VAULT = "vault"  # HashiCorp Vault
    AWS_SSM = "aws_ssm"  # AWS Systems Manager Parameter Store
    AWS_SECRETS = "aws_secrets"  # AWS Secrets Manager
    AZURE_KEYVAULT = "azure_keyvault"  # Azure Key Vault
    MEMORY = "memory"  # In-memory store for testing


@dataclass
class SecretMetadata:
    """Metadata about a secret"""
    name: str
    backend: SecretBackend
    created_at: datetime
    accessed_at: datetime
    access_count: int = 0
    ttl_seconds: Optional[int] = None
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class CachedSecret:
    """Cached secret with metadata"""
    value: str
    metadata: SecretMetadata
    cached_at: float
    encrypted: bool = False


class SecretBackendInterface(ABC):
    """Abstract interface for secret backends"""
    
    @abstractmethod
    async def get_secret(self, name: str) -> Optional[str]:
        """Retrieve secret value by name"""
        pass
    
    @abstractmethod
    async def list_secrets(self) -> List[str]:
        """List available secret names"""
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """Check if backend is available"""
        pass


class EnvironmentSecretBackend(SecretBackendInterface):
    """Environment variables and .env file backend"""
    
    def __init__(self, env_file: Optional[str] = None):
        self.env_file = Path(env_file) if env_file else None
        self.env_vars: Dict[str, str] = {}
        self._load_env_file()
    
    def _load_env_file(self):
        """Load environment variables from .env file"""
        if not self.env_file or not self.env_file.exists():
            return
        
        try:
            with open(self.env_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        self.env_vars[key.strip()] = value.strip().strip('"\'')
            
            logger.info(f"Loaded {len(self.env_vars)} environment variables from {self.env_file}")
            
        except Exception as e:
            logger.error(f"Failed to load env file {self.env_file}: {e}")
    
    async def get_secret(self, name: str) -> Optional[str]:
        """Get secret from environment or .env file"""
        # Check OS environment first
        value = os.getenv(name)
        if value:
            return value
        
        # Check .env file
        return self.env_vars.get(name)
    
    async def list_secrets(self) -> List[str]:
        """List all available environment variables"""
        env_keys = set(os.environ.keys())
        file_keys = set(self.env_vars.keys())
        return sorted(env_keys.union(file_keys))
    
    async def health_check(self) -> bool:
        """Environment backend is always available"""
        return True


class VaultSecretBackend(SecretBackendInterface):
    """HashiCorp Vault backend"""
    
    def __init__(
        self,
        vault_url: str = "http://localhost:8200",
        vault_token: Optional[str] = None,
        mount_point: str = "secret"
    ):
        self.vault_url = vault_url.rstrip('/')
        self.vault_token = vault_token or os.getenv("VAULT_TOKEN")
        self.mount_point = mount_point
        self._client = None
    
    async def _get_client(self):
        """Get or create Vault client"""
        if self._client is None:
            try:
                import hvac
                self._client = hvac.Client(
                    url=self.vault_url,
                    token=self.vault_token
                )
            except ImportError:
                raise RuntimeError("hvac library required for Vault backend. Install with: pip install hvac")
        
        return self._client
    
    async def get_secret(self, name: str) -> Optional[str]:
        """Get secret from Vault KV store"""
        try:
            client = await self._get_client()
            
            # Try KV v2 first
            try:
                response = client.secrets.kv.v2.read_secret_version(
                    path=name,
                    mount_point=self.mount_point
                )
                data = response.get('data', {}).get('data', {})
                
                # Return 'value' field or the whole data as JSON
                if 'value' in data:
                    return data['value']
                elif data:
                    return json.dumps(data)
                    
            except Exception:
                # Fallback to KV v1
                response = client.secrets.kv.v1.read_secret(
                    path=name,
                    mount_point=self.mount_point
                )
                data = response.get('data', {})
                
                if 'value' in data:
                    return data['value']
                elif data:
                    return json.dumps(data)
            
            return None
            
        except Exception as e:
            logger.error(f"Vault secret retrieval failed for '{name}': {e}")
            return None
    
    async def list_secrets(self) -> List[str]:
        """List secrets in Vault"""
        try:
            client = await self._get_client()
            
            # Try KV v2 first
            try:
                response = client.secrets.kv.v2.list_secrets(
                    path="",
                    mount_point=self.mount_point
                )
            except Exception:
                # Fallback to KV v1
                response = client.secrets.kv.v1.list_secrets(
                    path="",
                    mount_point=self.mount_point
                )
            
            return response.get('data', {}).get('keys', [])
            
        except Exception as e:
            logger.error(f"Vault secret listing failed: {e}")
            return []
    
    async def health_check(self) -> bool:
        """Check Vault connectivity"""
        try:
            client = await self._get_client()
            return client.sys.is_sealed() is False
        except Exception:
            return False


class AWSSSMSecretBackend(SecretBackendInterface):
    """AWS Systems Manager Parameter Store backend"""
    
    def __init__(self, region: Optional[str] = None, prefix: str = "/super-alita/"):
        self.region = region or os.getenv("AWS_DEFAULT_REGION", "us-east-1")
        self.prefix = prefix
        self._client = None
    
    async def _get_client(self):
        """Get or create SSM client"""
        if self._client is None:
            try:
                import boto3
                self._client = boto3.client('ssm', region_name=self.region)
            except ImportError:
                raise RuntimeError("boto3 library required for AWS backend. Install with: pip install boto3")
        
        return self._client
    
    async def get_secret(self, name: str) -> Optional[str]:
        """Get parameter from SSM Parameter Store"""
        try:
            client = await self._get_client()
            parameter_name = f"{self.prefix}{name}"
            
            response = client.get_parameter(
                Name=parameter_name,
                WithDecryption=True
            )
            
            return response['Parameter']['Value']
            
        except client.exceptions.ParameterNotFound:
            return None
        except Exception as e:
            logger.error(f"AWS SSM parameter retrieval failed for '{name}': {e}")
            return None
    
    async def list_secrets(self) -> List[str]:
        """List parameters in SSM"""
        try:
            client = await self._get_client()
            
            paginator = client.get_paginator('describe_parameters')
            parameters = []
            
            for page in paginator.paginate(
                ParameterFilters=[
                    {
                        'Key': 'Name',
                        'Option': 'BeginsWith',
                        'Values': [self.prefix]
                    }
                ]
            ):
                for param in page['Parameters']:
                    name = param['Name']
                    if name.startswith(self.prefix):
                        name = name[len(self.prefix):]
                    parameters.append(name)
            
            return sorted(parameters)
            
        except Exception as e:
            logger.error(f"AWS SSM parameter listing failed: {e}")
            return []
    
    async def health_check(self) -> bool:
        """Check AWS SSM connectivity"""
        try:
            client = await self._get_client()
            # Simple operation to test connectivity
            client.describe_parameters(MaxResults=1)
            return True
        except Exception:
            return False


class MemorySecretBackend(SecretBackendInterface):
    """In-memory secret backend for testing"""
    
    def __init__(self, initial_secrets: Optional[Dict[str, str]] = None):
        self.secrets = initial_secrets or {}
    
    async def get_secret(self, name: str) -> Optional[str]:
        """Get secret from memory"""
        return self.secrets.get(name)
    
    async def list_secrets(self) -> List[str]:
        """List all secrets in memory"""
        return sorted(self.secrets.keys())
    
    async def health_check(self) -> bool:
        """Memory backend is always available"""
        return True
    
    def set_secret(self, name: str, value: str):
        """Set secret in memory (for testing)"""
        self.secrets[name] = value
    
    def clear_secrets(self):
        """Clear all secrets (for testing)"""
        self.secrets.clear()


class SecretManager:
    """Main secret management system with multiple backends"""
    
    def __init__(
        self,
        backends: Optional[List[SecretBackendInterface]] = None,
        cache_ttl: int = 300,  # 5 minutes
        enable_encryption: bool = True
    ):
        self.backends = backends or self._detect_backends()
        self.cache_ttl = cache_ttl
        self.enable_encryption = enable_encryption
        
        # Setup encryption for cache
        if enable_encryption:
            self.encryption_key = os.getenv("SUPER_ALITA_SECRET_CACHE_KEY")
            if not self.encryption_key:
                self.encryption_key = Fernet.generate_key().decode()
                logger.warning("Generated ephemeral encryption key for secret cache")
            self.fernet = Fernet(self.encryption_key.encode())
        else:
            self.fernet = None
        
        # Cache and metrics
        self.cache: Dict[str, CachedSecret] = {}
        self.access_stats: Dict[str, int] = {}
        self.backend_health: Dict[str, bool] = {}
        
        # Audit logging
        self.audit_file = Path("logs/secret_audit.jsonl")
        self.audit_file.parent.mkdir(parents=True, exist_ok=True)
    
    def _detect_backends(self) -> List[SecretBackendInterface]:
        """Auto-detect available backends based on environment"""
        backends = []
        
        # Always include environment backend
        env_file = os.getenv("SUPER_ALITA_ENV_FILE", ".env")
        backends.append(EnvironmentSecretBackend(env_file))
        
        # Check for Vault
        if os.getenv("VAULT_ADDR") or os.getenv("VAULT_TOKEN"):
            vault_url = os.getenv("VAULT_ADDR", "http://localhost:8200")
            vault_token = os.getenv("VAULT_TOKEN")
            mount_point = os.getenv("VAULT_MOUNT_POINT", "secret")
            backends.append(VaultSecretBackend(vault_url, vault_token, mount_point))
        
        # Check for AWS
        if os.getenv("AWS_ACCESS_KEY_ID") or os.getenv("AWS_PROFILE"):
            region = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
            prefix = os.getenv("SUPER_ALITA_SSM_PREFIX", "/super-alita/")
            backends.append(AWSSSMSecretBackend(region, prefix))
        
        logger.info(f"Detected {len(backends)} secret backends")
        return backends
    
    async def get_secret(self, name: str) -> Optional[str]:
        """Get secret from first available backend"""
        # Check cache first
        cached = self._get_cached_secret(name)
        if cached:
            await self._audit_log("cache_hit", name, success=True)
            return cached
        
        # Try backends in order
        for backend in self.backends:
            try:
                value = await backend.get_secret(name)
                if value:
                    # Cache the result
                    await self._cache_secret(name, value, backend)
                    await self._audit_log("secret_retrieved", name, backend_type=type(backend).__name__, success=True)
                    return value
            except Exception as e:
                logger.error(f"Backend {type(backend).__name__} failed for secret '{name}': {e}")
                await self._audit_log("backend_error", name, backend_type=type(backend).__name__, error=str(e), success=False)
        
        await self._audit_log("secret_not_found", name, success=False)
        return None
    
    async def get_secret_with_fallback(self, name: str, fallback: str) -> str:
        """Get secret with fallback value"""
        value = await self.get_secret(name)
        return value if value is not None else fallback
    
    async def list_secrets(self) -> Dict[str, List[str]]:
        """List secrets from all backends"""
        result = {}
        for backend in self.backends:
            try:
                secrets = await backend.list_secrets()
                result[type(backend).__name__] = secrets
            except Exception as e:
                logger.error(f"Backend {type(backend).__name__} listing failed: {e}")
                result[type(backend).__name__] = []
        
        return result
    
    async def health_check(self) -> Dict[str, bool]:
        """Check health of all backends"""
        health = {}
        for backend in self.backends:
            try:
                is_healthy = await backend.health_check()
                health[type(backend).__name__] = is_healthy
                self.backend_health[type(backend).__name__] = is_healthy
            except Exception as e:
                logger.error(f"Health check failed for {type(backend).__name__}: {e}")
                health[type(backend).__name__] = False
                self.backend_health[type(backend).__name__] = False
        
        return health
    
    def _get_cached_secret(self, name: str) -> Optional[str]:
        """Get secret from cache if valid"""
        if name not in self.cache:
            return None
        
        cached = self.cache[name]
        
        # Check TTL
        if time.time() - cached.cached_at > self.cache_ttl:
            del self.cache[name]
            return None
        
        # Update access stats
        cached.metadata.accessed_at = datetime.now(UTC)
        cached.metadata.access_count += 1
        self.access_stats[name] = self.access_stats.get(name, 0) + 1
        
        # Decrypt if needed
        if cached.encrypted and self.fernet:
            try:
                return self.fernet.decrypt(cached.value.encode()).decode()
            except Exception as e:
                logger.error(f"Failed to decrypt cached secret '{name}': {e}")
                del self.cache[name]
                return None
        
        return cached.value
    
    async def _cache_secret(self, name: str, value: str, backend: SecretBackendInterface):
        """Cache secret with optional encryption"""
        metadata = SecretMetadata(
            name=name,
            backend=SecretBackend.ENV,  # Map backend to enum
            created_at=datetime.now(UTC),
            accessed_at=datetime.now(UTC),
            access_count=1
        )
        
        # Encrypt if enabled
        cached_value = value
        encrypted = False
        if self.fernet:
            try:
                cached_value = self.fernet.encrypt(value.encode()).decode()
                encrypted = True
            except Exception as e:
                logger.error(f"Failed to encrypt secret '{name}': {e}")
        
        self.cache[name] = CachedSecret(
            value=cached_value,
            metadata=metadata,
            cached_at=time.time(),
            encrypted=encrypted
        )
        
        # Limit cache size
        if len(self.cache) > 1000:
            # Remove oldest 20% of entries
            sorted_items = sorted(
                self.cache.items(),
                key=lambda x: x[1].cached_at
            )
            keep_count = int(len(sorted_items) * 0.8)
            self.cache = dict(sorted_items[-keep_count:])
    
    async def _audit_log(self, action: str, secret_name: str, **kwargs):
        """Log secret access for audit purposes"""
        try:
            audit_entry = {
                "timestamp": datetime.now(UTC).isoformat(),
                "action": action,
                "secret_name": secret_name,
                "success": kwargs.get("success", True),
                **kwargs
            }
            
            # Don't log actual secret values
            if "value" in audit_entry:
                del audit_entry["value"]
            
            with open(self.audit_file, 'a') as f:
                f.write(json.dumps(audit_entry) + '\n')
                
        except Exception as e:
            logger.error(f"Failed to write secret audit log: {e}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get usage metrics"""
        return {
            "cache_size": len(self.cache),
            "cache_ttl_seconds": self.cache_ttl,
            "total_accesses": sum(self.access_stats.values()),
            "unique_secrets_accessed": len(self.access_stats),
            "backend_count": len(self.backends),
            "backend_health": self.backend_health,
            "encryption_enabled": self.enable_encryption
        }
    
    def clear_cache(self):
        """Clear the secret cache"""
        self.cache.clear()
        logger.info("Secret cache cleared")


# Convenience functions for common use cases
_global_secret_manager: Optional[SecretManager] = None

def get_global_secret_manager() -> SecretManager:
    """Get or create global secret manager instance"""
    global _global_secret_manager
    if _global_secret_manager is None:
        _global_secret_manager = SecretManager()
    return _global_secret_manager

async def get_secret(name: str) -> Optional[str]:
    """Convenience function to get secret using global manager"""
    manager = get_global_secret_manager()
    return await manager.get_secret(name)

async def get_secret_with_fallback(name: str, fallback: str) -> str:
    """Convenience function to get secret with fallback"""
    manager = get_global_secret_manager()
    return await manager.get_secret_with_fallback(name, fallback)


# Example usage and testing
if __name__ == "__main__":
    import asyncio
    
    async def test_secret_manager():
        """Test secret manager functionality"""
        # Create test manager with memory backend
        memory_backend = MemorySecretBackend({
            "test_api_key": "sk-test-12345",
            "database_password": "super-secret-password",
            "encryption_key": "32-byte-key-for-encryption-here"
        })
        
        manager = SecretManager([memory_backend])
        
        print("Secret Manager Test Results:")
        print("=" * 40)
        
        # Test secret retrieval
        test_secrets = ["test_api_key", "database_password", "nonexistent_secret"]
        
        for secret_name in test_secrets:
            value = await manager.get_secret(secret_name)
            status = "✅ Found" if value else "❌ Not Found"
            display_value = value[:10] + "..." if value and len(value) > 10 else value
            print(f"{status} {secret_name}: {display_value}")
        
        # Test caching (second access should be faster)
        start_time = time.time()
        await manager.get_secret("test_api_key")
        cache_time = (time.time() - start_time) * 1000
        print(f"Cache access time: {cache_time:.2f}ms")
        
        # Test health check
        health = await manager.health_check()
        print(f"Backend health: {health}")
        
        # Show metrics
        print("\nMetrics:")
        metrics = manager.get_metrics()
        for key, value in metrics.items():
            print(f"  {key}: {value}")
    
    asyncio.run(test_secret_manager())