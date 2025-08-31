# Configuration Management - Agent Instructions

## Overview
The `src/config/` directory contains configuration management utilities and settings:
- **Configuration Loading** - Environment and file-based configuration
- **Prompt Management** - Centralized prompt templates and configuration
- **Settings Validation** - Configuration schema validation and defaults
- **Environment Management** - Environment-specific configuration handling

## Key Files & Responsibilities

### Configuration Components
- `prompts/` - Centralized prompt templates and management
- `__init__.py` - Module initialization and configuration exports

## Development Guidelines

### Configuration Loading Patterns
```python
import os
import json
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field

@dataclass
class SystemConfig:
    """System-wide configuration"""
    debug_mode: bool = False
    log_level: str = "INFO"
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    max_workers: int = 4
    timeout_seconds: int = 30
    data_dir: str = "./data"

    @classmethod
    def from_env(cls) -> 'SystemConfig':
        """Load configuration from environment variables"""
        return cls(
            debug_mode=os.getenv('DEBUG', '').lower() == 'true',
            log_level=os.getenv('LOG_LEVEL', 'INFO'),
            redis_host=os.getenv('REDIS_HOST', 'localhost'),
            redis_port=int(os.getenv('REDIS_PORT', '6379')),
            redis_db=int(os.getenv('REDIS_DB', '0')),
            max_workers=int(os.getenv('MAX_WORKERS', '4')),
            timeout_seconds=int(os.getenv('TIMEOUT_SECONDS', '30')),
            data_dir=os.getenv('SUPER_ALITA_DATA_DIR', './data')
        )

    @classmethod
    def from_file(cls, config_path: str) -> 'SystemConfig':
        """Load configuration from JSON file"""
        if not Path(config_path).exists():
            return cls()

        with open(config_path, 'r') as f:
            config_data = json.load(f)

        return cls(**config_data)

class ConfigManager:
    """Centralized configuration management"""

    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file
        self._config_cache: Dict[str, Any] = {}
        self._system_config: Optional[SystemConfig] = None

    def get_system_config(self) -> SystemConfig:
        """Get system configuration"""
        if self._system_config is None:
            if self.config_file and Path(self.config_file).exists():
                self._system_config = SystemConfig.from_file(self.config_file)
            else:
                self._system_config = SystemConfig.from_env()
        return self._system_config

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value with caching"""
        if key in self._config_cache:
            return self._config_cache[key]

        # Try environment variable
        env_key = key.upper().replace('.', '_')
        env_value = os.getenv(env_key)
        if env_value is not None:
            self._config_cache[key] = env_value
            return env_value

        # Try system config
        system_config = self.get_system_config()
        if hasattr(system_config, key):
            value = getattr(system_config, key)
            self._config_cache[key] = value
            return value

        # Use default
        self._config_cache[key] = default
        return default

    def set(self, key: str, value: Any):
        """Set configuration value"""
        self._config_cache[key] = value

    def reload(self):
        """Reload configuration from sources"""
        self._config_cache.clear()
        self._system_config = None
```

### Plugin Configuration
```python
@dataclass
class PluginConfig:
    """Plugin-specific configuration"""
    name: str
    enabled: bool = True
    priority: int = 1
    config: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)

    def get_config_value(self, key: str, default: Any = None) -> Any:
        """Get plugin configuration value"""
        return self.config.get(key, default)

    def validate(self) -> List[str]:
        """Validate plugin configuration"""
        errors = []

        if not self.name:
            errors.append("Plugin name is required")

        if not isinstance(self.enabled, bool):
            errors.append("Plugin enabled must be boolean")

        if not isinstance(self.priority, int) or self.priority < 1:
            errors.append("Plugin priority must be positive integer")

        return errors

class PluginConfigManager:
    """Manage plugin configurations"""

    def __init__(self, config_dir: str = "./config/plugins"):
        self.config_dir = Path(config_dir)
        self.plugin_configs: Dict[str, PluginConfig] = {}

    def load_plugin_configs(self):
        """Load all plugin configurations"""
        if not self.config_dir.exists():
            return

        for config_file in self.config_dir.glob("*.json"):
            try:
                with open(config_file) as f:
                    config_data = json.load(f)

                plugin_config = PluginConfig(**config_data)
                validation_errors = plugin_config.validate()

                if validation_errors:
                    logger.error(f"Invalid config for {config_file}: {validation_errors}")
                    continue

                self.plugin_configs[plugin_config.name] = plugin_config

            except Exception as e:
                logger.error(f"Failed to load plugin config {config_file}: {e}")

    def get_plugin_config(self, plugin_name: str) -> Optional[PluginConfig]:
        """Get configuration for specific plugin"""
        return self.plugin_configs.get(plugin_name)

    def register_plugin_config(self, config: PluginConfig):
        """Register plugin configuration"""
        validation_errors = config.validate()
        if validation_errors:
            raise ValueError(f"Invalid plugin config: {validation_errors}")

        self.plugin_configs[config.name] = config
```

### Environment Configuration
```python
from enum import Enum

class Environment(Enum):
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"

class EnvironmentConfig:
    """Environment-specific configuration management"""

    def __init__(self, environment: Environment = None):
        self.environment = environment or self._detect_environment()

    def _detect_environment(self) -> Environment:
        """Auto-detect environment from various sources"""
        # Check environment variable
        env_name = os.getenv('SUPER_ALITA_ENV', '').lower()
        if env_name:
            try:
                return Environment(env_name)
            except ValueError:
                pass

        # Check for specific files/conditions
        if Path('.env.local').exists():
            return Environment.DEVELOPMENT
        elif os.getenv('CI'):
            return Environment.TESTING
        elif os.getenv('STAGING'):
            return Environment.STAGING
        elif os.getenv('PRODUCTION'):
            return Environment.PRODUCTION
        else:
            return Environment.DEVELOPMENT

    def get_config_overrides(self) -> Dict[str, Any]:
        """Get environment-specific configuration overrides"""
        if self.environment == Environment.DEVELOPMENT:
            return {
                'debug_mode': True,
                'log_level': 'DEBUG',
                'redis_db': 1,
                'timeout_seconds': 60
            }
        elif self.environment == Environment.TESTING:
            return {
                'debug_mode': True,
                'log_level': 'DEBUG',
                'redis_db': 15,
                'timeout_seconds': 10
            }
        elif self.environment == Environment.STAGING:
            return {
                'debug_mode': False,
                'log_level': 'INFO',
                'timeout_seconds': 30
            }
        elif self.environment == Environment.PRODUCTION:
            return {
                'debug_mode': False,
                'log_level': 'WARNING',
                'timeout_seconds': 30
            }
        else:
            return {}
```

## Security Guidelines

### Secure Configuration Loading
```python
import re
from typing import Set

class SecureConfigLoader:
    """Secure configuration loading with validation"""

    def __init__(self):
        self.sensitive_keys: Set[str] = {
            'password', 'secret', 'key', 'token', 'credential'
        }
        self.allowed_config_files: Set[str] = {
            'config.json', 'settings.json', 'environment.json'
        }

    def is_sensitive_key(self, key: str) -> bool:
        """Check if configuration key contains sensitive data"""
        key_lower = key.lower()
        return any(sensitive in key_lower for sensitive in self.sensitive_keys)

    def sanitize_config_for_logging(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize configuration for safe logging"""
        sanitized = {}

        for key, value in config.items():
            if self.is_sensitive_key(key):
                sanitized[key] = "***REDACTED***"
            elif isinstance(value, dict):
                sanitized[key] = self.sanitize_config_for_logging(value)
            else:
                sanitized[key] = value

        return sanitized

    def validate_config_file_path(self, file_path: str) -> bool:
        """Validate configuration file path for security"""
        path = Path(file_path)

        # Check file extension
        if path.suffix not in {'.json', '.yaml', '.yml', '.toml'}:
            return False

        # Check for path traversal
        try:
            path.resolve().relative_to(Path.cwd())
        except ValueError:
            return False

        # Check filename against allowlist
        return path.name in self.allowed_config_files

    def load_secure_config(self, file_path: str) -> Dict[str, Any]:
        """Load configuration with security validation"""
        if not self.validate_config_file_path(file_path):
            raise SecurityError(f"Invalid configuration file path: {file_path}")

        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {file_path}")

        with open(path) as f:
            config = json.load(f)

        # Log sanitized config
        sanitized = self.sanitize_config_for_logging(config)
        logger.info(f"Loaded configuration: {sanitized}")

        return config
```

### Configuration Validation
```python
from pydantic import BaseModel, validator, Field
from typing import Union, List

class DatabaseConfig(BaseModel):
    """Database configuration schema"""
    host: str = Field(..., min_length=1)
    port: int = Field(..., ge=1, le=65535)
    database: str = Field(..., min_length=1)
    username: str = Field(..., min_length=1)
    password: str = Field(..., min_length=1)

    @validator('host')
    def validate_host(cls, v):
        # Basic hostname validation
        if not re.match(r'^[a-zA-Z0-9.-]+$', v):
            raise ValueError('Invalid hostname format')
        return v

class RedisConfig(BaseModel):
    """Redis configuration schema"""
    host: str = "localhost"
    port: int = Field(6379, ge=1, le=65535)
    database: int = Field(0, ge=0, le=15)
    password: Optional[str] = None

class LoggingConfig(BaseModel):
    """Logging configuration schema"""
    level: str = Field("INFO", regex=r'^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$')
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_path: Optional[str] = None
    max_file_size: int = Field(10485760, ge=1024)  # 10MB default

class AppConfig(BaseModel):
    """Application configuration schema"""
    debug: bool = False
    database: DatabaseConfig
    redis: RedisConfig
    logging: LoggingConfig
    plugins: List[str] = []

    @validator('plugins')
    def validate_plugins(cls, v):
        # Validate plugin names
        for plugin in v:
            if not re.match(r'^[a-zA-Z][a-zA-Z0-9_]*$', plugin):
                raise ValueError(f'Invalid plugin name: {plugin}')
        return v
```

## Testing Guidelines

### Configuration Testing
```python
import pytest
import tempfile
import json
from pathlib import Path
from src.config.config_manager import ConfigManager, SystemConfig

def test_system_config_from_env(monkeypatch):
    """Test system configuration loading from environment"""
    # Set environment variables
    monkeypatch.setenv('DEBUG', 'true')
    monkeypatch.setenv('LOG_LEVEL', 'DEBUG')
    monkeypatch.setenv('REDIS_PORT', '6380')

    config = SystemConfig.from_env()

    assert config.debug_mode is True
    assert config.log_level == 'DEBUG'
    assert config.redis_port == 6380

def test_system_config_from_file():
    """Test system configuration loading from file"""
    config_data = {
        'debug_mode': True,
        'log_level': 'DEBUG',
        'redis_host': 'redis.example.com',
        'redis_port': 6379,
        'max_workers': 8
    }

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(config_data, f)
        temp_path = f.name

    try:
        config = SystemConfig.from_file(temp_path)

        assert config.debug_mode is True
        assert config.log_level == 'DEBUG'
        assert config.redis_host == 'redis.example.com'
        assert config.max_workers == 8

    finally:
        Path(temp_path).unlink()

def test_config_manager_caching():
    """Test configuration caching"""
    manager = ConfigManager()

    # Mock environment variable
    import os
    os.environ['TEST_CONFIG_KEY'] = 'test_value'

    try:
        # First access should read from environment
        value1 = manager.get('test_config_key')
        assert value1 == 'test_value'

        # Second access should use cache
        value2 = manager.get('test_config_key')
        assert value2 == 'test_value'

        # Verify caching
        assert 'test_config_key' in manager._config_cache

    finally:
        del os.environ['TEST_CONFIG_KEY']

def test_plugin_config_validation():
    """Test plugin configuration validation"""
    from src.config.config_manager import PluginConfig

    # Valid config
    valid_config = PluginConfig(
        name="test_plugin",
        enabled=True,
        priority=1,
        config={"setting": "value"}
    )

    assert valid_config.validate() == []

    # Invalid config
    invalid_config = PluginConfig(
        name="",  # Empty name
        enabled="yes",  # Wrong type
        priority=0  # Invalid priority
    )

    errors = invalid_config.validate()
    assert len(errors) > 0
```

### Security Testing
```python
def test_secure_config_loader():
    """Test secure configuration loading"""
    from src.config.config_manager import SecureConfigLoader

    loader = SecureConfigLoader()

    # Test sensitive key detection
    assert loader.is_sensitive_key('database_password') is True
    assert loader.is_sensitive_key('api_key') is True
    assert loader.is_sensitive_key('username') is False

    # Test configuration sanitization
    config = {
        'username': 'admin',
        'password': 'secret123',
        'api_key': 'abc123',
        'debug': True
    }

    sanitized = loader.sanitize_config_for_logging(config)

    assert sanitized['username'] == 'admin'
    assert sanitized['password'] == '***REDACTED***'
    assert sanitized['api_key'] == '***REDACTED***'
    assert sanitized['debug'] is True

def test_config_file_path_validation():
    """Test configuration file path validation"""
    from src.config.config_manager import SecureConfigLoader

    loader = SecureConfigLoader()

    # Valid paths
    assert loader.validate_config_file_path('config.json') is True
    assert loader.validate_config_file_path('./config/settings.json') is True

    # Invalid paths
    assert loader.validate_config_file_path('../config.json') is False
    assert loader.validate_config_file_path('/etc/passwd') is False
    assert loader.validate_config_file_path('malicious.exe') is False
```

## Performance Guidelines

### Configuration Caching
```python
import time
from functools import lru_cache
from typing import Any, Dict

class PerformantConfigManager:
    """Configuration manager optimized for performance"""

    def __init__(self):
        self._config_cache: Dict[str, tuple] = {}  # (value, timestamp)
        self._cache_ttl = 300  # 5 minutes

    @lru_cache(maxsize=128)
    def get_cached_env_var(self, key: str) -> Optional[str]:
        """Get environment variable with LRU caching"""
        return os.getenv(key)

    def get_with_ttl_cache(self, key: str, default: Any = None) -> Any:
        """Get configuration with TTL cache"""
        current_time = time.time()

        if key in self._config_cache:
            value, timestamp = self._config_cache[key]
            if current_time - timestamp < self._cache_ttl:
                return value

        # Load fresh value
        value = self._load_config_value(key, default)
        self._config_cache[key] = (value, current_time)

        return value

    def _load_config_value(self, key: str, default: Any) -> Any:
        """Load configuration value from source"""
        # Implementation for loading from various sources
        return os.getenv(key.upper(), default)
```

## Common Patterns

### Configuration Hierarchies
```python
class HierarchicalConfig:
    """Configuration with multiple layers and inheritance"""

    def __init__(self):
        self.layers = [
            self._load_defaults(),
            self._load_file_config(),
            self._load_env_config(),
            self._load_runtime_config()
        ]

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value from hierarchy (last layer wins)"""
        for layer in reversed(self.layers):
            if key in layer:
                return layer[key]
        return default

    def _load_defaults(self) -> Dict[str, Any]:
        """Load default configuration values"""
        return {
            'log_level': 'INFO',
            'timeout': 30,
            'max_retries': 3
        }

    def _load_file_config(self) -> Dict[str, Any]:
        """Load configuration from files"""
        # Implementation for file-based config
        return {}

    def _load_env_config(self) -> Dict[str, Any]:
        """Load configuration from environment variables"""
        # Implementation for environment-based config
        return {}

    def _load_runtime_config(self) -> Dict[str, Any]:
        """Load runtime configuration overrides"""
        return {}
```

### Dynamic Configuration
```python
class DynamicConfigManager:
    """Configuration manager with runtime updates"""

    def __init__(self):
        self._config: Dict[str, Any] = {}
        self._watchers: Dict[str, List[Callable]] = {}

    def set_config(self, key: str, value: Any):
        """Set configuration value and notify watchers"""
        old_value = self._config.get(key)
        self._config[key] = value

        if old_value != value:
            self._notify_watchers(key, old_value, value)

    def watch_config(self, key: str, callback: Callable[[str, Any, Any], None]):
        """Watch for configuration changes"""
        if key not in self._watchers:
            self._watchers[key] = []
        self._watchers[key].append(callback)

    def _notify_watchers(self, key: str, old_value: Any, new_value: Any):
        """Notify watchers of configuration changes"""
        for callback in self._watchers.get(key, []):
            try:
                callback(key, old_value, new_value)
            except Exception as e:
                logger.error(f"Error in config watcher for {key}: {e}")
```

## Debugging Tips
- **Configuration tracing** - Log configuration loading and resolution
- **Cache monitoring** - Monitor configuration cache hit rates
- **Validation logging** - Log configuration validation results
- **Environment debugging** - Debug environment variable resolution
- **Hot reload testing** - Test configuration updates at runtime
