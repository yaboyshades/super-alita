#!/usr/bin/env python3
"""
DTA 2.0 Configuration Management - Cognitive Airlock Edition

Enhanced configuration management for DTA 2.0 components with
cognitive processing capabilities and circuit breaker patterns.
"""

import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


def substitute_env_vars(value: str) -> str:
    """Substitute environment variables in string values."""
    if not isinstance(value, str):
        return value

    # Pattern to match ${VAR} or ${VAR:default}
    pattern = r"\$\{([^:}]+)(?::([^}]*))?\}"

    def replace_var(match):
        var_name = match.group(1)
        default_value = match.group(2) or ""
        return os.environ.get(var_name, default_value)

    return re.sub(pattern, replace_var, value)


def process_config_dict(config_dict: dict[str, Any]) -> dict[str, Any]:
    """Process configuration dictionary to substitute environment variables."""
    if isinstance(config_dict, dict):
        return {k: process_config_dict(v) for k, v in config_dict.items()}
    if isinstance(config_dict, list):
        return [process_config_dict(item) for item in config_dict]
    if isinstance(config_dict, str):
        return substitute_env_vars(config_dict)
    return config_dict


# DTA 2.0 Cognitive Airlock Configuration Classes
@dataclass
class CircuitBreakerConfig:
    """Enhanced circuit breaker configuration for cognitive processing."""

    enabled: bool = True
    failure_threshold: int = 5
    timeout_seconds: int = 60
    recovery_timeout: int = 60
    half_open_max_calls: int = 5


@dataclass
class LLMConfig:
    """Enhanced LLM configuration for cognitive processing."""

    provider: str = "gemini"
    model: str = "gemini-1.5-pro"
    model_name: str = "gemini-1.5-flash"  # Backward compatibility
    api_key: str | None = None
    timeout_seconds: float = 30.0
    max_tokens: int = 4000
    temperature: float = 0.7
    retry_attempts: int = 3


@dataclass
class CognitiveTurnConfig:
    """Configuration for cognitive turn processing."""

    enabled: bool = True
    max_execution_time: float = 60.0
    validation_enabled: bool = True
    confidence_threshold: float = 0.7
    memory_retention_hours: int = 24


@dataclass
class MonitoringConfig:
    """Configuration for monitoring component."""

    enabled: bool = True
    log_level: str = "INFO"
    metrics_port: int | None = None


@dataclass
class CacheConfig:
    """Configuration for cache component."""

    enabled: bool = True
    backend: str = "memory"
    max_size: int = 10000
    default_ttl: int = 3600
    redis_url: str = "redis://localhost:6379"


@dataclass
class ValidationConfig:
    """Configuration for validation component."""

    enabled: bool = True
    level: str = "normal"


@dataclass
class DTAConfig:
    """Complete DTA 2.0 configuration with cognitive processing capabilities."""

    environment: str = "development"
    debug: bool = False
    circuit_breaker: CircuitBreakerConfig = field(default_factory=CircuitBreakerConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    cognitive_turn: CognitiveTurnConfig = field(default_factory=CognitiveTurnConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)

    def __init__(self, **kwargs):
        # Handle dict-style initialization
        self.environment = kwargs.get("environment", "development")
        self.debug = kwargs.get("debug", False)

        # Initialize circuit breaker config
        circuit_breaker_data = kwargs.get("circuit_breaker", {})
        if isinstance(circuit_breaker_data, dict):
            self.circuit_breaker = CircuitBreakerConfig(**circuit_breaker_data)
        else:
            self.circuit_breaker = circuit_breaker_data or CircuitBreakerConfig()

        # Initialize LLM config
        llm_data = kwargs.get("llm", {})
        if isinstance(llm_data, dict):
            # Process environment variables
            llm_data = process_config_dict(llm_data)
            self.llm = LLMConfig(**llm_data)
        else:
            self.llm = llm_data or LLMConfig()

        # Initialize cognitive turn config
        cognitive_turn_data = kwargs.get("cognitive_turn", {})
        if isinstance(cognitive_turn_data, dict):
            self.cognitive_turn = CognitiveTurnConfig(**cognitive_turn_data)
        else:
            self.cognitive_turn = cognitive_turn_data or CognitiveTurnConfig()

        # Initialize monitoring config
        monitoring_data = kwargs.get("monitoring", {})
        if isinstance(monitoring_data, dict):
            self.monitoring = MonitoringConfig(**monitoring_data)
        else:
            self.monitoring = monitoring_data or MonitoringConfig()

        # Initialize cache config
        cache_data = kwargs.get("cache", {})
        if isinstance(cache_data, dict):
            self.cache = CacheConfig(**cache_data)
        else:
            self.cache = cache_data or CacheConfig()

        # Initialize validation config
        validation_data = kwargs.get("validation", {})
        if isinstance(validation_data, dict):
            self.validation = ValidationConfig(**validation_data)
        else:
            self.validation = validation_data or ValidationConfig()

    @classmethod
    def from_file(cls, config_path: str) -> "DTAConfig":
        """Load configuration from YAML file with enhanced error handling."""
        try:
            config_file = Path(config_path)
            if not config_file.exists():
                logger.warning(f"Config file {config_path} not found, using defaults")
                return cls()

            with open(config_file) as f:
                data = yaml.safe_load(f)

            # Process environment variables in the loaded data
            data = process_config_dict(data)
            return cls(**data)

        except ImportError:
            logger.error("PyYAML not available, using default configuration")
            return cls()
        except Exception as e:
            logger.error(f"Error loading config from {config_path}: {e}")
            return cls()


def create_default_config() -> DTAConfig:
    """Create a DTAConfig instance with sensible defaults."""
    return DTAConfig()


def load_config_from_yaml(file_path: str) -> DTAConfig:
    """Load configuration from YAML file (simplified implementation)."""
    try:
        import yaml

        with open(file_path) as f:
            data = yaml.safe_load(f)
        # Process environment variables in the loaded data
        data = process_config_dict(data)
        return DTAConfig(**data)
    except ImportError:
        # Fallback if PyYAML not available
        return DTAConfig()
    except Exception:
        # Fallback on any error
        return DTAConfig()


# Example usage
if __name__ == "__main__":
    config = create_default_config()
    print(f"Default config: {config}")
