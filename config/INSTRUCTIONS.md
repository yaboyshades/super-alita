# Configuration Instructions

This directory contains configuration files and templates for Super Alita components.

## Overview

The config directory provides centralized configuration management for various aspects of the Super Alita system:

- **Environment Configuration**: Development, testing, and production settings
- **Service Configuration**: External service integration settings
- **Feature Configuration**: Component-specific configuration options
- **Template Configuration**: Configuration templates and examples

## Configuration Files

### Core Configuration
- **Application Settings**: Main application configuration
- **Environment Variables**: Environment-specific settings
- **Service Endpoints**: External service connection details
- **Feature Flags**: Dynamic feature enablement

### Component Configuration
- **Plugin Settings**: Plugin-specific configuration
- **LLM Configuration**: Language model provider settings
- **Event Bus**: Event system configuration
- **MCP Server**: Model Context Protocol server settings

## Configuration Management

### Environment-Specific Configuration
```yaml
# config/environments/development.yaml
environment: development

server:
  host: "0.0.0.0"
  port: 8080
  debug: true

database:
  url: "sqlite:///dev.db"
  echo: true

logging:
  level: DEBUG
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# config/environments/production.yaml
environment: production

server:
  host: "0.0.0.0"
  port: 8080
  debug: false

database:
  url: "${DATABASE_URL}"
  echo: false

logging:
  level: INFO
  format: "%(asctime)s - %(levelname)s - %(message)s"
```

### Service Configuration
```yaml
# config/services.yaml
llm_providers:
  gemini:
    api_key: "${GEMINI_API_KEY}"
    model: "gemini-1.5-pro"
    timeout: 30
    
  openai:
    api_key: "${OPENAI_API_KEY}"
    model: "gpt-4"
    timeout: 30
    
  anthropic:
    api_key: "${ANTHROPIC_API_KEY}"
    model: "claude-3-sonnet"
    timeout: 30

external_services:
  redis:
    url: "${REDIS_URL}"
    timeout: 5
    
  puter:
    base_url: "${PUTER_BASE_URL}"
    api_key: "${PUTER_API_KEY}"
```

### Plugin Configuration
```yaml
# config/plugins.yaml
plugins:
  memory_manager:
    enabled: true
    storage_path: "./data/memory"
    max_entries: 10000
    
  web_agent:
    enabled: true
    search_provider: "serpapi"
    api_key: "${SERPAPI_KEY}"
    
  creator:
    enabled: true
    output_path: "./generated"
    validation: true
```

## Configuration Loading

### Configuration Hierarchy
The configuration system follows a priority hierarchy:

1. **Command Line Arguments**: Highest priority
2. **Environment Variables**: Override configuration files
3. **Environment-Specific Config**: Development/production specific
4. **Default Configuration**: Base configuration values

### Loading Configuration
```python
from pathlib import Path
import yaml
import os

class ConfigManager:
    """Centralized configuration management."""
    
    def __init__(self, config_dir: Path = Path("config")):
        self.config_dir = config_dir
        self.config = self._load_configuration()
    
    def _load_configuration(self) -> dict:
        """Load configuration from multiple sources."""
        config = {}
        
        # Load base configuration
        base_config = self.config_dir / "base.yaml"
        if base_config.exists():
            config.update(yaml.safe_load(base_config.read_text()))
        
        # Load environment-specific configuration
        env = os.getenv("ENVIRONMENT", "development")
        env_config = self.config_dir / "environments" / f"{env}.yaml"
        if env_config.exists():
            config.update(yaml.safe_load(env_config.read_text()))
        
        # Apply environment variable overrides
        config = self._apply_env_overrides(config)
        
        return config
    
    def _apply_env_overrides(self, config: dict) -> dict:
        """Apply environment variable overrides."""
        def replace_env_vars(value):
            if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
                env_var = value[2:-1]
                return os.getenv(env_var, value)
            elif isinstance(value, dict):
                return {k: replace_env_vars(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [replace_env_vars(item) for item in value]
            return value
        
        return replace_env_vars(config)
    
    def get(self, key: str, default=None):
        """Get configuration value using dot notation."""
        keys = key.split(".")
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
```

## Environment Variables

### Required Variables
```bash
# Core application
ENVIRONMENT=development
PYTHONPATH=./src

# LLM Providers (at least one required)
GEMINI_API_KEY=your_gemini_key_here
OPENAI_API_KEY=your_openai_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here

# External Services
REDIS_URL=redis://localhost:6379
PUTER_BASE_URL=https://puter.com
PUTER_API_KEY=your_puter_key_here

# Optional Services
SERPAPI_KEY=your_serpapi_key_here
BING_API_KEY=your_bing_key_here
```

### Optional Variables
```bash
# Execution Settings
REUG_MAX_TOOL_CALLS=5
REUG_EXEC_TIMEOUT_S=20.0
REUG_MODEL_STREAM_TIMEOUT_S=60.0

# Logging and Debugging
LOG_LEVEL=INFO
DEBUG_MODE=false

# Performance Settings
CACHE_TTL=86400
MAX_CONCURRENT_REQUESTS=10
```

## Configuration Validation

### Schema Validation
```python
from pydantic import BaseModel, Field
from typing import Optional

class ServerConfig(BaseModel):
    """Server configuration schema."""
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8080, ge=1, le=65535)
    debug: bool = Field(default=False)

class LLMProviderConfig(BaseModel):
    """LLM provider configuration schema."""
    api_key: str
    model: str
    timeout: int = Field(default=30, ge=1)
    max_tokens: Optional[int] = Field(default=None, ge=1)

class ApplicationConfig(BaseModel):
    """Main application configuration schema."""
    server: ServerConfig
    llm_providers: dict[str, LLMProviderConfig]
    environment: str = Field(default="development")

def validate_configuration(config_dict: dict) -> ApplicationConfig:
    """Validate configuration against schema."""
    try:
        return ApplicationConfig(**config_dict)
    except ValidationError as e:
        raise ConfigurationError(f"Invalid configuration: {e}")
```

### Configuration Testing
```python
def test_configuration():
    """Test configuration loading and validation."""
    config_manager = ConfigManager()
    
    # Test required values are present
    assert config_manager.get("server.host") is not None
    assert config_manager.get("server.port") is not None
    
    # Test environment-specific overrides
    env = config_manager.get("environment")
    assert env in ["development", "testing", "production"]
    
    # Test LLM provider configuration
    providers = config_manager.get("llm_providers", {})
    assert len(providers) > 0, "At least one LLM provider must be configured"
```

## Configuration Templates

### Development Template
```yaml
# config/templates/development.yaml
environment: development

server:
  host: "0.0.0.0"
  port: 8080
  debug: true
  reload: true

database:
  url: "sqlite:///dev.db"
  echo: true

logging:
  level: DEBUG
  file: "logs/dev.log"

features:
  telemetry: true
  debug_mode: true
  auto_reload: true
```

### Production Template
```yaml
# config/templates/production.yaml
environment: production

server:
  host: "0.0.0.0"
  port: 8080
  debug: false
  workers: 4

database:
  url: "${DATABASE_URL}"
  pool_size: 20
  echo: false

logging:
  level: INFO
  file: "logs/app.log"
  rotation: "1 day"

features:
  telemetry: true
  debug_mode: false
  monitoring: true

security:
  cors_origins: ["https://app.example.com"]
  api_key_required: true
```

## Best Practices

### Configuration Security
- **Sensitive Data**: Never commit API keys or passwords to version control
- **Environment Variables**: Use environment variables for sensitive information
- **Encryption**: Encrypt sensitive configuration files when necessary
- **Access Control**: Restrict access to configuration files

### Configuration Management
- **Version Control**: Track configuration templates and schemas
- **Documentation**: Document all configuration options and their effects
- **Validation**: Validate configuration at startup
- **Defaults**: Provide sensible defaults for all options

### Environment Separation
- **Clear Separation**: Maintain separate configurations for different environments
- **Consistent Structure**: Keep configuration structure consistent across environments
- **Environment Detection**: Automatically detect environment when possible
- **Override Capability**: Allow environment-specific overrides

## Troubleshooting

### Common Configuration Issues

#### Missing Configuration Files
```bash
# Check if configuration files exist
ls -la config/
ls -la config/environments/

# Verify configuration loading
python -c "
from config.manager import ConfigManager
config = ConfigManager()
print(config.config)
"
```

#### Environment Variable Issues
```bash
# Check environment variables
env | grep -E "(GEMINI|OPENAI|ANTHROPIC|REDIS|PUTER)"

# Test environment variable substitution
python -c "
import os
print('GEMINI_API_KEY:', os.getenv('GEMINI_API_KEY', 'NOT_SET'))
"
```

#### Configuration Validation Errors
```python
# Debug configuration validation
try:
    config = validate_configuration(config_dict)
except ConfigurationError as e:
    print(f"Configuration error: {e}")
    # Check specific fields that failed validation
```

### Debugging Configuration
```bash
# Dump current configuration
python -c "
from config.manager import ConfigManager
import json
config = ConfigManager()
print(json.dumps(config.config, indent=2))
"

# Test specific configuration values
python -c "
from config.manager import ConfigManager
config = ConfigManager()
print('Server config:', config.get('server'))
print('LLM providers:', config.get('llm_providers'))
"
```

For specific configuration questions or issues, refer to the configuration schema documentation and example templates in this directory.