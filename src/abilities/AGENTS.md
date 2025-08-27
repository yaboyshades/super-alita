# Abilities System - Agent Instructions

## Overview
The `src/abilities/` directory contains capability implementations that extend Super Alita's core functionality:
- **Code Generation** - AI-powered code generation using external models
- **Capability Abstractions** - High-level ability interfaces and implementations
- **Model Integration** - Integration with external AI models and services

## Key Files & Responsibilities

### Code Generation Abilities
- `gemini_codegen_ability.py` - Google Gemini AI integration for code generation
- `__init__.py` - Module initialization and exports

## Development Guidelines

### Creating New Abilities
```python
from src.core.plugin_interface import PluginInterface
from src.utils.guardrails import hash_top_files, repo_download_integrity
from src.utils.telemetry import ReadLedger, start_timer, end_timer

class YourAbility(PluginInterface):
    """Custom ability implementation"""
    
    def __init__(self, event_bus, config=None):
        super().__init__(event_bus, config)
        self.name = "your_ability"
        
    async def execute_ability(self, request_data):
        """Execute the core ability logic"""
        timer = start_timer("ability_execution")
        try:
            # Implementation here
            result = await self._process_request(request_data)
            return result
        finally:
            end_timer(timer)
```

### Security Guidelines
- **Input validation** - Always validate external API inputs
- **Rate limiting** - Implement appropriate rate limits for external services
- **API key management** - Use environment variables for sensitive credentials
- **Output sanitization** - Sanitize generated code before execution

### Integration Patterns
```python
# Ability registration pattern
async def register_ability(self, capability_name: str):
    """Register ability with the system"""
    event = create_event(
        "ability_registered",
        capability=capability_name,
        source_plugin=self.name
    )
    await self.event_bus.publish(event)

# Telemetry integration
from src.utils.telemetry import ReadLedger, telemetry_footer

async def track_ability_usage(self, ability_name: str, result: dict):
    """Track ability usage for analytics"""
    ledger = ReadLedger()
    ledger.log_ability_execution(ability_name, result)
    await ledger.commit()
```

## Testing Guidelines

### Ability Testing Pattern
```python
import pytest
from unittest.mock import AsyncMock, MagicMock
from src.abilities.your_ability import YourAbility

@pytest.mark.asyncio
async def test_ability_execution():
    """Test ability execution flow"""
    event_bus = AsyncMock()
    ability = YourAbility(event_bus)
    
    # Test successful execution
    result = await ability.execute_ability({"input": "test"})
    assert result["success"] is True
    
    # Verify event emission
    event_bus.publish.assert_called()

@pytest.mark.asyncio
async def test_ability_error_handling():
    """Test ability error handling"""
    event_bus = AsyncMock()
    ability = YourAbility(event_bus)
    
    # Test error scenarios
    with pytest.raises(ValueError):
        await ability.execute_ability({"invalid": "input"})
```

### External Service Testing
- Use mocks for external API calls during testing
- Implement integration tests with test API keys
- Test rate limiting and timeout scenarios
- Verify error handling for service failures

## Performance Guidelines
- **Async operations** - All external API calls must be async
- **Connection pooling** - Reuse HTTP connections when possible
- **Caching** - Cache expensive operations and model responses
- **Timeout handling** - Implement proper timeouts for external services

## Common Patterns

### Error Handling
```python
async def safe_ability_call(self, ability_func, *args, **kwargs):
    """Safe wrapper for ability execution"""
    try:
        return await ability_func(*args, **kwargs)
    except aiohttp.ClientError as e:
        logger.error(f"HTTP client error: {e}")
        raise
    except Exception as e:
        logger.error(f"Ability execution failed: {e}")
        raise
```

### Configuration Management
```python
def load_ability_config(self, config_name: str) -> dict:
    """Load ability-specific configuration"""
    config_path = os.getenv(f"{config_name.upper()}_CONFIG")
    if config_path:
        return json.loads(Path(config_path).read_text())
    return {}
```

## Debugging Tips
- **API logging** - Enable debug logging for external API interactions
- **Telemetry monitoring** - Monitor ability execution metrics
- **Error tracking** - Track and analyze ability failure patterns
- **Performance profiling** - Profile expensive ability operations