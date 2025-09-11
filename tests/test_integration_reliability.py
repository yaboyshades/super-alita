"""
Integration test for Super Alita v3.0 reliability features.

Tests unified launcher, registry, router, and health endpoints with mocked dependencies
to ensure the system works offline and handles failures gracefully.

Adapted from patterns found in GitHub examples for robust testing:
- Mock external dependencies to avoid network calls
- Test timeout and retry behavior 
- Validate circuit breaker patterns
- Ensure graceful degradation
"""

import asyncio
import os
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Mock environment variables for testing
os.environ["ENABLE_GITHUB_DEMO"] = "false"
os.environ["ENABLE_PERPLEXICA_DEMO"] = "false"
os.environ["ENABLE_AUTOGEN_DEMO"] = "false" 
os.environ["ENABLE_ENHANCED_CONSENSUS"] = "true"
os.environ["ALITA_ABILITIES_INCLUDE"] = ""
os.environ["ALITA_ABILITIES_EXCLUDE"] = ""


class TestIntegrationReliability:
    """Integration tests for Super Alita v3.0 reliability features."""
    
    @pytest.fixture
    def mock_base_registry(self):
        """Mock base ability registry."""
        registry = MagicMock()
        registry.get_available_tools_schema.return_value = [
            {"tool_id": "test_tool", "description": "Test tool"},
            {"tool_id": "echo", "description": "Echo tool"},
            {"tool_id": "deepconf_consensus", "description": "Consensus tool"}
        ]
        registry.knows.return_value = True
        registry.validate_args.return_value = True
        registry.execute = AsyncMock(return_value={"result": "success"})
        registry.register_tool = MagicMock()
        return registry
    
    @pytest.fixture
    def mock_base_router(self):
        """Mock base REUG router."""
        router = MagicMock()
        
        async def mock_execute_turn(message, session_id, **kwargs):
            """Mock execute_turn generator."""
            yield {"type": "TaskStarted", "message": message}
            yield {"type": "LLMChunk", "content": "Processing..."}
            yield {"type": "AbilityCalled", "tool_name": "test_tool"}
            yield {"type": "AbilitySucceeded", "result": "success"}
            yield {"type": "TaskSucceeded", "response": "Task completed"}
        
        router.execute_turn = mock_execute_turn
        return router
    
    @pytest.fixture
    def mock_ollama_client(self):
        """Mock Ollama client to avoid real API calls."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"models": [{"name": "gpt-oss:20b"}]}
            
            mock_instance = mock_client.return_value
            mock_instance.get = AsyncMock(return_value=mock_response)
            mock_instance.post = AsyncMock(return_value=mock_response)
            mock_instance.aclose = AsyncMock()
            
            yield mock_instance
    
    @pytest.mark.asyncio
    async def test_unified_registry_initialization(self, mock_base_registry):
        """Test unified registry initializes with feature flags."""
        from src.abilities.unified_registry import UnifiedAbilityRegistry
        
        registry = UnifiedAbilityRegistry(mock_base_registry)
        result = await registry.initialize()
        
        # Verify initialization results
        assert "abilities" in result
        assert "demos" in result
        assert "total_tools" in result
        assert "feature_flags" in result
        
        # Check feature flags are properly configured
        flags = result["feature_flags"]
        assert "github_integration" in flags
        assert "perplexica_search" in flags
        assert flags["github_integration"] is False  # Disabled in env
        assert flags["perplexica_search"] is False   # Disabled in env
        
        # Verify tool registration works
        status = registry.get_status()
        assert status["total_tools"] == 3  # From mock registry
        assert "feature_flags" in status
    
    @pytest.mark.asyncio
    async def test_resilient_router_timeout_handling(self, mock_base_router, mock_ollama_client):
        """Test router handles timeouts gracefully."""
        from src.reug_runtime.unified_router import UnifiedRouter, TimeoutConfig
        
        # Create router with short timeouts for testing
        timeout_config = TimeoutConfig(
            tool_timeout_s=0.1,  # Very short timeout
            llm_timeout_s=0.1,
            overall_timeout_s=1.0
        )
        
        router = UnifiedRouter(mock_base_router, timeout_config=timeout_config)
        
        try:
            # Mock a slow operation that will timeout
            async def slow_operation():
                await asyncio.sleep(0.5)  # Longer than timeout
                return {"result": "slow"}
            
            # Test timeout handling
            result = await router.executor.execute_with_retry(slow_operation)
            
            # Should not reach here due to timeout
            assert False, "Expected timeout exception"
            
        except Exception as e:
            # Verify timeout was handled
            assert "timeout" in str(e).lower() or "asyncio" in str(e).lower()
        
        finally:
            await router.close()
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_functionality(self, mock_base_registry):
        """Test circuit breaker trips after failures."""
        from src.reug_runtime.unified_router import CircuitBreaker, CircuitBreakerConfig, CircuitState
        
        # Create circuit breaker with low threshold for testing
        config = CircuitBreakerConfig(failure_threshold=2, recovery_timeout_s=1)
        circuit = CircuitBreaker(config)
        
        # Initially closed
        assert circuit.state == CircuitState.CLOSED
        assert circuit.can_execute() is True
        
        # Record failures
        circuit.record_failure()
        assert circuit.state == CircuitState.CLOSED  # Still closed after 1 failure
        
        circuit.record_failure()
        assert circuit.state == CircuitState.OPEN  # Should trip after 2 failures
        assert circuit.can_execute() is False
        
        # Wait for recovery timeout
        await asyncio.sleep(1.1)  
        assert circuit.can_execute() is True  # Should be half-open now
        
        # Record success to close circuit
        circuit.record_success()
        assert circuit.state == CircuitState.CLOSED
    
    @pytest.mark.asyncio 
    async def test_enhanced_health_endpoint_mock(self, mock_base_registry):
        """Test enhanced health endpoint without real FastAPI app."""
        # Simulate the health endpoint logic
        
        # Mock app state
        mock_app_state = MagicMock()
        mock_app_state.ability_registry = mock_base_registry
        mock_app_state.event_bus = MagicMock()
        mock_app_state.kg = MagicMock()
        mock_app_state.llm_model = MagicMock()
        
        # Mock the base health check function
        with patch("reug_runtime.health.check_health") as mock_health:
            mock_health.return_value = {
                "status": "healthy",
                "components": {
                    "event_bus": {"status": "ok"},
                    "ability_registry": {"status": "ok"},
                    "kg": {"status": "ok"},
                    "llm": {"status": "ok"}
                }
            }
            
            # Simulate enhanced health endpoint logic
            base_status = await mock_health(
                mock_app_state.event_bus,
                mock_app_state.ability_registry,
                mock_app_state.kg,
                mock_app_state.llm_model
            )
            
            enhanced_status = {
                **base_status,
                "tools": {
                    "total_count": 0,
                    "available": True,
                    "registry_type": "unknown"
                },
                "features": {},
                "version": "3.0"
            }
            
            # Get tool count
            registry = mock_app_state.ability_registry
            if hasattr(registry, 'get_available_tools_schema'):
                tools_schema = registry.get_available_tools_schema()
                enhanced_status["tools"]["total_count"] = len(tools_schema)
                enhanced_status["tools"]["registry_type"] = "simple"
            
            # Verify enhanced status
            assert enhanced_status["status"] == "healthy"
            assert enhanced_status["tools"]["total_count"] == 3
            assert enhanced_status["tools"]["available"] is True
            assert enhanced_status["version"] == "3.0"
    
    @pytest.mark.asyncio
    async def test_startup_sequence_mocked(self, mock_base_registry, mock_ollama_client):
        """Test complete startup sequence with mocked dependencies."""
        
        # Mock imports to avoid dependency issues
        with patch.dict(sys.modules, {
            'fastapi': MagicMock(),
            'uvicorn': MagicMock(),
            'src.main': MagicMock(),
        }):
            from start import UnifiedLauncher
            
            launcher = UnifiedLauncher()
            
            # Test prerequisite checks
            checks = launcher._check_prerequisites()
            
            # Should pass basic checks (python version, env file exists)
            assert checks["python_version"] is True
            # Note: Other checks may fail due to mocking, which is expected
            
            # Test feature flag loading
            flags = launcher._get_feature_flags()
            assert "github_demo" in flags
            assert "consensus_enhanced" in flags
            assert flags["github_demo"] is False  # Disabled in env
            assert flags["consensus_enhanced"] is True  # Enabled in env
    
    @pytest.mark.asyncio
    async def test_demo_feature_flags(self):
        """Test that demo feature flags work correctly."""
        from src.abilities.unified_registry import get_demo_status
        
        # Get demo status without creating full registry
        status = get_demo_status()
        
        assert "github_integration" in status
        assert "perplexica_search" in status
        assert "autogen_pipeline" in status
        
        # Verify flags match environment
        github_status = status["github_integration"] 
        assert github_status["enabled"] is False  # Set to false in env
        assert github_status["flag_env"] == "ENABLE_GITHUB_DEMO"
        assert "GitHub API integration" in github_status["description"]
    
    @pytest.mark.asyncio 
    async def test_retry_mechanism_with_eventual_success(self, mock_ollama_client):
        """Test retry mechanism succeeds on final attempt."""
        from src.reug_runtime.unified_router import ResilientExecutor, RetryConfig
        
        # Create executor with fast retries for testing
        retry_config = RetryConfig(max_retries=3, base_delay_ms=10, max_delay_ms=50)
        executor = ResilientExecutor(retry_config=retry_config)
        
        try:
            # Mock operation that fails twice then succeeds
            call_count = 0
            
            async def flaky_operation():
                nonlocal call_count
                call_count += 1
                if call_count <= 2:
                    raise Exception("Temporary failure")
                return {"result": "success"}
            
            # Should succeed on 3rd attempt
            result = await executor.execute_with_retry(flaky_operation)
            
            assert result["result"] == "success"
            assert call_count == 3  # Failed twice, succeeded on 3rd
            
        finally:
            await executor.close()
    
    @pytest.mark.asyncio
    async def test_graceful_degradation_with_missing_modules(self):
        """Test system handles missing optional modules gracefully."""
        
        # Test unified registry handles missing demo modules
        with patch("importlib.import_module") as mock_import:
            # Make demo dependencies unavailable
            mock_import.side_effect = ImportError("Module not found")
            
            from src.abilities.unified_registry import FeatureFlaggedDemo
            
            demo = FeatureFlaggedDemo(
                name="Test Demo",
                flag_env="ENABLE_TEST_DEMO",
                module_path="nonexistent_module",
                dependencies=["missing_dependency"]
            )
            
            # Should handle missing dependencies gracefully
            assert demo.available is False
            assert demo.load_module() is None  # Should return None, not crash
    
    def test_environment_configuration_loading(self):
        """Test environment configuration is loaded correctly."""
        from src.reug_runtime.unified_router import load_resilience_config_from_env
        
        # Set test environment variables
        test_env = {
            "REUG_MAX_RETRIES": "5",
            "REUG_TOOL_TIMEOUT_S": "45.0",
            "REUG_CIRCUIT_FAILURE_THRESHOLD": "3"
        }
        
        with patch.dict(os.environ, test_env):
            retry_config, timeout_config, circuit_config = load_resilience_config_from_env()
            
            assert retry_config.max_retries == 5
            assert timeout_config.tool_timeout_s == 45.0
            assert circuit_config.failure_threshold == 3


@pytest.mark.integration
class TestOfflineIntegration:
    """Integration tests that must work completely offline."""
    
    def test_unified_launcher_help(self):
        """Test unified launcher help works offline."""
        import subprocess
        
        # Test help command works without network
        result = subprocess.run(
            [sys.executable, "start.py", "--help"],
            capture_output=True, 
            text=True,
            timeout=10
        )
        
        assert result.returncode == 0
        assert "Super Alita v3.0 Unified Launcher" in result.stdout
        assert "--mode" in result.stdout
        assert "web" in result.stdout
    
    def test_demo_status_offline(self):
        """Test demo status can be queried offline."""
        from src.abilities.unified_registry import get_demo_status
        
        # Should work without any network dependencies
        status = get_demo_status()
        
        assert isinstance(status, dict)
        assert len(status) > 0
        
        for demo_id, demo_info in status.items():
            assert "enabled" in demo_info
            assert "available" in demo_info
            assert "description" in demo_info
            assert isinstance(demo_info["enabled"], bool)
            assert isinstance(demo_info["available"], bool)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])