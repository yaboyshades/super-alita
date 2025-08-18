"""
Integration tests for Super Alita agent.
"""

import pytest
import asyncio
import tempfile
import os
from unittest.mock import patch, Mock

from src.main import SuperAlita


@pytest.mark.asyncio
async def test_agent_initialization():
    """Test agent initialization process."""
    
    # Create temporary config file
    config_content = """
agent:
  name: "Test Alita"
  version: "1.0.0"

plugins:
  semantic_memory:
    enabled: false
  semantic_fsm:
    enabled: false
  skill_discovery:
    enabled: false

logging:
  level: "WARNING"
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(config_content)
        config_path = f.name
    
    try:
        agent = SuperAlita(config_path)
        
        # Test initialization
        await agent.initialize()
        
        assert agent.config["agent"]["name"] == "Test Alita"
        assert len(agent.plugins) == 0  # All plugins disabled
        assert agent.event_bus is not None
        assert agent.neural_store is not None
        assert agent.genealogy_tracer is not None
        
    finally:
        os.unlink(config_path)


@pytest.mark.asyncio
async def test_agent_with_plugins():
    """Test agent with plugins enabled."""
    
    config_content = """
agent:
  name: "Test Alita"

plugins:
  semantic_memory:
    enabled: true
    max_memories: 100
  semantic_fsm:
    enabled: true
  skill_discovery:
    enabled: true
    discovery_interval_minutes: 1

logging:
  level: "ERROR"
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(config_content)
        config_path = f.name
    
    try:
        # Mock SentenceTransformer to avoid downloading models in tests
        with patch('src.plugins.semantic_memory_plugin.SentenceTransformer') as mock_st:
            mock_encoder = Mock()
            mock_encoder.encode.return_value = [0.1, 0.2, 0.3]
            mock_encoder.get_sentence_embedding_dimension.return_value = 3
            mock_st.return_value = mock_encoder
            
            with patch('src.plugins.semantic_fsm_plugin.SentenceTransformer', return_value=mock_encoder):
                agent = SuperAlita(config_path)
                
                await agent.initialize()
                
                # Check plugins are registered
                assert "semantic_memory" in agent.plugins
                assert "semantic_fsm" in agent.plugins
                assert "skill_discovery" in agent.plugins
                
                # Test plugin health before start
                for plugin_name, plugin in agent.plugins.items():
                    health = await plugin.health_check()
                    assert health["status"] == "stopped"
    
    finally:
        os.unlink(config_path)


@pytest.mark.asyncio
async def test_agent_lifecycle():
    """Test complete agent lifecycle."""
    
    # Use minimal config for faster testing
    config_content = """
agent:
  name: "Lifecycle Test"

plugins:
  semantic_memory:
    enabled: false
  semantic_fsm:
    enabled: false
  skill_discovery:
    enabled: false

logging:
  level: "ERROR"
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(config_content)
        config_path = f.name
    
    try:
        agent = SuperAlita(config_path)
        
        # Initialize
        await agent.initialize()
        assert not agent.is_running
        
        # Start
        await agent.start()
        assert agent.is_running
        assert agent.start_time is not None
        
        # Get stats
        stats = await agent.get_agent_stats()
        assert stats["agent"]["name"] == "Lifecycle Test"
        assert stats["agent"]["is_running"] is True
        
        # Process commands
        result = await agent.process_command("status")
        assert "agent" in result
        
        result = await agent.process_command("health")
        assert "health_reports" in result
        
        # Test unknown command
        result = await agent.process_command("unknown_command")
        assert "error" in result
        
        # Shutdown
        await agent.shutdown()
        assert not agent.is_running
    
    finally:
        os.unlink(config_path)


@pytest.mark.asyncio
async def test_agent_command_processing():
    """Test agent command processing functionality."""
    
    config_content = """
agent:
  name: "Command Test"

plugins:
  semantic_memory:
    enabled: false
  semantic_fsm:
    enabled: false  
  skill_discovery:
    enabled: false

logging:
  level: "ERROR"
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(config_content)
        config_path = f.name
    
    try:
        agent = SuperAlita(config_path)
        await agent.initialize()
        await agent.start()
        
        # Test status command
        result = await agent.process_command("status")
        assert "agent" in result
        assert result["agent"]["name"] == "Command Test"
        
        # Test health command
        result = await agent.process_command("health")
        assert "health_reports" in result
        
        # Test genealogy export command
        result = await agent.process_command("export_genealogy")
        assert "message" in result
        assert "genealogy_export_" in result["message"]
        
        # Test event emission command
        result = await agent.process_command("emit_event test_event", test_data="value")
        assert result["message"] == "Event 'test_event' emitted"
        
        # Test skill discovery trigger
        result = await agent.process_command("trigger_skill_discovery")
        assert result["message"] == "Skill discovery triggered"
        
        # Test evolution trigger
        result = await agent.process_command("trigger_evolution")
        assert result["message"] == "Evolution triggered"
        
        await agent.shutdown()
    
    finally:
        os.unlink(config_path)


@pytest.mark.asyncio 
async def test_agent_error_handling():
    """Test agent error handling."""
    
    # Test with invalid config path
    agent = SuperAlita("nonexistent_config.yaml")
    
    # Should not crash, should use default config
    await agent.initialize()
    assert agent.config["agent"]["name"] == "Super Alita"  # Default name
    
    await agent.shutdown()


@pytest.mark.asyncio
async def test_agent_plugin_failure_handling():
    """Test handling of plugin failures."""
    
    # Create a plugin that fails during setup
    class FailingPlugin:
        @property
        def name(self):
            return "failing_plugin"
        
        @property 
        def version(self):
            return "1.0.0"
        
        async def setup(self, event_bus, store, config):
            raise Exception("Setup failed")
        
        async def start(self):
            pass
        
        async def stop(self):
            pass
        
        async def shutdown(self):
            pass
        
        async def health_check(self):
            return {"status": "failed"}
    
    agent = SuperAlita()
    agent.plugins["failing"] = FailingPlugin()
    
    # Should handle plugin failure gracefully
    await agent._initialize_plugins()
    
    # Failed plugin should be removed
    assert "failing" not in agent.plugins


def test_agent_configuration_loading():
    """Test configuration loading with various scenarios."""
    
    # Test with valid config
    config_content = """
agent:
  name: "Config Test"
  custom_setting: "test_value"

plugins:
  test_plugin:
    enabled: true
    setting: "value"
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(config_content)
        config_path = f.name
    
    try:
        agent = SuperAlita(config_path)
        agent.load_config()
        
        assert agent.config["agent"]["name"] == "Config Test"
        assert agent.config["agent"]["custom_setting"] == "test_value"
        assert agent.config["plugins"]["test_plugin"]["setting"] == "value"
    
    finally:
        os.unlink(config_path)
    
    # Test with invalid YAML
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("invalid: yaml: content:")
        invalid_config_path = f.name
    
    try:
        agent = SuperAlita(invalid_config_path)
        agent.load_config()
        
        # Should fall back to default config
        assert agent.config["agent"]["name"] == "Super Alita"
    
    finally:
        os.unlink(invalid_config_path)


if __name__ == "__main__":
    pytest.main([__file__])
