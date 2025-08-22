"""
End-to-end cognitive test for Super Alita agent.

Tests the full cognitive loop:
- Load minimal config (memory + conversation enabled)
- Inject a user message via EventBus
- Assert the agent emits an AgentResponseEvent with non-empty text
- Verify plugins are working together correctly
"""

import asyncio
import tempfile
from datetime import datetime
from pathlib import Path
from typing import List

import pytest
import yaml
from src.core.event_bus import EventBus
from src.core.events import AgentResponseEvent, ConversationEvent, SystemEvent
from src.main import SuperAlita


class TestAgentCognitiveLoop:
    """Full agent cognition integration tests."""

    @pytest.fixture
    async def minimal_config(self):
        """Create a minimal test configuration."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = {
                "logging": {"level": "INFO"},
                "redis": {"host": "localhost", "port": 6379},
                "neural_store": {"learning_rate": 0.01},
                "plugins": {
                    "event_bus": {"enabled": True},
                    "semantic_memory": {
                        "enabled": True,
                        "db_path": str(Path(tmp_dir) / "test_chroma"),
                        "collection_name": "test_memory",
                    },
                    "conversation": {"enabled": True, "llm_model": "gemini-2.5-pro"},
                },
            }

            # Write config to temp file
            config_path = Path(tmp_dir) / "test_agent.yaml"
            with open(config_path, "w") as f:
                yaml.dump(config, f)

            yield config_path

    @pytest.mark.asyncio
    async def test_agent_startup_and_shutdown(self, minimal_config):
        """Test that the agent can start up and shut down cleanly."""
        alita = SuperAlita(cfg_path=minimal_config)

        # Start agent in background
        agent_task = asyncio.create_task(alita.run())

        # Give it time to initialize
        await asyncio.sleep(2.0)

        # Agent should be running
        assert not agent_task.done()

        # Shutdown
        await alita.shutdown()

        # Wait for clean shutdown
        try:
            await asyncio.wait_for(agent_task, timeout=5.0)
        except asyncio.TimeoutError:
            agent_task.cancel()
            pytest.fail("Agent failed to shutdown within timeout")

    @pytest.mark.asyncio
    async def test_agent_response_to_conversation(self, minimal_config):
        """Test that agent responds to conversation events."""
        # Start agent
        alita = SuperAlita(cfg_path=minimal_config)
        agent_task = asyncio.create_task(alita.run())

        # Give agent time to start
        await asyncio.sleep(3.0)

        try:
            # Create separate event bus for testing
            test_bus = EventBus(wire_format="json")
            await test_bus.connect()
            await test_bus.start()

            # Collect agent responses
            agent_responses: List[AgentResponseEvent] = []

            async def response_handler(event):
                if hasattr(event, "response"):
                    agent_responses.append(event)

            # Subscribe to agent responses
            await test_bus.subscribe("agent_response", response_handler)
            await test_bus.subscribe("conversation_response", response_handler)

            # Send conversation event
            conversation = ConversationEvent(
                source_plugin="test_client",
                session_id="test_session",
                user_message="Hello, can you hear me?",
                message_id="test_msg_001",
                timestamp=datetime.now().isoformat(),
            )

            await test_bus.publish(conversation)

            # Wait for agent to process and respond
            await asyncio.sleep(5.0)

            # Should receive at least one response
            assert len(agent_responses) > 0, "Agent should respond to conversation"

            # Response should have content
            response = agent_responses[0]
            assert hasattr(response, "response") or hasattr(response, "message")

            if hasattr(response, "response"):
                assert response.response.strip(), "Response should not be empty"
            elif hasattr(response, "message"):
                assert response.message.strip(), "Message should not be empty"

            await test_bus.shutdown()

        finally:
            await alita.shutdown()
            try:
                await asyncio.wait_for(agent_task, timeout=5.0)
            except asyncio.TimeoutError:
                agent_task.cancel()

    @pytest.mark.asyncio
    async def test_multiple_conversations(self, minimal_config):
        """Test agent handling multiple conversation turns."""
        alita = SuperAlita(cfg_path=minimal_config)
        agent_task = asyncio.create_task(alita.run())

        await asyncio.sleep(3.0)

        try:
            test_bus = EventBus(wire_format="json")
            await test_bus.connect()
            await test_bus.start()

            responses = []

            async def handler(event):
                responses.append(event)

            await test_bus.subscribe("agent_response", handler)
            await test_bus.subscribe("conversation_response", handler)

            # Send multiple messages
            messages = [
                "What is your name?",
                "How are you today?",
                "Tell me about yourself.",
            ]

            for i, msg in enumerate(messages):
                conv = ConversationEvent(
                    source_plugin="test_client",
                    session_id=f"session_{i}",
                    user_message=msg,
                    message_id=f"msg_{i}",
                    timestamp=datetime.now().isoformat(),
                )
                await test_bus.publish(conv)
                await asyncio.sleep(2.0)  # Space out messages

            # Wait for all responses
            await asyncio.sleep(5.0)

            # Should get multiple responses
            assert len(responses) >= len(
                messages
            ), f"Expected at least {len(messages)} responses, got {len(responses)}"

            await test_bus.shutdown()

        finally:
            await alita.shutdown()
            try:
                await asyncio.wait_for(agent_task, timeout=5.0)
            except asyncio.TimeoutError:
                agent_task.cancel()

    @pytest.mark.asyncio
    async def test_system_events_processing(self, minimal_config):
        """Test that agent processes system events correctly."""
        alita = SuperAlita(cfg_path=minimal_config)
        agent_task = asyncio.create_task(alita.run())

        await asyncio.sleep(3.0)

        try:
            test_bus = EventBus(wire_format="json")
            await test_bus.connect()
            await test_bus.start()

            system_responses = []

            async def system_handler(event):
                system_responses.append(event)

            await test_bus.subscribe("system", system_handler)

            # Send system event
            system_event = SystemEvent(
                source_plugin="test_monitor",
                level="info",
                message="Test system status check",
                component="integration_test",
            )

            await test_bus.publish(system_event)
            await asyncio.sleep(2.0)

            # System should process the event (may or may not respond)
            # The test passes if no errors occur

            await test_bus.shutdown()

        finally:
            await alita.shutdown()
            try:
                await asyncio.wait_for(agent_task, timeout=5.0)
            except asyncio.TimeoutError:
                agent_task.cancel()


class TestPluginIntegration:
    """Test integration between different plugins."""

    @pytest.mark.asyncio
    async def test_memory_conversation_integration(self, minimal_config):
        """Test that conversation plugin works with semantic memory."""
        # This test would verify that the conversation plugin
        # can store and retrieve memories via the semantic memory plugin
        # For now, it's a placeholder for future implementation

    @pytest.mark.asyncio
    async def test_fsm_conversation_integration(self, minimal_config):
        """Test FSM state changes during conversation."""
        # This test would verify that the FSM plugin changes states
        # appropriately during conversation flow


# Test configuration
@pytest.fixture(scope="session")
def event_loop():
    """Create an event loop for the test session."""
    policy = asyncio.get_event_loop_policy()
    loop = policy.new_event_loop()
    yield loop
    loop.close()


if __name__ == "__main__":
    # Run with: python -m pytest tests/integration/test_agent_loop.py -v -s
    import sys

    sys.exit(pytest.main([__file__, "-v", "-s"]))
