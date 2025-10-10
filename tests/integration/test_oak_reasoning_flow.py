"""
End-to-end integration test for the OaK Reasoning Flow.

Tests the full flow from a high-level goal to a tool call through the OaK Tactical Layer.
"""

import asyncio
import importlib
import tempfile
from pathlib import Path
from typing import Any

import pytest
import yaml

from src.core.event_bus import EventBus
from src.core.events import GoalReceivedEvent, ToolCallEvent
from src.core.plugin_interface import PluginInterface
from src.main_unified import UnifiedSuperAlita

pytestmark = pytest.mark.integration_redis

if importlib.util.find_spec("redis") is None:
    pytest.skip("redis not installed", allow_module_level=True)


def _redis_running(host: str = "localhost", port: int = 6379) -> bool:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        try:
            sock.connect((host, port))
            return True
        except OSError:
            return False


if not _redis_running():
    pytest.skip("Redis server not available", allow_module_level=True)


# A mock tool for testing purposes
class EchoTool(PluginInterface):
    @property
    def name(self) -> str:
        return "echo_tool"

    async def setup(
        self, event_bus: Any, store: Any, config: dict[str, Any]
    ) -> None:
        await super().setup(event_bus, store, config)

    async def start(self) -> None:
        await super().start()
        await self.subscribe("tool_call", self._on_tool_call)

    async def _on_tool_call(self, event: ToolCallEvent) -> None:
        if event.tool_name != self.name:
            return
        # Echo the parameters back in an event for the test to catch
        await self.emit_event("echo_tool_called", **event.parameters)


class TestOakReasoningFlow:
    @pytest.fixture
    async def oak_config(self):
        """Create a minimal test configuration with OaK plugins enabled."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = {
                "plugins": {
                    "planner_v2": {"enabled": True},
                    "oak_coordinator": {"enabled": True},
                    "option_executor": {
                        "enabled": True,
                        "option_mapping": {
                            "test-option": {
                                "tool_name": "echo_tool",
                                "parameters": {"message": "{goal}"},
                            }
                        },
                    },
                    "echo_tool": {"enabled": True},  # Our mock tool
                    # Disable other plugins to isolate the test
                    "memory_manager": {"enabled": False},
                    "tool_executor": {"enabled": False},
                    "creator_plugin": {"enabled": False},
                    "llm_planner": {"enabled": False},
                    "puter": {"enabled": False},
                    "conversation": {"enabled": False},
                    "web_agent": {"enabled": False},
                    "perplexica_search": {"enabled": False},
                },
            }

            config_path = Path(tmp_dir) / "test_agent.yaml"
            with open(config_path, "w") as f:
                yaml.dump(config, f)

            yield config_path

    @pytest.mark.asyncio
    async def test_oak_end_to_end_flow(self, oak_config):
        """Test the full OaK reasoning flow from goal to tool call."""

        # Add our mock tool to the list of available plugins
        from src.main_unified import AVAILABLE_PLUGINS, PLUGIN_ORDER

        AVAILABLE_PLUGINS["echo_tool"] = EchoTool
        if "echo_tool" not in PLUGIN_ORDER:
            PLUGIN_ORDER.append("echo_tool")

        alita = UnifiedSuperAlita(cfg_path=oak_config)
        agent_task = asyncio.create_task(alita.run())
        await asyncio.sleep(3.0)  # Give agent time to start

        try:
            test_bus = EventBus()
            await test_bus.connect()
            await test_bus.start()

            collected_events = {
                "subgoal_defined": [],
                "tool_call": [],
                "echo_tool_called": [],
            }

            async def handler(event):
                event_type = (
                    event.event_type
                    if hasattr(event, "event_type")
                    else event.get("event_type")
                )
                if event_type in collected_events:
                    collected_events[event_type].append(event)

            await test_bus.subscribe("subgoal_defined", handler)
            await test_bus.subscribe("tool_call", handler)
            await test_bus.subscribe("echo_tool_called", handler)

            # This is a hack for the test. The OaK PlanningEngine needs at least one option
            # to be available. We'll manually add a dummy option to the OptionTrainer.
            # In a real scenario, options would be learned.
            option_trainer = alita.plugins["oak_coordinator"].option_trainer
            option_trainer.options["test-option"] = {"id": "test-option"}

            goal = "echo this message"
            await test_bus.publish(
                GoalReceivedEvent(
                    source_plugin="test_client",
                    goal=goal,
                    session_id="test_session_oak",
                )
            )

            await asyncio.sleep(5.0)  # Wait for processing

            # 1. Assert that a subgoal was defined
            assert (
                len(collected_events["subgoal_defined"]) > 0
            ), "A subgoal should have been defined"
            subgoal_event = collected_events["subgoal_defined"][0]
            assert subgoal_event.subgoal.description == goal

            # 2. Assert that the correct tool was called
            assert (
                len(collected_events["tool_call"]) > 0
            ), "A tool call should have been made"
            tool_call_event = collected_events["tool_call"][0]
            assert tool_call_event.tool_name == "echo_tool"
            assert tool_call_event.parameters["message"] == goal

            # 3. Assert that our mock tool was executed
            assert (
                len(collected_events["echo_tool_called"]) > 0
            ), "The mock tool should have been called"
            echo_event = collected_events["echo_tool_called"][0]
            assert echo_event.message == goal

        finally:
            await alita.shutdown()
            try:
                await asyncio.wait_for(agent_task, timeout=5.0)
            except TimeoutError:
                agent_task.cancel()
                pytest.fail("Agent failed to shutdown within timeout")
