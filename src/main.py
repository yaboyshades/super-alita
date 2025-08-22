"""
Super Alita orchestrator.
Loads config, wires dependencies, starts / stops the agent.
"""

import asyncio
import contextlib
import json
import logging
import os
import signal
import sys
from pathlib import Path
from typing import Any

# Add the parent directory to Python path so we can import from 'src'
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import yaml
except ImportError:
    print("[ERROR] PyYAML not installed. Run: pip install PyYAML")
    sys.exit(1)

try:
    from dotenv import load_dotenv
except ImportError:
    print("[ERROR] python-dotenv not installed. Run: pip install python-dotenv")
    sys.exit(1)

from src.core.event_bus import EventBus

# REUG v9.0 imports
from src.core.execution_flow import REUGExecutionFlow
from src.core.neural_atom import NeuralStore
from src.core.plugin_interface import PluginInterface

# CRITICAL: Load environment variables from .env file
print("[CONFIG] Loading environment variables from .env file...")
load_dotenv()

# Validate that critical environment variables are loaded
api_key = os.getenv("GEMINI_API_KEY")
if api_key:
    # Avoid printing any portion of the API key for security
    print("[OK] GEMINI_API_KEY loaded successfully.")
else:
    print("[WARNING] GEMINI_API_KEY not found in environment!")
    print("   This will cause LLM plugins to fail. Check your .env file.")

# Telemetry integration - safe imports
telemetry_available = False
EventTypes: Any | None = None
broadcast_agent_event: Any | None = None
get_broadcaster: Any | None = None

try:
    from src.telemetry import EventTypes, broadcast_agent_event, get_broadcaster

    telemetry_available = True
    print("[OK] Telemetry broadcaster available")
except ImportError as e:
    print(f"[WARNING] Telemetry not available: {e}")

# --- Plugin registry: order = boot order -----------------------------
PLUGIN_ORDER: list[type[PluginInterface]] = []


def _load_plugins() -> None:
    """Load all available plugins into the global PLUGIN_ORDER list."""
    plugin_specs = [
        (
            "src.plugins.pythonic_preprocessor_plugin",
            "PythonicPreprocessorPlugin",
        ),  # NEW: DTA 2.0 Preprocessor
        ("src.plugins.event_bus_plugin", "EventBusPlugin"),
        # ("src.plugins.semantic_memory_plugin", "SemanticMemoryPlugin"),  # Disabled due to Google AI import issues
        # ("src.plugins.semantic_fsm_plugin", "SemanticFSMPlugin"),  # Disabled due to Google AI import issues
        ("src.plugins.ladder_aog_plugin", "LADDERAOGPlugin"),
        ("src.plugins.skill_discovery_plugin", "SkillDiscoveryPlugin"),
        ("src.plugins.self_heal_plugin", "SelfHealPlugin"),
        ("src.plugins.system_introspection_plugin", "SystemIntrospectionPlugin"),
        ("src.plugins.tool_executor_plugin", "ToolExecutorPlugin"),
        ("src.plugins.atom_tools_plugin", "AtomToolsPlugin"),
        ("src.plugins.brainstorm_plugin", "BrainstormPlugin"),
        ("src.plugins.compose_plugin", "ComposePlugin"),
        ("src.plugins.auto_tools_plugin", "AutoToolsPlugin"),
        ("src.plugins.atom_creator_plugin", "AtomCreatorPlugin"),
        ("src.plugins.atom_executor_plugin", "AtomExecutorPlugin"),
        ("src.plugins.tool_lifecycle_plugin", "ToolLifecyclePlugin"),
        (
            "src.atoms.memory_manager_atom",
            "MemoryManagerAtom",
        ),  # Added memory manager atom
        (
            "src.plugins.memory_manager_plugin",
            "MemoryManagerPlugin",
        ),  # NEW: Memory management plugin
        ("src.plugins.calculator_plugin", "CalculatorPlugin"),  # NEW: Calculator plugin
        (
            "src.plugins.core_utils_plugin_dynamic",
            "CoreUtilsPlugin",
        ),  # NEW: Dynamic core utilities plugin
        (
            "src.plugins.creator_plugin",
            "CreatorPlugin",
        ),  # NEW: Auto tool creation plugin
        (
            "src.plugins.llm_planner_plugin",
            "LLMPlannerPlugin",
        ),  # NEW: LLM-based planning plugin (SINGLE SOURCE OF TRUTH)
        (
            "src.plugins.self_reflection_plugin",
            "SelfReflectionPlugin",
        ),  # NEW: Self-reflection and introspection plugin
        # ENHANCEMENT PLUGINS - Cognitive Capabilities
        (
            "src.plugins.adaptive_neural_atom_plugin",
            "AdaptiveNeuralAtomPlugin",
        ),  # NEW: Adaptive learning neural atoms
        (
            "src.plugins.predictive_world_model_plugin",
            "PredictiveWorldModelPlugin",
        ),  # NEW: Predictive world modeling and strategic planning
        (
            "src.plugins.meta_learning_creator_plugin",
            "MetaLearningCreatorPlugin",
        ),  # NEW: Meta-learning and autonomous tool creation
        ("src.atoms.web_agent_atom", "WebAgentAtom"),
        ("src.plugins.conversation_plugin", "ConversationPlugin"),
    ]

    for module_path, class_name in plugin_specs:
        try:
            module = __import__(module_path, fromlist=[class_name])
            plugin_class = getattr(module, class_name)
            PLUGIN_ORDER.append(plugin_class)
        except (ImportError, AttributeError) as e:
            # Only warn if logger exists
            if "logger" in globals():
                logger.warning(f"Could not load plugin {module_path}.{class_name}: {e}")


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
for noisy in ("httpx", "chromadb"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

logger = logging.getLogger("super_alita")

# Load plugins after logger is initialized
_load_plugins()


class SuperAlita:
    def __init__(self, cfg_path: str | Path | None = None) -> None:
        """Initialize Super Alita agent with configuration."""
        if cfg_path is None:
            # Determine config path relative to current script location
            script_dir = Path(__file__).parent
            cfg_path = script_dir / "config" / "agent.yaml"
            # If that doesn't exist, try relative to working directory
            if not cfg_path.exists():
                cfg_path = Path("src/config/agent.yaml")
            # If that doesn't exist either, try from project root
            if not cfg_path.exists():
                cfg_path = Path("config/agent.yaml")
        else:
            cfg_path = Path(cfg_path)

        if not cfg_path.exists():
            raise FileNotFoundError(f"Config file not found: {cfg_path}")

        self.cfg = yaml.safe_load(cfg_path.read_text())
        redis_config = self.cfg.get("redis", {})
        self.bus = EventBus(
            host=redis_config.get("host", "localhost"),
            port=redis_config.get("port", 6379),
            wire_format=redis_config.get("wire_format", "json"),
        )
        self.store = NeuralStore(**self.cfg.get("neural_store", {}))
        self.plugins: dict[str, PluginInterface] = {}
        self._stop = asyncio.Event()

        # REUG v9.0 Execution Flow
        self.execution_flow: REUGExecutionFlow | None = None

    async def run(self) -> None:
        """Run the Super Alita agent with all configured plugins."""
        enabled = {
            k: v for k, v in self.cfg.get("plugins", {}).items() if v.get("enabled")
        }
        logger.info("Bootstrapping Super Alita ...")

        # Initialize telemetry broadcaster
        if (
            telemetry_available
            and get_broadcaster
            and broadcast_agent_event
            and EventTypes
        ):
            try:
                broadcaster = get_broadcaster()
                await broadcaster.start()
                await broadcast_agent_event(
                    event_type=EventTypes.PLUGIN_STARTUP,
                    source="super_alita_orchestrator",
                    data={
                        "status": "starting",
                        "enabled_plugins": list(enabled.keys()),
                    },
                )
                logger.info("[OK] Telemetry broadcaster started")
            except Exception as e:
                logger.warning(f"[WARNING] Telemetry broadcaster failed to start: {e}")

        try:
            # Start event bus first
            await self.bus.start()

            for cls in PLUGIN_ORDER:
                # Handle special name mappings
                if cls.__name__ == "WebAgentAtom":
                    name = "web_agent"
                elif cls.__name__ == "MemoryManagerPlugin":
                    name = "memory_manager"
                elif cls.__name__ == "CalculatorPlugin":
                    name = "calculator"
                elif cls.__name__ == "CoreUtilsPlugin":
                    name = "core_utils"
                elif cls.__name__ == "CreatorPlugin":
                    name = "creator"
                elif cls.__name__ == "LLMPlannerPlugin":
                    name = "llm_planner"
                elif cls.__name__ == "SelfReflectionPlugin":
                    name = "self_reflection"
                elif cls.__name__ == "PythonicPreprocessorPlugin":
                    name = "pythonic_preprocessor"  # NEW: DTA 2.0 Preprocessor mapping
                elif cls.__name__ == "PredictiveWorldModelPlugin":
                    name = (
                        "predictive_world_model"  # NEW: Predictive World Model mapping
                    )
                else:
                    name = cls.__name__.lower().replace("plugin", "")

                if name not in enabled:
                    logger.debug("Skipping disabled plugin %s", name)
                    continue

                inst = cls()

                # Special configuration for SelfReflectionPlugin
                plugin_config = enabled[name].copy()
                if name == "self_reflection":
                    plugin_config["orchestrator_instance"] = self

                try:
                    await inst.setup(self.bus, self.store, {name: plugin_config})
                    self.plugins[name] = inst
                except Exception:
                    logger.exception(f"Failed to setup plugin {name}")
                    continue

            # Start plugins with enhanced error handling
            for name, plugin in self.plugins.items():
                try:
                    await plugin.start()
                    logger.info("Plugin %s started", name)
                except Exception:
                    logger.exception(f"Failed to start plugin {name}")
                    # Continue with other plugins instead of crashing

            # Initialize REUG v9.0 Execution Flow after plugins are loaded
            self.execution_flow = REUGExecutionFlow(self.bus, self.plugins)
            await self.execution_flow.start_session("main_session")
            logger.info("🧠 REUG v9.0 Execution Flow initialized")

            logger.info("Super Alita is LIVE")
            await self._stop.wait()

        except KeyboardInterrupt:
            logger.info("Received shutdown signal")
        except Exception:
            logger.exception("Unexpected error during startup")
            raise
        finally:
            await self.shutdown()

    async def shutdown(self) -> None:
        """Gracefully shutdown all plugins and services."""
        if self._stop.is_set():
            return
        logger.info("Shutting down ...")
        self._stop.set()

        # Shutdown REUG execution flow first
        if self.execution_flow:
            try:
                await self.execution_flow.shutdown()
                logger.debug("REUG execution flow stopped")
            except Exception:
                logger.exception("Error shutting down REUG execution flow")

        # Shutdown plugins in reverse order with error handling
        for name, plugin in reversed(list(self.plugins.items())):
            try:
                await plugin.shutdown()
                logger.debug("Plugin %s stopped", name)
            except Exception:
                logger.exception(f"Error shutting down plugin {name}")

        # Shutdown event bus
        try:
            await self.bus.shutdown()
        except Exception:
            logger.exception("Error shutting down event bus")

        logger.info("Shutdown complete")

    async def process_user_message(self, user_input: str) -> dict[str, Any]:
        """
        Process user input through REUG v9.0 execution flow

        Args:
            user_input: User's message

        Returns:
            dict: Response and execution metadata
        """
        if not self.execution_flow:
            return {
                "response": "System not ready. Please wait for initialization to complete.",
                "success": False,
                "error": "execution_flow_not_initialized",
            }

        try:
            result = await self.execution_flow.process_user_input(user_input)
            return result
        except Exception as e:
            logger.error(f"Error processing user message: {e}")
            return {
                "response": "I encountered an error processing your message. Please try again.",
                "success": False,
                "error": str(e),
            }

    def request_shutdown(self) -> None:
        """Request a graceful shutdown (thread-safe)."""
        self._stop.set()


class ChatInterface:
    """Integrated chat interface for Super Alita."""

    def __init__(
        self,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        agent: SuperAlita | None = None,
    ):
        self.redis_host = redis_host
        self.redis_port = redis_port
        self.agent = agent  # Reference to SuperAlita instance for REUG processing
        self.redis = None
        self._stop = asyncio.Event()
        self._listener_task: asyncio.Task | None = None

    async def start(self) -> None:
        """Start the chat interface."""
        try:
            import redis.asyncio as redis

            self.redis = redis.Redis(
                host=self.redis_host, port=self.redis_port, decode_responses=True
            )

            # Test Redis connection
            await self.redis.ping()
            logger.info("✅ Chat interface connected to Redis")
        except ImportError:
            logger.exception(
                "[ERROR] redis package not installed. Run: pip install redis"
            )
            return
        except Exception:
            logger.exception("[ERROR] Chat interface failed to connect to Redis")
            return

        # Start listener task
        self._listener_task = asyncio.create_task(self._listen_agent())

        # Start chat input task
        chat_task = asyncio.create_task(self._chat_loop())

        logger.info("💬 Super Alita Chat Interface Ready (type 'quit' to exit)")

        # Wait for either chat to end or stop signal
        done, pending = await asyncio.wait(
            [chat_task, asyncio.create_task(self._stop.wait())],
            return_when=asyncio.FIRST_COMPLETED,
        )

        # Cancel remaining tasks
        for task in pending:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    async def _chat_loop(self) -> None:
        """Main chat input loop."""
        print("\n" + "=" * 50)
        print("💬 SUPER ALITA CHAT INTERFACE")
        print("=" * 50)
        print("Type your messages and press Enter.")
        print("Type 'quit', 'exit', or 'bye' to stop the chat.")
        print("Use Ctrl+C to stop the entire system.")
        print("=" * 50)

        while not self._stop.is_set():
            try:
                # Use asyncio.to_thread for non-blocking input
                line = await asyncio.to_thread(input, "\n🧑 You: ")

                if line.strip().lower() in ["quit", "exit", "bye"]:
                    print("👋 Ending chat session...")
                    break

                if line.strip():  # Only send non-empty messages
                    await self._send_message(line.strip())

            except EOFError:
                break
            except KeyboardInterrupt:
                print("\n🔄 Use 'quit' to end chat or Ctrl+C again to stop the system")
                break

    async def _send_message(self, text: str) -> None:
        """Send message to agent via REUG execution flow or Redis fallback."""
        if not self.redis:
            return

        try:
            # NEW: Use REUG v9.0 execution flow if available
            if self.agent and self.agent.execution_flow:
                print(f"📤 Processing: {text}")
                result = await self.agent.process_user_message(text)

                if result.get("success", False):
                    response = result.get("response", "No response generated")
                    print(f"\n🤖 Alita: {response}")

                    # Also publish to Redis for other listeners
                    await self.redis.publish(
                        "agent_reply",
                        json.dumps(
                            {
                                "text": response,
                                "source": "reug_execution_flow",
                                "metadata": result.get("metadata", {}),
                            }
                        ),
                    )
                else:
                    error_response = result.get(
                        "response", "I encountered an error processing your request."
                    )
                    print(f"\n🤖 Alita: {error_response}")
                    print(f"   ⚠️  Error: {result.get('error', 'Unknown error')}")

                return

            # FALLBACK: Legacy Redis event publishing
            # Create comprehensive event data
            event_data = {
                "event_type": "conversation_message",
                "source_plugin": "chat_interface",
                "session_id": "integrated_chat",
                "user_message": text,
                "message_id": f"msg_{asyncio.get_event_loop().time()}",
                "conversation_id": "integrated_chat",
                "timestamp": asyncio.get_event_loop().time(),
            }

            # Create simple event data for compatibility
            simple_event = {
                "text": text,
                "type": "user_message",
                "source": "chat_interface",
            }

            # Publish to multiple channels for maximum compatibility
            json_data = json.dumps(event_data)
            simple_json = json.dumps(simple_event)

            await self.redis.publish("conversation_message", json_data)
            await self.redis.publish("user_message", simple_json)

            print(f"📤 Message sent: {text}")

        except Exception:
            logger.exception("[ERROR] Error sending message")

    async def _listen_agent(self) -> None:
        """Listen for agent responses."""
        if not self.redis:
            return

        try:
            pubsub = self.redis.pubsub()
            await pubsub.subscribe("agent_reply")
            logger.info("🎧 Chat interface listening for agent responses")

            async for msg in pubsub.listen():
                if self._stop.is_set():
                    break

                if msg["type"] == "message":
                    try:
                        data = json.loads(str(msg["data"]))
                        response_text = data.get("text", str(data))
                        print(f"\n🤖 Alita: {response_text}")
                    except json.JSONDecodeError:
                        # Handle plain text responses
                        print(f"\n🤖 Alita: {msg['data']}")
                    except Exception as e:
                        logger.error(f"Error processing agent response: {e}")

        except Exception:
            logger.exception("[ERROR] Error in agent response listener")

    async def shutdown(self) -> None:
        """Shutdown the chat interface."""
        self._stop.set()

        if self._listener_task and not self._listener_task.done():
            self._listener_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._listener_task

        if self.redis:
            await self.redis.aclose()


class SuperAlitaWithChat:
    """Super Alita agent with integrated chat interface."""

    def __init__(
        self, cfg_path: Path = Path("src/config/agent.yaml"), enable_chat: bool = True
    ):
        self.alita = SuperAlita(cfg_path)
        self.enable_chat = enable_chat
        self.chat = None
        if enable_chat:
            redis_config = self.alita.cfg.get("redis", {})
            self.chat = ChatInterface(
                redis_host=redis_config.get("host", "localhost"),
                redis_port=redis_config.get("port", 6379),
                agent=self.alita,  # Pass agent reference for REUG processing
            )

    async def run(self) -> None:
        """Run both the agent and chat interface."""
        tasks = []

        # Start the agent
        agent_task = asyncio.create_task(self.alita.run())
        tasks.append(("agent", agent_task))

        if self.enable_chat and self.chat:
            # Give agent a moment to start up
            await asyncio.sleep(2)

            # Start the chat interface
            chat_task = asyncio.create_task(self.chat.start())
            tasks.append(("chat", chat_task))

        try:
            # Wait for any task to complete
            done, pending = await asyncio.wait(
                [task for _, task in tasks], return_when=asyncio.FIRST_COMPLETED
            )

            # If any task completes, shutdown everything
            logger.info("One component finished, initiating shutdown...")

        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt, shutting down...")
        except Exception:
            logger.exception("Unexpected error")
        finally:
            await self.shutdown()

            # Cancel and wait for remaining tasks
            for _name, task in tasks:
                if not task.done():
                    task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await task

    async def shutdown(self) -> None:
        """Shutdown both components."""
        if self.chat:
            await self.chat.shutdown()
        await self.alita.shutdown()


def main() -> None:
    """Main entry point for the Super Alita agent with integrated chat."""
    # Check if chat should be disabled via command line argument
    enable_chat = "--no-chat" not in sys.argv

    alita_with_chat = SuperAlitaWithChat(enable_chat=enable_chat)

    def signal_handler(signum: int, _frame: Any) -> None:
        """Handle shutdown signals gracefully."""
        logger.info(f"Received signal {signum}, initiating shutdown...")
        # Use the public shutdown method
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.call_soon_threadsafe(alita_with_chat.alita.request_shutdown)
            else:
                alita_with_chat.alita.request_shutdown()
        except RuntimeError:
            # No event loop running, call directly
            alita_with_chat.alita.request_shutdown()

    # Register signal handlers for graceful shutdown
    try:
        signal.signal(signal.SIGINT, signal_handler)
        if hasattr(signal, "SIGTERM"):
            signal.signal(signal.SIGTERM, signal_handler)
    except (AttributeError, OSError):
        # Fallback for systems that don't support certain signals
        signal.signal(signal.SIGINT, signal_handler)

    try:
        if enable_chat:
            print("[LAUNCH] Starting Super Alita with integrated chat interface...")
            print("         Use --no-chat flag to disable chat interface")
        else:
            print("[LAUNCH] Starting Super Alita in agent-only mode...")

        # Use asyncio.run() which is the modern way to run async code
        asyncio.run(alita_with_chat.run())
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, shutting down...")
    except Exception:
        logger.exception("Agent failed with error")
        raise


if __name__ == "__main__":
    main()
