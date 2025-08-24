# Version: 3.0.0
# Description: The main entry point and orchestrator for the Super Alita agent.

import asyncio
import logging
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Import core services for unified architecture
from src.core.global_workspace import AttentionLevel, GlobalWorkspace  # noqa: E402
from src.core.neural_atom import NeuralStore  # noqa: E402
from src.core.plugin_interface import PluginInterface  # noqa: E402
from src.core.schemas import SystemState  # noqa: E402

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
for noisy in ("httpx", "chromadb"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

logger = logging.getLogger("super_alita_unified")

# This list defines the order of plugin initialization and startup.
PLUGIN_ORDER = [
    "memory_manager",  # Initialize memory first for other plugins
    "tool_executor",  # Tool execution capability
    "creator_plugin",  # Tool creation capability
    "llm_planner",  # LLM-based planning and routing
    "puter",  # Cloud environment integration
    "conversation",  # User interaction (legacy)
    "web_agent",  # Web search capability (legacy)
]

# A map of plugin names to their class definitions.
AVAILABLE_PLUGINS = {
    # Will be populated dynamically
}


def _load_unified_plugins():
    """Load plugins for the unified architecture."""
    global AVAILABLE_PLUGINS

    plugin_specs = [
        # Unified cognitive plugins
        ("src.plugins.llm_planner_plugin_unified", "LLMPlannerPlugin", "llm_planner"),
        ("src.plugins.creator_plugin_unified", "CreatorPlugin", "creator_plugin"),
        (
            "src.plugins.memory_manager_plugin_unified",
            "MemoryManagerPlugin",
            "memory_manager",
        ),
        (
            "src.plugins.tool_executor_plugin_unified",
            "ToolExecutorPlugin",
            "tool_executor",
        ),
        # Cloud integration plugins
        ("src.plugins.puter_plugin", "PuterPlugin", "puter"),
        # Legacy plugins (fallback compatibility)
        ("src.plugins.conversation_plugin", "ConversationPlugin", "conversation"),
        ("src.atoms.web_agent_atom", "WebAgentAtom", "web_agent"),
    ]

    for module_path, class_name, plugin_name in plugin_specs:
        try:
            module = __import__(module_path, fromlist=[class_name])
            plugin_class = getattr(module, class_name)
            AVAILABLE_PLUGINS[plugin_name] = plugin_class
            logger.debug(f"Loaded plugin: {plugin_name} -> {class_name}")
        except (ImportError, AttributeError) as e:
            logger.warning(f"Could not load plugin {module_path}.{class_name}: {e}")


class UnifiedSuperAlita:
    """
    Unified Super Alita orchestrator implementing Version 3.0 architecture.

    Coordinates the Global Workspace, Neural Store, and cognitive plugins
    to create a consciousness-inspired AI agent.
    """

    def __init__(self, config_path: Path = Path("src/config/agent.yaml")):
        """
        Initialize the unified Super Alita agent.

        Args:
            config_path: Path to agent configuration file
        """
        self.config_path = config_path
        self.config = self._load_configuration()

        # Initialize core services
        self.workspace = self._initialize_global_workspace()
        self.store = self._initialize_neural_store()

        # Plugin management
        self.plugins: dict[str, PluginInterface] = {}
        self._shutdown_event = asyncio.Event()

        # System state tracking
        self.system_state = SystemState(
            state_id="system_initial", cognitive_load=0.0, neural_atoms_active=0
        )

        logger.info("Unified Super Alita initialized with Version 3.0 architecture")

    def _load_configuration(self) -> dict[str, Any]:
        """Load and validate configuration from agent.yaml."""
        try:
            if self.config_path.exists():
                with open(self.config_path) as f:
                    config = yaml.safe_load(f)
                logger.info(f"Configuration loaded from {self.config_path}")
                return config
            logger.warning(f"Config file {self.config_path} not found, using defaults")
            return self._get_default_config()
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            return self._get_default_config()

    def _get_default_config(self) -> dict[str, Any]:
        """Get default configuration for the unified architecture."""
        return {
            "neural_atoms": {
                "registry_type": "semantic_vector",
                "max_concurrent_execution": 10,
                "performance_tracking": True,
                "auto_optimization": True,
                "similarity_threshold": 0.7,
            },
            "global_workspace": {
                "max_events": 10000,
                "attention_decay_rate": 0.95,
                "broadcast_timeout": 5.0,
                "consciousness_layer_enabled": True,
            },
            "cognitive_cycle": {
                "world_model_enabled": True,
                "meta_learning_enabled": True,
                "safety_checks_enabled": True,
                "reflection_frequency": "per_cycle",
                "prediction_lookahead": 3,
            },
            "creator_engine": {
                "enabled": True,
                "max_rectification_attempts": 3,
                "sandbox_timeout": 30.0,
                "validation_strictness": "high",
            },
            "safety": {
                "recursive_improvement": True,
                "sandbox_execution": True,
                "improvement_validation": "strict",
                "policy_engine": "opa",
            },
            "memory": {
                "working_memory_capacity": 128,
                "vector_store_type": "qdrant",
                "embedding_model": "text-embedding-3-small",
                "long_term_storage": "redis",
                "episodic_memory_enabled": True,
            },
            "world_model": {
                "prediction_enabled": True,
                "model_type": "transformer",
                "context_window": 512,
                "update_frequency": "continuous",
                "confidence_threshold": 0.6,
            },
            "plugins": {
                "memory_manager": {"enabled": True},
                "tool_executor": {"enabled": True},
                "creator_plugin": {"enabled": True},
                "llm_planner": {"enabled": True},
                "puter": {
                    "enabled": True,
                    "puter_api_url": "https://api.puter.com",
                    "puter_api_key": "",
                    "puter_workspace_id": "default",
                },
                "conversation": {"enabled": True},
                "web_agent": {"enabled": True},
            },
        }

    def _initialize_global_workspace(self) -> GlobalWorkspace:
        """Initialize the Global Workspace (consciousness layer)."""
        workspace_config = self.config.get("global_workspace", {})
        max_events = workspace_config.get("max_events", 10000)

        workspace = GlobalWorkspace(max_events=max_events)
        logger.info("Global Workspace initialized as consciousness layer")
        return workspace

    def _initialize_neural_store(self) -> NeuralStore:
        """Initialize the Neural Store (memory and reasoning substrate)."""
        memory_config = self.config.get("memory", {})
        learning_rate = memory_config.get("learning_rate", 0.01)

        store = NeuralStore(learning_rate=learning_rate)
        logger.info("Neural Store initialized as cognitive substrate")
        return store

    async def run(self):
        """Run the unified Super Alita agent."""
        try:
            logger.info(
                "🚀 Bootstrapping Super Alita (Unified Next-Generation Architecture)..."
            )

            # Load unified plugins
            _load_unified_plugins()

            # Get enabled plugins from configuration
            enabled_plugins = {
                k: v
                for k, v in self.config.get("plugins", {}).items()
                if v.get("enabled", False)
            }

            # Initialize plugins in order
            for plugin_name in PLUGIN_ORDER:
                if plugin_name in enabled_plugins and plugin_name in AVAILABLE_PLUGINS:
                    await self._initialize_plugin(
                        plugin_name, enabled_plugins[plugin_name]
                    )
                else:
                    logger.debug(
                        f"Skipping plugin '{plugin_name}' (disabled or not available)"
                    )

            # Start all plugins
            for plugin_name, plugin in self.plugins.items():
                await plugin.start()
                logger.info(f"✅ Plugin '{plugin_name}' started")

            # Update system state
            await self._update_system_state()

            # Broadcast system ready event
            await self.workspace.update(
                data={
                    "status": "system_ready",
                    "plugins_active": list(self.plugins.keys()),
                },
                source="system",
                attention_level=AttentionLevel.HIGH,
            )

            logger.info(
                "🧠 Super Alita is LIVE with unified consciousness architecture"
            )

            # Run until shutdown
            await self._shutdown_event.wait()

        except KeyboardInterrupt:
            logger.info("Received shutdown signal")
        except Exception as e:
            logger.error(f"System error: {e}", exc_info=True)
            raise
        finally:
            await self.shutdown()

    async def _initialize_plugin(self, plugin_name: str, plugin_config: dict[str, Any]):
        """Initialize a single plugin with the unified architecture."""
        try:
            plugin_class = AVAILABLE_PLUGINS[plugin_name]
            instance = plugin_class()

            # Merge environment variables for plugin-specific configuration
            final_config = plugin_config.copy()
            
            # Special handling for Puter plugin - load from environment variables
            if plugin_name == "puter":
                import os
                env_config = {
                    "puter_api_url": os.getenv("PUTER_API_URL", plugin_config.get("puter_api_url", "https://api.puter.com")),
                    "puter_api_key": os.getenv("PUTER_API_KEY", plugin_config.get("puter_api_key", "")),
                    "puter_workspace_id": os.getenv("PUTER_WORKSPACE_ID", plugin_config.get("puter_workspace_id", "default")),
                }
                final_config.update(env_config)
                logger.info(f"Puter plugin configured with API URL: {env_config['puter_api_url']}")

            # Setup plugin with unified dependencies
            await instance.setup(self.workspace, self.store, final_config)

            self.plugins[plugin_name] = instance
            logger.info(f"✅ Plugin '{plugin_name}' initialized")

        except Exception as e:
            logger.error(f"Failed to initialize plugin '{plugin_name}': {e}")
            # Continue with other plugins rather than failing entirely

    async def _update_system_state(self):
        """Update the current system state."""
        self.system_state = SystemState(
            state_id=f"system_{len(self.plugins)}_plugins",
            cognitive_load=len(self.plugins) / 10.0,  # Simple heuristic
            active_tasks=[],
            memory_usage={"neural_atoms": len(getattr(self.store, "_atoms", {}))},
            neural_atoms_active=len(getattr(self.store, "_atoms", {})),
            attention_focus=[],
            performance_metrics={},
        )

    async def shutdown(self):
        """Gracefully shutdown the unified agent."""
        if self._shutdown_event.is_set():
            return

        logger.info("🔄 Shutting down Super Alita...")
        self._shutdown_event.set()

        # Shutdown plugins in reverse order
        for plugin_name in reversed(list(self.plugins.keys())):
            try:
                plugin = self.plugins[plugin_name]
                await plugin.shutdown()
                logger.info(f"✅ Plugin '{plugin_name}' stopped")
            except Exception as e:
                logger.error(f"Error shutting down plugin '{plugin_name}': {e}")

        # Broadcast shutdown event
        try:
            await self.workspace.update(
                data={"status": "system_shutdown"},
                source="system",
                attention_level=AttentionLevel.CRITICAL,
            )
        except Exception as e:
            logger.error(f"Error broadcasting shutdown event: {e}")

        logger.info("🏁 Super Alita shutdown complete")

    def get_system_stats(self) -> dict[str, Any]:
        """Get comprehensive system statistics."""
        workspace_stats = self.workspace.get_workspace_stats()
        store_stats = self.store.get_topology_summary()

        return {
            "system_state": self.system_state.dict(),
            "workspace_stats": workspace_stats,
            "neural_store_stats": store_stats,
            "plugins_active": list(self.plugins.keys()),
            "config_summary": {
                "neural_atoms_enabled": self.config.get("neural_atoms", {}).get(
                    "registry_type"
                ),
                "consciousness_enabled": self.config.get("global_workspace", {}).get(
                    "consciousness_layer_enabled"
                ),
                "creator_enabled": self.config.get("creator_engine", {}).get("enabled"),
                "safety_enabled": self.config.get("safety", {}).get(
                    "recursive_improvement"
                ),
            },
        }


async def main():
    """Main entry point for the unified Super Alita agent."""
    alita = UnifiedSuperAlita()

    try:
        await alita.run()
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, shutting down...")
    except Exception as e:
        logger.error(f"Agent failed with error: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
