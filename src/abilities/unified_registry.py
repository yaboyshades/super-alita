"""
Unified ability registry with auto-discovery and feature-flagged demos.

Combines the functionality of registry_auto.py with feature flag support
for demo scripts and enhanced capability management.

Pattern adapted from GitHub patterns found in:
- Feature flag environment variable patterns
- Plugin auto-discovery with include/exclude lists
- Graceful degradation when modules are unavailable
"""

import importlib
import inspect
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class FeatureFlaggedDemo:
    """Represents a demo that can be enabled/disabled via feature flags."""

    def __init__(
        self,
        name: str,
        flag_env: str,
        module_path: str,
        description: str = "",
        dependencies: list[str] | None = None,
    ):
        self.name = name
        self.flag_env = flag_env
        self.module_path = module_path
        self.description = description
        self.dependencies = dependencies or []
        self._enabled = None
        self._module = None

    @property
    def enabled(self) -> bool:
        """Check if demo is enabled via environment variable."""
        if self._enabled is None:
            self._enabled = os.getenv(self.flag_env, "false").lower() == "true"
        return self._enabled

    @property
    def available(self) -> bool:
        """Check if demo dependencies are available."""
        try:
            for dep in self.dependencies:
                importlib.import_module(dep)
            return True
        except ImportError:
            return False

    def load_module(self):
        """Load the demo module if enabled and available."""
        if not self.enabled:
            logger.debug(f"Demo {self.name} disabled via {self.flag_env}")
            return None

        if not self.available:
            logger.warning(
                f"Demo {self.name} dependencies not available: {self.dependencies}"
            )
            return None

        try:
            self._module = importlib.import_module(self.module_path)
            logger.info(f"Loaded demo: {self.name}")
            return self._module
        except Exception as e:
            logger.error(f"Failed to load demo {self.name}: {e}")
            return None


class UnifiedAbilityRegistry:
    """
    Unified registry that combines auto-discovery with feature-flagged demos.

    Maintains compatibility with SimpleAbilityRegistry.register_tool(contract, executor)
    while adding enhanced capability management.
    """

    def __init__(self, base_registry: Any):
        self.base_registry = base_registry
        self.demos = self._register_demos()
        self.discovered_abilities = {}
        self.demo_tools = {}

    def _register_demos(self) -> dict[str, FeatureFlaggedDemo]:
        """Register available feature-flagged demos."""
        return {
            "github_integration": FeatureFlaggedDemo(
                name="GitHub Integration",
                flag_env="ENABLE_GITHUB_DEMO",
                module_path="demo_github_integration",
                description="GitHub API integration with event processing",
                dependencies=[
                    "src.core.github_priority_calculator",
                    "src.integration.github_api",
                ],
            ),
            "perplexica_search": FeatureFlaggedDemo(
                name="Perplexica Search",
                flag_env="ENABLE_PERPLEXICA_DEMO",
                module_path="demo_perplexica_integration",
                description="AI-powered search capabilities",
                dependencies=["src.plugins.perplexica_search_plugin"],
            ),
            "autogen_pipeline": FeatureFlaggedDemo(
                name="AutoGen Pipeline",
                flag_env="ENABLE_AUTOGEN_DEMO",
                module_path="test_autogen_registry",
                description="AutoGen integration for multi-agent workflows",
                dependencies=["autogen"],
            ),
            "consensus_chat": FeatureFlaggedDemo(
                name="Consensus Chat",
                flag_env="ENABLE_CONSENSUS_DEMO",
                module_path="demo_consensus_chat",
                description="Enhanced consensus algorithms demonstration",
                dependencies=["src.abilities.enhanced_consensus_ability"],
            ),
            "ollama_integration": FeatureFlaggedDemo(
                name="Ollama Integration",
                flag_env="ENABLE_OLLAMA_DEMO",
                module_path="demo_ollama_working",
                description="Local Ollama model integration",
                dependencies=["httpx"],
            ),
        }

    async def auto_discover_abilities(self) -> dict[str, Any]:
        """
        Enhanced auto-discovery that respects include/exclude environment variables.

        Compatible with existing registry_auto.py patterns.
        """
        abilities_dir = Path(__file__).parent
        include = {
            x.strip()
            for x in (os.getenv("ALITA_ABILITIES_INCLUDE", "") or "").split(
                ","
            )
            if x.strip()
        }
        exclude = {
            x.strip()
            for x in (os.getenv("ALITA_ABILITIES_EXCLUDE", "") or "").split(
                ","
            )
            if x.strip()
        }

        discovered = {}

        # Scan abilities directory
        for file in abilities_dir.glob("*.py"):
            name = file.stem

            # Skip private modules and self
            if name.startswith("_") or name == "unified_registry":
                continue

            # Apply include/exclude filters
            if include and name not in include:
                continue
            if name in exclude:
                continue

            try:
                mod_name = f"src.abilities.{name}"
                mod = importlib.import_module(mod_name)
                discovered[name] = await self._process_ability_module(
                    mod, name
                )

            except Exception as e:
                logger.warning(f"Failed to discover ability {name}: {e}")
                continue

        self.discovered_abilities = discovered
        return discovered

    async def _process_ability_module(
        self, module: Any, name: str
    ) -> dict[str, Any]:
        """Process a discovered ability module using existing patterns."""
        result = {
            "module": module,
            "tools": [],
            "contracts": [],
            "executors": {},
        }

        # Pattern 1: register_abilities(registry) function
        try:
            register_fn = getattr(module, "register_abilities", None)
            if callable(register_fn):
                # Call with base registry to maintain compatibility
                res = register_fn(self.base_registry)
                if inspect.isawaitable(res):
                    await res
                result["registration_method"] = "function"
                result["tools"] = (
                    self.base_registry.get_available_tools_schema()
                )
                return result
        except Exception as e:
            logger.warning(f"Function registration failed for {name}: {e}")

        # Pattern 2: ABILITY_CONTRACTS + EXECUTORS mapping
        try:
            contracts = getattr(module, "ABILITY_CONTRACTS", None)
            executors = getattr(module, "EXECUTORS", None)

            if isinstance(contracts, list) and isinstance(executors, dict):
                for contract in contracts:
                    tool_id = contract.get("tool_id")
                    if tool_id and tool_id in executors:
                        executor = executors[tool_id]
                        if callable(executor):
                            # Register with base registry
                            self.base_registry.register_tool(
                                contract=contract, executor=executor
                            )
                            result["contracts"].append(contract)
                            result["executors"][tool_id] = executor

                result["registration_method"] = "contracts_executors"
                return result
        except Exception as e:
            logger.warning(
                f"Contracts/executors registration failed for {name}: {e}"
            )

        # Pattern 3: Provider classes with get_tools()/get_tool_executor()
        try:
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (
                    inspect.isclass(attr)
                    and hasattr(attr, "get_tools")
                    and hasattr(attr, "get_tool_executor")
                ):

                    provider = attr()
                    tools = provider.get_tools()

                    for tool in tools:
                        tool_name = tool.get("name") or tool.get("tool_id")
                        if tool_name:
                            executor = provider.get_tool_executor(tool_name)
                            if callable(executor):
                                # Create compatible contract
                                contract = {
                                    "tool_id": tool_name,
                                    "description": tool.get("description", ""),
                                    "input_schema": tool.get(
                                        "input_schema", {"type": "object"}
                                    ),
                                    "output_schema": tool.get(
                                        "output_schema", {"type": "object"}
                                    ),
                                }

                                self.base_registry.register_tool(
                                    contract=contract, executor=executor
                                )
                                result["contracts"].append(contract)
                                result["executors"][tool_name] = executor

                    result["registration_method"] = "provider_class"
                    return result
        except Exception as e:
            logger.warning(
                f"Provider class registration failed for {name}: {e}"
            )

        result["registration_method"] = "none"
        return result

    async def load_demo_tools(self) -> dict[str, Any]:
        """Load tools from feature-flagged demos."""
        demo_results = {}

        for demo_id, demo in self.demos.items():
            if not demo.enabled:
                logger.debug(f"Demo {demo.name} disabled")
                continue

            module = demo.load_module()
            if module:
                try:
                    # Try to extract tools from demo module
                    tools = await self._extract_demo_tools(module, demo)
                    if tools:
                        demo_results[demo_id] = {
                            "demo": demo,
                            "module": module,
                            "tools": tools,
                        }
                        self.demo_tools[demo_id] = tools
                        logger.info(
                            f"Loaded {len(tools)} tools from demo {demo.name}"
                        )

                except Exception as e:
                    logger.error(
                        f"Error loading tools from demo {demo.name}: {e}"
                    )

        return demo_results

    async def _extract_demo_tools(
        self, module: Any, demo: FeatureFlaggedDemo
    ) -> list[dict[str, Any]]:
        """Extract tools from a demo module."""
        tools = []

        # Look for main function that can be wrapped as a tool
        if hasattr(module, "main") and callable(module.main):
            tool = {
                "tool_id": f"demo_{demo.name.lower().replace(' ', '_')}",
                "description": f"Run {demo.description}",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "args": {
                            "type": "array",
                            "description": "Command line arguments",
                        }
                    },
                },
                "output_schema": {"type": "object"},
            }

            async def demo_executor(args=None):
                """Execute demo main function."""
                try:
                    # Mock sys.argv for demo
                    import sys

                    old_argv = sys.argv
                    sys.argv = ["demo"] + (args or [])

                    result = module.main()
                    if inspect.isawaitable(result):
                        result = await result

                    sys.argv = old_argv
                    return {"status": "success", "result": str(result)}

                except Exception as e:
                    if "old_argv" in locals():
                        sys.argv = old_argv
                    return {"status": "error", "error": str(e)}

            # Register with base registry
            self.base_registry.register_tool(
                contract=tool, executor=demo_executor
            )
            tools.append(tool)

        # Look for demo-specific tool registration patterns
        for attr_name in ["DEMO_TOOLS", "TOOLS", "get_tools"]:
            if hasattr(module, attr_name):
                attr = getattr(module, attr_name)
                if callable(attr):
                    try:
                        demo_tools = attr()
                        if isinstance(demo_tools, list):
                            tools.extend(demo_tools)
                    except Exception as e:
                        logger.warning(
                            f"Error calling {attr_name} in {demo.name}: {e}"
                        )
                elif isinstance(attr, list):
                    tools.extend(attr)

        return tools

    async def initialize(self) -> dict[str, Any]:
        """Initialize the unified registry with full discovery."""
        logger.info("Initializing unified ability registry...")

        # Discover standard abilities
        abilities = await self.auto_discover_abilities()
        logger.info(f"Discovered {len(abilities)} standard abilities")

        # Load demo tools if enabled
        demos = await self.load_demo_tools()
        logger.info(f"Loaded {len(demos)} demo modules")

        # Get final tool count from base registry
        total_tools = len(self.base_registry.get_available_tools_schema())
        logger.info(f"Total tools registered: {total_tools}")

        return {
            "abilities": abilities,
            "demos": demos,
            "total_tools": total_tools,
            "feature_flags": {
                demo_id: demo.enabled for demo_id, demo in self.demos.items()
            },
        }

    def get_status(self) -> dict[str, Any]:
        """Get current registry status including feature flags."""
        return {
            "discovered_abilities": len(self.discovered_abilities),
            "loaded_demos": len(self.demo_tools),
            "total_tools": len(
                self.base_registry.get_available_tools_schema()
            ),
            "feature_flags": {
                demo_id: {
                    "enabled": demo.enabled,
                    "available": demo.available,
                    "description": demo.description,
                }
                for demo_id, demo in self.demos.items()
            },
            "environment_filters": {
                "include": os.getenv("ALITA_ABILITIES_INCLUDE", ""),
                "exclude": os.getenv("ALITA_ABILITIES_EXCLUDE", ""),
            },
        }

    # Delegate methods to maintain compatibility
    def register_tool(self, contract: dict[str, Any], executor: callable):
        """Delegate to base registry for compatibility."""
        return self.base_registry.register_tool(contract, executor)

    def get_available_tools_schema(self):
        """Delegate to base registry."""
        return self.base_registry.get_available_tools_schema()

    def knows(self, name: str) -> bool:
        """Delegate to base registry."""
        return self.base_registry.knows(name)

    def validate_args(self, name: str, args: dict) -> bool:
        """Delegate to base registry."""
        return self.base_registry.validate_args(name, args)

    async def execute(self, name: str, args: dict):
        """Delegate to base registry."""
        return await self.base_registry.execute(name, args)


async def create_unified_registry(
    base_registry: Any,
) -> UnifiedAbilityRegistry:
    """
    Factory function to create and initialize a unified registry.

    Args:
        base_registry: Existing SimpleAbilityRegistry instance

    Returns:
        Initialized UnifiedAbilityRegistry
    """
    registry = UnifiedAbilityRegistry(base_registry)
    await registry.initialize()
    return registry


def get_demo_status() -> dict[str, dict[str, Any]]:
    """Get current demo status without creating registry."""
    registry = UnifiedAbilityRegistry(None)
    return {
        demo_id: {
            "enabled": demo.enabled,
            "available": demo.available,
            "description": demo.description,
            "flag_env": demo.flag_env,
            "dependencies": demo.dependencies,
        }
        for demo_id, demo in registry.demos.items()
    }
