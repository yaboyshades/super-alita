"""
UnifiedToolRegistry: auto-discovers abilities and conditionally integrates demos via feature flags.
Feature flags (env):
  ENABLE_GITHUB_DEMO=1
  ENABLE_PERPLEXICA_DEMO=1
  ENABLE_AUTOGEN_DEMO=1
"""
import asyncio, importlib, inspect, logging, os
from pathlib import Path
from typing import Any, Dict, Callable

logger = logging.getLogger(__name__)
ToolExecutor = Callable[[dict], Any]

class UnifiedToolRegistry:
    def __init__(self) -> None:
        self._tools: Dict[str, Dict[str, Any]] = {}
        self._executors: Dict[str, ToolExecutor] = {}
        self._health_checks: Dict[str, Callable[[], Any]] = {}

    def register_tool(self, contract: Dict[str, Any], executor: ToolExecutor) -> None:
        tool_id = contract.get("tool_id")
        if not tool_id:
            logger.warning("Skipping tool without tool_id")
            return
        self._tools[tool_id] = contract
        self._executors[tool_id] = executor

    def get_tools(self): return dict(self._tools)
    def get_available_tools(self): return list(self._tools.keys())

    async def execute(self, tool_id: str, args: dict) -> Any:
        ex = self._executors[tool_id]
        return await ex(args) if inspect.iscoroutinefunction(ex) else ex(args)

    async def health_check_all(self) -> Dict[str, Any]:
        """Run health checks for all registered tools."""
        results = {}
        for tool_id, health_check in self._health_checks.items():
            try:
                if inspect.iscoroutinefunction(health_check):
                    result = await health_check()
                else:
                    result = health_check()
                results[tool_id] = {"status": "healthy", "details": result}
            except Exception as e:
                results[tool_id] = {"status": "unhealthy", "error": str(e)}
        return results

    async def auto_discover_all(self) -> None:
        await self._discover_abilities()
        await self._integrate_demos_if_enabled()

    async def _discover_abilities(self) -> None:
        abilities_dir = Path(__file__).resolve().parent
        for py in abilities_dir.glob("*.py"):
            stem = py.stem
            if stem.startswith("_") or stem in {"unified_registry", "registry_auto"}:
                continue
            mod_name = f"src.abilities.{stem}"
            try:
                mod = importlib.import_module(mod_name)
            except Exception as e:
                logger.warning("Failed to import %s: %s", mod_name, e)
                continue
            if hasattr(mod, "register_abilities"):
                try: 
                    result = mod.register_abilities(self)
                    if inspect.iscoroutinefunction(mod.register_abilities):
                        await result
                except Exception as e: logger.warning("register_abilities failed for %s: %s", stem, e)

    async def _integrate_demos_if_enabled(self) -> None:
        if os.getenv("ENABLE_GITHUB_DEMO", "0") == "1":
            try:
                demo = importlib.import_module("demo_github_integration")
                inst = demo.GitHubCliTool()
                self.register_tool({"tool_id": "github_cli"}, inst.execute)
            except Exception as e: logger.warning("GitHub demo failed: %s", e)
        if os.getenv("ENABLE_PERPLEXICA_DEMO", "0") == "1":
            try:
                demo = importlib.import_module("demo_perplexica_integration")
                inst = demo.PerplexicaSearchPlugin()
                self.register_tool({"tool_id": "perplexica_search"}, lambda a, i=inst: i.execute_tool("search", a))
            except Exception as e: logger.warning("Perplexica demo failed: %s", e)
        if os.getenv("ENABLE_AUTOGEN_DEMO", "0") == "1":
            try:
                reg = importlib.import_module("test_autogen_registry")
                self.register_tool({"tool_id": "autogen_code_pipeline"}, reg.autogen_any)
            except Exception as e: logger.warning("AutoGen demo failed: %s", e)