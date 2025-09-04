"""
UnifiedToolRouter: resilient execution wrapper around registry tools.
"""
import asyncio, logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

class UnifiedToolRouter:
    def __init__(self, registry, *, timeout_s: float = 30.0, max_retries: int = 2) -> None:
        self.registry = registry
        self.timeout_s = timeout_s
        self.max_retries = max_retries

    async def execute_tool_with_recovery(self, tool_id: str, args: Dict[str, Any]) -> Dict[str, Any]:
        tries = 0; last_err: Optional[Exception] = None
        while tries <= self.max_retries:
            try:
                res = await asyncio.wait_for(self.registry.execute(tool_id, args), timeout=self.timeout_s)
                return {"success": True, "result": res}
            except asyncio.TimeoutError as e:
                last_err = e; logger.warning("Tool %s timed out", tool_id)
            except ConnectionError as e:
                last_err = e; logger.warning("Conn error for %s", tool_id)
                fb = await self._maybe_consensus_fallback(tool_id, args)
                if fb: return {"success": True, "result": fb}
            except Exception as e:
                last_err = e; logger.error("Tool %s failed: %s", tool_id, e)
            tries += 1; await asyncio.sleep(0.5 * tries)
        return {"success": False, "error": str(last_err) if last_err else "unknown error"}

    async def _maybe_consensus_fallback(self, tool_id: str, args: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if tool_id != "deepconf_consensus": return None
        fallback = dict(args); fallback["method"] = "simple_vote"
        try: return {"consensus": await self.registry.execute(tool_id, fallback)}
        except Exception: return None