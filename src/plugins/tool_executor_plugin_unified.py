# Version: 3.0.0
# Description: Executes Neural Atoms and manages tool orchestration for unified architecture.

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from typing import Any

from src.core.global_workspace import AttentionLevel, GlobalWorkspace, WorkspaceEvent
from src.core.neural_atom import NeuralAtom, NeuralAtomMetadata, NeuralStore
from src.core.plugin_interface import PluginInterface
from src.core.schemas import (
    ExecutionStatus,
    ToolExecutionRequest,
    ToolExecutionResult,
)

logger = logging.getLogger(__name__)


class ToolExecutorPlugin(PluginInterface):
    """
    Executes Neural Atoms and manages concurrent tool orchestration.

    Features:
    - Semantic tool selection and execution
    - Concurrent execution management
    - Performance monitoring and optimization
    - Safe execution environment
    - Result aggregation and broadcasting
    """

    def __init__(self):
        super().__init__()
        self.workspace: GlobalWorkspace | None = None
        self.store: NeuralStore | None = None

        # Execution management
        self.thread_pool: ThreadPoolExecutor | None = None
        self.max_concurrent_executions = 10
        self.execution_timeout = 30.0

        # Active execution tracking
        self.active_executions: dict[str, dict[str, Any]] = {}
        self.execution_lock = Lock()

        # Performance and safety
        self.execution_stats = {
            "total_executions": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "timeout_executions": 0,
            "average_execution_time": 0.0,
            "concurrent_executions_peak": 0,
        }

        # Tool discovery cache
        self.tool_cache: dict[str, NeuralAtom] = {}
        self.cache_ttl = 300  # 5 minutes
        self.cache_timestamps: dict[str, float] = {}

    async def setup(
        self, workspace: GlobalWorkspace, store: NeuralStore, config: dict[str, Any]
    ):
        """Initialize the Tool Executor Plugin with workspace and store."""
        await super().setup(workspace, store, config)

        self.workspace = workspace
        self.store = store

        # Configure execution settings
        self.max_concurrent_executions = config.get("max_concurrent_execution", 10)
        self.execution_timeout = config.get("execution_timeout", 30.0)
        self.cache_ttl = config.get("tool_cache_ttl", 300)

        # Initialize thread pool for sync tool execution
        self.thread_pool = ThreadPoolExecutor(
            max_workers=self.max_concurrent_executions,
            thread_name_prefix="ToolExecutor",
        )

        logger.info(
            f"Tool Executor Plugin initialized with {self.max_concurrent_executions} max concurrent executions"
        )

    async def start(self):
        """Start the Tool Executor Plugin and subscribe to workspace events."""
        await super().start()

        if self.workspace:
            self.workspace.subscribe("tool_executor", self._handle_workspace_event)
            logger.info("Tool Executor Plugin subscribed to Global Workspace")

    async def shutdown(self):
        """Gracefully shutdown the Tool Executor Plugin."""
        # Wait for active executions to complete
        await self._wait_for_active_executions()

        # Shutdown thread pool
        if self.thread_pool:
            self.thread_pool.shutdown(wait=True)

        await super().shutdown()
        logger.info("Tool Executor Plugin shutdown complete")

    async def _handle_workspace_event(self, event: WorkspaceEvent):
        """Handle events from the Global Workspace."""
        try:
            if isinstance(event.data, dict):
                event_type = event.data.get("type")

                if event_type == "tool_execution_request":
                    request = ToolExecutionRequest(**event.data)
                    await self._handle_execution_request(request)
                elif event_type == "batch_execution_request":
                    await self._handle_batch_execution_request(event.data)
                elif event_type == "tool_discovery_request":
                    await self._handle_tool_discovery_request(event.data)

        except Exception as e:
            logger.error(f"Error handling workspace event in Tool Executor: {e}")

    async def _handle_execution_request(self, request: ToolExecutionRequest):
        """Handle individual tool execution request."""
        execution_id = request.execution_id or f"exec_{int(time.time() * 1000)}"

        logger.info(f"⚡ EXECUTOR: Processing execution request {execution_id}")

        try:
            # Check execution limits
            if len(self.active_executions) >= self.max_concurrent_executions:
                await self._send_execution_result(
                    execution_id,
                    request,
                    False,
                    error="Maximum concurrent executions reached",
                )
                return

            # Find suitable Neural Atom
            neural_atom = await self._find_neural_atom(
                request.tool_name, request.task_description
            )
            if not neural_atom:
                await self._send_execution_result(
                    execution_id,
                    request,
                    False,
                    error=f"No suitable Neural Atom found for: {request.tool_name or request.task_description}",
                )
                return

            # Execute the Neural Atom
            await self._execute_neural_atom(execution_id, request, neural_atom)

        except Exception as e:
            logger.error(f"Execution request failed: {e}")
            await self._send_execution_result(
                execution_id, request, False, error=f"Execution failed: {e}"
            )

    async def _find_neural_atom(
        self, tool_name: str | None, task_description: str | None
    ) -> NeuralAtom | None:
        """Find the best Neural Atom for the given request."""
        try:
            # Check cache first
            cache_key = tool_name or task_description or "unknown"
            if cache_key in self.tool_cache:
                if time.time() - self.cache_timestamps[cache_key] < self.cache_ttl:
                    return self.tool_cache[cache_key]
                # Cache expired
                del self.tool_cache[cache_key]
                del self.cache_timestamps[cache_key]

            # Search by exact name first
            if tool_name:
                exact_match = self.store.get_by_name(tool_name)
                if exact_match:
                    self._cache_tool(cache_key, exact_match)
                    return exact_match

            # Semantic search based on task description
            if task_description:
                best_atom = None
                best_score = 0.0

                for atom in self.store.get_all():
                    confidence_score = atom.can_handle(task_description)
                    if (
                        confidence_score > best_score and confidence_score > 0.5
                    ):  # Minimum confidence threshold
                        best_score = confidence_score
                        best_atom = atom

                if best_atom:
                    self._cache_tool(cache_key, best_atom)
                    return best_atom

            # Fallback: get any available atom if nothing specific requested
            if not tool_name and not task_description:
                all_atoms = self.store.get_all()
                if all_atoms:
                    return all_atoms[0]

            return None

        except Exception as e:
            logger.error(f"Error finding Neural Atom: {e}")
            return None

    def _cache_tool(self, cache_key: str, atom: NeuralAtom):
        """Cache a tool for future use."""
        self.tool_cache[cache_key] = atom
        self.cache_timestamps[cache_key] = time.time()

    async def _execute_neural_atom(
        self, execution_id: str, request: ToolExecutionRequest, atom: NeuralAtom
    ):
        """Execute a Neural Atom with monitoring and safety."""
        start_time = time.time()

        # Track active execution
        with self.execution_lock:
            self.active_executions[execution_id] = {
                "request": request,
                "atom": atom,
                "start_time": start_time,
                "status": "running",
            }
            current_concurrent = len(self.active_executions)
            self.execution_stats["concurrent_executions_peak"] = max(
                self.execution_stats["concurrent_executions_peak"], current_concurrent
            )

        self.execution_stats["total_executions"] += 1

        try:
            logger.info(f"Executing Neural Atom: {atom.metadata.name}")

            # Execute with timeout
            result = await asyncio.wait_for(
                atom.safe_execute(request.parameters), timeout=self.execution_timeout
            )

            execution_time = time.time() - start_time
            success = result.get("success", False)

            if success:
                self.execution_stats["successful_executions"] += 1
                await self._send_execution_result(
                    execution_id,
                    request,
                    True,
                    result=result.get("result"),
                    execution_time=execution_time,
                    atom_metadata=atom.metadata,
                )
            else:
                self.execution_stats["failed_executions"] += 1
                await self._send_execution_result(
                    execution_id,
                    request,
                    False,
                    error=result.get("error", "Unknown execution error"),
                    execution_time=execution_time,
                    atom_metadata=atom.metadata,
                )

            self._update_execution_stats(execution_time)

        except TimeoutError:
            self.execution_stats["timeout_executions"] += 1
            execution_time = time.time() - start_time
            await self._send_execution_result(
                execution_id,
                request,
                False,
                error=f"Execution timed out after {self.execution_timeout}s",
                execution_time=execution_time,
                atom_metadata=atom.metadata,
            )

        except Exception as e:
            self.execution_stats["failed_executions"] += 1
            execution_time = time.time() - start_time
            await self._send_execution_result(
                execution_id,
                request,
                False,
                error=f"Execution error: {e}",
                execution_time=execution_time,
                atom_metadata=atom.metadata,
            )

        finally:
            # Remove from active executions
            with self.execution_lock:
                if execution_id in self.active_executions:
                    del self.active_executions[execution_id]

    async def _send_execution_result(
        self,
        execution_id: str,
        request: ToolExecutionRequest,
        success: bool,
        result: Any = None,
        error: str = None,
        execution_time: float = 0.0,
        atom_metadata: NeuralAtomMetadata = None,
    ):
        """Send execution result back to the workspace."""
        try:
            execution_result = ToolExecutionResult(
                execution_id=execution_id,
                request_id=request.request_id,
                success=success,
                result=result,
                error=error,
                execution_time=execution_time,
                status=ExecutionStatus.COMPLETED if success else ExecutionStatus.FAILED,
                atom_used=atom_metadata.name if atom_metadata else None,
            )

            await self.workspace.update(
                data={"type": "tool_execution_result", **execution_result.model_dump()},
                source="tool_executor",
                attention_level=(
                    AttentionLevel.HIGH if success else AttentionLevel.CRITICAL
                ),
            )

            logger.info(
                f"✅ EXECUTOR: Sent result for {execution_id} - Success: {success}"
            )

        except Exception as e:
            logger.error(f"Failed to send execution result: {e}")

    async def _handle_batch_execution_request(self, request_data: dict[str, Any]):
        """Handle batch execution of multiple tools."""
        batch_id = request_data.get("batch_id", f"batch_{int(time.time())}")
        requests = request_data.get("requests", [])

        logger.info(
            f"🔄 EXECUTOR: Processing batch {batch_id} with {len(requests)} requests"
        )

        try:
            # Convert to ToolExecutionRequest objects
            execution_requests = []
            for req_data in requests:
                execution_requests.append(ToolExecutionRequest(**req_data))

            # Execute all requests concurrently
            tasks = []
            for request in execution_requests:
                task = asyncio.create_task(self._handle_execution_request(request))
                tasks.append(task)

            # Wait for all executions to complete
            await asyncio.gather(*tasks, return_exceptions=True)

            # Send batch completion notification
            await self.workspace.update(
                data={
                    "type": "batch_execution_completed",
                    "batch_id": batch_id,
                    "total_requests": len(requests),
                },
                source="tool_executor",
                attention_level=AttentionLevel.MEDIUM,
            )

        except Exception as e:
            logger.error(f"Batch execution failed: {e}")
            await self.workspace.update(
                data={
                    "type": "batch_execution_failed",
                    "batch_id": batch_id,
                    "error": str(e),
                },
                source="tool_executor",
                attention_level=AttentionLevel.CRITICAL,
            )

    async def _handle_tool_discovery_request(self, request_data: dict[str, Any]):
        """Handle tool discovery and capability matching requests."""
        query = request_data.get("query", "")
        limit = request_data.get("limit", 10)

        logger.info(f"🔍 EXECUTOR: Tool discovery for: {query}")

        try:
            discovered_tools = []

            # Get all available Neural Atoms
            all_atoms = self.store.get_all()

            # Score each atom for the query
            scored_atoms = []
            for atom in all_atoms:
                confidence = atom.can_handle(query)
                if confidence > 0.1:  # Minimum relevance threshold
                    scored_atoms.append((atom, confidence))

            # Sort by confidence score
            scored_atoms.sort(key=lambda x: x[1], reverse=True)

            # Build discovery results
            for atom, confidence in scored_atoms[:limit]:
                discovered_tools.append(
                    {
                        "name": atom.metadata.name,
                        "description": atom.metadata.description,
                        "capabilities": atom.metadata.capabilities,
                        "confidence_score": confidence,
                        "version": atom.metadata.version,
                        "tags": list(atom.metadata.tags) if atom.metadata.tags else [],
                        "usage_count": atom.metadata.usage_count,
                        "success_rate": atom.metadata.success_rate,
                    }
                )

            # Send discovery results
            await self.workspace.update(
                data={
                    "type": "tool_discovery_result",
                    "query": query,
                    "tools": discovered_tools,
                    "total_found": len(discovered_tools),
                },
                source="tool_executor",
                attention_level=AttentionLevel.MEDIUM,
            )

        except Exception as e:
            logger.error(f"Tool discovery failed: {e}")
            await self.workspace.update(
                data={"type": "tool_discovery_failed", "query": query, "error": str(e)},
                source="tool_executor",
                attention_level=AttentionLevel.MEDIUM,
            )

    async def _wait_for_active_executions(self, timeout: float = 30.0):
        """Wait for all active executions to complete."""
        start_time = time.time()

        while self.active_executions and (time.time() - start_time) < timeout:
            logger.info(
                f"Waiting for {len(self.active_executions)} active executions to complete..."
            )
            await asyncio.sleep(1.0)

        if self.active_executions:
            logger.warning(
                f"Timeout waiting for executions: {list(self.active_executions.keys())}"
            )

    def _update_execution_stats(self, execution_time: float):
        """Update execution statistics."""
        # Update average execution time
        alpha = 0.1
        if self.execution_stats["average_execution_time"] == 0.0:
            self.execution_stats["average_execution_time"] = execution_time
        else:
            self.execution_stats["average_execution_time"] = (
                alpha * execution_time
                + (1 - alpha) * self.execution_stats["average_execution_time"]
            )

    def get_execution_stats(self) -> dict[str, Any]:
        """Get current execution statistics."""
        return {
            **self.execution_stats,
            "active_executions": len(self.active_executions),
            "cached_tools": len(self.tool_cache),
            "thread_pool_active": self.thread_pool is not None,
            "max_concurrent_executions": self.max_concurrent_executions,
        }

    async def execute_tool_directly(
        self, tool_name: str, parameters: dict[str, Any]
    ) -> dict[str, Any]:
        """Direct tool execution interface for programmatic use."""
        ToolExecutionRequest(
            request_id=f"direct_{int(time.time() * 1000)}",
            tool_name=tool_name,
            parameters=parameters,
        )

        # Find and execute the tool
        neural_atom = await self._find_neural_atom(tool_name, None)
        if not neural_atom:
            return {"success": False, "error": f"Tool not found: {tool_name}"}

        try:
            result = await neural_atom.safe_execute(parameters)
            return result
        except Exception as e:
            return {"success": False, "error": f"Direct execution failed: {e}"}

    async def get_available_tools(self) -> list[dict[str, Any]]:
        """Get list of all available tools/Neural Atoms."""
        tools = []

        for atom in self.store.get_all():
            tools.append(
                {
                    "name": atom.metadata.name,
                    "description": atom.metadata.description,
                    "capabilities": atom.metadata.capabilities,
                    "version": atom.metadata.version,
                    "tags": list(atom.metadata.tags) if atom.metadata.tags else [],
                    "usage_count": atom.metadata.usage_count,
                    "success_rate": atom.metadata.success_rate,
                    "avg_execution_time": atom.metadata.avg_execution_time,
                }
            )

        return tools
