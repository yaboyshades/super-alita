"""Copilot Todos API Integration

Provides bi-directional synchronization between VS Code Copilot Todos and
the Knowledge Graph Task Manager for enhanced task management.
"""

import asyncio
import contextlib
import logging
from datetime import UTC, datetime
from typing import Any

import httpx
from pydantic import BaseModel, Field

from .kg_task_manager import (
    GraphTaskNode,
    KnowledgeGraphTaskManager,
    TaskPriority,
    TaskStatus,
)
from .plugin_interface import PluginInterface

logger = logging.getLogger(__name__)


class CopilotTodoItem(BaseModel):
    """Copilot Todos API item model."""

    id: str = Field(..., description="Todo item ID")
    title: str = Field(..., description="Todo title")
    description: str = Field(default="", description="Todo description")
    completed: bool = Field(default=False, description="Completion status")
    priority: str = Field(default="medium", description="Priority level")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")
    due_date: datetime | None = Field(default=None, description="Due date")
    tags: list[str] = Field(default_factory=list, description="Associated tags")
    context: dict[str, Any] = Field(
        default_factory=dict, description="Additional context"
    )


class CopilotTodosIntegration(PluginInterface):
    """
    Copilot Todos API Integration Plugin

    Provides bi-directional synchronization between VS Code Copilot Todos
    and the Knowledge Graph Task Manager. Enables:

    - Automatic mirroring of Copilot todos to knowledge graph
    - Graph-driven enrichment of todo items
    - Priority synchronization based on graph metrics
    - Context augmentation from related entities
    """

    def __init__(self, event_bus, config: dict[str, Any] | None = None):
        super().__init__(event_bus, config)
        self.config = config or {}

        # API configuration
        self.base_url = self.config.get("copilot_api_base", "http://localhost:3000")
        self.api_timeout = self.config.get("api_timeout", 30.0)
        self.sync_interval = self.config.get("sync_interval", 300)  # 5 minutes

        # Integration components
        self.kg_task_manager: KnowledgeGraphTaskManager | None = None
        self.http_client: httpx.AsyncClient | None = None

        # State tracking
        self.todo_to_task_mapping: dict[str, str] = {}  # todo_id -> task_id
        self.task_to_todo_mapping: dict[str, str] = {}  # task_id -> todo_id
        self.last_sync: datetime | None = None

        # Background tasks
        self._sync_task: asyncio.Task | None = None

    @property
    def name(self) -> str:
        return "CopilotTodosIntegration"

    async def initialize(self):
        """Initialize the Copilot Todos integration."""
        logger.info("Initializing Copilot Todos Integration")

        # Initialize HTTP client
        self.http_client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=self.api_timeout,
            headers={"Content-Type": "application/json"},
        )

        # Get reference to Knowledge Graph Task Manager
        await self._find_kg_task_manager()

        # Register for events
        await self.event_bus.subscribe("task_created", self._handle_task_created)
        await self.event_bus.subscribe("task_updated", self._handle_task_updated)
        await self.event_bus.subscribe("task_completed", self._handle_task_completed)

        # Start background sync if enabled
        if self.config.get("auto_sync", True):
            self._sync_task = asyncio.create_task(self._periodic_sync())

        # Perform initial sync
        await self._initial_sync()

        logger.info("Copilot Todos Integration initialized")

    async def shutdown(self):
        """Shutdown the integration and cleanup resources."""
        logger.info("Shutting down Copilot Todos Integration")

        if self._sync_task:
            self._sync_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._sync_task

        if self.http_client:
            await self.http_client.aclose()

        logger.info("Copilot Todos Integration shutdown complete")

    # API Methods

    async def create_todo(self, todo_item: CopilotTodoItem) -> CopilotTodoItem | None:
        """Create a new todo item via Copilot API."""
        try:
            if not self.http_client:
                logger.error("HTTP client not initialized")
                return None

            response = await self.http_client.post(
                "/api/todos/create", json=todo_item.model_dump()
            )
            response.raise_for_status()

            result = response.json()
            logger.info(f"Created Copilot todo: {result.get('id', 'unknown')}")
            return CopilotTodoItem(**result)

        except Exception as e:
            logger.error(f"Failed to create Copilot todo: {e}")
            return None

    async def update_todo(
        self, todo_id: str, updates: dict[str, Any]
    ) -> CopilotTodoItem | None:
        """Update an existing todo item via Copilot API."""
        try:
            if not self.http_client:
                logger.error("HTTP client not initialized")
                return None

            response = await self.http_client.put(f"/api/todos/{todo_id}", json=updates)
            response.raise_for_status()

            result = response.json()
            logger.info(f"Updated Copilot todo: {todo_id}")
            return CopilotTodoItem(**result)

        except Exception as e:
            logger.error(f"Failed to update Copilot todo {todo_id}: {e}")
            return None

    async def get_todos(self, limit: int = 100) -> list[CopilotTodoItem]:
        """Retrieve todos from Copilot API."""
        try:
            if not self.http_client:
                logger.error("HTTP client not initialized")
                return []

            response = await self.http_client.get(
                "/api/todos/list", params={"limit": limit}
            )
            response.raise_for_status()

            result = response.json()
            todos = [CopilotTodoItem(**item) for item in result.get("todos", [])]
            logger.info(f"Retrieved {len(todos)} Copilot todos")
            return todos

        except Exception as e:
            logger.error(f"Failed to retrieve Copilot todos: {e}")
            return []

    async def delete_todo(self, todo_id: str) -> bool:
        """Delete a todo item via Copilot API."""
        try:
            if not self.http_client:
                logger.error("HTTP client not initialized")
                return False

            response = await self.http_client.delete(f"/api/todos/{todo_id}")
            response.raise_for_status()

            logger.info(f"Deleted Copilot todo: {todo_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to delete Copilot todo {todo_id}: {e}")
            return False

    # Synchronization Methods

    async def sync_task_to_todo(self, task: GraphTaskNode) -> CopilotTodoItem | None:
        """Sync a knowledge graph task to Copilot todo."""
        try:
            # Check if task already has a corresponding todo
            todo_id = self.task_to_todo_mapping.get(task.task_id)

            # Enrich todo with graph context
            enriched_context = await self._enrich_task_context(task)

            todo_data = {
                "title": task.title,
                "description": task.description,
                "completed": task.status == TaskStatus.COMPLETED,
                "priority": self._map_task_priority_to_todo(task.priority),
                "due_date": task.due_date.isoformat() if task.due_date else None,
                "tags": list(task.tags),
                "context": {
                    **task.context,
                    **enriched_context,
                    "kg_node_id": task.kg_node_id,
                    "task_type": task.task_type.value,
                    "dependency_depth": task.dependency_depth,
                    "centrality_score": task.centrality_score,
                },
            }

            if todo_id:
                # Update existing todo
                todo_item = await self.update_todo(todo_id, todo_data)
            else:
                # Create new todo
                todo_item = CopilotTodoItem(
                    id="",  # Will be assigned by API
                    created_at=task.created_at,
                    updated_at=task.updated_at,
                    **todo_data,
                )
                todo_item = await self.create_todo(todo_item)

                if todo_item:
                    # Update mappings
                    self.task_to_todo_mapping[task.task_id] = todo_item.id
                    self.todo_to_task_mapping[todo_item.id] = task.task_id

            return todo_item

        except Exception as e:
            logger.error(f"Failed to sync task {task.task_id} to todo: {e}")
            return None

    async def sync_todo_to_task(self, todo: CopilotTodoItem) -> GraphTaskNode | None:
        """Sync a Copilot todo to knowledge graph task."""
        try:
            if not self.kg_task_manager:
                logger.error("Knowledge Graph Task Manager not available")
                return None

            # Check if todo already has a corresponding task
            task_id = self.todo_to_task_mapping.get(todo.id)

            if task_id:
                # Update existing task
                task_updates = {
                    "title": todo.title,
                    "description": todo.description,
                    "status": TaskStatus.COMPLETED
                    if todo.completed
                    else TaskStatus.NOT_STARTED,
                    "priority": self._map_todo_priority_to_task(todo.priority),
                    "due_date": todo.due_date,
                    "tags": set(todo.tags),
                    "context": todo.context,
                    "last_modified_by": "copilot_todos_integration",
                }

                task = await self.kg_task_manager.update_task(task_id, **task_updates)
            else:
                # Create new task
                from .schemas import TaskType

                task = await self.kg_task_manager.create_task(
                    title=todo.title,
                    description=todo.description,
                    task_type=TaskType.PLANNING,  # Default type for imported todos
                    priority=self._map_todo_priority_to_task(todo.priority),
                    due_date=todo.due_date,
                    tags=set(todo.tags),
                    context={
                        **todo.context,
                        "copilot_todo_id": todo.id,
                        "imported_from": "copilot_todos",
                    },
                    created_by="copilot_todos_integration",
                )

                if task:
                    # Update mappings
                    self.todo_to_task_mapping[todo.id] = task.task_id
                    self.task_to_todo_mapping[task.task_id] = todo.id

            return task

        except Exception as e:
            logger.error(f"Failed to sync todo {todo.id} to task: {e}")
            return None

    async def full_sync(self) -> dict[str, int]:
        """Perform a full bidirectional synchronization."""
        stats = {"todos_synced": 0, "tasks_synced": 0, "errors": 0}

        try:
            # Sync todos to tasks
            todos = await self.get_todos()
            for todo in todos:
                try:
                    await self.sync_todo_to_task(todo)
                    stats["todos_synced"] += 1
                except Exception as e:
                    logger.error(f"Error syncing todo {todo.id}: {e}")
                    stats["errors"] += 1

            # Sync tasks to todos
            if self.kg_task_manager:
                for task in self.kg_task_manager.tasks.values():
                    try:
                        await self.sync_task_to_todo(task)
                        stats["tasks_synced"] += 1
                    except Exception as e:
                        logger.error(f"Error syncing task {task.task_id}: {e}")
                        stats["errors"] += 1

            self.last_sync = datetime.now(UTC)
            logger.info(f"Full sync completed: {stats}")

        except Exception as e:
            logger.error(f"Full sync failed: {e}")
            stats["errors"] += 1

        return stats

    # Helper Methods

    async def _find_kg_task_manager(self):
        """Find the Knowledge Graph Task Manager plugin."""
        # This would integrate with the plugin registry to find the task manager
        # For now, this is a placeholder
        logger.info("Looking for Knowledge Graph Task Manager plugin")

    async def _initial_sync(self):
        """Perform initial synchronization on startup."""
        logger.info("Performing initial sync with Copilot Todos")
        stats = await self.full_sync()
        logger.info(f"Initial sync completed: {stats}")

    async def _periodic_sync(self):
        """Background task for periodic synchronization."""
        while True:
            try:
                await asyncio.sleep(self.sync_interval)
                await self.full_sync()
                logger.debug("Periodic sync completed")
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in periodic sync: {e}")

    async def _enrich_task_context(self, task: GraphTaskNode) -> dict[str, Any]:
        """Enrich task context with knowledge graph information."""
        enrichment = {
            "graph_metrics": {
                "centrality_score": task.centrality_score,
                "dependency_depth": task.dependency_depth,
                "estimated_effort": task.estimated_effort,
            },
            "dependencies": {
                "depends_on": list(task.depends_on),
                "blocks": list(task.blocks),
                "related_to": list(task.related_to),
            },
            "kg_enrichment": {
                "entity_type": task.kg_entity_type,
                "node_id": task.kg_node_id,
                "last_updated": task.updated_at.isoformat(),
            },
        }

        # Add related entities from knowledge graph if available
        # This would query the KG for related entities

        return enrichment

    def _map_task_priority_to_todo(self, priority: TaskPriority) -> str:
        """Map task priority to Copilot todo priority."""
        mapping = {
            TaskPriority.CRITICAL: "critical",
            TaskPriority.HIGH: "high",
            TaskPriority.MEDIUM: "medium",
            TaskPriority.LOW: "low",
            TaskPriority.DEFERRED: "low",
        }
        return mapping.get(priority, "medium")

    def _map_todo_priority_to_task(self, priority: str) -> TaskPriority:
        """Map Copilot todo priority to task priority."""
        mapping = {
            "critical": TaskPriority.CRITICAL,
            "high": TaskPriority.HIGH,
            "medium": TaskPriority.MEDIUM,
            "low": TaskPriority.LOW,
        }
        return mapping.get(priority.lower(), TaskPriority.MEDIUM)

    # Event Handlers

    async def _handle_task_created(self, event):
        """Handle task creation events."""
        try:
            task_data = event.get("task_data", {})
            if task_data:
                # Find the actual task object
                if self.kg_task_manager:
                    task = self.kg_task_manager.tasks.get(event.get("task_id"))
                    if task:
                        await self.sync_task_to_todo(task)
        except Exception as e:
            logger.error(f"Error handling task created event: {e}")

    async def _handle_task_updated(self, event):
        """Handle task update events."""
        try:
            task_data = event.get("task_data", {})
            if task_data:
                # Find the actual task object
                if self.kg_task_manager:
                    task = self.kg_task_manager.tasks.get(event.get("task_id"))
                    if task:
                        await self.sync_task_to_todo(task)
        except Exception as e:
            logger.error(f"Error handling task updated event: {e}")

    async def _handle_task_completed(self, event):
        """Handle task completion events."""
        try:
            task_data = event.get("task_data", {})
            if task_data:
                # Find the actual task object
                if self.kg_task_manager:
                    task = self.kg_task_manager.tasks.get(event.get("task_id"))
                    if task:
                        await self.sync_task_to_todo(task)
        except Exception as e:
            logger.error(f"Error handling task completed event: {e}")


# Utility Functions for Integration


async def discover_copilot_endpoints() -> dict[str, str]:
    """Discover available Copilot Todos API endpoints."""
    endpoints = {}

    try:
        async with httpx.AsyncClient() as client:
            # Try common endpoints
            base_urls = [
                "http://localhost:3000",
                "http://127.0.0.1:3000",
                "http://localhost:8080",
                "http://127.0.0.1:8080",
            ]

            for base_url in base_urls:
                try:
                    response = await client.get(f"{base_url}/api/health", timeout=5.0)
                    if response.status_code == 200:
                        endpoints[base_url] = "available"
                        logger.info(f"Found Copilot API at: {base_url}")
                except Exception:
                    endpoints[base_url] = "unavailable"

    except Exception as e:
        logger.error(f"Error discovering Copilot endpoints: {e}")

    return endpoints


async def test_copilot_api_connection(base_url: str) -> dict[str, Any]:
    """Test connection and capabilities of Copilot Todos API."""
    result = {
        "connected": False,
        "endpoints": {},
        "version": None,
        "features": [],
    }

    try:
        async with httpx.AsyncClient(base_url=base_url, timeout=10.0) as client:
            # Test health endpoint
            health_response = await client.get("/api/health")
            if health_response.status_code == 200:
                result["connected"] = True
                result["health"] = health_response.json()

            # Test todos endpoints
            test_endpoints = [
                "/api/todos/list",
                "/api/todos/create",
                "/api/todos/stats",
            ]

            for endpoint in test_endpoints:
                try:
                    # Use HEAD request to test endpoint existence
                    response = await client.head(endpoint)
                    result["endpoints"][endpoint] = {
                        "available": response.status_code < 500,
                        "status_code": response.status_code,
                    }
                except Exception as e:
                    result["endpoints"][endpoint] = {
                        "available": False,
                        "error": str(e),
                    }

            logger.info(f"Copilot API test completed for {base_url}")

    except Exception as e:
        logger.error(f"Failed to test Copilot API at {base_url}: {e}")
        result["error"] = str(e)

    return result
