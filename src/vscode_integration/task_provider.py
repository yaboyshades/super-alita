"""VS Code Task Provider Integration for LADDER Planner

This module provides a bridge between the Enhanced LADDER Planner and VS Code's
experimental todos/task provider API. It enables bi-directional synchronization
between planner tasks and VS Code's todo interface.

Key Features:
- TaskProvider implementation for VS Code API
- Bi-directional sync between planner and VS Code todos
- Real-time task updates via event system
- Support for task dependencies and priorities
"""

import asyncio
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

from ..core.events import create_event
from ..core.plugin_interface import PluginInterface
from ..cortex.config.planner_config import PlannerConfig
from ..cortex.planner.ladder_enhanced import EnhancedLadderPlanner

logger = logging.getLogger(__name__)


class VSCodeTask(BaseModel):
    """VS Code Task representation for todos API."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    title: str
    description: str = ""
    completed: bool = False
    priority: str = "medium"  # low, medium, high
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    due_date: datetime | None = None
    tags: list[str] = Field(default_factory=list)
    context: dict[str, Any] = Field(default_factory=dict)

    # LADDER-specific fields
    ladder_task_id: str | None = None
    ladder_stage: str | None = None
    energy_required: float | None = None
    depends_on: list[str] = Field(default_factory=list)


class VSCodeTaskProvider(PluginInterface):
    """Task Provider for VS Code experimental todos integration."""

    def __init__(self, event_bus, config: dict[str, Any] | None = None):
        super().__init__(event_bus)
        self.config = config or {}
        self.planner_config = PlannerConfig()
        self.enhanced_planner: EnhancedLadderPlanner | None = None
        self.tasks: dict[str, VSCodeTask] = {}
        self.workspace_folder = Path.cwd()
        self.todos_file = self.workspace_folder / ".vscode" / "todos.json"
        self.sync_interval = self.config.get("sync_interval", 30)  # seconds
        self._sync_task: asyncio.Task | None = None

    @property
    def name(self) -> str:
        return "vscode_task_provider"

    async def initialize(self):
        """Initialize the task provider and start sync."""
        logger.info("Initializing VS Code Task Provider")

        # Initialize enhanced planner
        self.enhanced_planner = EnhancedLadderPlanner(
            event_bus=self.event_bus, config=self.planner_config.model_dump()
        )
        await self.enhanced_planner.initialize()

        # Load existing tasks from VS Code todos
        await self._load_vscode_todos()

        # Subscribe to planner events
        if hasattr(self.event_bus, "subscribe"):
            await self.event_bus.subscribe(
                "ladder_task_created", self._on_ladder_task_created
            )
            await self.event_bus.subscribe(
                "ladder_task_updated", self._on_ladder_task_updated
            )
            await self.event_bus.subscribe(
                "ladder_task_completed", self._on_ladder_task_completed
            )

        # Start periodic sync
        self._sync_task = asyncio.create_task(self._periodic_sync())

        logger.info("VS Code Task Provider initialized successfully")

    async def shutdown(self):
        """Cleanup task provider."""
        if self._sync_task and not self._sync_task.done():
            self._sync_task.cancel()
            try:
                await self._sync_task
            except asyncio.CancelledError:
                pass

        if self.enhanced_planner:
            await self.enhanced_planner.shutdown()

        logger.info("VS Code Task Provider shutdown complete")

    async def provide_tasks(self) -> list[dict[str, Any]]:
        """Provide tasks for VS Code task provider API."""
        tasks = []
        for task in self.tasks.values():
            task_data = {
                "id": task.id,
                "title": task.title,
                "description": task.description,
                "completed": task.completed,
                "priority": task.priority,
                "createdAt": task.created_at.isoformat(),
                "updatedAt": task.updated_at.isoformat(),
                "tags": task.tags,
                "context": {
                    **task.context,
                    "ladder_task_id": task.ladder_task_id,
                    "ladder_stage": task.ladder_stage,
                    "energy_required": task.energy_required,
                    "depends_on": task.depends_on,
                },
            }
            if task.due_date:
                task_data["dueDate"] = task.due_date.isoformat()
            tasks.append(task_data)

        return tasks

    async def create_task(self, task_data: dict[str, Any]) -> VSCodeTask:
        """Create a new task from VS Code todos."""
        task = VSCodeTask(
            title=task_data.get("title", ""),
            description=task_data.get("description", ""),
            priority=task_data.get("priority", "medium"),
            tags=task_data.get("tags", []),
            context=task_data.get("context", {}),
        )

        # Store task
        self.tasks[task.id] = task

        # Create corresponding LADDER task if planner is available
        if self.enhanced_planner:
            await self._create_ladder_task_from_vscode(task)

        # Save to VS Code todos
        await self._save_vscode_todos()

        # Emit event
        event = create_event(
            "vscode_task_created",
            task_id=task.id,
            title=task.title,
            description=task.description,
        )
        if hasattr(self.event_bus, "publish"):
            await self.event_bus.publish(event)

        logger.info(f"Created VS Code task: {task.title}")
        return task

    async def update_task(
        self, task_id: str, updates: dict[str, Any]
    ) -> VSCodeTask | None:
        """Update an existing task."""
        if task_id not in self.tasks:
            return None

        task = self.tasks[task_id]

        # Apply updates
        for key, value in updates.items():
            if hasattr(task, key):
                setattr(task, key, value)

        task.updated_at = datetime.now(UTC)

        # Update corresponding LADDER task
        if task.ladder_task_id and self.enhanced_planner:
            await self._update_ladder_task_from_vscode(task)

        # Save to VS Code todos
        await self._save_vscode_todos()

        # Emit event
        event = create_event("vscode_task_updated", task_id=task.id, updates=updates)
        if hasattr(self.event_bus, "publish"):
            await self.event_bus.publish(event)

        logger.info(f"Updated VS Code task: {task.title}")
        return task

    async def complete_task(self, task_id: str) -> VSCodeTask | None:
        """Mark a task as completed."""
        return await self.update_task(task_id, {"completed": True})

    async def delete_task(self, task_id: str) -> bool:
        """Delete a task."""
        if task_id not in self.tasks:
            return False

        task = self.tasks[task_id]

        # Delete corresponding LADDER task
        if task.ladder_task_id and self.enhanced_planner:
            try:
                # Note: Enhanced planner doesn't have explicit delete,
                # so we complete it instead
                await self.enhanced_planner.complete_task(task.ladder_task_id)
            except Exception as e:
                logger.warning(
                    f"Failed to delete LADDER task {task.ladder_task_id}: {e}"
                )

        # Remove from tasks
        del self.tasks[task_id]

        # Save to VS Code todos
        await self._save_vscode_todos()

        # Emit event
        event = create_event("vscode_task_deleted", task_id=task.id, title=task.title)
        if hasattr(self.event_bus, "publish"):
            await self.event_bus.publish(event)

        logger.info(f"Deleted VS Code task: {task.title}")
        return True

    # Event handlers for LADDER planner events

    async def _on_ladder_task_created(self, event):
        """Handle LADDER task creation."""
        try:
            task_data = event.get("task_data", {})
            title = task_data.get("title", "LADDER Task")
            description = task_data.get("description", "")
            ladder_task_id = task_data.get("task_id")

            # Create VS Code task
            vscode_task = VSCodeTask(
                title=title,
                description=description,
                ladder_task_id=ladder_task_id,
                ladder_stage=task_data.get("stage", "not_started"),
                energy_required=task_data.get("energy_required"),
                context={"source": "ladder_planner", "stage": task_data.get("stage")},
            )

            self.tasks[vscode_task.id] = vscode_task
            await self._save_vscode_todos()

            logger.info(f"Created VS Code task from LADDER: {title}")

        except Exception as e:
            logger.error(f"Error handling LADDER task creation: {e}")

    async def _on_ladder_task_updated(self, event):
        """Handle LADDER task updates."""
        try:
            task_data = event.get("task_data", {})
            ladder_task_id = task_data.get("task_id")

            # Find corresponding VS Code task
            vscode_task = None
            for task in self.tasks.values():
                if task.ladder_task_id == ladder_task_id:
                    vscode_task = task
                    break

            if vscode_task:
                # Update VS Code task
                vscode_task.ladder_stage = task_data.get("stage")
                vscode_task.updated_at = datetime.now(UTC)
                if "title" in task_data:
                    vscode_task.title = task_data["title"]
                if "description" in task_data:
                    vscode_task.description = task_data["description"]

                await self._save_vscode_todos()
                logger.info(f"Updated VS Code task from LADDER: {vscode_task.title}")

        except Exception as e:
            logger.error(f"Error handling LADDER task update: {e}")

    async def _on_ladder_task_completed(self, event):
        """Handle LADDER task completion."""
        try:
            task_data = event.get("task_data", {})
            ladder_task_id = task_data.get("task_id")

            # Find and complete corresponding VS Code task
            for task in self.tasks.values():
                if task.ladder_task_id == ladder_task_id:
                    task.completed = True
                    task.updated_at = datetime.now(UTC)
                    await self._save_vscode_todos()
                    logger.info(f"Completed VS Code task from LADDER: {task.title}")
                    break

        except Exception as e:
            logger.error(f"Error handling LADDER task completion: {e}")

    # VS Code todos file management

    async def _load_vscode_todos(self):
        """Load existing todos from VS Code todos.json."""
        try:
            if self.todos_file.exists():
                with open(self.todos_file, encoding="utf-8") as f:
                    data = json.load(f)

                todo_list = data.get("todoList", [])
                for todo_data in todo_list:
                    task = VSCodeTask(
                        id=str(todo_data.get("id", uuid4())),
                        title=todo_data.get("title", ""),
                        description=todo_data.get("description", ""),
                        completed=todo_data.get("status") == "completed",
                        context={
                            "source": "vscode_todos",
                            "original_status": todo_data.get("status"),
                        },
                    )
                    self.tasks[task.id] = task

                logger.info(f"Loaded {len(todo_list)} tasks from VS Code todos")

        except Exception as e:
            logger.error(f"Error loading VS Code todos: {e}")

    async def _save_vscode_todos(self):
        """Save current tasks to VS Code todos.json."""
        try:
            # Ensure .vscode directory exists
            self.todos_file.parent.mkdir(exist_ok=True)

            # Convert tasks to VS Code todos format
            todo_list = []
            for task in self.tasks.values():
                todo_item = {
                    "id": int(task.id) if task.id.isdigit() else hash(task.id) % 10000,
                    "title": task.title,
                    "description": task.description,
                    "status": "completed" if task.completed else "not-started",
                }
                todo_list.append(todo_item)

            data = {
                "todoList": todo_list,
                "lastModified": datetime.now(UTC).isoformat(),
                "version": "1.0",
            }

            with open(self.todos_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            logger.debug(f"Saved {len(todo_list)} tasks to VS Code todos")

        except Exception as e:
            logger.error(f"Error saving VS Code todos: {e}")

    # LADDER planner integration

    async def _create_ladder_task_from_vscode(self, vscode_task: VSCodeTask):
        """Create a LADDER planner task from VS Code task."""
        try:
            if not self.enhanced_planner:
                return

            # Create LADDER task
            ladder_task = await self.enhanced_planner.create_task(
                title=vscode_task.title,
                description=vscode_task.description,
                metadata={
                    "vscode_task_id": vscode_task.id,
                    "priority": vscode_task.priority,
                    "tags": vscode_task.tags,
                    "source": "vscode_todos",
                },
            )

            # Update VS Code task with LADDER task ID
            vscode_task.ladder_task_id = ladder_task["task_id"]
            vscode_task.ladder_stage = ladder_task.get("stage", "not_started")

            logger.info(f"Created LADDER task from VS Code: {vscode_task.title}")

        except Exception as e:
            logger.error(f"Error creating LADDER task from VS Code: {e}")

    async def _update_ladder_task_from_vscode(self, vscode_task: VSCodeTask):
        """Update LADDER planner task from VS Code task changes."""
        try:
            if not self.enhanced_planner or not vscode_task.ladder_task_id:
                return

            # Update LADDER task (if planner supports updates)
            # Note: Current enhanced planner doesn't have explicit update method
            # This would need to be implemented in the planner
            logger.debug(
                f"Would update LADDER task {vscode_task.ladder_task_id} from VS Code changes"
            )

        except Exception as e:
            logger.error(f"Error updating LADDER task from VS Code: {e}")

    async def _periodic_sync(self):
        """Periodic synchronization between planner and VS Code todos."""
        while True:
            try:
                await asyncio.sleep(self.sync_interval)

                # Sync tasks between planner and VS Code
                if self.enhanced_planner:
                    # Get current LADDER tasks
                    try:
                        ladder_tasks = await self.enhanced_planner.get_tasks()

                        # Sync any new LADDER tasks to VS Code
                        for ladder_task in ladder_tasks:
                            # Check if VS Code task exists for this LADDER task
                            ladder_task_id = ladder_task.get("task_id")
                            vscode_task_exists = any(
                                task.ladder_task_id == ladder_task_id
                                for task in self.tasks.values()
                            )

                            if not vscode_task_exists:
                                # Create VS Code task for this LADDER task
                                await self._on_ladder_task_created(
                                    {"task_data": ladder_task}
                                )

                    except Exception as e:
                        logger.debug(f"Error getting LADDER tasks during sync: {e}")

                logger.debug("Periodic sync completed")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in periodic sync: {e}")


# VS Code extension JavaScript/TypeScript interface
# This would be the companion code for the VS Code extension
VSCODE_EXTENSION_CODE = """
// VS Code extension code for LADDER planner integration
// File: extension.ts

import * as vscode from 'vscode';
import * as path from 'path';
import * as fs from 'fs';

interface LadderTask {
    id: string;
    title: string;
    description: string;
    completed: boolean;
    priority: string;
    createdAt: string;
    updatedAt: string;
    tags: string[];
    context: any;
}

class LadderTaskProvider implements vscode.TaskProvider {
    private tasks: LadderTask[] = [];
    private workspaceFolder: vscode.WorkspaceFolder;
    private pythonProcess?: any;

    constructor(workspaceFolder: vscode.WorkspaceFolder) {
        this.workspaceFolder = workspaceFolder;
    }

    async provideTasks(): Promise<vscode.Task[]> {
        // Get tasks from Python task provider
        const tasks = await this.getTasksFromPython();
        
        return tasks.map(task => {
            const definition: vscode.TaskDefinition = {
                type: 'ladder',
                task: task.id,
                title: task.title
            };

            const taskItem = new vscode.Task(
                definition,
                this.workspaceFolder,
                task.title,
                'ladder',
                new vscode.ShellExecution('echo', [`"Task: ${task.title}"`])
            );

            taskItem.detail = task.description;
            taskItem.group = vscode.TaskGroup.Build;
            
            return taskItem;
        });
    }

    async resolveTask(task: vscode.Task): Promise<vscode.Task | undefined> {
        // Resolve task definition
        return task;
    }

    private async getTasksFromPython(): Promise<LadderTask[]> {
        try {
            // Call Python task provider via MCP or direct communication
            const pythonPath = path.join(this.workspaceFolder.uri.fsPath, '.venv', 'Scripts', 'python.exe');
            const scriptPath = path.join(this.workspaceFolder.uri.fsPath, 'src', 'vscode_integration', 'task_provider.py');
            
            // This would use the MCP server or direct Python execution
            // For now, read from todos.json as fallback
            const todosPath = path.join(this.workspaceFolder.uri.fsPath, '.vscode', 'todos.json');
            
            if (fs.existsSync(todosPath)) {
                const todosData = JSON.parse(fs.readFileSync(todosPath, 'utf8'));
                return todosData.todoList?.map((todo: any) => ({
                    id: todo.id.toString(),
                    title: todo.title,
                    description: todo.description,
                    completed: todo.status === 'completed',
                    priority: 'medium',
                    createdAt: new Date().toISOString(),
                    updatedAt: new Date().toISOString(),
                    tags: [],
                    context: {}
                })) || [];
            }
            
            return [];
        } catch (error) {
            console.error('Error getting tasks from Python:', error);
            return [];
        }
    }
}

export function activate(context: vscode.ExtensionContext) {
    // Register task provider for LADDER planner
    const workspaceFolder = vscode.workspace.workspaceFolders?.[0];
    if (workspaceFolder) {
        const taskProvider = new LadderTaskProvider(workspaceFolder);
        const disposable = vscode.tasks.registerTaskProvider('ladder', taskProvider);
        context.subscriptions.push(disposable);
    }

    // Register commands for task management
    const createTaskCommand = vscode.commands.registerCommand('ladder.createTask', async () => {
        const title = await vscode.window.showInputBox({
            prompt: 'Enter task title',
            placeHolder: 'Task title'
        });
        
        if (title) {
            const description = await vscode.window.showInputBox({
                prompt: 'Enter task description',
                placeHolder: 'Task description'
            });
            
            // Create task via Python task provider
            // This would communicate with the Python backend
            vscode.window.showInformationMessage(`Created task: ${title}`);
        }
    });

    context.subscriptions.push(createTaskCommand);

    // Watch for changes to todos.json and refresh tasks
    const todosWatcher = vscode.workspace.createFileSystemWatcher('**/.vscode/todos.json');
    todosWatcher.onDidChange(() => {
        // Refresh task provider
        vscode.commands.executeCommand('workbench.action.tasks.runTask');
    });
    context.subscriptions.push(todosWatcher);
}
"""
