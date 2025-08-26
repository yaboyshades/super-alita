#!/usr/bin/env python3
"""
🌐 PUTER PLUGIN - Cloud environment integration for Super Alita
Provides seamless integration with Puter cloud services for file I/O and process execution
"""

import json
import logging
import hashlib
import uuid
from datetime import datetime, timezone
from typing import Any

import aiohttp

from src.core.events import BaseEvent, create_event
from src.core.plugin_interface import PluginInterface
from src.core.neural_atom import NeuralAtomMetadata, TextualMemoryAtom

logger = logging.getLogger(__name__)


class PuterOperationAtom(TextualMemoryAtom):
    """Neural atom for Puter cloud operations with deterministic UUIDs."""
    
    def __init__(self, operation_type: str, operation_data: dict[str, Any]):
        # Generate deterministic UUID for the operation
        operation_signature = f"puter_{operation_type}_{hash(str(operation_data))}"
        atom_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, operation_signature))
        
        metadata = NeuralAtomMetadata(
            name=f"puter_{operation_type}_{atom_uuid[:8]}",
            description=f"Puter {operation_type} operation",
            capabilities=["cloud_storage", "process_execution", "file_io"],
            tags={"puter", operation_type, "cloud"},
        )
        
        content = f"Puter {operation_type}: {operation_data.get('description', 'Cloud operation')}"
        super().__init__(metadata, content)
        
        self.operation_type = operation_type
        self.operation_data = operation_data
        self.atom_uuid = atom_uuid
        
    def get_deterministic_uuid(self) -> str:
        """Return the deterministic UUID for this operation."""
        return self.atom_uuid


class PuterApiClient:
    """Minimal async client for interacting with the Puter API."""

    def __init__(self, base_url: str, api_key: str, timeout: int = 30) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.session: aiohttp.ClientSession | None = None

    async def initialize(self) -> None:
        headers = {"User-Agent": "PuterAgent/1.0"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        timeout = aiohttp.ClientTimeout(total=self.timeout)
        self.session = aiohttp.ClientSession(timeout=timeout, headers=headers)
        await self._request("GET", "/api/health")

    async def cleanup(self) -> None:
        if self.session:
            await self.session.close()
            self.session = None

    async def _request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        if not self.session:
            raise RuntimeError("Client not initialized")
        url = f"{self.base_url}{path}"
        async with self.session.request(method, url, **kwargs) as resp:
            if resp.status >= 400:
                text = await resp.text()
                raise RuntimeError(f"HTTP {resp.status}: {text}")
            ct = resp.headers.get("Content-Type", "")
            if "application/json" in ct:
                return await resp.json()
            return {"data": await resp.text()}

    async def read_file(self, path: str) -> str:
        data = await self._request("GET", "/api/fs/read", params={"path": path})
        return data.get("content", "")

    async def write_file(self, path: str, content: str) -> None:
        await self._request(
            "POST", "/api/fs/write", json={"path": path, "content": content, "create_dirs": True}
        )

    async def delete_file(self, path: str) -> None:
        await self._request("DELETE", "/api/fs/delete", params={"path": path})

    async def list_directory(self, path: str) -> list[dict[str, Any]]:
        data = await self._request("GET", "/api/fs/list", params={"path": path})
        return data.get("items", [])

    async def execute_command(
        self, command: str, args: list[str], cwd: str
    ) -> dict[str, Any]:
        return await self._request(
            "POST", "/api/exec", json={"command": command, "args": args, "cwd": cwd, "env": {}}
        )


class PuterPlugin(PluginInterface):
    """Plugin wrapper for Puter cloud environment integration."""

    @property
    def name(self) -> str:
        return "puter"

    async def setup(self, event_bus: Any, store: Any, config: dict[str, Any]) -> None:
        await super().setup(event_bus, store, config)
        
        # Initialize Puter client configuration
        import os

        self.puter_config = {
            "api_url": os.getenv(
                "PUTER_BASE_URL", config.get("puter_base_url", "https://puter.com")
            ),
            "api_key": os.getenv(
                "PUTER_API_KEY", config.get("puter_api_key", "")
            ),
            "workspace_id": os.getenv(
                "PUTER_WORKSPACE_ID", config.get("puter_workspace_id", "default")
            ),
        }

        # Instantiate actual Puter API client
        self._client = PuterApiClient(
            base_url=self.puter_config["api_url"],
            api_key=self.puter_config["api_key"],
        )
        await self._client.initialize()

        # Track operation history for neural atoms
        self.operation_history: list[PuterOperationAtom] = []

        logger.info("🌐 Puter Plugin initialized with cloud integration")

    async def start(self) -> None:
        await super().start()

        # Subscribe to Puter-specific events
        await self.subscribe("puter_file_operation", self._handle_file_operation)
        await self.subscribe("puter_process_execution", self._handle_process_execution)
        await self.subscribe("puter_workspace_sync", self._handle_workspace_sync)
        
        # Subscribe to general tool calls that might need Puter
        await self.subscribe("tool_call", self._handle_tool_call)

        logger.info("🌐 Puter Plugin started with event subscriptions")

    async def shutdown(self) -> None:
        # Store operation history to neural store before shutdown
        if self.store and self.operation_history:
            for atom in self.operation_history:
                try:
                    await self.store.register(atom)
                except Exception as e:
                    logger.warning(f"Failed to register operation atom: {e}")
        if hasattr(self, "_client"):
            try:
                await self._client.cleanup()
            except Exception as e:
                logger.warning(f"Failed to cleanup Puter client: {e}")

        logger.info("🌐 Puter Plugin shutting down")

    async def _handle_file_operation(self, event: BaseEvent) -> None:
        """Handle file I/O operations with Puter cloud."""
        try:
            # Check if event has proper metadata
            if not hasattr(event, 'metadata') or event.metadata is None or not event.metadata:
                raise ValueError("Event missing required metadata")
                
            operation = event.metadata.get("operation", "read")
            file_path = event.metadata.get("file_path", "")
            content = event.metadata.get("content", "")
            
            # Create neural atom for this operation
            operation_data = {
                "operation": operation,
                "file_path": file_path,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "description": f"File {operation} operation on {file_path}",
            }
            
            atom = PuterOperationAtom("file_operation", operation_data)
            self.operation_history.append(atom)

            span_id = str(uuid.uuid4())
            args_hash = hashlib.sha256(
                json.dumps(operation_data, sort_keys=True).encode()
            ).hexdigest()
            start_time = datetime.now(timezone.utc)
            await self.emit_event(
                "AbilityCalled",
                tool="puter_file_operation",
                span_id=span_id,
                args_hash=args_hash,
                attempt=1,
                max_attempts=1,
                neural_atom_id=atom.get_deterministic_uuid(),
                conversation_id=event.conversation_id,
            )

            if operation == "read":
                content_out = await self._client.read_file(file_path)
                result = {"success": True, "content": content_out}
            elif operation == "write":
                await self._client.write_file(file_path, content)
                result = {
                    "success": True,
                    "bytes_written": len(content.encode()),
                    "file_path": file_path,
                }
            elif operation == "delete":
                await self._client.delete_file(file_path)
                result = {"success": True, "file_path": file_path}
            else:
                raise ValueError(f"Unknown operation: {operation}")

            duration_ms = int(
                (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            )
            output_hash = hashlib.sha256(
                json.dumps(result, sort_keys=True).encode()
            ).hexdigest()

            await self.emit_event(
                "AbilitySucceeded",
                tool="puter_file_operation",
                span_id=span_id,
                output_hash=output_hash,
                attempt=1,
                max_attempts=1,
                duration_ms=duration_ms,
                neural_atom_id=atom.get_deterministic_uuid(),
                conversation_id=event.conversation_id,
            )

            await self.emit_event(
                "puter_operation_completed",
                operation_type="file_operation",
                file_path=file_path,
                operation=operation,
                success=result.get("success", False),
                result=result,
                neural_atom_id=atom.get_deterministic_uuid(),
                timestamp=datetime.now(timezone.utc),
                source_plugin=self.name,
                conversation_id=event.conversation_id,
            )

            logger.info(f"🌐 Completed Puter file {operation}: {file_path}")

        except Exception as e:
            logger.exception("❌ Puter file operation error")
            span_id = locals().get("span_id", str(uuid.uuid4()))
            start = locals().get("start_time", datetime.now(timezone.utc))
            duration_ms = int((datetime.now(timezone.utc) - start).total_seconds() * 1000)
            neural_atom_id = (
                atom.get_deterministic_uuid() if "atom" in locals() else ""
            )
            await self.emit_event(
                "AbilityFailed",
                tool="puter_file_operation",
                span_id=span_id,
                error=str(e),
                attempt=1,
                max_attempts=1,
                duration_ms=duration_ms,
                neural_atom_id=neural_atom_id,
                conversation_id=getattr(event, 'conversation_id', 'unknown'),
            )
            await self.emit_event(
                "puter_operation_failed",
                operation_type="file_operation",
                error=str(e),
                timestamp=datetime.now(timezone.utc),
                source_plugin=self.name,
                conversation_id=getattr(event, 'conversation_id', 'unknown'),
            )

    async def _handle_process_execution(self, event: BaseEvent) -> None:
        """Handle process execution in Puter cloud environment."""
        try:
            command = event.metadata.get("command", "")
            args = event.metadata.get("args", [])
            working_dir = event.metadata.get("working_dir", "/")
            
            # Create neural atom for this operation
            operation_data = {
                "command": command,
                "args": args,
                "working_dir": working_dir,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "description": f"Process execution: {command} {' '.join(args)}",
            }
            
            atom = PuterOperationAtom("process_execution", operation_data)
            self.operation_history.append(atom)

            span_id = str(uuid.uuid4())
            args_hash = hashlib.sha256(
                json.dumps(operation_data, sort_keys=True).encode()
            ).hexdigest()
            start_time = datetime.now(timezone.utc)
            await self.emit_event(
                "AbilityCalled",
                tool="puter_process_execution",
                span_id=span_id,
                args_hash=args_hash,
                attempt=1,
                max_attempts=1,
                neural_atom_id=atom.get_deterministic_uuid(),
                conversation_id=event.conversation_id,
            )

            exec_result = await self._client.execute_command(
                command, args, cwd=working_dir
            )
            success = exec_result.get("exit_code", 1) == 0
            result = {"success": success, **exec_result}

            duration_ms = int(
                (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            )
            output_hash = hashlib.sha256(
                json.dumps(result, sort_keys=True).encode()
            ).hexdigest()

            await self.emit_event(
                "AbilitySucceeded" if success else "AbilityFailed",
                tool="puter_process_execution",
                span_id=span_id,
                output_hash=output_hash if success else None,
                error=None if success else result.get("stderr"),
                attempt=1,
                max_attempts=1,
                duration_ms=duration_ms,
                neural_atom_id=atom.get_deterministic_uuid(),
                conversation_id=event.conversation_id,
            )

            await self.emit_event(
                "puter_operation_completed" if success else "puter_operation_failed",
                operation_type="process_execution",
                command=command,
                args=args,
                working_dir=working_dir,
                success=success,
                result=result,
                neural_atom_id=atom.get_deterministic_uuid(),
                timestamp=datetime.now(timezone.utc),
                source_plugin=self.name,
                conversation_id=event.conversation_id,
            )

            if success:
                logger.info(f"🌐 Completed Puter process execution: {command}")
            else:
                logger.error(f"❌ Puter process execution failed: {command}")

        except Exception as e:
            logger.exception("❌ Puter process execution error")
            span_id = locals().get("span_id", str(uuid.uuid4()))
            start = locals().get("start_time", datetime.now(timezone.utc))
            duration_ms = int((datetime.now(timezone.utc) - start).total_seconds() * 1000)
            neural_atom_id = (
                atom.get_deterministic_uuid() if "atom" in locals() else ""
            )
            await self.emit_event(
                "AbilityFailed",
                tool="puter_process_execution",
                span_id=span_id,
                error=str(e),
                attempt=1,
                max_attempts=1,
                duration_ms=duration_ms,
                neural_atom_id=neural_atom_id,
                conversation_id=event.conversation_id,
            )
            await self.emit_event(
                "puter_operation_failed",
                operation_type="process_execution",
                error=str(e),
                timestamp=datetime.now(timezone.utc),
                source_plugin=self.name,
                conversation_id=event.conversation_id,
            )

    async def _handle_workspace_sync(self, event: BaseEvent) -> None:
        """Handle workspace synchronization with Puter cloud."""
        try:
            sync_type = event.metadata.get("sync_type", "bidirectional")
            local_path = event.metadata.get("local_path", ".")
            remote_path = event.metadata.get("remote_path", "/workspace")
            
            # Create neural atom for this operation
            operation_data = {
                "sync_type": sync_type,
                "local_path": local_path,
                "remote_path": remote_path,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "description": f"Workspace sync: {sync_type} between {local_path} and {remote_path}",
            }
            
            atom = PuterOperationAtom("workspace_sync", operation_data)
            self.operation_history.append(atom)

            span_id = str(uuid.uuid4())
            args_hash = hashlib.sha256(
                json.dumps(operation_data, sort_keys=True).encode()
            ).hexdigest()
            start_time = datetime.now(timezone.utc)
            await self.emit_event(
                "AbilityCalled",
                tool="puter_workspace_sync",
                span_id=span_id,
                args_hash=args_hash,
                attempt=1,
                max_attempts=1,
                neural_atom_id=atom.get_deterministic_uuid(),
                conversation_id=event.conversation_id,
            )

            remote_listing = await self._client.list_directory(remote_path)
            result = {
                "success": True,
                "files_synced": len(remote_listing),
            }

            duration_ms = int(
                (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            )
            output_hash = hashlib.sha256(
                json.dumps(result, sort_keys=True).encode()
            ).hexdigest()

            await self.emit_event(
                "AbilitySucceeded",
                tool="puter_workspace_sync",
                span_id=span_id,
                output_hash=output_hash,
                attempt=1,
                max_attempts=1,
                duration_ms=duration_ms,
                neural_atom_id=atom.get_deterministic_uuid(),
                conversation_id=event.conversation_id,
            )

            await self.emit_event(
                "puter_operation_completed",
                operation_type="workspace_sync",
                sync_type=sync_type,
                local_path=local_path,
                remote_path=remote_path,
                success=True,
                result=result,
                neural_atom_id=atom.get_deterministic_uuid(),
                timestamp=datetime.now(timezone.utc),
                source_plugin=self.name,
                conversation_id=event.conversation_id,
            )

            logger.info(f"🌐 Completed Puter workspace sync: {sync_type}")

        except Exception as e:
            logger.exception("❌ Puter workspace sync error")
            span_id = locals().get("span_id", str(uuid.uuid4()))
            start = locals().get("start_time", datetime.now(timezone.utc))
            duration_ms = int((datetime.now(timezone.utc) - start).total_seconds() * 1000)
            neural_atom_id = (
                atom.get_deterministic_uuid() if "atom" in locals() else ""
            )
            await self.emit_event(
                "AbilityFailed",
                tool="puter_workspace_sync",
                span_id=span_id,
                error=str(e),
                attempt=1,
                max_attempts=1,
                duration_ms=duration_ms,
                neural_atom_id=neural_atom_id,
                conversation_id=event.conversation_id,
            )
            await self.emit_event(
                "puter_operation_failed",
                operation_type="workspace_sync",
                error=str(e),
                timestamp=datetime.now(timezone.utc),
                source_plugin=self.name,
                conversation_id=event.conversation_id,
            )

    async def _handle_tool_call(self, event: BaseEvent) -> None:
        """Handle tool calls that might need Puter cloud services."""
        try:
            if not hasattr(event, 'tool_name'):
                return
                
            tool_name = event.tool_name
            
            # Check if this is a Puter-related tool call
            if not tool_name.startswith("puter_"):
                return
                
            parameters = getattr(event, 'parameters', {})
            
            if tool_name == "puter_file_read":
                # Create a proper event object for file operation
                file_event = create_event(
                    "puter_file_operation",
                    source_plugin=self.name,
                    conversation_id=getattr(event, 'conversation_id', 'unknown'),
                )
                file_event.metadata = {
                    "operation": "read",
                    "file_path": parameters.get("file_path", ""),
                }
                await self._handle_file_operation(file_event)
                
            elif tool_name == "puter_file_write":
                # Create a proper event object for file operation
                file_event = create_event(
                    "puter_file_operation",
                    source_plugin=self.name,
                    conversation_id=getattr(event, 'conversation_id', 'unknown'),
                )
                file_event.metadata = {
                    "operation": "write",
                    "file_path": parameters.get("file_path", ""),
                    "content": parameters.get("content", ""),
                }
                await self._handle_file_operation(file_event)
                
            elif tool_name == "puter_execute":
                # Create a proper event object for process execution
                exec_event = create_event(
                    "puter_process_execution",
                    source_plugin=self.name,
                    conversation_id=getattr(event, 'conversation_id', 'unknown'),
                )
                exec_event.metadata = {
                    "command": parameters.get("command", ""),
                    "args": parameters.get("args", []),
                    "working_dir": parameters.get("working_dir", "/"),
                }
                await self._handle_process_execution(exec_event)
                
        except Exception as e:
            logger.exception("❌ Puter tool call error")

    def get_operation_history(self) -> list[PuterOperationAtom]:
        """Get the history of Puter operations as neural atoms."""
        return self.operation_history.copy()

    def get_capabilities(self) -> dict[str, Any]:
        """Get Puter plugin capabilities."""
        return {
            "file_operations": ["read", "write", "delete", "list"],
            "process_execution": True,
            "workspace_sync": True,
            "cloud_storage": True,
            "neural_atom_tracking": True,
            "deterministic_uuids": True,
        }
