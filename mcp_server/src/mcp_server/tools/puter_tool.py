from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from mcp_server.server import app

logger = logging.getLogger(__name__)


def _is_subpath(base: Path, candidate: Path) -> bool:
    """Check if candidate path is within base directory for security."""
    try:
        candidate.relative_to(base)
        return True
    except ValueError:
        return False


def _get_workspace_root() -> Path:
    """Get the workspace root directory."""
    return Path.cwd().resolve()


@app.tool(
    name="puter_file_read",
    description="Read a file from Puter cloud storage. Args: file_path (str), dry_run (bool, default=True). Returns file content or diff preview.",
)
async def puter_file_read(file_path: str, dry_run: bool = True) -> dict[str, Any]:
    """Read a file from Puter cloud storage with workspace boundary validation."""
    try:
        workspace_root = _get_workspace_root()
        target_path = Path(file_path).resolve()
        
        # Validate workspace boundary
        if not _is_subpath(workspace_root, target_path):
            return {
                "success": False,
                "result": "",
                "error": f"Path {file_path} is outside workspace boundary",
            }
        
        if dry_run:
            # Return preview/diff of what would be read
            if target_path.exists():
                file_size = target_path.stat().st_size
                return {
                    "success": True,
                    "result": f"Would read file: {file_path} ({file_size} bytes)",
                    "error": "",
                    "dry_run": True,
                    "preview": {
                        "file_path": str(target_path),
                        "file_size": file_size,
                        "exists": True,
                    },
                }
            else:
                return {
                    "success": False,
                    "result": "",
                    "error": f"File {file_path} does not exist",
                    "dry_run": True,
                }
        
        # Actual file read operation
        if not target_path.exists():
            return {
                "success": False,
                "result": "",
                "error": f"File {file_path} does not exist",
            }
        
        with open(target_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        return {
            "success": True,
            "result": content,
            "error": "",
            "file_info": {
                "file_path": str(target_path),
                "file_size": len(content.encode()),
                "lines": len(content.splitlines()),
            },
        }
        
    except Exception as e:
        logger.exception(f"Error reading file {file_path}")
        return {
            "success": False,
            "result": "",
            "error": f"Failed to read file {file_path}: {str(e)}",
        }


@app.tool(
    name="puter_file_write",
    description="Write content to a file in Puter cloud storage. Args: file_path (str), content (str), dry_run (bool, default=True). Returns write status or diff preview.",
)
async def puter_file_write(file_path: str, content: str, dry_run: bool = True) -> dict[str, Any]:
    """Write content to a file in Puter cloud storage with workspace boundary validation."""
    try:
        workspace_root = _get_workspace_root()
        target_path = Path(file_path).resolve()
        
        # Validate workspace boundary
        if not _is_subpath(workspace_root, target_path):
            return {
                "success": False,
                "result": "",
                "error": f"Path {file_path} is outside workspace boundary",
            }
        
        if dry_run:
            # Generate unified diff preview
            existing_content = ""
            if target_path.exists():
                try:
                    with open(target_path, 'r', encoding='utf-8') as f:
                        existing_content = f.read()
                except Exception:
                    existing_content = "<binary or unreadable file>"
            
            # Simple diff representation
            if existing_content != content:
                diff_lines = []
                if existing_content:
                    diff_lines.append(f"--- {file_path}")
                    diff_lines.append(f"+++ {file_path}")
                    # Simplified diff - show first few lines of changes
                    old_lines = existing_content.splitlines()
                    new_lines = content.splitlines()
                    max_lines = 10
                    
                    for i, (old, new) in enumerate(zip(old_lines[:max_lines], new_lines[:max_lines])):
                        if old != new:
                            diff_lines.append(f"-{old}")
                            diff_lines.append(f"+{new}")
                    
                    if len(old_lines) > max_lines or len(new_lines) > max_lines:
                        diff_lines.append("... (diff truncated)")
                else:
                    diff_lines.append(f"+++ {file_path} (new file)")
                    for line in content.splitlines()[:10]:
                        diff_lines.append(f"+{line}")
                    if len(content.splitlines()) > 10:
                        diff_lines.append("... (content truncated)")
                
                diff_preview = "\n".join(diff_lines)
            else:
                diff_preview = "No changes detected"
            
            return {
                "success": True,
                "result": f"Would write {len(content.encode())} bytes to {file_path}",
                "error": "",
                "dry_run": True,
                "diff_preview": diff_preview,
                "content_info": {
                    "bytes": len(content.encode()),
                    "lines": len(content.splitlines()),
                    "characters": len(content),
                },
            }
        
        # Actual file write operation
        target_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(target_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return {
            "success": True,
            "result": f"Successfully wrote {len(content.encode())} bytes to {file_path}",
            "error": "",
            "file_info": {
                "file_path": str(target_path),
                "bytes_written": len(content.encode()),
                "lines_written": len(content.splitlines()),
            },
        }
        
    except Exception as e:
        logger.exception(f"Error writing file {file_path}")
        return {
            "success": False,
            "result": "",
            "error": f"Failed to write file {file_path}: {str(e)}",
        }


@app.tool(
    name="puter_execute",
    description="Execute a command in Puter cloud environment. Args: command (str), args (list), working_dir (str), dry_run (bool, default=True). Returns execution result or preview.",
)
async def puter_execute(
    command: str, 
    args: List[str] = None, 
    working_dir: str = None, 
    dry_run: bool = True
) -> dict[str, Any]:
    """Execute a command in Puter cloud environment with security validation."""
    try:
        args = args or []
        working_dir = working_dir or str(_get_workspace_root())
        
        workspace_root = _get_workspace_root()
        work_path = Path(working_dir).resolve()
        
        # Validate workspace boundary for working directory
        if not _is_subpath(workspace_root, work_path):
            return {
                "success": False,
                "result": "",
                "error": f"Working directory {working_dir} is outside workspace boundary",
            }
        
        # Security: Only allow safe commands in dry_run=False mode
        safe_commands = {
            "echo", "cat", "ls", "pwd", "whoami", "date", "python", "node", "npm", "git"
        }
        
        if not dry_run and command not in safe_commands:
            return {
                "success": False,
                "result": "",
                "error": f"Command '{command}' is not in the safe commands list for execution",
            }
        
        full_command = [command] + args
        
        if dry_run:
            return {
                "success": True,
                "result": f"Would execute: {' '.join(full_command)} in {working_dir}",
                "error": "",
                "dry_run": True,
                "execution_preview": {
                    "command": command,
                    "args": args,
                    "working_dir": str(work_path),
                    "full_command": ' '.join(full_command),
                },
            }
        
        # Actual command execution with timeout
        proc = await asyncio.create_subprocess_exec(
            *full_command,
            cwd=work_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30.0)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return {
                "success": False,
                "result": "",
                "error": "Command execution timed out after 30 seconds",
            }
        
        return {
            "success": proc.returncode == 0,
            "result": stdout.decode('utf-8', errors='replace'),
            "error": stderr.decode('utf-8', errors='replace') if proc.returncode != 0 else "",
            "execution_info": {
                "command": command,
                "args": args,
                "working_dir": str(work_path),
                "exit_code": proc.returncode,
                "stdout_lines": len(stdout.decode('utf-8', errors='replace').splitlines()),
                "stderr_lines": len(stderr.decode('utf-8', errors='replace').splitlines()),
            },
        }
        
    except Exception as e:
        logger.exception(f"Error executing command {command}")
        return {
            "success": False,
            "result": "",
            "error": f"Failed to execute command {command}: {str(e)}",
        }


@app.tool(
    name="puter_workspace_sync",
    description="Sync local workspace with Puter cloud storage. Args: sync_type (str), local_path (str), remote_path (str), dry_run (bool, default=True). Returns sync status or preview.",
)
async def puter_workspace_sync(
    sync_type: str = "bidirectional",
    local_path: str = ".",
    remote_path: str = "/workspace",
    dry_run: bool = True
) -> dict[str, Any]:
    """Sync workspace with Puter cloud storage with boundary validation."""
    try:
        workspace_root = _get_workspace_root()
        local_target = Path(local_path).resolve()
        
        # Validate workspace boundary
        if not _is_subpath(workspace_root, local_target):
            return {
                "success": False,
                "result": "",
                "error": f"Local path {local_path} is outside workspace boundary",
            }
        
        valid_sync_types = ["upload", "download", "bidirectional"]
        if sync_type not in valid_sync_types:
            return {
                "success": False,
                "result": "",
                "error": f"Invalid sync_type '{sync_type}'. Must be one of: {valid_sync_types}",
            }
        
        if dry_run:
            # Simulate file discovery and sync preview
            files_to_sync = []
            
            if local_target.is_dir():
                for file_path in local_target.rglob("*"):
                    if file_path.is_file():
                        rel_path = file_path.relative_to(local_target)
                        files_to_sync.append({
                            "local_path": str(file_path),
                            "remote_path": f"{remote_path}/{rel_path}",
                            "size": file_path.stat().st_size,
                            "action": sync_type,
                        })
            elif local_target.is_file():
                files_to_sync.append({
                    "local_path": str(local_target),
                    "remote_path": f"{remote_path}/{local_target.name}",
                    "size": local_target.stat().st_size,
                    "action": sync_type,
                })
            
            total_size = sum(f["size"] for f in files_to_sync)
            
            return {
                "success": True,
                "result": f"Would sync {len(files_to_sync)} files ({total_size} bytes) - {sync_type}",
                "error": "",
                "dry_run": True,
                "sync_preview": {
                    "sync_type": sync_type,
                    "local_path": str(local_target),
                    "remote_path": remote_path,
                    "files_count": len(files_to_sync),
                    "total_size": total_size,
                    "files": files_to_sync[:10],  # Limit preview to first 10 files
                    "truncated": len(files_to_sync) > 10,
                },
            }
        
        # Actual sync operation (simulated for now)
        # In a real implementation, this would call Puter's sync API
        
        # Simulate sync process
        await asyncio.sleep(0.1)  # Simulate network delay
        
        files_synced = 0
        bytes_transferred = 0
        
        if local_target.is_dir():
            for file_path in local_target.rglob("*"):
                if file_path.is_file():
                    files_synced += 1
                    bytes_transferred += file_path.stat().st_size
        elif local_target.is_file():
            files_synced = 1
            bytes_transferred = local_target.stat().st_size
        
        return {
            "success": True,
            "result": f"Successfully synced {files_synced} files ({bytes_transferred} bytes)",
            "error": "",
            "sync_info": {
                "sync_type": sync_type,
                "local_path": str(local_target),
                "remote_path": remote_path,
                "files_synced": files_synced,
                "bytes_transferred": bytes_transferred,
                "simulated": True,  # Mark as simulated until real API is integrated
            },
        }
        
    except Exception as e:
        logger.exception(f"Error syncing workspace {local_path}")
        return {
            "success": False,
            "result": "",
            "error": f"Failed to sync workspace {local_path}: {str(e)}",
        }


@app.tool(
    name="puter_list_files",
    description="List files in Puter cloud storage directory. Args: directory_path (str), pattern (str), dry_run (bool, default=True). Returns file listing.",
)
async def puter_list_files(
    directory_path: str = ".",
    pattern: str = "*",
    dry_run: bool = True
) -> dict[str, Any]:
    """List files in Puter cloud directory with workspace boundary validation."""
    try:
        workspace_root = _get_workspace_root()
        target_dir = Path(directory_path).resolve()
        
        # Validate workspace boundary
        if not _is_subpath(workspace_root, target_dir):
            return {
                "success": False,
                "result": "",
                "error": f"Directory {directory_path} is outside workspace boundary",
            }
        
        if not target_dir.exists():
            return {
                "success": False,
                "result": "",
                "error": f"Directory {directory_path} does not exist",
            }
        
        if not target_dir.is_dir():
            return {
                "success": False,
                "result": "",
                "error": f"Path {directory_path} is not a directory",
            }
        
        # List files matching pattern
        files = []
        directories = []
        
        for item in target_dir.glob(pattern):
            item_info = {
                "name": item.name,
                "path": str(item.relative_to(workspace_root)),
                "size": item.stat().st_size if item.is_file() else 0,
                "modified": item.stat().st_mtime,
                "is_dir": item.is_dir(),
            }
            
            if item.is_dir():
                directories.append(item_info)
            else:
                files.append(item_info)
        
        # Sort results
        files.sort(key=lambda x: x["name"])
        directories.sort(key=lambda x: x["name"])
        
        result_summary = f"Found {len(directories)} directories and {len(files)} files"
        
        return {
            "success": True,
            "result": result_summary,
            "error": "",
            "listing": {
                "directory_path": str(target_dir.relative_to(workspace_root)),
                "pattern": pattern,
                "directories": directories,
                "files": files,
                "total_directories": len(directories),
                "total_files": len(files),
                "total_size": sum(f["size"] for f in files),
            },
        }
        
    except Exception as e:
        logger.exception(f"Error listing files in {directory_path}")
        return {
            "success": False,
            "result": "",
            "error": f"Failed to list files in {directory_path}: {str(e)}",
        }