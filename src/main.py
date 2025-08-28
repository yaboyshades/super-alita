#!/usr/bin/env python3
"""FastAPI entrypoint for the REUG runtime."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import hashlib
import json
import logging
import os
import re
import sys
import urllib.request
from collections.abc import AsyncGenerator, Callable
from logging.config import dictConfig
from pathlib import Path
from typing import Any
from uuid import uuid4

# Add conditional imports for FastAPI dependencies
try:
    import uvicorn
    from fastapi import APIRouter, Body, FastAPI, Request
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse, StreamingResponse

    FASTAPI_AVAILABLE = True
except ImportError:
    # Create stub classes when FastAPI is not available
    FastAPI = None  # type: ignore
    Request = None  # type: ignore
    APIRouter = None  # type: ignore
    Body = None  # type: ignore
    CORSMiddleware = None  # type: ignore
    JSONResponse = None  # type: ignore
    StreamingResponse = None  # type: ignore
    uvicorn = None  # type: ignore
    FASTAPI_AVAILABLE = False

# Event bus imports (moved up to avoid E402)
from src.core.events import create_event
from src.reug_runtime.event_bus import BaseEventBus, FileEventBus, make_event_bus
from src.reug_runtime.llm_client import LLMClient, get_llm_client

logger = logging.getLogger(__name__)


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:  # pragma: no cover - simple
        data = {
            "time": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage(),
        }
        return json.dumps(data, ensure_ascii=False)


def _configure_logging() -> Path:
    log_dir = Path(os.getenv("REUG_LOG_DIR", "./logs"))
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "runtime.log"
    dictConfig(
        {
            "version": 1,
            "formatters": {"json": {"()": JsonFormatter}},
            "handlers": {
                "file": {
                    "class": "logging.FileHandler",
                    "filename": str(log_file),
                    "formatter": "json",
                    "encoding": "utf-8",
                },
                "console": {
                    "class": "logging.StreamHandler",
                    "formatter": "json",
                },
            },
            "root": {
                "level": os.getenv("REUG_LOG_LEVEL", "INFO"),
                "handlers": ["file", "console"],
            },
        }
    )
    return log_file


def _hash_json(obj: Any) -> str:
    with contextlib.suppress(Exception):
        h = hashlib.sha256(json.dumps(obj, sort_keys=True).encode("utf-8")).hexdigest()
        return h[:16]
    return "na"


# --- Resolve reug_runtime from local src if not installed ---
ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
# Ensure the PROJECT ROOT (parent of 'src') is on sys.path so that
# package imports like 'src.core.events' resolve. Previously we only
# inserted the 'src' directory itself which makes top-level packages
# (core, agents, etc.) importable, but breaks fully-qualified
# 'src.*' imports used throughout the codebase.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# (Optionally also ensure direct 'src' path for simpler 'core.*' imports)
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# REUG runtime routers (streaming agent + toolbox)
try:
    from reug_runtime.config import SETTINGS
    from reug_runtime.router import router as agent_router
    from reug_runtime.router_tools import tools as tools_router
except Exception as e:  # pragma: no cover
    # Fallback: minimal routers to allow boot/health during development
    print("[WARN] reug_runtime import failed; falling back to minimal routers:", e)

    if not FASTAPI_AVAILABLE:
        # Create minimal stubs when FastAPI is not available
        agent_router = None  # type: ignore
        tools_router = None  # type: ignore
        SETTINGS = type("Settings", (), {"api_prefix": ""})()  # type: ignore
    else:
        agent_router = APIRouter(prefix="/v1", tags=["agent"])  # type: ignore

        @agent_router.post("/chat/stream")  # type: ignore
        async def chat_stream(_: Request) -> StreamingResponse:  # type: ignore
            async def gen() -> AsyncGenerator[str, None]:
                yield "Thinking... "
                yield '<final_answer>{"content":"hello","citations":[]}</final_answer>'

            return StreamingResponse(gen(), media_type="text/plain")  # type: ignore

        tools_router = APIRouter(prefix="/tools", tags=["tools"])  # type: ignore

        @tools_router.get("/health")  # type: ignore
        async def tools_health() -> dict[str, str]:
            return {"status": "ok"}

        # Create minimal settings
        SETTINGS = type("Settings", (), {"api_prefix": ""})()  # type: ignore


# --- Ability registry (minimal adapter; replace with your real one) ---
class SimpleAbilityRegistry:
    """
    Minimal, schema-friendly registry:
      - knows(): does this tool exist?
      - validate_args(): shallow "type-ish" validation
      - register(): dynamic tool creation (contract-first)
      - execute(): your dispatch to MCP / SDK / code
    """

    def __init__(self) -> None:
        # Seed with initial tools
        self._known: set[str] = {
            "echo",
            "brainstorm_mcp_stub",
            "fetch_github_raw",
            "secure_scan_code",
            "full_cycle_prototype",
            # DeepCode abilities
            "analyze_code_file",
            "analyze_code_directory",
            "get_code_quality_score",
            "detect_code_patterns",
            "analyze_workspace_context",
            "analyze_file_context",
            "check_file_support",
            "get_supported_extensions",
            "understand_code_structure",
            # Enhanced Copilot abilities
            "analyze_and_suggest_repos",
            "automated_problem_solver",
            "repository_deep_analysis",
            "enhanced_code_review",
            "discover_github_upgrades",
        }
        self._contracts: dict[str, dict[str, Any]] = {
            "echo": {
                "tool_id": "echo",
                "description": "Echo back the provided payload",
                "input_schema": {
                    "type": "object",
                    "properties": {"payload": {"type": "string"}},
                },
                "output_schema": {"type": "object"},
            },
            "brainstorm_mcp_stub": {
                "tool_id": "brainstorm_mcp_stub",
                "description": (
                    "Lightweight brainstorming helper returning a structured "
                    "idea list for a task."
                ),
                "input_schema": {
                    "type": "object",
                    "required": ["task"],
                    "properties": {"task": {"type": "string"}},
                },
                "output_schema": {
                    "type": "object",
                    "properties": {
                        "task": {"type": "string"},
                        "ideas": {"type": "array", "items": {"type": "string"}},
                    },
                },
            },
            "fetch_github_raw": {
                "tool_id": "fetch_github_raw",
                "description": (
                    "Fetch a raw file from GitHub (best-effort; graceful "
                    "fallback if network disabled)."
                ),
                "input_schema": {
                    "type": "object",
                    "required": ["owner", "repo", "path"],
                    "properties": {
                        "owner": {"type": "string"},
                        "repo": {"type": "string"},
                        "path": {"type": "string"},
                        "ref": {"type": "string"},
                        "truncate": {"type": "integer"},
                    },
                },
                "output_schema": {
                    "type": "object",
                    "properties": {
                        "content": {"type": "string"},
                        "url": {"type": "string"},
                        "truncated": {"type": "boolean"},
                        "error": {"type": "string"},
                    },
                },
            },
            "secure_scan_code": {
                "tool_id": "secure_scan_code",
                "description": (
                    "Naive static scan (Bandit-inspired heuristics) for "
                    "obvious risky patterns."
                ),
                "input_schema": {
                    "type": "object",
                    "required": ["code"],
                    "properties": {"code": {"type": "string"}},
                },
                "output_schema": {
                    "type": "object",
                    "properties": {
                        "issues": {"type": "array", "items": {"type": "object"}},
                        "issue_count": {"type": "integer"},
                    },
                },
            },
            "full_cycle_prototype": {
                "tool_id": "full_cycle_prototype",
                "description": (
                    "Synthetic build-run cycle: optional fetch then produce "
                    "run instructions."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "task": {"type": "string"},
                        "owner": {"type": "string"},
                        "repo": {"type": "string"},
                        "path": {"type": "string"},
                    },
                },
                "output_schema": {"type": "object"},
            },
            # DeepCode Analysis Tools
            "analyze_code_file": {
                "tool_id": "analyze_code_file",
                "description": "Perform deep analysis on a single code file",
                "input_schema": {
                    "type": "object",
                    "required": ["file_path"],
                    "properties": {
                        "file_path": {"type": "string"},
                        "analysis_level": {
                            "type": "string",
                            "enum": [
                                "syntax",
                                "semantic",
                                "security",
                                "performance",
                                "architecture",
                                "deep",
                            ],
                            "default": "semantic",
                        },
                    },
                },
                "output_schema": {"type": "object"},
            },
            "analyze_code_directory": {
                "tool_id": "analyze_code_directory",
                "description": "Analyze all supported code files in a directory",
                "input_schema": {
                    "type": "object",
                    "required": ["directory_path"],
                    "properties": {
                        "directory_path": {"type": "string"},
                        "analysis_level": {
                            "type": "string",
                            "enum": [
                                "syntax",
                                "semantic",
                                "security",
                                "performance",
                                "architecture",
                                "deep",
                            ],
                            "default": "semantic",
                        },
                        "generate_report": {"type": "boolean", "default": True},
                    },
                },
                "output_schema": {"type": "object"},
            },
            "get_code_quality_score": {
                "tool_id": "get_code_quality_score",
                "description": "Calculate quality score for code analysis results",
                "input_schema": {
                    "type": "object",
                    "required": ["target_path"],
                    "properties": {
                        "target_path": {"type": "string"},
                        "include_metrics": {"type": "boolean", "default": False},
                    },
                },
                "output_schema": {"type": "object"},
            },
            "detect_code_patterns": {
                "tool_id": "detect_code_patterns",
                "description": "Detect specific patterns and anti-patterns in code",
                "input_schema": {
                    "type": "object",
                    "required": ["file_path"],
                    "properties": {
                        "file_path": {"type": "string"},
                        "pattern_types": {
                            "type": "array",
                            "items": {"type": "string"},
                            "default": ["security", "performance"],
                        },
                    },
                },
                "output_schema": {"type": "object"},
            },
            # DeepCode Integration Tools
            "analyze_workspace_context": {
                "tool_id": "analyze_workspace_context",
                "description": "Analyze workspace to understand project structure and context",
                "input_schema": {
                    "type": "object",
                    "required": ["workspace_path"],
                    "properties": {
                        "workspace_path": {"type": "string"},
                        "max_files": {"type": "integer", "default": 20},
                        "include_quality_metrics": {"type": "boolean", "default": True},
                    },
                },
                "output_schema": {"type": "object"},
            },
            "analyze_file_context": {
                "tool_id": "analyze_file_context",
                "description": "Analyze a specific file with full context understanding",
                "input_schema": {
                    "type": "object",
                    "required": ["file_path"],
                    "properties": {
                        "file_path": {"type": "string"},
                        "include_suggestions": {"type": "boolean", "default": True},
                    },
                },
                "output_schema": {"type": "object"},
            },
            "check_file_support": {
                "tool_id": "check_file_support",
                "description": "Check if a file type is supported by deepcode analysis",
                "input_schema": {
                    "type": "object",
                    "required": ["file_path"],
                    "properties": {"file_path": {"type": "string"}},
                },
                "output_schema": {"type": "object"},
            },
            "get_supported_extensions": {
                "tool_id": "get_supported_extensions",
                "description": "Get list of file extensions supported by deepcode",
                "input_schema": {
                    "type": "object",
                    "properties": {},
                },
                "output_schema": {"type": "object"},
            },
            "understand_code_structure": {
                "tool_id": "understand_code_structure",
                "description": "Analyze code structure and provide architectural insights",
                "input_schema": {
                    "type": "object",
                    "required": ["target_path"],
                    "properties": {
                        "target_path": {"type": "string"},
                        "focus_area": {
                            "type": "string",
                            "enum": [
                                "architecture",
                                "dependencies",
                                "complexity",
                                "patterns",
                            ],
                            "default": "architecture",
                        },
                    },
                },
                "output_schema": {"type": "object"},
            },
            # Enhanced Copilot Tools
            "analyze_and_suggest_repos": {
                "tool_id": "analyze_and_suggest_repos",
                "description": "Analyze code problems and suggest GitHub repositories that can help solve them",
                "input_schema": {
                    "type": "object",
                    "required": ["problem_description"],
                    "properties": {
                        "problem_description": {"type": "string"},
                        "code_context": {"type": "string", "default": ""},
                        "language_preference": {"type": "string", "default": "python"},
                        "max_results": {"type": "integer", "default": 5},
                    },
                },
                "output_schema": {"type": "object"},
            },
            "automated_problem_solver": {
                "tool_id": "automated_problem_solver",
                "description": "End-to-end automated problem solver that finds repos, analyzes code, and provides implementation guidance",
                "input_schema": {
                    "type": "object",
                    "required": ["task_description"],
                    "properties": {
                        "task_description": {"type": "string"},
                        "workspace_path": {"type": "string", "default": "."},
                        "include_code_generation": {"type": "boolean", "default": True},
                        "analyze_existing_code": {"type": "boolean", "default": True},
                    },
                },
                "output_schema": {"type": "object"},
            },
            "repository_deep_analysis": {
                "tool_id": "repository_deep_analysis",
                "description": "Perform deep analysis on a specific GitHub repository to understand its capabilities",
                "input_schema": {
                    "type": "object",
                    "required": ["repo_url"],
                    "properties": {
                        "repo_url": {"type": "string"},
                        "analysis_focus": {
                            "type": "string",
                            "enum": [
                                "architecture",
                                "security",
                                "performance",
                                "usability",
                                "all",
                            ],
                            "default": "all",
                        },
                        "include_dependencies": {"type": "boolean", "default": True},
                    },
                },
                "output_schema": {"type": "object"},
            },
            "enhanced_code_review": {
                "tool_id": "enhanced_code_review",
                "description": "Comprehensive code review with GitHub repository context and DeepCode analysis",
                "input_schema": {
                    "type": "object",
                    "required": ["code_path"],
                    "properties": {
                        "code_path": {"type": "string"},
                        "review_type": {
                            "type": "string",
                            "enum": [
                                "security",
                                "performance",
                                "best_practices",
                                "comprehensive",
                            ],
                            "default": "comprehensive",
                        },
                        "suggest_improvements": {"type": "boolean", "default": True},
                    },
                },
                "output_schema": {"type": "object"},
            },
        }

    def get_available_tools_schema(self) -> list[dict[str, Any]]:
        return list(self._contracts.values())

    def knows(self, tool_name: str) -> bool:
        return tool_name in self._known

    def validate_args(self, tool_name: str, args: dict[str, Any]) -> bool:
        # Simple: require "payload" string for echo; otherwise permissive
        if tool_name == "echo":
            return isinstance(args.get("payload"), str)
        return self.knows(tool_name)

    async def health_check(self, _: dict[str, Any]) -> bool:
        # In real setups, ping MCP, SDK, HTTP endpoint, etc.
        return True

    async def register(self, contract: dict[str, Any]) -> None:
        tid = contract["tool_id"]
        self._contracts[tid] = contract
        self._known.add(tid)

    async def execute(self, tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
        # Implement your actual bindings here (MCP, HTTP APIs, Python functions).
        if tool_name == "echo":
            return {"echo": args.get("payload", "")}
        if tool_name == "brainstorm_mcp_stub":
            task = (args.get("task") or "").strip()
            base = task or "unspecified task"
            ideas = [
                f"Outline {base}",
                f"List core primitives for {base}",
                f"Write minimal happy-path example for {base}",
                f"Add edge case tests for {base}",
                f"Refactor {base} for clarity",
            ]
            return {"task": task, "ideas": ideas}
        if tool_name == "fetch_github_raw":
            owner = args.get("owner")
            repo = args.get("repo")
            path = args.get("path")
            ref = args.get("ref") or "main"
            truncate_at = int(args.get("truncate") or 4000)
            url_main = f"https://raw.githubusercontent.com/{owner}/{repo}/{ref}/{path}"
            content = ""
            err: str | None = None
            truncated = False
            for candidate_ref in [ref, "main", "master"]:
                try:
                    url = f"https://raw.githubusercontent.com/{owner}/{repo}/{candidate_ref}/{path}"
                    with urllib.request.urlopen(url, timeout=5) as resp:  # nosec B310
                        raw_bytes = resp.read()
                    content = raw_bytes.decode("utf-8", errors="replace")
                    if len(content) > truncate_at:
                        content = content[:truncate_at]
                        truncated = True
                    url_main = url
                    break
                except Exception as e:  # pragma: no cover - network variability
                    err = str(e)
            return {
                "content": content,
                "url": url_main,
                "truncated": truncated,
                **({"error": err} if err and not content else {}),
            }
        if tool_name == "secure_scan_code":
            code = args.get("code") or ""
            patterns: list[tuple[str, str, str]] = [
                (r"\beval\(", "HIGH", "Use of eval() can be unsafe"),
                (r"\bexec\(", "HIGH", "Use of exec() can be unsafe"),
                (
                    r"subprocess\.Popen\(",
                    "MEDIUM",
                    "subprocess.Popen without sanitization",
                ),
                (r"os\.system\(", "MEDIUM", "os.system() call detected"),
                (r"pickle\.loads\(", "MEDIUM", "Untrusted pickle.loads()"),
            ]
            issues: list[dict[str, Any]] = []
            for pat, severity, message in patterns:
                for m in re.finditer(pat, code):
                    line = code.count("\n", 0, m.start()) + 1
                    issues.append(
                        {
                            "severity": severity,
                            "message": message,
                            "line": line,
                            "pattern": pat,
                        }
                    )
            return {"issues": issues, "issue_count": len(issues)}
        if tool_name == "full_cycle_prototype":
            # Combine brainstorming + optional fetch to produce a synthetic plan
            task = (args.get("task") or "hello world").strip()
            owner = args.get("owner")
            repo = args.get("repo")
            path = args.get("path")
            fetched: dict[str, Any] | None = None
            if owner and repo and path:
                fetched = await self.execute(
                    "fetch_github_raw",
                    {"owner": owner, "repo": repo, "path": path},
                )
            brainstorm = await self.execute("brainstorm_mcp_stub", {"task": task})
            run_instructions = [
                "# 1. Create virtual environment",
                "python -m venv .venv",
                "# 2. Activate venv",
                "# (Windows) .venv\\Scripts\\activate",
                "# 3. Write prototype code (hello.py)",
                'print("Hello World")',
                "# 4. Run it",
                "python hello.py",
            ]
            return {
                "task": task,
                "brainstorm": brainstorm,
                "fetched": fetched,
                "plan": run_instructions,
            }

        # DeepCode tool execution
        if tool_name in [
            "analyze_code_file",
            "analyze_code_directory",
            "get_code_quality_score",
            "detect_code_patterns",
            "analyze_workspace_context",
            "analyze_file_context",
            "check_file_support",
            "get_supported_extensions",
            "understand_code_structure",
        ]:
            return await self._execute_deepcode_tool(tool_name, args)

        # Enhanced Copilot tool execution
        if tool_name in [
            "analyze_and_suggest_repos",
            "automated_problem_solver",
            "repository_deep_analysis",
            "enhanced_code_review",
            "discover_github_upgrades",
        ]:
            return await self._execute_enhanced_copilot_tool(tool_name, args)

        # Fallback generic - echo contract
        return {"ok": True, "tool": tool_name, "args": args}

    async def _execute_deepcode_tool(
        self, tool_name: str, args: dict[str, Any]
    ) -> dict[str, Any]:
        """Execute deepcode tools by delegating to the appropriate ability"""
        try:
            # Import abilities dynamically to avoid circular imports
            from src.abilities.deepcode_analysis_ability import DeepCodeAnalysisAbility
            from src.abilities.deepcode_integration_ability import (
                DeepCodeIntegrationAbility,
            )

            # Determine which ability to use
            analysis_tools = [
                "analyze_code_file",
                "analyze_code_directory",
                "get_code_quality_score",
                "detect_code_patterns",
            ]
            integration_tools = [
                "analyze_workspace_context",
                "analyze_file_context",
                "check_file_support",
                "get_supported_extensions",
                "understand_code_structure",
            ]

            if tool_name in analysis_tools:
                ability = DeepCodeAnalysisAbility()
                await ability.setup(None, None, {})
                return await ability._execute_tool(tool_name, args)
            elif tool_name in integration_tools:
                ability = DeepCodeIntegrationAbility()
                await ability.setup(None, None, {})
                return await ability._execute_tool(tool_name, args)
            else:
                return {"error": f"Unknown deepcode tool: {tool_name}"}

        except Exception as e:
            logger.exception(f"DeepCode tool execution failed for {tool_name}: {e}")
            return {"error": f"Tool execution failed: {str(e)}"}

    async def _execute_enhanced_copilot_tool(
        self, tool_name: str, args: dict[str, Any]
    ) -> dict[str, Any]:
        """Execute enhanced copilot tools by delegating to the enhanced copilot ability"""
        try:
            # Import ability dynamically to avoid circular imports
            from src.abilities.enhanced_copilot_ability import EnhancedCopilotAbility

            # Create and setup ability
            ability = EnhancedCopilotAbility()
            await ability.setup(None, None, {})

            # Execute the tool
            return await ability._execute_tool(tool_name, args)

        except Exception as e:
            logger.exception(
                f"Enhanced Copilot tool execution failed for {tool_name}: {e}"
            )
            return {"error": f"Tool execution failed: {str(e)}"}


# --- Knowledge graph (minimal; replace with your store/driver) ---
class SimpleKG:
    def __init__(self) -> None:
        self.atoms: list[dict[str, Any]] = []
        self.bonds: list[dict[str, Any]] = []

    async def retrieve_relevant_context(self, user_message: str) -> str:
        return f"(context for: {user_message[:48]}...)"

    async def get_goal_for_session(self, session_id: str) -> dict[str, Any]:
        return {
            "id": f"goal_{session_id}",
            "description": f"Assist session {session_id}",
        }

    async def create_atom(self, atom_type: str, content: Any) -> dict[str, Any]:
        atom: dict[str, Any] = {
            "id": f"atom_{len(self.atoms)}",
            "type": atom_type,
            "content": content,
        }
        self.atoms.append(atom)
        return atom

    async def create_bond(
        self, bond_type: str, source_atom_id: str, target_atom_id: str
    ) -> None:
        self.bonds.append(
            {"type": bond_type, "src": source_atom_id, "tgt": target_atom_id}
        )


# --- FastAPI factory ---
def create_app(*, event_bus: BaseEventBus | None = None) -> Any:
    """Create FastAPI app or return None if FastAPI not available."""
    if not FASTAPI_AVAILABLE:
        print(
            "[ERROR] FastAPI not available - install with: "
            "'pip install fastapi uvicorn'"
        )
        return None

    _configure_logging()
    
    from contextlib import asynccontextmanager
    
    @asynccontextmanager
    async def _lifespan(_: Any) -> AsyncGenerator[None, None]:  # type: ignore
        # Startup: initialize plugins from manifest and emit runtime events
        plugins_loaded = []
        
        # Load OaK integration plugins first
        with contextlib.suppress(Exception):
            from src.core.plugin_loader import discover_plugins, load_plugin_manifest
            
            try:
                plugin_configs = load_plugin_manifest("plugins.yaml")
                if plugin_configs:
                    plugin_classes = discover_plugins(plugin_configs)
                    
                    # Initialize plugins in priority order
                    for plugin_name, plugin_class in plugin_classes:
                        try:
                            # Find config for this plugin
                            plugin_config = next(
                                (p for p in plugin_configs if p["name"] == plugin_name), 
                                {}
                            )
                            config = plugin_config.get("config", {})
                            
                            plugin_instance = plugin_class()
                            await plugin_instance.setup(app.state.event_bus, store=None, config=config)  # type: ignore
                            await plugin_instance.start()
                            plugins_loaded.append(plugin_instance)
                            logger.info(f"Loaded OaK plugin: {plugin_name}")
                        except Exception as e:
                            logger.warning(f"Failed to load plugin {plugin_name}: {e}")
                            
                    logger.info(f"Loaded {len(plugins_loaded)} OaK plugins from manifest")
                else:
                    logger.info("No plugins.yaml found, loading minimal plugins")
            except Exception as e:
                logger.warning(f"Plugin manifest loading failed: {e}")
        
        # Fallback: initialize optional DeepCode plugins if manifest loading failed
        if not plugins_loaded:
            with contextlib.suppress(Exception):
                from src.plugins.deepcode_generator_plugin import (
                    DeepCodeGeneratorBridgePlugin,
                )
                from src.plugins.deepcode_orchestrator_plugin import (
                    DeepCodeOrchestratorPlugin,
                )

                gen = DeepCodeGeneratorBridgePlugin()
                orch = DeepCodeOrchestratorPlugin()
                await gen.setup(app.state.event_bus, store=None, config={})  # type: ignore
                await orch.setup(app.state.event_bus, store=None, config={})  # type: ignore
                await gen.start()
                await orch.start()
                plugins_loaded = [gen, orch]
                
        app.state.plugins = plugins_loaded  # type: ignore

        # Emit startup events
        try:
            corr = str(uuid4())
            logging.getLogger().info("runtime startup")
            await app.state.event_bus.emit(  # type: ignore
                {
                    "type": "STATE_TRANSITION",
                    "from": "BOOT",
                    "to": "READY",
                    "correlation_id": corr,
                }
            )
            await app.state.event_bus.emit(  # type: ignore
                {
                    "type": "TaskStarted",
                    "correlation_id": corr,
                    "goal": "startup",
                    "user_msg_hash": _hash_json("startup"),
                }
            )
        except Exception:
            pass  # best-effort; keep service up even if telemetry fails

        yield

        # Shutdown: stop plugins gracefully
        with contextlib.suppress(Exception):
            for p in getattr(app.state, "plugins", []):  # type: ignore
                stop = getattr(p, "stop", None)
                if callable(stop):
                    await stop()
    
    # Create FastAPI app with lifespan handler
    app = FastAPI(title="REUG Runtime", version="0.2.0", lifespan=_lifespan)  # type: ignore

    # CORS (tweak as needed)
    app.add_middleware(  # type: ignore
        CORSMiddleware,  # type: ignore
        allow_origins=os.getenv("CORS_ALLOW_ORIGINS", "*").split(","),
        allow_methods=["*"],
        allow_headers=["*"],
        allow_credentials=True,
    )

    # Health for Dockerfile/compose
    from reug_runtime.health import check_health

    @app.get("/healthz")  # type: ignore
    async def health_check() -> JSONResponse:  # type: ignore
        status = await check_health(
            app.state.event_bus,  # type: ignore
            app.state.ability_registry,  # type: ignore
            app.state.kg,  # type: ignore
            app.state.llm_model,  # type: ignore
        )
        code = 200 if status["status"] == "healthy" else 503
        return JSONResponse(status_code=code, content=status)  # type: ignore

    # Alternative health endpoint
    @app.get("/health")  # type: ignore
    async def health_check_alt() -> JSONResponse:  # type: ignore
        status = await check_health(
            app.state.event_bus,  # type: ignore
            app.state.ability_registry,  # type: ignore
            app.state.kg,  # type: ignore
            app.state.llm_model,  # type: ignore
        )
        code = 200 if status["status"] == "healthy" else 503
        if (
            isinstance(app.state.event_bus, FileEventBus)  # type: ignore
            and isinstance(app.state.ability_registry, SimpleAbilityRegistry)  # type: ignore
            and isinstance(app.state.kg, SimpleKG)  # type: ignore
            and isinstance(app.state.llm_model, LLMClient)  # type: ignore
        ):
            minimal: dict[str, Any] = {
                "status": status["status"],
                "service": "super-alita",
            }
            return JSONResponse(status_code=code, content=minimal)  # type: ignore
        return JSONResponse(status_code=code, content=status)  # type: ignore

    # Inject dependencies for the REUG router
    app.state.event_bus = event_bus if event_bus is not None else make_event_bus()  # type: ignore
    app.state.ability_registry = SimpleAbilityRegistry()  # type: ignore
    app.state.kg = SimpleKG()  # type: ignore
    app.state.llm_model = get_llm_client(os.getenv("LLM_MODEL"))  # type: ignore

    # Lifespan handler (replaces deprecated on_event startup/shutdown)
    app.state.plugins = []  # type: ignore

    # Mount routers
    prefix = SETTINGS.api_prefix
    if prefix and prefix != "/":
        if not prefix.startswith("/"):
            prefix = f"/{prefix}"
        prefix = prefix.rstrip("/")
        app.include_router(agent_router, prefix=prefix)  # type: ignore
        app.include_router(tools_router, prefix=prefix)  # type: ignore

        # Automatic message optimization middleware (HTTP level)
        @app.middleware("http")  # type: ignore
        async def _optimize_incoming(
            request: Request,
            call_next: Callable,  # type: ignore
        ) -> Any:
            with contextlib.suppress(Exception):
                # Only process JSON chat route
                if (
                    request.headers.get("content-type", "").startswith(  # type: ignore
                        "application/json"
                    )
                    and "/chat/stream" in request.url.path  # type: ignore
                ):
                    with contextlib.suppress(Exception):
                        from reug_runtime.config import (
                            SETTINGS as RT_SETTINGS,  # type: ignore
                        )
                        from reug_runtime.message_mw import (  # type: ignore
                            MessageContext,
                            apply_all,
                        )

                        # Ensure amplifier is registered when enabled
                        if RT_SETTINGS.message_optimizer_enabled:
                            with contextlib.suppress(Exception):
                                import src.plugins.message_amplifier_plugin  # noqa: F401

                    # If optimizer is enabled, attempt to rewrite body
                    if (
                        RT_SETTINGS is not None
                        and getattr(RT_SETTINGS, "message_optimizer_enabled", False)
                        and apply_all is not None
                    ):
                        raw = await request.body()  # type: ignore
                        with contextlib.suppress(Exception):
                            payload = json.loads(raw.decode("utf-8") or "{}")  # type: ignore
                        msg = payload.get("message")
                        if isinstance(msg, str) and msg:
                            session_id = payload.get("session_id") or "default"
                            optimized, steps = apply_all(  # type: ignore
                                msg,
                                MessageContext(session_id=session_id),  # type: ignore
                            )
                            if getattr(
                                RT_SETTINGS, "message_optimizer_emit_telemetry", True
                            ):
                                with contextlib.suppress(Exception):
                                    await app.state.event_bus.emit(  # type: ignore
                                        {
                                            "type": "MessageOptimized",
                                            "correlation_id": f"http-{session_id}",
                                            "len_in": len(msg),
                                            "len_out": len(optimized),
                                            "steps": steps,
                                            "source": "http_mw",
                                        }
                                    )
                            max_len = getattr(
                                RT_SETTINGS, "message_optimizer_max_len", 6000
                            )
                            if len(optimized) > max_len:
                                optimized = optimized[:max_len]
                            payload["message"] = optimized
                            new_body = json.dumps(payload).encode("utf-8")

                            # Rebuild request with new body
                            async def _receive() -> dict[str, Any]:
                                return {
                                    "type": "http.request",
                                    "body": new_body,
                                    "more_body": False,
                                }

                            request = Request(request.scope, _receive)  # type: ignore
            return await call_next(request)

    else:
        app.include_router(agent_router)  # type: ignore
        app.include_router(tools_router)  # type: ignore

    # Startup events are handled via lifespan; on_event is deprecated

    # DeepCode trigger endpoint (fire-and-forget). Accept generic JSON to
    # reduce tight coupling / avoid Pydantic forward issues across versions.
    @app.post("/deepcode/request")  # type: ignore
    async def deepcode_request(req: Request) -> dict[str, Any]:  # type: ignore
        body = {}  # type: ignore[var-annotated]
        with contextlib.suppress(Exception):
            body = await req.json()  # type: ignore
        payload: dict[str, Any] = {
            "source_plugin": "http_gateway",
            "task_kind": body.get("task_kind", "generic"),
            "requirements": body.get("requirements", ""),
            "repo_path": body.get("repo_path", "."),
            "conversation_id": body.get("conversation_id"),
        }
        evt = create_event("deepcode_request", **payload)
        # Use publish if available to notify orchestrator
        if hasattr(app.state.event_bus, "publish"):  # type: ignore
            await app.state.event_bus.publish(evt.model_dump())  # type: ignore
        else:
            await app.state.event_bus.emit(evt.model_dump())  # type: ignore
        return {"status": "accepted", "request": payload}

    # Generic ability execution endpoint (internal tools registry exposure)
    @app.post("/ability/execute/{tool_id}")  # type: ignore
    async def execute_ability(tool_id: str, req: Request) -> JSONResponse:  # type: ignore
        args: dict[str, Any] = {}
        with contextlib.suppress(Exception):
            parsed = await req.json()  # type: ignore
            if isinstance(parsed, dict):
                args = parsed
        registry: SimpleAbilityRegistry = app.state.ability_registry  # type: ignore
        if not registry.knows(tool_id):
            return JSONResponse(
                status_code=404, content={"error": "unknown_tool", "tool": tool_id}
            )  # type: ignore
        if not registry.validate_args(tool_id, args):
            return JSONResponse(
                status_code=400,
                content={"error": "invalid_args", "tool": tool_id, "args": args},
            )  # type: ignore
        result = await registry.execute(tool_id, args)
        return JSONResponse(  # type: ignore
            status_code=200,
            content={"tool": tool_id, "result": result},
        )

    # Bandit stats snapshot (consumed by IDE agent)
    @app.get("/bandit/stats")  # type: ignore
    async def bandit_stats() -> JSONResponse:  # type: ignore
        try:
            from cortex.bandit_stats_store import get_snapshot

            snap = get_snapshot()
            return JSONResponse(
                status_code=200,
                content={
                    "tools": snap.tools,
                    "generated_at": snap.generated_at,
                },
            )  # type: ignore
        except Exception as e:  # pragma: no cover
            return JSONResponse(status_code=200, content={"tools": [], "error": str(e)})  # type: ignore

    # DeepCode latest retrieval
    @app.get("/deepcode/latest")  # type: ignore
    async def deepcode_latest() -> JSONResponse:  # type: ignore
        plugin = None
        for p in getattr(app.state, "plugins", []):  # type: ignore
            if getattr(p, "name", None) == "deepcode_orchestrator":
                plugin = p
                break
        if not plugin or not hasattr(plugin, "get_latest"):
            return JSONResponse(
                status_code=404, content={"error": "orchestrator_not_ready"}
            )  # type: ignore
        latest = plugin.get_latest()  # type: ignore
        if not latest:
            return JSONResponse(status_code=404, content={"error": "no_latest"})  # type: ignore
        return JSONResponse(status_code=200, content=latest)  # type: ignore

    # Bandit endpoints: decide + feedback
    @app.post("/bandit/decide")  # type: ignore
    async def bandit_decide(req: Request) -> JSONResponse:  # type: ignore
        body = await req.json()  # type: ignore
        policy_id = body.get("policy_id", "default")
        from cortex.bandit_service import decide as bandit_decide_impl

        result = bandit_decide_impl(policy_id)
        return JSONResponse(status_code=200, content=result)  # type: ignore

    @app.post("/bandit/feedback")  # type: ignore
    async def bandit_feedback(req: Request) -> JSONResponse:  # type: ignore
        body = await req.json()  # type: ignore
        decision_id = body.get("decision_id")
        reward = float(body.get("reward", 0.0))
        source = body.get("source")
        if not decision_id:
            return JSONResponse(
                status_code=400, content={"error": "missing decision_id"}
            )  # type: ignore
        from cortex.bandit_service import feedback as bandit_feedback_impl

        result = bandit_feedback_impl(decision_id, reward, source)
        return JSONResponse(status_code=200, content=result)  # type: ignore

    # DeepCode apply endpoint (delegates to orchestrator; guardian applies elsewhere)
    @app.post("/deepcode/apply")  # type: ignore
    async def deepcode_apply(req: Request) -> JSONResponse:  # type: ignore
        raw = await req.json()  # type: ignore
        body = raw if isinstance(raw, dict) else {}
        filter_paths = (
            body.get("paths") if isinstance(body.get("paths"), list) else None
        )
        plugin = None
        for p in getattr(app.state, "plugins", []):  # type: ignore
            if getattr(p, "name", None) == "deepcode_orchestrator":
                plugin = p
                break
        if not plugin or not hasattr(plugin, "apply_latest"):
            return JSONResponse(
                status_code=404, content={"error": "orchestrator_not_ready"}
            )  # type: ignore
        result = await plugin.apply_latest(filter_paths)  # type: ignore
        return JSONResponse(status_code=200, content=result)  # type: ignore

    # Simplified minimal health (no deep dependency checks)
    @app.get("/health/simple")  # type: ignore
    async def health_simple() -> dict[str, str]:  # type: ignore
        return {"status": "ok"}

    # Route enumeration (debug only)
    @app.get("/routes")  # type: ignore
    async def list_routes() -> list[dict[str, str]]:  # type: ignore
        info: list[dict[str, str]] = []
        for r in app.router.routes:  # type: ignore[attr-defined]
            path = getattr(r, "path", "")
            methods = ",".join(sorted(getattr(r, "methods", []) or []))
            if path:
                info.append({"path": path, "methods": methods})
        return info

    return app


app = create_app()

# Optional CLI entry (e.g., python src/main.py --no-chat just validates startup)
if __name__ == "__main__":
    if not FASTAPI_AVAILABLE:
        print("[ERROR] FastAPI dependencies not available. Install with:")
        print("pip install fastapi uvicorn")
        sys.exit(1)

    if app is None:
        print("[ERROR] Failed to create app")
        sys.exit(1)

    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument(
        "--no-chat",
        action="store_true",
        help="Boot only; don't open sockets beyond uvicorn",
    )
    ap.add_argument(
        "--reload",
        action="store_true",
        help="Reload server on code changes (dev mode)",
    )
    args = ap.parse_args()

    async def _dependency_health() -> dict[str, bool]:
        results: dict[str, bool] = {}
        with contextlib.suppress(Exception):
            await app.state.event_bus.emit({"event": "health_check"})  # type: ignore
            results["event_bus"] = True
        with contextlib.suppress(Exception):
            contract = app.state.ability_registry.get_available_tools_schema()[0]  # type: ignore
            results["ability_registry"] = await app.state.ability_registry.health_check(
                contract
            )  # type: ignore
        with contextlib.suppress(Exception):
            await app.state.kg.get_goal_for_session("health")  # type: ignore
            results["kg"] = True
        with contextlib.suppress(Exception):
            stream_gen = app.state.llm_model.stream_chat([], timeout=1)  # type: ignore
            await stream_gen.__anext__()
            results["llm_model"] = True
        return results

    if args.no_chat:
        checks = asyncio.run(_dependency_health())
        print(json.dumps(checks))
        raise SystemExit(0)

    # Just start the ASGI server; REUG handles single-turn streaming internally
    uvicorn.run("main:app", host=args.host, port=args.port, reload=args.reload)  # type: ignore
