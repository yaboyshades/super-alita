"""Tools module for REUG runtime.

This module contains tool catalog definitions and services for
managing dynamic tool registration and catalog operations.
"""

# Static tool catalog from router_tools.py
TOOL_CATALOG = [
    {
        "name": "reug_start_turn",
        "description": "Start a single-turn REUG agent run with streaming. Returns a run_id and initial stream chunk(s).",
        "input_schema": {
            "type": "object",
            "properties": {
                "message": {"type": "string"},
                "session_id": {"type": "string"},
            },
            "required": ["message"],
        },
        "output_schema": {
            "type": "object",
            "properties": {
                "run_id": {"type": "string"},
                "stream_begun": {"type": "boolean"},
            },
            "required": ["run_id", "stream_begun"],
        },
    },
    {
        "name": "reug_stream_next",
        "description": "Fetch next streamed chunk(s) for a run. Ends when final answer emitted.",
        "input_schema": {
            "type": "object",
            "properties": {"run_id": {"type": "string"}},
            "required": ["run_id"],
        },
        "output_schema": {
            "type": "object",
            "properties": {
                "chunks": {"type": "array", "items": {"type": "string"}},
                "finished": {"type": "boolean"},
            },
            "required": ["chunks", "finished"],
        },
    },
    {
        "name": "pytest_run",
        "description": "Run pytest with optional target path and markers.",
        "input_schema": {
            "type": "object",
            "properties": {
                "target": {"type": "string"},
                "markers": {"type": "string"},
                "quiet": {"type": "boolean", "default": True},
            },
        },
        "output_schema": {
            "type": "object",
            "properties": {
                "ok": {"type": "boolean"},
                "exit_code": {"type": "integer"},
                "stdout": {"type": "string"},
                "stderr": {"type": "string"},
            },
            "required": ["ok", "exit_code"],
        },
    },
    {
        "name": "fs_read",
        "description": "Read a UTF-8 text file.",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
        "output_schema": {
            "type": "object",
            "properties": {"content": {"type": "string"}},
            "required": ["content"],
        },
    },
    {
        "name": "fs_write",
        "description": "Write UTF-8 content to a file (creates/overwrites).",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["path", "content"],
        },
        "output_schema": {
            "type": "object",
            "properties": {"ok": {"type": "boolean"}},
            "required": ["ok"],
        },
    },
    {
        "name": "super_alita_v4_instructions",
        "description": "Return structured runtime setup instructions with constitutional validation metadata.",
        "input_schema": {"type": "object", "properties": {}},
        "output_schema": {
            "type": "object",
            "properties": {
                "profile": {"type": "object"},
                "sections": {
                    "type": "array",
                    "items": {"type": "object"},
                },
                "validation": {"type": "object"},
            },
            "required": ["profile", "sections", "validation"],
        },
    },
    {
        "name": "git_apply_patch",
        "description": "Apply a unified diff patch to the repo.",
        "input_schema": {
            "type": "object",
            "properties": {"patch": {"type": "string"}},
            "required": ["patch"],
        },
        "output_schema": {
            "type": "object",
            "properties": {
                "ok": {"type": "boolean"},
                "stdout": {"type": "string"},
                "stderr": {"type": "string"},
            },
            "required": ["ok"],
        },
    },
    {
        "name": "deepconf_consensus",
        "description": "Perform consensus sampling using multiple LLM calls",
        "input_schema": {
            "type": "object",
            "required": ["prompt"],
            "properties": {
                "prompt": {"type": "string"},
                "num_samples": {"type": "integer", "default": 3},
                "temperature": {"type": "number", "default": 0.7},
                "max_tokens": {"type": "integer", "default": 512},
            },
        },
        "output_schema": {
            "type": "object",
            "properties": {
                "consensus_text": {"type": "string"},
                "consensus_confidence": {"type": "number"},
                "aggregation_method": {"type": "string"},
                "individual_responses": {"type": "array"},
                "metadata": {"type": "object"},
            },
        },
    },
]

from .service import ToolCatalogService

__all__ = [
    "TOOL_CATALOG",
    "ToolCatalogService",
]
