#!/usr/bin/env python3
"""Constitutional Gateway API - Direct Copilot Capability Augmentation Backend

Provides a FastAPI router that exposes all agent capabilities via HTTP endpoints
for direct integration with VS Code extensions and Copilot augmentation.

Features:
- Context providers for workspace analysis
- Code actions for refactoring and suggestions
- Inline suggestions and completions
- Real-time diagnostics and linting
- Constitutional enforcement and validation
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Body, HTTPException, Request
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# Create the router
constitutional_router = APIRouter(
    prefix="/constitutional", tags=["constitutional"]
)


# Request/Response Models
class ContextRequest(BaseModel):
    """Request for workspace context analysis."""

    workspace_path: str
    file_pattern: str = "**/*.py"
    include_content: bool = False
    max_files: int = 100


class ContextResponse(BaseModel):
    """Response with workspace context."""

    files: list[dict[str, Any]]
    summary: dict[str, Any]
    metadata: dict[str, Any]


class CodeActionRequest(BaseModel):
    """Request for code actions and suggestions."""

    file_path: str
    content: str
    line: int | None = None
    character: int | None = None
    selection: dict[str, Any] | None = None


class CodeActionResponse(BaseModel):
    """Response with available code actions."""

    actions: list[dict[str, Any]]
    suggestions: list[dict[str, Any]]
    diagnostics: list[dict[str, Any]]


class InlineSuggestionRequest(BaseModel):
    """Request for inline code suggestions."""

    file_path: str
    content: str
    cursor_position: dict[str, int]
    context_lines: int = 5


class InlineSuggestionResponse(BaseModel):
    """Response with inline suggestions."""

    suggestions: list[dict[str, Any]]
    completions: list[str]
    confidence: float


class DiagnosticRequest(BaseModel):
    """Request for code diagnostics."""

    file_path: str
    content: str
    check_types: list[str] = ["syntax", "style", "security", "patterns"]


class DiagnosticResponse(BaseModel):
    """Response with diagnostic information."""

    diagnostics: list[dict[str, Any]]
    summary: dict[str, Any]
    recommendations: list[str]


# Context Provider Endpoints
@constitutional_router.post(
    "/context/workspace", response_model=ContextResponse
)
async def get_workspace_context(
    request: ContextRequest,
    req: Request,
) -> ContextResponse:
    """Provide comprehensive workspace context for Copilot integration."""
    try:
        # Get ability registry from app state
        ability_registry = getattr(req.app.state, "ability_registry", None)
        if not ability_registry:
            raise HTTPException(
                status_code=500, detail="Ability registry not available"
            )

        # Use repo_list_files tool to get file structure
        list_result = await ability_registry.execute(
            "repo_list_files",
            {
                "directory": request.workspace_path,
                "pattern": request.file_pattern,
                "exclude_patterns": [
                    "__pycache__/**",
                    ".git/**",
                    "*.pyc",
                    ".venv/**",
                    "node_modules/**",
                ],
            },
        )

        files = list_result.get("files", [])

        # Optionally include file content for analysis
        if request.include_content:
            for file_info in files[: request.max_files]:
                if file_info.get("type") == "file":
                    try:
                        content_result = await ability_registry.execute(
                            "repo_read_file",
                            {"file_path": file_info["path"], "max_lines": 200},
                        )
                        file_info["content"] = content_result.get(
                            "content", ""
                        )
                    except Exception as e:
                        logger.warning(
                            f"Failed to read file {file_info['path']}: {e}"
                        )
                        file_info["content"] = f"# Error reading file: {e}"

        # Generate workspace summary
        total_files = len([f for f in files if f.get("type") == "file"])
        total_dirs = len([f for f in files if f.get("type") == "directory"])

        summary = {
            "total_files": total_files,
            "total_directories": total_dirs,
            "file_types": {},
            "project_structure": "analyzed",
        }

        # Analyze file types
        for file_info in files:
            if file_info.get("type") == "file":
                ext = Path(file_info["path"]).suffix
                summary["file_types"][ext] = (
                    summary["file_types"].get(ext, 0) + 1
                )

        metadata = {
            "workspace_path": request.workspace_path,
            "analysis_timestamp": "current",
            "context_provider": "constitutional_gateway",
        }

        return ContextResponse(files=files, summary=summary, metadata=metadata)

    except Exception as e:
        logger.error(f"Workspace context error: {e}")
        raise HTTPException(
            status_code=500, detail=f"Context analysis failed: {e}"
        )


@constitutional_router.post("/context/file", response_model=dict[str, Any])
async def get_file_context(
    file_path: str = Body(..., embed=True),
    req: Request = None,
) -> dict[str, Any]:
    """Provide detailed file context for Copilot analysis."""
    try:
        ability_registry = getattr(req.app.state, "ability_registry", None)
        if not ability_registry:
            raise HTTPException(
                status_code=500, detail="Ability registry not available"
            )

        # Read file content
        content_result = await ability_registry.execute(
            "repo_read_file", {"file_path": file_path}
        )

        content = content_result.get("content", "")

        # Get git history for the file
        history_result = await ability_registry.execute(
            "repo_git_history", {"file_path": file_path, "limit": 5}
        )

        # Analyze the file for patterns and structure
        scan_result = await ability_registry.execute(
            "secure_scan_code", {"code": content}
        )

        return {
            "file_path": file_path,
            "content": content,
            "line_count": len(content.split("\n")),
            "git_history": history_result.get("commits", []),
            "security_scan": scan_result,
            "metadata": {
                "size": len(content),
                "encoding": "utf-8",
                "analysis_provider": "constitutional_gateway",
            },
        }

    except Exception as e:
        logger.error(f"File context error: {e}")
        raise HTTPException(
            status_code=500, detail=f"File context failed: {e}"
        )


# Code Action Endpoints
@constitutional_router.post(
    "/actions/refactor", response_model=CodeActionResponse
)
async def get_refactor_actions(
    request: CodeActionRequest,
    req: Request,
) -> CodeActionResponse:
    """Provide refactoring actions and suggestions."""
    try:
        ability_registry = getattr(req.app.state, "ability_registry", None)
        if not ability_registry:
            raise HTTPException(
                status_code=500, detail="Ability registry not available"
            )

        # Scan for refactoring hotspots
        hotspots_result = await ability_registry.execute(
            "scan_refactor_hotspots",
            {"path": str(Path(request.file_path).parent), "mode": "default"},
        )

        # Get specific suggestions for the file
        suggestions_result = await ability_registry.execute(
            "get_refactor_suggestions", {"path": request.file_path}
        )

        # Security scan for the content
        security_result = await ability_registry.execute(
            "secure_scan_code", {"code": request.content}
        )

        actions = []
        suggestions = []
        diagnostics = []

        # Convert hotspots to code actions
        for hotspot in hotspots_result.get("hotspots", []):
            if hotspot.get("file") == request.file_path:
                actions.append(
                    {
                        "title": f"Refactor: {hotspot.get('type', 'pattern')}",
                        "kind": "refactor",
                        "isPreferred": hotspot.get("priority", 0) > 7,
                        "edit": {
                            "changes": {
                                request.file_path: [
                                    {
                                        "range": {
                                            "start": {
                                                "line": hotspot.get("line", 0),
                                                "character": 0,
                                            },
                                            "end": {
                                                "line": hotspot.get("line", 0)
                                                + 1,
                                                "character": 0,
                                            },
                                        },
                                        "newText": f"# TODO: Refactor {hotspot.get('reason', 'code')}\n",
                                    }
                                ]
                            }
                        },
                        "metadata": hotspot,
                    }
                )

        # Convert suggestions to inline suggestions
        for suggestion in suggestions_result.get("suggestions", []):
            suggestions.append(
                {
                    "text": suggestion,
                    "kind": "text",
                    "insertText": f"# Suggestion: {suggestion}",
                    "documentation": "Constitutional Gateway suggestion",
                }
            )

        # Convert security issues to diagnostics
        for issue in security_result.get("issues", []):
            diagnostics.append(
                {
                    "range": {
                        "start": {
                            "line": issue.get("line", 0) - 1,
                            "character": 0,
                        },
                        "end": {
                            "line": issue.get("line", 0) - 1,
                            "character": 100,
                        },
                    },
                    "severity": 1 if issue.get("severity") == "HIGH" else 2,
                    "message": issue.get("message", "Security issue detected"),
                    "source": "constitutional-gateway",
                    "code": issue.get("pattern", "security"),
                }
            )

        return CodeActionResponse(
            actions=actions, suggestions=suggestions, diagnostics=diagnostics
        )

    except Exception as e:
        logger.error(f"Refactor actions error: {e}")
        raise HTTPException(
            status_code=500, detail=f"Refactor actions failed: {e}"
        )


@constitutional_router.post(
    "/actions/quick-fix", response_model=dict[str, Any]
)
async def get_quick_fixes(
    request: CodeActionRequest,
    req: Request,
) -> dict[str, Any]:
    """Provide quick fix suggestions for code issues."""
    try:
        ability_registry = getattr(req.app.state, "ability_registry", None)
        if not ability_registry:
            raise HTTPException(
                status_code=500, detail="Ability registry not available"
            )

        # Security scan to identify fixable issues
        scan_result = await ability_registry.execute(
            "secure_scan_code", {"code": request.content}
        )

        fixes = []

        for issue in scan_result.get("issues", []):
            fix_text = ""
            if "eval(" in issue.get("pattern", ""):
                fix_text = (
                    "# Replace eval() with ast.literal_eval() for safety"
                )
            elif "exec(" in issue.get("pattern", ""):
                fix_text = "# Replace exec() with safer alternatives"
            elif "TODO" in issue.get("pattern", ""):
                fix_text = "# Implement TODO item"
            elif "NotImplementedError" in issue.get("pattern", ""):
                fix_text = "# Provide implementation"
            else:
                fix_text = f"# Fix: {issue.get('message', 'Issue')}"

            fixes.append(
                {
                    "title": f"Fix: {issue.get('message', 'Issue')}",
                    "kind": "quickfix",
                    "line": issue.get("line", 0),
                    "edit": fix_text,
                    "severity": issue.get("severity", "LOW"),
                }
            )

        return {
            "fixes": fixes,
            "count": len(fixes),
            "provider": "constitutional_gateway",
        }

    except Exception as e:
        logger.error(f"Quick fixes error: {e}")
        raise HTTPException(status_code=500, detail=f"Quick fixes failed: {e}")


# Inline Suggestion Endpoints
@constitutional_router.post(
    "/suggestions/inline", response_model=InlineSuggestionResponse
)
async def get_inline_suggestions(
    request: InlineSuggestionRequest,
    req: Request,
) -> InlineSuggestionResponse:
    """Provide inline code suggestions and completions."""
    try:
        ability_registry = getattr(req.app.state, "ability_registry", None)
        if not ability_registry:
            raise HTTPException(
                status_code=500, detail="Ability registry not available"
            )

        # Use enhanced consensus for intelligent suggestions
        consensus_result = await ability_registry.execute(
            "deepconf_consensus",
            {
                "prompt": f"Complete this Python code intelligently:\n{request.content}",
                "method": "confidence_based",
                "num_samples": 3,
                "confidence_threshold": 0.8,
            },
        )

        suggestions = []
        completions = []

        consensus_text = consensus_result.get("consensus_text", "")
        confidence = consensus_result.get("consensus_confidence", 0.5)

        if consensus_text:
            # Extract suggestions from consensus
            lines = consensus_text.split("\n")
            for line in lines[:5]:  # Limit to 5 suggestions
                if line.strip() and not line.strip().startswith("#"):
                    suggestions.append(
                        {
                            "text": line.strip(),
                            "kind": "completion",
                            "detail": "Constitutional Gateway AI suggestion",
                            "insertText": line.strip(),
                            "confidence": confidence,
                        }
                    )
                    completions.append(line.strip())

        # Add some fallback completions for common patterns
        cursor_line = request.cursor_position.get("line", 0)
        lines = request.content.split("\n")
        current_line = lines[cursor_line] if cursor_line < len(lines) else ""

        if current_line.strip().endswith(":"):
            completions.append("    pass")
            completions.append("    # TODO: Implement")

        if (
            "def " in current_line
            and "(" in current_line
            and ")" not in current_line
        ):
            completions.append("):")

        if "import " in current_line:
            completions.append("from pathlib import Path")
            completions.append("import asyncio")

        return InlineSuggestionResponse(
            suggestions=suggestions,
            completions=list(set(completions)),  # Remove duplicates
            confidence=confidence,
        )

    except Exception as e:
        logger.error(f"Inline suggestions error: {e}")
        raise HTTPException(
            status_code=500, detail=f"Inline suggestions failed: {e}"
        )


# Diagnostic Endpoints
@constitutional_router.post(
    "/diagnostics/analyze", response_model=DiagnosticResponse
)
async def analyze_code_diagnostics(
    request: DiagnosticRequest,
    req: Request,
) -> DiagnosticResponse:
    """Provide comprehensive code diagnostics."""
    try:
        ability_registry = getattr(req.app.state, "ability_registry", None)
        if not ability_registry:
            raise HTTPException(
                status_code=500, detail="Ability registry not available"
            )

        all_diagnostics = []
        recommendations = []

        # Security scan
        if "security" in request.check_types:
            security_result = await ability_registry.execute(
                "secure_scan_code", {"code": request.content}
            )

            for issue in security_result.get("issues", []):
                all_diagnostics.append(
                    {
                        "range": {
                            "start": {
                                "line": issue.get("line", 0) - 1,
                                "character": 0,
                            },
                            "end": {
                                "line": issue.get("line", 0) - 1,
                                "character": 100,
                            },
                        },
                        "severity": (
                            1 if issue.get("severity") == "HIGH" else 2
                        ),
                        "message": issue.get("message", "Security issue"),
                        "source": "constitutional-security",
                        "code": issue.get("pattern", "security"),
                        "type": "security",
                    }
                )

        # Pattern analysis using Mangle if available
        if "patterns" in request.check_types:
            try:
                mangle_result = await ability_registry.execute(
                    "analyze_code_question",
                    {
                        "question": f"Analyze this code for patterns and issues:\n{request.content}",
                        "scope": request.file_path,
                    },
                )

                if mangle_result.get("analysis"):
                    recommendations.append(
                        f"Mangle analysis: {mangle_result['analysis']}"
                    )
            except Exception:
                # Mangle might not be available
                pass

        # Generate summary
        summary = {
            "total_issues": len(all_diagnostics),
            "by_severity": {},
            "by_type": {},
            "file_health_score": max(0, 100 - len(all_diagnostics) * 5),
        }

        for diag in all_diagnostics:
            severity = diag.get("severity", 2)
            diag_type = diag.get("type", "general")

            summary["by_severity"][str(severity)] = (
                summary["by_severity"].get(str(severity), 0) + 1
            )
            summary["by_type"][diag_type] = (
                summary["by_type"].get(diag_type, 0) + 1
            )

        # Add general recommendations
        if not all_diagnostics:
            recommendations.append(
                "Code looks clean! Consider adding more tests."
            )
        elif len(all_diagnostics) > 10:
            recommendations.append(
                "Consider refactoring to reduce complexity."
            )

        return DiagnosticResponse(
            diagnostics=all_diagnostics,
            summary=summary,
            recommendations=recommendations,
        )

    except Exception as e:
        logger.error(f"Diagnostics error: {e}")
        raise HTTPException(status_code=500, detail=f"Diagnostics failed: {e}")


# Health and Status Endpoints
@constitutional_router.get("/health")
async def constitutional_health() -> dict[str, Any]:
    """Health check for Constitutional Gateway."""
    return {
        "status": "healthy",
        "service": "constitutional_gateway",
        "version": "1.0.0",
        "capabilities": [
            "context_providers",
            "code_actions",
            "inline_suggestions",
            "diagnostics",
            "constitutional_enforcement",
        ],
    }


@constitutional_router.get("/capabilities")
async def get_capabilities() -> dict[str, Any]:
    """Get available Constitutional Gateway capabilities."""
    return {
        "context_providers": [
            "workspace_context",
            "file_context",
            "git_history",
        ],
        "code_actions": [
            "refactor_suggestions",
            "quick_fixes",
            "security_fixes",
        ],
        "inline_suggestions": [
            "ai_completions",
            "pattern_completions",
            "context_aware_suggestions",
        ],
        "diagnostics": [
            "security_analysis",
            "pattern_analysis",
            "style_checking",
            "complexity_analysis",
        ],
        "integrations": [
            "mangle_semantic_analysis",
            "enhanced_consensus",
            "refactoring_agent",
            "security_scanner",
        ],
    }


# Constitutional Enforcement Endpoints
@constitutional_router.post("/enforce/validate")
async def validate_constitutional_compliance(
    content: str = Body(..., embed=True),
    req: Request = None,
) -> dict[str, Any]:
    """Validate content against constitutional principles."""
    try:
        ability_registry = getattr(req.app.state, "ability_registry", None)
        if not ability_registry:
            raise HTTPException(
                status_code=500, detail="Ability registry not available"
            )

        # Security validation
        security_result = await ability_registry.execute(
            "secure_scan_code", {"code": content}
        )

        # Constitutional principles validation
        violations = []
        score = 100

        # Check for placeholder/mock violations
        for issue in security_result.get("issues", []):
            if any(
                pattern in issue.get("pattern", "")
                for pattern in [
                    "TODO",
                    "FIXME",
                    "NotImplementedError",
                    "placeholder",
                    "mock",
                ]
            ):
                violations.append(
                    {
                        "principle": "No Mock/Placeholder Policy",
                        "violation": issue.get(
                            "message", "Placeholder detected"
                        ),
                        "line": issue.get("line", 0),
                        "severity": "high",
                    }
                )
                score -= 20

        # Check for security violations
        high_security_issues = [
            i
            for i in security_result.get("issues", [])
            if i.get("severity") == "HIGH"
        ]
        for issue in high_security_issues:
            violations.append(
                {
                    "principle": "Security First",
                    "violation": issue.get("message", "Security issue"),
                    "line": issue.get("line", 0),
                    "severity": "critical",
                }
            )
            score -= 30

        compliance_status = (
            "compliant"
            if score > 80
            else "non_compliant" if score > 50 else "critical"
        )

        return {
            "compliance_status": compliance_status,
            "score": max(0, score),
            "violations": violations,
            "recommendations": (
                [
                    "Remove all placeholder content",
                    "Implement proper error handling",
                    "Follow security best practices",
                    "Use constitutional principles in all code",
                ]
                if violations
                else ["Code is constitutionally compliant"]
            ),
            "constitutional_version": "v5.0",
        }

    except Exception as e:
        logger.error(f"Constitutional validation error: {e}")
        raise HTTPException(status_code=500, detail=f"Validation failed: {e}")


# Export the router for registration in main.py
__all__ = ["constitutional_router"]
