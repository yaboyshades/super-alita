"""
Code Reasoning Module for Unified Intelligence Layer

This module provides mangle-style code analysis capabilities:
- AST-based code ingestion and fact extraction
- SQL-based rule engine for detecting code quality issues
- Integration with unified intelligence pipeline

Based on mangle_code_scaffold_v2 architecture.
"""

from .ingester import CodeIngester
from .models import CodeAnalysisRequest, CodeAnalysisResponse, Finding
from .rules import RuleEngine

__all__ = [
    "CodeIngester",
    "RuleEngine",
    "CodeAnalysisRequest",
    "CodeAnalysisResponse",
    "Finding",
]
