"""Specification-Driven Development (SDD) Framework.

This module implements the SDD workflow with constitutional validation,
providing /specify, /plan, and /tasks endpoints with integrated constitutional
compliance checking at each gate.
"""

from .constitutional_pipeline import ConstitutionalSDDPipeline
from .enhanced_sdd_framework import EnhancedSDDFramework
from .mangle_integration import MangleFactGenerator

# Mangle integration modules
from .mangle_reasoner import MangleReasoner
from .mangle_rules import MANGLE_RULES, get_available_queries, get_query_for_question
from .models import (
    PlanRequest,
    PlanResponse,
    SpecifyRequest,
    SpecifyResponse,
    TasksRequest,
    TasksResponse,
)
from .router import create_sdd_router

__all__ = [
    "ConstitutionalSDDPipeline",
    "SpecifyRequest",
    "SpecifyResponse",
    "PlanRequest",
    "PlanResponse",
    "TasksRequest",
    "TasksResponse",
    "create_sdd_router",
    # Mangle integration
    "MangleReasoner",
    "EnhancedSDDFramework",
    "MangleFactGenerator",
    "MANGLE_RULES",
    "get_query_for_question",
    "get_available_queries",
]

__version__ = "1.0.0"
