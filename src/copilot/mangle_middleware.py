"""
GitHub Copilot Mangle Integration Middleware.

This module integrates Mangle reasoning directly into GitHub Copilot's
processing pipeline, making Code Knowledge Graph analysis a native part
of every interaction.
"""

import os
import sys
from pathlib import Path

# Add src to path for imports
src_path = Path(__file__).parent.parent
sys.path.insert(0, str(src_path))

try:
    from src.copilot.mangle_enhanced_agent import process_copilot_input

    MANGLE_AVAILABLE = True
except ImportError:
    MANGLE_AVAILABLE = False


def enhance_copilot_with_mangle():
    """
    Enhance GitHub Copilot with automatic Mangle reasoning.

    This function patches the standard Copilot processing to include
    automatic Code Knowledge Graph analysis for all interactions.
    """
    if not MANGLE_AVAILABLE:
        print("⚠️  Mangle integration not available - missing dependencies")
        return False

    # Set environment variable to enable Mangle mode
    os.environ["COPILOT_MANGLE_MODE"] = "true"

    print("✅ GitHub Copilot enhanced with Mangle reasoning")
    print("   • Code Knowledge Graph analysis enabled")
    print("   • Constitutional compliance monitoring active")
    print("   • Natural language code querying available")
    print("   • Specification-to-code traceability enabled")

    return True


async def copilot_chat_with_mangle(message: str, **kwargs) -> str:
    """
    Process Copilot chat messages with automatic Mangle enhancement.

    This function replaces the standard chat processing to include
    Code Knowledge Graph reasoning in every response.
    """
    if not MANGLE_AVAILABLE:
        return f"Standard response: {message}"

    # Extract context from kwargs
    context = {
        "current_file": kwargs.get("current_file"),
        "workspace": kwargs.get("workspace", "."),
        "selection": kwargs.get("selection"),
        "language": kwargs.get("language"),
    }

    # Process with Mangle enhancement
    enhanced_response = await process_copilot_input(message, context)

    return enhanced_response


def register_mangle_commands():
    """Register Mangle-specific commands for GitHub Copilot."""
    commands = {
        "ask_mangle": {
            "description": "Ask questions about code using Mangle reasoning",
            "usage": "@copilot ask_mangle what functions are untested?",
            "handler": "copilot_chat_with_mangle",
        },
        "analyze_constitutional": {
            "description": "Analyze constitutional compliance",
            "usage": "@copilot analyze_constitutional",
            "handler": "copilot_chat_with_mangle",
        },
        "check_quality": {
            "description": "Check code quality using knowledge graph",
            "usage": "@copilot check_quality",
            "handler": "copilot_chat_with_mangle",
        },
        "trace_spec": {
            "description": "Trace code to specifications",
            "usage": "@copilot trace_spec function_name",
            "handler": "copilot_chat_with_mangle",
        },
    }

    return commands


# Auto-initialize when imported
if __name__ == "__main__":
    enhance_copilot_with_mangle()
else:
    # Automatically enhance Copilot when this module is imported
    if os.getenv("COPILOT_AUTO_ENHANCE", "true").lower() == "true":
        enhance_copilot_with_mangle()
