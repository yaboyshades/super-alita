"""GitHub Copilot integration package."""

from .mangle_enhanced_agent import MangleEnhancedAgent, process_copilot_input
from .mangle_middleware import enhance_copilot_with_mangle

__all__ = [
    "MangleEnhancedAgent",
    "process_copilot_input",
    "enhance_copilot_with_mangle",
]
