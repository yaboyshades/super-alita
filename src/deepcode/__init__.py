"""
DeepCode package for advanced code analysis and intelligence
"""

from .analyzer_simple import AnalysisLevel, SeverityLevel, create_deepcode_engine
from .integration import (
    analyze_current_file,
    analyze_workspace_sample,
    get_deepcode_integration,
    is_supported_file,
)
from .templates import (
    get_templates_dir,
    list_deepcode_templates,
    load_deepcode_template,
    render_deepcode_template,
)

__all__ = [
    "get_deepcode_integration",
    "analyze_current_file",
    "analyze_workspace_sample",
    "is_supported_file",
    "AnalysisLevel",
    "SeverityLevel",
    "create_deepcode_engine",
    "get_templates_dir",
    "list_deepcode_templates",
    "load_deepcode_template",
    "render_deepcode_template",
]
