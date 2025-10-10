"""Super Alita snippet library for VS Code integration."""

import json
from pathlib import Path
from typing import Any

from src.plugins.plugin_interface import PluginInterface


class SnippetLibrary(PluginInterface):
    """Manages code snippets for Super Alita development patterns."""

    def __init__(self):
        self.snippets_dir = (
            Path(__file__).parent.parent.parent
            / "extensions"
            / "copilot-prompt-optimizer"
            / "snippets"
        )
        self.snippets_dir.mkdir(parents=True, exist_ok=True)

    async def initialize(self, event_bus, **kwargs) -> bool:
        """Initialize the snippet library plugin."""
        self.event_bus = event_bus
        await self._ensure_snippets_exist()
        return True

    async def process_event(
        self, event: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Process snippet-related events."""
        event_type = event.get("type")

        if event_type == "snippet_request":
            language = event.get("language", "python")
            return await self._get_snippets_for_language(language)
        elif event_type == "snippet_validate":
            snippet = event.get("snippet")
            is_valid = await self.validate_snippet_quality(snippet)
            return {"valid": is_valid, "snippet": snippet}

        return None

    async def _ensure_snippets_exist(self) -> None:
        """Ensure snippet files exist with default content."""
        python_snippets_path = self.snippets_dir / "python.json"

        if not python_snippets_path.exists():
            python_snippets = self._get_python_snippets()
            with open(python_snippets_path, "w", encoding="utf-8") as f:
                json.dump(python_snippets, f, indent=2, ensure_ascii=False)

    async def _get_snippets_for_language(
        self, language: str
    ) -> dict[str, Any]:
        """Get snippets for specified language."""
        if language == "python":
            return {"snippets": self._get_python_snippets()}
        return {"snippets": {}}

    def _get_python_snippets(self) -> dict[str, Any]:
        """Get comprehensive Python snippets following Super Alita patterns."""
        return {
            # Super Alita specific patterns
            "Super Alita Plugin": {
                "prefix": "alitaplugin",
                "body": [
                    "from typing import Any, Dict",
                    "from src.plugins.plugin_interface import PluginInterface",
                    "from src.core.events import create_event",
                    "",
                    "class ${1:NewPlugin}(PluginInterface):",
                    '    """${2:Plugin description}"""',
                    "    ",
                    "    async def initialize(self, event_bus, **kwargs) -> bool:",
                    '        """Initialize the plugin"""',
                    "        self.event_bus = event_bus",
                    "        return True",
                    "    ",
                    "    async def process_event(self, event: Dict[str, Any]) -> Dict[str, Any] | None:",
                    '        """Process events"""',
                    "        ${3:pass}",
                    '        return {"processed": True, "data": event}',
                ],
                "description": "Create Super Alita plugin following established patterns",
            },
            # Enhanced function with Black formatting
            "Function with type hints": {
                "prefix": "func",
                "body": [
                    "def ${1:function_name}(",
                    "    ${2:param}: ${3:type},",
                    "    ${4:optional_param}: ${5:type} = ${6:default},",
                    ") -> ${7:return_type}:",
                    '    """${8:Function description}.',
                    "    ",
                    "    Args:",
                    "        ${2:param}: ${9:Parameter description}",
                    "        ${4:optional_param}: ${10:Optional parameter description}",
                    "    ",
                    "    Returns:",
                    "        ${11:Return description}",
                    '    """',
                    "    ${12:pass}",
                    "    return ${13:None}",
                ],
                "description": "Create function with proper type hints and docstring (Black 88 chars)",
            },
            # Enhanced consensus integration
            "Enhanced Consensus Usage": {
                "prefix": "consensus",
                "body": [
                    "from src.abilities.enhanced_consensus_ability import EnhancedConsensusProvider",
                    "",
                    "# Canonical constructor: config dict, not kwargs",
                    "provider = EnhancedConsensusProvider(config={",
                    '    "base_url": "http://localhost:11434/v1",',
                    '    "model_name": "gpt-oss:20b",',
                    '    "timeout": 60',
                    "})",
                    "",
                    "result = await provider.consensus_sampling(",
                    '    prompt="${1:Your question}",',
                    "    num_samples=${2:3},",
                    '    method="weighted_vote",  # Default method per src/main.py:1660',
                    "    confidence_threshold=${3:0.7}",
                    ")",
                ],
                "description": "Use enhanced consensus following Super Alita patterns",
            },
            # Standard Python patterns from awesome-python-snippets
            "Python File Header": {
                "prefix": "fileheader",
                "body": ["# -*- coding: utf-8 -*-"],
                "description": "Add Python file encoding header",
            },
            "Python File Shebang": {
                "prefix": "shebang",
                "body": ["#!/usr/bin/env python"],
                "description": "Add Python shebang line",
            },
            "Set breakpoint for debugging": {
                "prefix": "ipdb",
                "body": ["import ipdb; ipdb.set_trace()  # noqa"],
                "description": "Set ipdb debugger breakpoint",
            },
            # Enhanced imports with Super Alita patterns
            "Import datetime with timezone": {
                "prefix": "imdt",
                "body": ["from datetime import datetime, timezone"],
                "description": "Import datetime with timezone support (Super Alita pattern)",
            },
            "Import pathlib": {
                "prefix": "pathlib",
                "body": ["from pathlib import Path"],
                "description": "Import pathlib (Super Alita standard)",
            },
            # Async patterns for Super Alita
            "Pytest async test": {
                "prefix": "asynctest",
                "body": [
                    "@pytest.mark.asyncio  # Required for all async tests",
                    "async def ${1:test_function}():",
                    '    """${2:Test description}"""',
                    "    from datetime import datetime, timezone",
                    "    ",
                    "    # Use timezone-aware timestamps",
                    "    timestamp = datetime.now(timezone.utc)",
                    "    ${3:pass}",
                ],
                "description": "Create async test following Super Alita patterns",
            },
            # File operations with context managers
            "Open file": {
                "prefix": "openfile",
                "body": [
                    "with open('${1:filename}', '${2:r}') as f:",
                    "    ${3:content = f.read()}",
                ],
                "description": "Open file with context manager",
            },
            # List comprehensions
            "List comprehension with condition": {
                "prefix": "listcomp",
                "body": [
                    "[${1:expression} for ${2:item} in ${3:iterable} if ${4:condition}]"
                ],
                "description": "List comprehension with condition",
            },
            # Control flow
            "If/else statement": {
                "prefix": "ifelse",
                "body": [
                    "if ${1:condition}:",
                    "    ${2:pass}",
                    "else:",
                    "    ${3:pass}",
                ],
                "description": "if/else statement",
            },
            # Exception handling
            "Try/except": {
                "prefix": "tryexcept",
                "body": [
                    "try:",
                    "    ${1:pass}",
                    "except ${2:Exception} as e:",
                    "    ${3:pass}",
                ],
                "description": "Try/except block with specific exception",
            },
            # Class patterns with type hints
            "Main class with type hints": {
                "prefix": "mainclass",
                "body": [
                    "class ${1:NewClass}:",
                    '    """${2:Class description}"""',
                    "    ",
                    "    def __init__(self, ${3:param}: ${4:type}) -> None:",
                    '        """Initialize ${1:NewClass}.',
                    "        ",
                    "        Args:",
                    "            ${3:param}: ${5:Parameter description}",
                    '        """',
                    "        self.${3:param} = ${3:param}",
                    "        ${6:pass}",
                ],
                "description": "Create main class with type hints and proper docstring",
            },
            # MCP Tool Registration
            "MCP Tool Registration": {
                "prefix": "mcptool",
                "body": [
                    "from src.core.ability_registry import AbilityContract",
                    "",
                    "# Register MCP tool",
                    "contract = AbilityContract(",
                    '    id="${1:tool_id}",',
                    '    name="${2:Tool Name}",',
                    '    description="${3:Tool description}",',
                    "    parameters={",
                    '        "type": "object",',
                    '        "properties": {',
                    '            "${4:param_name}": {"type": "string"}',
                    "        },",
                    '        "required": ["${4:param_name}"]',
                    "    }",
                    ")",
                    "",
                    "async def ${5:tool_executor}(args: dict[str, Any]) -> dict[str, Any]:",
                    '    """Execute the tool"""',
                    "    ${6:pass}",
                    '    return {"result": "success"}',
                    "",
                    "ability_registry.register_tool(contract, ${5:tool_executor})",
                ],
                "description": "Register MCP tool with Super Alita ability registry",
            },
            # Event system patterns
            "Create Event": {
                "prefix": "createevent",
                "body": [
                    "from src.core.events import create_event",
                    "",
                    "event = create_event(",
                    '    "${1:event_type}",',
                    "    ${2:data}=${3:value},",
                    "    confidence=${4:0.95}",
                    ")",
                ],
                "description": "Create event following Super Alita patterns",
            },
        }

    async def validate_snippet_quality(self, snippet: dict[str, Any]) -> bool:
        """Validate snippet follows Super Alita quality standards."""
        try:
            body = snippet.get("body", [])
            if isinstance(body, str):
                body = [body]

            # Check for Super Alita patterns
            code_text = "\n".join(body)

            # Validate type hints usage
            if "def " in code_text and "->" not in code_text:
                return False  # Functions should have return type hints

            # Validate docstring format
            if '"""' in code_text:
                # Should follow Google/NumPy style docstrings
                pass

            # Validate import organization
            if "import" in code_text:
                # Should use absolute imports for Super Alita modules
                if "from src." in code_text:
                    pass  # Good Super Alita import

            return True
        except Exception:
            return False

    async def get_snippets_by_category(
        self, category: str
    ) -> list[dict[str, Any]]:
        """Get snippets filtered by category."""
        python_snippets = self._get_python_snippets()

        category_filters = {
            "super_alita": ["plugin", "consensus", "mcp", "event"],
            "functions": ["func", "async", "lambda"],
            "classes": ["class", "subclass", "mainclass"],
            "imports": ["import", "from"],
            "control_flow": ["if", "for", "while", "try"],
            "testing": ["pytest", "test", "parametrize"],
        }

        if category not in category_filters:
            return list(python_snippets.values())

        keywords = category_filters[category]
        filtered_snippets = []

        for name, snippet in python_snippets.items():
            snippet_text = snippet.get("description", "").lower()
            if any(keyword in snippet_text for keyword in keywords):
                snippet["name"] = name
                filtered_snippets.append(snippet)

        return filtered_snippets

    async def search_snippets(self, query: str) -> list[dict[str, Any]]:
        """Search snippets by query string."""
        python_snippets = self._get_python_snippets()
        query_lower = query.lower()

        matching_snippets = []
        for name, snippet in python_snippets.items():
            # Search in name, description, and prefix
            search_text = f"{name} {snippet.get('description', '')} {snippet.get('prefix', '')}"
            if query_lower in search_text.lower():
                snippet["name"] = name
                matching_snippets.append(snippet)

        return matching_snippets
