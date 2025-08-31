# Prompt Management & Templates - Agent Instructions

## Overview
The `src/prompts/` directory contains prompt templates and management utilities:
- **System Prompts** - Core system prompt templates
- **Conversation Prompts** - Conversation management prompts
- **Router Prompts** - Request routing and decision prompts
- **Planner Prompts** - Task planning and execution prompts

## Key Files & Responsibilities

### Prompt Templates
- `planner_system_prompt.txt` - System prompt for planning operations
- `router_system_prompt.txt` - System prompt for request routing
- `conversation_finalizer_system_prompt.txt` - Conversation finalization prompt

## Development Guidelines

### Prompt Template Management
```python
from typing import Dict, Any, List, Optional
from pathlib import Path
import jinja2
from dataclasses import dataclass
from enum import Enum

class PromptType(Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    FUNCTION = "function"

@dataclass
class PromptTemplate:
    """Prompt template structure"""
    name: str
    content: str
    prompt_type: PromptType
    variables: List[str] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.variables is None:
            self.variables = []
        if self.metadata is None:
            self.metadata = {}

class PromptManager:
    """Manager for prompt templates and rendering"""

    def __init__(self, prompts_dir: str = None):
        self.prompts_dir = Path(prompts_dir) if prompts_dir else Path(__file__).parent
        self.templates: Dict[str, PromptTemplate] = {}
        self.jinja_env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(str(self.prompts_dir)),
            undefined=jinja2.StrictUndefined
        )

    def load_templates(self):
        """Load all prompt templates from directory"""
        try:
            for prompt_file in self.prompts_dir.glob("*.txt"):
                template_name = prompt_file.stem
                content = prompt_file.read_text(encoding='utf-8')

                # Extract prompt type from filename or content
                prompt_type = self._infer_prompt_type(template_name, content)

                # Extract variables from template
                variables = self._extract_template_variables(content)

                template = PromptTemplate(
                    name=template_name,
                    content=content,
                    prompt_type=prompt_type,
                    variables=variables,
                    metadata={
                        "file_path": str(prompt_file),
                        "size": len(content),
                        "lines": len(content.splitlines())
                    }
                )

                self.templates[template_name] = template

            logger.info(f"Loaded {len(self.templates)} prompt templates")

        except Exception as e:
            logger.error(f"Failed to load prompt templates: {e}")

    def get_template(self, template_name: str) -> Optional[PromptTemplate]:
        """Get prompt template by name"""
        return self.templates.get(template_name)

    def render_prompt(self, template_name: str, **kwargs) -> str:
        """Render prompt template with variables"""
        template = self.get_template(template_name)

        if not template:
            raise ValueError(f"Template not found: {template_name}")

        try:
            # Use Jinja2 for rendering
            jinja_template = self.jinja_env.from_string(template.content)
            rendered = jinja_template.render(**kwargs)

            return rendered.strip()

        except jinja2.TemplateError as e:
            raise ValueError(f"Template rendering failed for {template_name}: {e}")

    def validate_template_variables(self, template_name: str, variables: Dict[str, Any]) -> List[str]:
        """Validate template variables"""
        template = self.get_template(template_name)

        if not template:
            return [f"Template not found: {template_name}"]

        errors = []

        # Check required variables
        for required_var in template.variables:
            if required_var not in variables:
                errors.append(f"Missing required variable: {required_var}")

        # Check for unexpected variables
        provided_vars = set(variables.keys())
        expected_vars = set(template.variables)
        unexpected_vars = provided_vars - expected_vars

        if unexpected_vars:
            errors.append(f"Unexpected variables: {', '.join(unexpected_vars)}")

        return errors

    def _infer_prompt_type(self, template_name: str, content: str) -> PromptType:
        """Infer prompt type from name and content"""
        name_lower = template_name.lower()

        if "system" in name_lower:
            return PromptType.SYSTEM
        elif "user" in name_lower:
            return PromptType.USER
        elif "assistant" in name_lower:
            return PromptType.ASSISTANT
        elif "function" in name_lower:
            return PromptType.FUNCTION
        else:
            # Default to system for most prompts
            return PromptType.SYSTEM

    def _extract_template_variables(self, content: str) -> List[str]:
        """Extract Jinja2 template variables"""
        # Simple regex-based extraction
        import re

        # Find {{ variable }} patterns
        variables = re.findall(r'\{\{\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\}\}', content)

        # Find {% for variable in ... %} patterns
        for_vars = re.findall(r'\{\%\s*for\s+([a-zA-Z_][a-zA-Z0-9_]*)\s+in\s+', content)
        variables.extend(for_vars)

        # Remove duplicates and return
        return list(set(variables))
```

### Specialized Prompt Builders
```python
class SystemPromptBuilder:
    """Builder for system prompts"""

    def __init__(self, prompt_manager: PromptManager):
        self.prompt_manager = prompt_manager
        self.context_sections: List[str] = []

    def add_context_section(self, section: str):
        """Add context section to prompt"""
        self.context_sections.append(section)
        return self

    def build_planner_prompt(self,
                           task_description: str,
                           available_tools: List[Dict[str, Any]],
                           context: Dict[str, Any] = None) -> str:
        """Build planner system prompt"""
        template_vars = {
            "task_description": task_description,
            "available_tools": available_tools,
            "context": context or {},
            "additional_context": "\n".join(self.context_sections)
        }

        return self.prompt_manager.render_prompt("planner_system_prompt", **template_vars)

    def build_router_prompt(self,
                          request: str,
                          available_routes: List[str],
                          routing_history: List[Dict[str, Any]] = None) -> str:
        """Build router system prompt"""
        template_vars = {
            "request": request,
            "available_routes": available_routes,
            "routing_history": routing_history or [],
            "additional_context": "\n".join(self.context_sections)
        }

        return self.prompt_manager.render_prompt("router_system_prompt", **template_vars)

    def build_conversation_finalizer_prompt(self,
                                          conversation_history: List[Dict[str, Any]],
                                          summary_requirements: List[str] = None) -> str:
        """Build conversation finalizer prompt"""
        template_vars = {
            "conversation_history": conversation_history,
            "summary_requirements": summary_requirements or [],
            "additional_context": "\n".join(self.context_sections)
        }

        return self.prompt_manager.render_prompt("conversation_finalizer_system_prompt", **template_vars)

class ConversationPromptManager:
    """Manager for conversation-specific prompts"""

    def __init__(self, prompt_manager: PromptManager):
        self.prompt_manager = prompt_manager
        self.conversation_context: Dict[str, Any] = {}

    def initialize_conversation(self, user_id: str, session_id: str, initial_context: Dict[str, Any] = None):
        """Initialize conversation context"""
        self.conversation_context = {
            "user_id": user_id,
            "session_id": session_id,
            "start_time": datetime.now(timezone.utc).isoformat(),
            "message_count": 0,
            "context": initial_context or {}
        }

    def create_user_prompt(self, message: str, context_additions: Dict[str, Any] = None) -> str:
        """Create user prompt with context"""
        if context_additions:
            self.conversation_context["context"].update(context_additions)

        self.conversation_context["message_count"] += 1

        return f"""User Message #{self.conversation_context['message_count']}:
{message}

Context:
{json.dumps(self.conversation_context['context'], indent=2)}"""

    def create_system_response_prompt(self, user_message: str, available_actions: List[str]) -> str:
        """Create system response prompt"""
        template_vars = {
            "user_message": user_message,
            "available_actions": available_actions,
            "conversation_context": self.conversation_context,
            "session_id": self.conversation_context["session_id"]
        }

        # Use a general response template or build one
        system_prompt = f"""You are Super Alita, an advanced AI assistant.

Current conversation context:
- Session ID: {template_vars['session_id']}
- Message count: {self.conversation_context['message_count']}

User message: {user_message}

Available actions: {', '.join(available_actions)}

Please provide a helpful, accurate, and contextually appropriate response. If you need to use tools or perform actions, clearly indicate which actions you want to take."""

        return system_prompt

    def finalize_conversation(self) -> Dict[str, Any]:
        """Finalize conversation and return summary"""
        self.conversation_context["end_time"] = datetime.now(timezone.utc).isoformat()

        duration = datetime.fromisoformat(self.conversation_context["end_time"]) - \
                  datetime.fromisoformat(self.conversation_context["start_time"])

        return {
            "session_summary": self.conversation_context,
            "duration_seconds": duration.total_seconds(),
            "message_count": self.conversation_context["message_count"]
        }

class DynamicPromptGenerator:
    """Generate prompts dynamically based on context"""

    def __init__(self, prompt_manager: PromptManager):
        self.prompt_manager = prompt_manager
        self.prompt_cache: Dict[str, str] = {}

    def generate_tool_execution_prompt(self,
                                     tool_name: str,
                                     tool_description: str,
                                     parameters: Dict[str, Any],
                                     expected_output: str = None) -> str:
        """Generate prompt for tool execution"""
        prompt = f"""You are about to execute the '{tool_name}' tool.

Tool Description: {tool_description}

Parameters:
{json.dumps(parameters, indent=2)}
"""

        if expected_output:
            prompt += f"\nExpected Output Format: {expected_output}"

        prompt += "\nPlease execute this tool and provide the results."

        return prompt

    def generate_error_handling_prompt(self,
                                     error_message: str,
                                     context: Dict[str, Any],
                                     suggested_actions: List[str] = None) -> str:
        """Generate prompt for error handling"""
        prompt = f"""An error has occurred that needs to be handled:

Error: {error_message}

Context:
{json.dumps(context, indent=2)}
"""

        if suggested_actions:
            prompt += f"\nSuggested actions:\n"
            for i, action in enumerate(suggested_actions, 1):
                prompt += f"{i}. {action}\n"

        prompt += "\nPlease analyze this error and determine the best course of action."

        return prompt

    def generate_code_analysis_prompt(self,
                                    code: str,
                                    analysis_type: str,
                                    specific_focus: List[str] = None) -> str:
        """Generate prompt for code analysis"""
        prompt = f"""Please analyze the following {analysis_type}:

```
{code}
```

Analysis type: {analysis_type}
"""

        if specific_focus:
            prompt += f"\nSpecific areas to focus on:\n"
            for focus in specific_focus:
                prompt += f"- {focus}\n"

        prompt += "\nProvide a detailed analysis with recommendations."

        return prompt
```

### Prompt Template Examples
```python
# Example prompt templates that would be in .txt files

PLANNER_SYSTEM_PROMPT_TEMPLATE = """
You are Super Alita's planning system. Your role is to break down user requests into actionable tasks and coordinate their execution.

Current Task: {{ task_description }}

Available Tools:
{% for tool in available_tools %}
- {{ tool.name }}: {{ tool.description }}
  Parameters: {{ tool.parameters | join(', ') }}
{% endfor %}

Context:
{{ context | tojson(indent=2) }}

{% if additional_context %}
Additional Context:
{{ additional_context }}
{% endif %}

Please create a step-by-step plan to accomplish the task. For each step:
1. Clearly state what needs to be done
2. Identify which tool(s) to use
3. Specify the exact parameters
4. Note any dependencies on previous steps

Respond with a structured plan in JSON format:
{
  "plan_id": "unique_identifier",
  "steps": [
    {
      "step_number": 1,
      "description": "What to do",
      "tool": "tool_name",
      "parameters": {...},
      "dependencies": []
    }
  ],
  "estimated_duration": "time_estimate",
  "confidence": 0.95
}
"""

ROUTER_SYSTEM_PROMPT_TEMPLATE = """
You are Super Alita's intelligent request router. Your role is to analyze incoming requests and route them to the most appropriate handler.

Current Request: {{ request }}

Available Routes:
{% for route in available_routes %}
- {{ route }}
{% endfor %}

{% if routing_history %}
Recent Routing History:
{% for entry in routing_history %}
Request: {{ entry.request }}
Route: {{ entry.route }}
Success: {{ entry.success }}
{% endfor %}
{% endif %}

{% if additional_context %}
Additional Context:
{{ additional_context }}
{% endif %}

Please analyze the request and determine the best route. Consider:
1. The type of request (question, task, command)
2. The complexity and requirements
3. Available capabilities of each route
4. Historical performance data

Respond with:
{
  "selected_route": "route_name",
  "confidence": 0.95,
  "reasoning": "Why this route was selected",
  "fallback_routes": ["alternative1", "alternative2"]
}
"""

CONVERSATION_FINALIZER_PROMPT_TEMPLATE = """
You are Super Alita's conversation finalizer. Your role is to create meaningful summaries and conclusions for conversations.

Conversation History:
{% for message in conversation_history %}
{{ message.role }}: {{ message.content }}
Timestamp: {{ message.timestamp }}
{% endfor %}

{% if summary_requirements %}
Summary Requirements:
{% for requirement in summary_requirements %}
- {{ requirement }}
{% endfor %}
{% endif %}

{% if additional_context %}
Additional Context:
{{ additional_context }}
{% endif %}

Please create a comprehensive summary including:
1. Main topics discussed
2. Key decisions made
3. Action items identified
4. Unresolved issues
5. Overall sentiment and outcome

Respond with:
{
  "summary": {
    "main_topics": [...],
    "key_decisions": [...],
    "action_items": [...],
    "unresolved_issues": [...],
    "sentiment": "positive/neutral/negative",
    "outcome": "successful/partial/unsuccessful"
  },
  "metadata": {
    "message_count": number,
    "duration": "time_span",
    "participants": [...]
  }
}
"""
```

## Testing Guidelines

### Prompt Testing Framework
```python
import pytest
from unittest.mock import patch, MagicMock
from src.prompts.prompt_manager import PromptManager, SystemPromptBuilder

@pytest.fixture
def prompt_manager(tmp_path):
    """Create prompt manager with test templates"""
    # Create test template files
    test_templates = {
        "test_system_prompt.txt": "You are a test assistant. Task: {{ task }}",
        "test_user_prompt.txt": "User input: {{ user_input }}",
        "test_complex_prompt.txt": """
Complex template with:
- Task: {{ task }}
- Tools: {% for tool in tools %}{{ tool.name }}{% endfor %}
- Context: {{ context | default('None') }}
"""
    }

    for filename, content in test_templates.items():
        (tmp_path / filename).write_text(content)

    manager = PromptManager(str(tmp_path))
    manager.load_templates()
    return manager

def test_prompt_template_loading(prompt_manager):
    """Test prompt template loading"""
    assert len(prompt_manager.templates) == 3
    assert "test_system_prompt" in prompt_manager.templates
    assert "test_user_prompt" in prompt_manager.templates
    assert "test_complex_prompt" in prompt_manager.templates

def test_prompt_rendering(prompt_manager):
    """Test prompt rendering with variables"""
    rendered = prompt_manager.render_prompt(
        "test_system_prompt",
        task="test task"
    )

    assert "You are a test assistant" in rendered
    assert "Task: test task" in rendered

def test_complex_prompt_rendering(prompt_manager):
    """Test complex prompt with loops and filters"""
    tools = [
        {"name": "tool1"},
        {"name": "tool2"}
    ]

    rendered = prompt_manager.render_prompt(
        "test_complex_prompt",
        task="complex task",
        tools=tools
    )

    assert "Task: complex task" in rendered
    assert "tool1tool2" in rendered  # Loop result
    assert "Context: None" in rendered  # Default filter

def test_template_variable_validation(prompt_manager):
    """Test template variable validation"""
    # Valid variables
    errors = prompt_manager.validate_template_variables(
        "test_system_prompt",
        {"task": "test"}
    )
    assert len(errors) == 0

    # Missing required variable
    errors = prompt_manager.validate_template_variables(
        "test_system_prompt",
        {}
    )
    assert len(errors) == 1
    assert "Missing required variable: task" in errors[0]

def test_system_prompt_builder():
    """Test system prompt builder"""
    mock_manager = MagicMock()
    mock_manager.render_prompt.return_value = "Rendered prompt"

    builder = SystemPromptBuilder(mock_manager)
    builder.add_context_section("Additional context")

    result = builder.build_planner_prompt(
        "Test task",
        [{"name": "test_tool", "description": "Test tool"}]
    )

    assert result == "Rendered prompt"
    mock_manager.render_prompt.assert_called_once()

@pytest.mark.asyncio
async def test_conversation_prompt_manager():
    """Test conversation prompt manager"""
    mock_manager = MagicMock()

    conv_manager = ConversationPromptManager(mock_manager)
    conv_manager.initialize_conversation("user123", "session456")

    # Test user prompt creation
    user_prompt = conv_manager.create_user_prompt("Hello")
    assert "User Message #1:" in user_prompt
    assert "Hello" in user_prompt

    # Test conversation finalization
    summary = conv_manager.finalize_conversation()
    assert summary["session_summary"]["user_id"] == "user123"
    assert summary["message_count"] == 1
```

### Prompt Quality Testing
```python
def test_prompt_clarity_and_structure():
    """Test prompt clarity and structure"""
    prompt_manager = PromptManager()
    prompt_manager.load_templates()

    for template_name, template in prompt_manager.templates.items():
        # Check for clear structure
        content = template.content

        # Should have clear instructions
        assert any(keyword in content.lower() for keyword in [
            "you are", "your role", "please", "respond with"
        ]), f"Template {template_name} lacks clear instructions"

        # Should not be too long
        assert len(content) < 5000, f"Template {template_name} is too long"

        # Should not have obvious typos (basic check)
        assert "teh" not in content.lower(), f"Template {template_name} has typos"

def test_prompt_variable_consistency():
    """Test prompt variable consistency"""
    prompt_manager = PromptManager()
    prompt_manager.load_templates()

    for template_name, template in prompt_manager.templates.items():
        variables = template.variables
        content = template.content

        # Check that all variables are actually used
        for var in variables:
            assert f"{{{{{var}}}}}" in content or f"{{% for {var}" in content, \
                f"Variable {var} declared but not used in {template_name}"

@pytest.mark.integration
def test_prompt_integration_with_llm():
    """Test prompt integration with actual LLM"""
    # This would test with a real or mock LLM service
    prompt_manager = PromptManager()
    prompt_manager.load_templates()

    # Test planner prompt
    planner_prompt = prompt_manager.render_prompt(
        "planner_system_prompt",
        task_description="Create a simple calculator",
        available_tools=[{
            "name": "code_generator",
            "description": "Generates code",
            "parameters": ["language", "specification"]
        }],
        context={"project_type": "utility"}
    )

    # Verify prompt is well-formed
    assert len(planner_prompt) > 100
    assert "Create a simple calculator" in planner_prompt
    assert "code_generator" in planner_prompt
```

## Security Guidelines

### Prompt Injection Prevention
```python
class PromptSecurityManager:
    """Security manager for prompt templates"""

    def __init__(self):
        self.dangerous_patterns = [
            r"ignore\s+previous\s+instructions",
            r"forget\s+everything",
            r"act\s+as\s+if",
            r"pretend\s+to\s+be",
            r"system\s*:\s*",
            r"assistant\s*:\s*",
            r"```.*?exec.*?```",
            r"```.*?eval.*?```"
        ]

    def validate_user_input(self, user_input: str) -> List[str]:
        """Validate user input for prompt injection attempts"""
        violations = []
        input_lower = user_input.lower()

        for pattern in self.dangerous_patterns:
            if re.search(pattern, input_lower, re.IGNORECASE):
                violations.append(f"Potential prompt injection: {pattern}")

        return violations

    def sanitize_template_variables(self, variables: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize template variables"""
        sanitized = {}

        for key, value in variables.items():
            if isinstance(value, str):
                # Remove potential injection attempts
                sanitized_value = value
                for pattern in self.dangerous_patterns:
                    sanitized_value = re.sub(pattern, "[FILTERED]", sanitized_value, flags=re.IGNORECASE)
                sanitized[key] = sanitized_value[:1000]  # Limit length
            elif isinstance(value, (list, dict)):
                # Recursively sanitize complex types
                sanitized[key] = self._sanitize_complex_value(value)
            else:
                sanitized[key] = value

        return sanitized

    def _sanitize_complex_value(self, value):
        """Sanitize complex values (lists, dicts)"""
        if isinstance(value, list):
            return [self._sanitize_complex_value(item) for item in value[:100]]  # Limit list size
        elif isinstance(value, dict):
            return {
                k: self._sanitize_complex_value(v)
                for k, v in list(value.items())[:50]  # Limit dict size
            }
        elif isinstance(value, str):
            return value[:500]  # Limit string length
        else:
            return value
```

### Template Access Control
```python
class TemplateAccessControl:
    """Access control for prompt templates"""

    def __init__(self):
        self.user_permissions: Dict[str, Set[str]] = {}
        self.template_permissions: Dict[str, Set[str]] = {}

    def set_user_permissions(self, user_id: str, permissions: Set[str]):
        """Set permissions for user"""
        self.user_permissions[user_id] = permissions

    def set_template_permissions(self, template_name: str, required_permissions: Set[str]):
        """Set required permissions for template"""
        self.template_permissions[template_name] = required_permissions

    def can_access_template(self, user_id: str, template_name: str) -> bool:
        """Check if user can access template"""
        user_perms = self.user_permissions.get(user_id, set())
        required_perms = self.template_permissions.get(template_name, set())

        # User must have all required permissions
        return required_perms.issubset(user_perms)

    def filter_accessible_templates(self, user_id: str, templates: List[str]) -> List[str]:
        """Filter templates accessible to user"""
        return [
            template for template in templates
            if self.can_access_template(user_id, template)
        ]
```

## Performance Guidelines

### Prompt Caching
```python
from functools import lru_cache
import hashlib

class CachedPromptManager(PromptManager):
    """Prompt manager with caching optimizations"""

    def __init__(self, prompts_dir: str = None, cache_size: int = 1000):
        super().__init__(prompts_dir)
        self.cache_size = cache_size
        self.render_cache: Dict[str, str] = {}

    @lru_cache(maxsize=1000)
    def _cached_template_load(self, template_path: str) -> str:
        """Cached template loading"""
        return Path(template_path).read_text(encoding='utf-8')

    def render_prompt(self, template_name: str, **kwargs) -> str:
        """Render prompt with caching"""
        # Create cache key from template name and variables
        cache_key = self._create_cache_key(template_name, kwargs)

        if cache_key in self.render_cache:
            return self.render_cache[cache_key]

        # Render prompt
        rendered = super().render_prompt(template_name, **kwargs)

        # Cache result
        if len(self.render_cache) >= self.cache_size:
            # Remove oldest entry (simple FIFO)
            oldest_key = next(iter(self.render_cache))
            del self.render_cache[oldest_key]

        self.render_cache[cache_key] = rendered
        return rendered

    def _create_cache_key(self, template_name: str, variables: Dict[str, Any]) -> str:
        """Create cache key for template and variables"""
        variables_str = json.dumps(variables, sort_keys=True)
        key_string = f"{template_name}:{variables_str}"
        return hashlib.md5(key_string.encode()).hexdigest()
```

## Common Patterns

### Prompt Composition
```python
class PromptComposer:
    """Compose complex prompts from multiple parts"""

    def __init__(self, prompt_manager: PromptManager):
        self.prompt_manager = prompt_manager
        self.sections: List[str] = []

    def add_section(self, template_name: str, **kwargs) -> 'PromptComposer':
        """Add section from template"""
        section = self.prompt_manager.render_prompt(template_name, **kwargs)
        self.sections.append(section)
        return self

    def add_text(self, text: str) -> 'PromptComposer':
        """Add raw text section"""
        self.sections.append(text)
        return self

    def add_separator(self, separator: str = "\n---\n") -> 'PromptComposer':
        """Add separator between sections"""
        self.sections.append(separator)
        return self

    def compose(self) -> str:
        """Compose final prompt"""
        return "\n".join(self.sections)

    def clear(self) -> 'PromptComposer':
        """Clear all sections"""
        self.sections.clear()
        return self

# Usage example:
def create_complex_prompt(prompt_manager: PromptManager) -> str:
    composer = PromptComposer(prompt_manager)

    return (composer
            .add_section("system_context", role="assistant")
            .add_separator()
            .add_section("task_definition", task="code analysis")
            .add_separator()
            .add_text("Additional instructions: Focus on security")
            .compose())
```

### Prompt Versioning
```python
class VersionedPromptManager(PromptManager):
    """Prompt manager with versioning support"""

    def __init__(self, prompts_dir: str = None):
        super().__init__(prompts_dir)
        self.prompt_versions: Dict[str, Dict[str, PromptTemplate]] = {}

    def load_versioned_templates(self):
        """Load templates with version support"""
        # Look for templates with version suffixes: template_v1.txt, template_v2.txt
        for prompt_file in self.prompts_dir.glob("*_v*.txt"):
            # Extract template name and version
            name_with_version = prompt_file.stem
            if "_v" in name_with_version:
                template_name, version = name_with_version.rsplit("_v", 1)

                content = prompt_file.read_text(encoding='utf-8')
                prompt_type = self._infer_prompt_type(template_name, content)
                variables = self._extract_template_variables(content)

                template = PromptTemplate(
                    name=template_name,
                    content=content,
                    prompt_type=prompt_type,
                    variables=variables,
                    metadata={"version": version, "file_path": str(prompt_file)}
                )

                if template_name not in self.prompt_versions:
                    self.prompt_versions[template_name] = {}

                self.prompt_versions[template_name][version] = template

    def get_template_version(self, template_name: str, version: str) -> Optional[PromptTemplate]:
        """Get specific version of template"""
        return self.prompt_versions.get(template_name, {}).get(version)

    def get_latest_template(self, template_name: str) -> Optional[PromptTemplate]:
        """Get latest version of template"""
        versions = self.prompt_versions.get(template_name, {})
        if not versions:
            return None

        # Sort versions and get latest
        latest_version = max(versions.keys(), key=lambda v: [int(x) for x in v.split('.')])
        return versions[latest_version]
```

## Debugging Tips
- **Template validation** - Validate template syntax and variable usage
- **Rendering testing** - Test prompt rendering with various input combinations
- **Variable tracking** - Track which variables are used in each template
- **Performance monitoring** - Monitor prompt rendering performance and cache hit rates
- **Security auditing** - Regularly audit prompts for potential security issues
