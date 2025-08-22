"""
CREATOR Plugin - Autonomous Tool Generation
Listens for AtomGapEvents and creates new tools on-demand using LLM generation.
"""

import ast
import keyword
import logging
import os
import re
from pathlib import Path
from typing import Any

import aiohttp

# Load environment variables
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass  # dotenv not available, environment variables should be set manually

# Try to import Google Generative AI, handle gracefully if missing
try:
    import google.generativeai as genai

    GENAI_AVAILABLE = True
except (ImportError, TypeError, Exception):
    genai = None
    GENAI_AVAILABLE = False

from src.core.events import (
    AtomGapEvent,
    AtomReadyEvent,
    ComposeRequestEvent,
    ShowCreatedToolRequest,
    ToolCreatedEvent,
)
from src.core.ids import deterministic_tool_id
from src.core.plugin_interface import PluginInterface
from src.core.secure_executor import get_tool_registry

logger = logging.getLogger(__name__)


class CreatorPlugin(PluginInterface):
    """Plugin that creates new tools on-demand when gaps are detected."""

    @property
    def name(self) -> str:
        return "creator"

    async def setup(self, event_bus, store, config: dict[str, Any]):
        """Initialize the CREATOR plugin."""
        await super().setup(event_bus, store, config)

        # Initialize LLM client for code generation
        try:
            # REUG 5.1 Fix: Secure & consistent environment variable loading
            # Following Sacred Law #2: Secure Environment Source-of-Truth

            api_key = None

            # First try nested config structure (as passed by orchestrator)
            # Config comes as {"creator": {"gemini_api_key": "...", ...}}
            for plugin_name, plugin_config in config.items():
                if isinstance(plugin_config, dict):
                    api_key = plugin_config.get("gemini_api_key")
                    if api_key:
                        logger.debug(
                            f"CREATOR: Found API key in nested config for {plugin_name}"
                        )
                        break

            # Fallback to direct config access
            if not api_key:
                api_key = config.get("gemini_api_key", "")
                if api_key:
                    logger.debug("CREATOR: Found API key in direct config")

            # Handle environment variable substitution (shouldn't be needed if main.py does it)
            if api_key and api_key.startswith("${") and api_key.endswith("}"):
                env_var_name = api_key[2:-1]  # Remove ${ and }
                api_key = os.getenv(env_var_name)
                logger.debug(f"CREATOR: Resolved environment variable {env_var_name}")

            # Final fallback to environment variable
            if not api_key:
                api_key = os.getenv("GEMINI_API_KEY", "")
                if api_key:
                    logger.debug("CREATOR: Using direct environment variable fallback")

            # Store API key and configure REST client
            if api_key:
                self.api_key = api_key
                self.model_name = config.get("llm_model", "gemini-1.5-flash") or next(
                    (
                        pc.get("llm_model", "gemini-1.5-flash")
                        for pc in config.values()
                        if isinstance(pc, dict)
                    ),
                    "gemini-1.5-flash",
                )
                self.llm_available = True
                logger.info(
                    f"CREATOR: Gemini API configured for model {self.model_name}"
                )
            else:
                self.llm_available = False
                logger.warning(
                    "CREATOR: No Gemini API key found - using template generation"
                )

        except Exception as e:
            logger.exception(f"CREATOR: Failed to initialize Gemini client: {e}")
            self.llm_available = False

        # Get tool registry reference
        self.registry = get_tool_registry()

        # Statistics
        self.stats = {
            "gaps_detected": 0,
            "tools_created": 0,
            "generation_errors": 0,
            "validation_errors": 0,
        }

        logger.info("CreatorPlugin setup complete")

    async def start(self):
        """Start the CREATOR plugin and subscribe to events."""
        await super().start()

        # Subscribe to gap detection events
        await self.subscribe("atom_gap", self._handle_gap_event)

        # Subscribe to "show me you created it" requests
        await self.subscribe("show_created_tool", self._handle_show_created_tool)

        # Subscribe to compose_request events for deterministic tool creation
        await self.subscribe("compose_request", self._handle_compose_request)

        logger.info("CreatorPlugin started - ready to create tools on-demand")

    async def shutdown(self):
        """Shutdown the CREATOR plugin."""
        logger.info(f"CreatorPlugin shutdown - Stats: {self.stats}")
        # Base class shutdown is abstract, so we don't call super()

    async def _handle_gap_event(self, event: AtomGapEvent):
        """Handle detected tool gaps by creating new tools."""
        try:
            self.stats["gaps_detected"] += 1

            logger.info(f"🔧 CREATOR: Handling gap for tool '{event.missing_tool}'")

            # Generate tool code
            code = await self._generate_tool_code(event.missing_tool, event.description)

            if not code:
                logger.error(
                    f"❌ CREATOR: Failed to generate code for '{event.missing_tool}'"
                )
                self.stats["generation_errors"] += 1
                return

            # Validate the generated code
            if not await self._validate_code(code):
                logger.error(
                    f"❌ CREATOR: Code validation failed for '{event.missing_tool}'"
                )
                self.stats["validation_errors"] += 1
                return

            # Register the tool
            await self._register_tool(event.missing_tool, code, event.description)

            # Emit AtomReadyEvent to notify other components
            ready_event = AtomReadyEvent(
                source_plugin=self.name,
                atom={
                    "name": event.missing_tool,
                    "description": event.description,
                    "code": code,
                    "gap_id": event.gap_id,
                },
            )

            await self.event_bus.publish(ready_event)

            self.stats["tools_created"] += 1
            logger.info(f"✅ CREATOR: Successfully created tool '{event.missing_tool}'")

        except Exception as e:
            logger.error(f"❌ CREATOR: Error handling gap event: {e}")
            self.stats["generation_errors"] += 1

    async def _handle_show_created_tool(self, event: ShowCreatedToolRequest):
        """Handle requests to show proof that a tool was created."""
        try:
            # Generate deterministic tool ID (use placeholder code for lookup)
            # Since we don't have the code yet, we'll check all generated tools
            generated_dir = Path("tools/generated")
            found_files = []

            if generated_dir.exists():
                # Look for files containing the tool name
                for tool_file in generated_dir.glob("*.py"):
                    try:
                        with tool_file.open("r") as f:
                            content = f.read()
                            if (
                                event.tool_name in content
                                or event.tool_name in tool_file.name
                            ):
                                found_files.append((tool_file, content))
                    except Exception:
                        continue

            if found_files:
                # Use the first match
                tool_path, tool_code = found_files[0]
                tool_id = deterministic_tool_id(event.tool_name, tool_code)

                # Emit response with proof of creation
                proof_event = ToolCreatedEvent(
                    source_plugin=self.name,
                    tool_id=tool_id,
                    name=event.tool_name,
                    code_path=str(tool_path),
                    contract={"description": "Generated tool", "size": len(tool_code)},
                    registry_key=event.tool_name,
                )

                await self.event_bus.publish(proof_event)
                logger.info(
                    f"📋 CREATOR: Provided proof for tool '{event.tool_name}' (ID: {tool_id})"
                )
            else:
                # Tool not found
                proof_event = ToolCreatedEvent(
                    source_plugin=self.name,
                    tool_id="not_found",
                    name=event.tool_name,
                    code_path="",
                    contract={"error": "Tool not found"},
                    registry_key=event.tool_name,
                )

                await self.event_bus.publish(proof_event)
                logger.warning(
                    f"🔍 CREATOR: Tool '{event.tool_name}' not found in generated tools"
                )

        except Exception as e:
            logger.error(f"❌ CREATOR: Error showing created tool: {e}")

    async def _handle_compose_request(self, event: ComposeRequestEvent):
        """Handle compose_request events by creating tools deterministically."""
        try:
            # Extract tool name from goal or params
            tool_name = event.params.get("tool_name", event.goal)
            description = event.params.get("description", f"Tool for: {event.goal}")

            logger.info(f"🔧 CREATOR: Handling compose_request for '{tool_name}'")

            # Generate tool code using the existing pipeline
            code = await self._generate_tool_code(tool_name, description)

            if not code:
                logger.error(f"❌ CREATOR: Failed to generate code for '{tool_name}'")
                self.stats["generation_errors"] += 1
                return

            # Validate the generated code
            if not await self._validate_code(code):
                logger.error(f"❌ CREATOR: Code validation failed for '{tool_name}'")
                self.stats["validation_errors"] += 1
                return

            # Register the tool
            await self._register_tool(tool_name, code, description)

            # Emit AtomReadyEvent to notify other components
            ready_event = AtomReadyEvent(
                source_plugin=self.name,
                atom={
                    "name": tool_name,
                    "description": description,
                    "code": code,
                    "goal": event.goal,
                    "compose_request_id": getattr(event, "request_id", "deterministic"),
                },
            )

            await self.event_bus.publish(ready_event)

            self.stats["tools_created"] += 1
            logger.info(
                f"✅ CREATOR: Successfully created tool '{tool_name}' from compose_request"
            )

        except Exception as e:
            logger.error(f"❌ CREATOR: Error handling compose_request: {e}")
            self.stats["generation_errors"] += 1

    async def _generate_tool_code(self, tool_name: str, description: str) -> str | None:
        """Generate Python code for the requested tool."""

        if self.llm_available:
            # Try LLM first with improved error handling
            try:
                llm_code = await self._generate_with_llm(tool_name, description)
                if llm_code:
                    cleaned_code = self._clean_llm_response(llm_code)
                    if cleaned_code and await self._validate_code(cleaned_code):
                        logger.info(
                            f"CREATOR: LLM generation successful for '{tool_name}'"
                        )
                        return cleaned_code
                    logger.warning(
                        f"CREATOR: LLM code validation failed for '{tool_name}', falling back to template"
                    )
                else:
                    logger.warning(
                        f"CREATOR: Empty LLM response for '{tool_name}', falling back to template"
                    )
            except Exception as e:
                logger.warning(
                    f"CREATOR: LLM generation failed for '{tool_name}': {e}, falling back to template"
                )

            # Fall back to template
            return self._generate_template_code(tool_name, description)
        logger.info(
            f"CREATOR: Using template generation for '{tool_name}' (no LLM available)"
        )
        return self._generate_template_code(tool_name, description)

    async def _generate_with_llm(self, tool_name: str, description: str) -> str | None:
        """Generate tool code using Gemini REST API."""
        try:
            # Sanitize tool name for function safety
            safe_name = self._sanitize_tool_name(tool_name)

            prompt = f"""Generate a Python function for a dynamic tool system.

Tool Name: {safe_name}
Description: {description}

Requirements:
1. Function must be named exactly: {safe_name}
2. Function signature: def {safe_name}(**kwargs):
3. Must return a dictionary with results
4. Include try/except error handling
5. Add docstring with description
6. Use only standard library imports (uuid, json, math, etc.)
7. No dangerous operations (os, sys, subprocess, eval, exec)

Template structure:
```python
import uuid

def {safe_name}(**kwargs):
    \"\"\"
    {description}
    \"\"\"
    try:
        # Implementation here
        result = {{"value": "implementation_result"}}
        return result
    except Exception as e:
        return {{"error": str(e)}}
```

Generate ONLY the function code, no markdown formatting or explanations."""

            # Use REST API to call Gemini
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent?key={self.api_key}"

            payload = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "temperature": 0.1,
                    "topP": 0.8,
                    "topK": 40,
                    "maxOutputTokens": 2048,
                },
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        if (
                            "candidates" in result
                            and len(result["candidates"]) > 0
                            and "content" in result["candidates"][0]
                            and "parts" in result["candidates"][0]["content"]
                            and len(result["candidates"][0]["content"]["parts"]) > 0
                        ):
                            raw_code = result["candidates"][0]["content"]["parts"][0][
                                "text"
                            ].strip()
                            cleaned_code = self._clean_llm_response(raw_code)
                            logger.info(
                                f"CREATOR: Successfully generated code for {tool_name} using Gemini API"
                            )
                            return cleaned_code
                        logger.warning(
                            f"CREATOR: Empty or malformed response from Gemini API for {tool_name}"
                        )
                    else:
                        error_text = await response.text()
                        logger.error(
                            f"CREATOR: Gemini API error {response.status}: {error_text}"
                        )

            return None

        except Exception:
            logger.exception(f"CREATOR: LLM generation failed for {tool_name}")
            return None

    def _clean_llm_response(self, raw_code: str) -> str:
        """Clean up LLM response by removing markdown formatting and other artifacts."""
        if not raw_code:
            return ""

        # Remove BOM if present first
        cleaned = raw_code
        if cleaned.startswith("\ufeff"):
            cleaned = cleaned[1:]

        # Split into lines for processing
        lines = cleaned.strip().split("\n")

        # Remove opening markdown block
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]

        # Remove closing markdown block and anything after it
        code_end_idx = len(lines)
        for i, line in enumerate(lines):
            if line.strip().startswith("```"):
                code_end_idx = i
                break
        lines = lines[:code_end_idx]

        # Join back and clean up
        cleaned = "\n".join(lines).strip()

        # Remove common leading text artifacts
        if cleaned.startswith("python\n") or cleaned.startswith("python "):
            cleaned = cleaned[7:]

        return cleaned

    def _sanitize_tool_name(self, tool_name: str) -> str:
        """Sanitize tool name to avoid Python keyword conflicts and ensure valid identifier."""

        # Remove special characters and spaces, replace with underscores
        sanitized = re.sub(r"[^a-zA-Z0-9_]", "_", tool_name)

        # Ensure it starts with a letter or underscore
        if not sanitized[0].isalpha() and sanitized[0] != "_":
            sanitized = "tool_" + sanitized

        # Avoid Python keywords
        if keyword.iskeyword(sanitized):
            sanitized = sanitized + "_tool"

        # Avoid common problematic names
        problematic = [
            "planner",
            "executor",
            "plugin",
            "main",
            "import",
            "class",
            "def",
        ]
        if sanitized.lower() in problematic:
            sanitized = sanitized + "_atom"

        return sanitized

    def _generate_template_code(self, tool_name: str, description: str) -> str:
        """Generate a bullet-proof template that always compiles and publishes events."""
        safe_name = self._sanitize_tool_name(tool_name)

        # Smart logic selection based on tool name and description
        core_logic = self._get_smart_logic(tool_name, description)

        return f'''import uuid
import asyncio
import logging
from src.core.events import ToolResultEvent

logger = logging.getLogger(__name__)

async def {safe_name}(event_bus=None, tool_call_id="", session_id="", conversation_id="", **kwargs):
    """
    {description}

    Auto-generated tool that publishes results via event bus.
    Generated by CREATOR Plugin with guaranteed event contract.
    """
    result = dict()
    success = False

    try:
        # {description}
        {core_logic}
        success = True
        logger.info(f"{tool_name} executed successfully")

    except Exception as e:
        logger.exception(f"{tool_name} execution failed")
        result = {{"error": str(e)}}
        success = False

    # Always emit ToolResultEvent if event_bus is available
    if event_bus and hasattr(event_bus, 'publish'):
        try:
            tool_result = ToolResultEvent(
                source_plugin="{tool_name}",
                tool_call_id=tool_call_id,
                session_id=session_id,
                conversation_id=conversation_id,
                success=success,
                result=result
            )

            # Handle async event publishing
            if asyncio.iscoroutinefunction(event_bus.publish):
                await event_bus.publish(tool_result)
            else:
                # Sync publishing fallback
                event_bus.publish(tool_result)

            logger.debug(f"{tool_name} result event published")

        except Exception as e:
            logger.error(f"{tool_name} failed to publish result event: {{str(e)}}")

    return result
'''

    def _get_smart_logic(self, tool_name: str, description: str) -> str:
        """Generate smart logic based on tool name and description patterns."""

        # Check for fibonacci pattern
        if "fibonacci" in tool_name.lower() or "fibonacci" in description.lower():
            return """        # Fibonacci calculation logic
        n = kwargs.get("n", kwargs.get("number", 10))  # Default to 10 if not specified
        if n <= 0:
            value = 0
        elif n == 1:
            value = 1
        else:
            a, b = 0, 1
            for _ in range(n):
                a, b = b, a + b
            value = a
        result = {"value": value, "n": n, "description": f"fibonacci({n}) = {value}"}"""

        # Check for calculator pattern
        if (
            "calculator" in tool_name.lower()
            or "math" in tool_name.lower()
            or "calculate" in description.lower()
        ):
            return """        # Calculator logic
        operation = kwargs.get("operation", "add")
        a = kwargs.get("a", kwargs.get("x", 1))
        b = kwargs.get("b", kwargs.get("y", 1))

        if operation == "add":
            value = a + b
        elif operation in ["subtract", "sub"]:
            value = a - b
        elif operation in ["multiply", "mul"]:
            value = a * b
        elif operation in ["divide", "div"]:
            value = a / b if b != 0 else 0
        else:
            value = a + b  # Default to addition
        result = {"value": value, "operation": operation, "a": a, "b": b}"""

        # Check for text processing pattern
        if (
            "text" in tool_name.lower()
            or "string" in tool_name.lower()
            or "reverse" in description.lower()
        ):
            return """        # Text processing logic
        text = kwargs.get("text", kwargs.get("input", "hello"))
        operation = kwargs.get("operation", "reverse")

        if operation == "reverse":
            value = text[::-1]
        elif operation == "upper":
            value = text.upper()
        elif operation == "lower":
            value = text.lower()
        else:
            value = text
        result = {"value": value, "original": text, "operation": operation}"""

        # Default generic logic
        return """        # Generic tool logic
        value = kwargs.get("input", "completed")
        if isinstance(value, str):
            result = {"value": value, "status": "completed", "description": f"Processed: {value}"}
        else:
            result = {"value": str(value), "status": "completed", "type": type(value).__name__}"""

    def _get_implementation_logic(self, tool_name: str, description: str) -> str:
        """Get specific implementation logic based on tool type."""

        # Smart template selection based on tool name patterns
        if "fibonacci" in tool_name.lower():
            return """        n = params.get("n", 0)
        value = 0
        if n <= 0:
            value = 0
        elif n == 1:
            value = 1
        else:
            a, b = 0, 1
            for _ in range(n):
                a, b = b, a + b
            value = a
        result = {"value": value}"""

        if "calculator" in tool_name.lower() or "math" in tool_name.lower():
            return """        operation = params.get("operation", "add")
        a = params.get("a", 0)
        b = params.get("b", 0)

        if operation == "add":
            value = a + b
        elif operation == "subtract":
            value = a - b
        elif operation == "multiply":
            value = a * b
        elif operation == "divide":
            value = a / b if b != 0 else 0
        else:
            value = 0
        result = {"value": value}"""

        if "text" in tool_name.lower() or "reverse" in tool_name.lower():
            return """        text = params.get("text", "")
        operation = params.get("operation", "reverse")

        if operation == "reverse":
            value = text[::-1]
        elif operation == "word_count":
            value = len(text.split())
        elif operation == "upper":
            value = text.upper()
        elif operation == "lower":
            value = text.lower()
        else:
            value = text.strip()
        result = {"value": value}"""

        if "doubler" in tool_name.lower():
            return """        number = params.get("number", 0)
        value = number * 2
        result = {"value": value}"""

        # Generic implementation
        return """        input_data = params.get("input", "")
        # Generic processing logic
        value = f"Processed: {input_data}"
        result = {"value": value}"""

    def _fibonacci_template(self, tool_name: str, description: str) -> str:
        """Template for Fibonacci sequence tools."""
        return f'''
def {tool_name}(n: int) -> int:
    """
    {description}
    Safe, iterative, non-recursive, O(n) Fibonacci implementation.

    Args:
        n: The position in the Fibonacci sequence (0-indexed)

    Returns:
        The Fibonacci number at position n
    """
    if not isinstance(n, int) or n < 0:
        raise ValueError("n must be a non-negative integer")

    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a
'''

    def _math_template(self, tool_name: str, description: str) -> str:
        """Template for mathematical tools."""
        return f'''
def {tool_name}(expression: str) -> float:
    """
    {description}

    Args:
        expression: Mathematical expression to evaluate

    Returns:
        The result of the calculation
    """
    import ast
    import operator

    # Safe evaluation of mathematical expressions
    operators = {{
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }}

    def eval_expr(node):
        if isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.BinOp):
            return operators[type(node.op)](eval_expr(node.left), eval_expr(node.right))
        elif isinstance(node, ast.UnaryOp):
            return operators[type(node.op)](eval_expr(node.operand))
        else:
            raise TypeError(f"Unsupported operation: {{type(node)}}")

    try:
        tree = ast.parse(expression, mode='eval')
        return eval_expr(tree.body)
    except Exception as e:
        raise ValueError(f"Invalid expression: {{e}}")
'''

    def _text_template(self, tool_name: str, description: str) -> str:
        """Template for text processing tools."""
        return f'''
def {tool_name}(text: str, operation: str = "process") -> str:
    """
    {description}

    Args:
        text: Input text to process
        operation: Type of operation to perform

    Returns:
        Processed text
    """
    if operation == "upper":
        return text.upper()
    elif operation == "lower":
        return text.lower()
    elif operation == "reverse":
        return text[::-1]
    elif operation == "word_count":
        return str(len(text.split()))
    else:
        return text.strip()
'''

    def _generic_template(self, tool_name: str, description: str) -> str:
        """Generic template for unknown tool types."""
        return f'''
def {tool_name}(input_data: str) -> str:
    """
    {description}

    Args:
        input_data: Input data for the tool

    Returns:
        Processed result
    """
    # Generic processing logic
    result = f"Processed: {{input_data}}"
    return result
'''

    async def _validate_code(self, code: str) -> bool:
        """Validate generated code for syntax and basic safety."""
        try:
            # Syntax validation
            tree = ast.parse(code)

            # Advanced safety checks using AST
            dangerous_functions = ["eval", "exec", "compile", "__import__"]
            dangerous_modules = ["os", "sys", "subprocess"]

            for node in ast.walk(tree):
                # Check for dangerous function calls
                if isinstance(node, ast.Call):
                    if (
                        isinstance(node.func, ast.Name)
                        and node.func.id in dangerous_functions
                    ):
                        logger.warning(
                            f"CREATOR: Dangerous function call '{node.func.id}' detected"
                        )
                        return False

                # Check for dangerous imports
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name in dangerous_modules:
                            logger.warning(
                                f"CREATOR: Dangerous import '{alias.name}' detected"
                            )
                            return False

                elif isinstance(node, ast.ImportFrom):
                    if node.module in dangerous_modules:
                        logger.warning(
                            f"CREATOR: Dangerous import from '{node.module}' detected"
                        )
                        return False

            # Compile check - Windows-friendly approach
            try:
                compile(code, "<string>", "exec")
                logger.info("CREATOR: Code compilation successful")
                return True
            except SyntaxError as e:
                logger.error(f"CREATOR: Syntax error in generated code: {e}")
                return False

        except Exception as e:
            logger.error(f"CREATOR: Code validation error: {e}")
            return False

    async def _register_tool(self, tool_name: str, code: str, description: str):
        """Register the generated tool in the dynamic registry."""
        try:
            # Generate deterministic tool ID
            tool_id = deterministic_tool_id(tool_name, code)

            # Create tools/generated directory if it doesn't exist
            generated_dir = Path("tools/generated")
            generated_dir.mkdir(parents=True, exist_ok=True)

            # Save to tools/generated/<tool_id>.py
            tool_file = generated_dir / f"{tool_id}.py"
            tool_file.write_text(code)

            # Register in the dynamic tool registry
            self.registry.register_tool(
                tool_name=tool_name,
                code=code,
                description=description,
                author="CREATOR Plugin",
            )

            # Emit ToolCreatedEvent for observability
            created_event = ToolCreatedEvent(
                source_plugin=self.name,
                tool_id=tool_id,
                name=tool_name,
                code_path=str(tool_file),
                contract={"description": description},
                registry_key=tool_name,
            )

            await self.event_bus.publish(created_event)

            logger.info(
                f"CREATOR: Tool '{tool_name}' (ID: {tool_id}) registered and saved to {tool_file}"
            )

        except Exception as e:
            logger.error(f"CREATOR: Failed to register tool '{tool_name}': {e}")
            raise
