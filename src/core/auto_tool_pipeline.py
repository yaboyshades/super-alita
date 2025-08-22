import json
import logging
import textwrap
from pathlib import Path
from typing import Any

from src.core.event_bus import EventBus
from src.core.neural_atom import NeuralStore
from src.core.prompt_manager import get_prompt_manager
from src.core.sandbox_runner import SandboxRunner

logger = logging.getLogger(__name__)


class AutoToolPipeline:
    """Receives 'atom_gap_request' events and emits fully-working atoms."""

    def __init__(self, store: NeuralStore, llm_client, sandbox: SandboxRunner):
        self.store = store
        self.llm = llm_client
        self.sandbox = sandbox
        self.event_bus = None
        self.prompt_manager = get_prompt_manager()

    async def start(self, event_bus: EventBus) -> None:
        """Start the pipeline and subscribe to events."""
        self.event_bus = event_bus
        await event_bus.subscribe("atom_gap_request", self._handle_gap)
        logger.info("🔧 AutoToolPipeline started - ready to generate tools")

    async def _handle_gap(self, event):
        """Handle requests for new atoms/tools."""
        try:
            data = event.model_dump() if hasattr(event, "model_dump") else event
            goal = data.get("goal", "")
            data.get("session_id", "default")
            if not goal:
                logger.warning("Received atom_gap_request with empty goal")
                return
            logger.info(f"🎯 Processing atom gap request: {goal}")
            # Generate tool specification
            spec = await self._brainstorm_spec(goal)
            logger.info(f"📋 Generated spec for tool: {spec.get('key', 'unknown')}")
            # Generate code
            code = await self._generate_code(spec)
            logger.info(f"💻 Generated code (length: {len(code)} chars)")
            # Test the code
            ok, log = await self._test_code(code, spec)
            if ok:
                logger.info("✅ Code test passed")
                await self._register_atom(code, spec)
                # Emit atom ready event
                if self.event_bus:
                    from src.core.events import AtomReadyEvent

                    ready_event = AtomReadyEvent(
                        source_plugin="auto_tools",
                        atom={
                            "key": spec["key"],
                            "name": spec["name"],
                            "description": spec["description"],
                            "signature": spec["signature"],
                        },
                    )
                    await self.event_bus.publish(ready_event)
                # Emit success event
                if self.event_bus:
                    from src.core.events import SystemEvent

                    success_event = SystemEvent(
                        source_plugin="auto_tools",
                        level="info",
                        message=f"Successfully created new tool: {spec['name']}",
                        component="auto_tool_pipeline",
                    )
                    await self.event_bus.publish(success_event)
            else:
                logger.error(f"❌ Code test failed: {log}")
                # Emit failure event
                if self.event_bus:
                    from src.core.events import SystemEvent

                    failure_event = SystemEvent(
                        source_plugin="auto_tools",
                        level="error",
                        message=f"Failed to create tool: {log}",
                        component="auto_tool_pipeline",
                    )
                    await self.event_bus.publish(failure_event)
        except Exception as e:
            logger.error(f"Error handling atom gap request: {e}")
            # Emit error event
            if self.event_bus:
                from src.core.events import SystemEvent

                error_event = SystemEvent(
                    source_plugin="auto_tools",
                    level="error",
                    message=f"Auto-tool pipeline error: {e!s}",
                    component="auto_tool_pipeline",
                )
                await self.event_bus.publish(error_event)

    async def _brainstorm_spec(self, goal: str) -> dict[str, Any]:
        """Generate tool specification using LLM."""
        # Get JSON schema from prompts.json
        json_schema = '{\n  "key": "<python_identifier>",\n  "name": "<human_readable_name>",\n  "description": "<what_it_does>",\n  "signature": {"param1": {"type": "str", "description": "parameter description"}},\n  "dependencies": ["requests"],\n  "example": {"param1": "example_value"}\n}'
        # Format the prompt using template
        prompt = self.prompt_manager.get_prompt(
            "auto_tool_pipeline.brainstorm_spec", goal=goal, json_schema=json_schema
        )
        try:
            raw = await self.llm.generate(prompt, max_tokens=400)
            logger.debug(f"🧠 LLM response: {raw[:200]}...")
            # Try to extract JSON from response
            raw = raw.strip()
            if "```json" in raw:
                raw = raw.split("```json")[1].split("```")[0].strip()
            elif "```" in raw:
                raw = raw.split("```")[1].split("```")[0].strip()
            spec = json.loads(raw)
            # Validate required fields
            required_fields = [
                "key",
                "name",
                "description",
                "signature",
                "dependencies",
                "example",
            ]
            for field in required_fields:
                if field not in spec:
                    raise ValueError(f"Missing required field: {field}")
            return spec
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON from LLM: {raw}")
            raise ValueError(f"Invalid JSON from LLM: {raw}") from e
        except Exception as e:
            logger.error(f"Error in _brainstorm_spec: {e}")
            raise

    async def _generate_code(self, spec: dict[str, Any]) -> str:
        """Generate Python code for the tool."""
        sig_params = []
        for param, info in spec["signature"].items():
            param_type = info.get("type", "str")
            sig_params.append(f"{param}: {param_type}")
        sig = ", ".join(sig_params)
        # Format the prompt using template
        prompt = self.prompt_manager.get_prompt(
            "auto_tool_pipeline.generate_code",
            class_name=f"{spec['key'].title()}Atom",
            tool_key=spec["key"],
            description=spec["description"],
            dependencies=spec["dependencies"],
            signature=sig,
            example=json.dumps(spec["example"]),
        )
        try:
            code = await self.llm.generate(prompt, max_tokens=800)
            logger.debug(f"💻 Generated code preview: {code[:150]}...")
            return code.strip()
        except Exception as e:
            logger.error(f"Error generating code: {e}")
            raise

    async def _test_code(self, code: str, spec: dict[str, Any]) -> tuple[bool, str]:
        """Test the generated code in sandbox."""
        # Create test module with robust class detection
        module_code = textwrap.dedent(
            f"""
import asyncio
import json
import sys
import traceback
# Generated Atom Code:
{code}
async def test_tool():
    try:
        # Dynamically find the atom class in globals
        expected_class_name = "{spec["key"].title()}Atom"
        atom_class = globals().get(expected_class_name)
        if atom_class is None:
            # Look for any class that ends with 'Atom'
            atom_classes = [name for name in globals() if name.endswith('Atom') and callable(globals()[name])]
            if atom_classes:
                atom_class = globals()[atom_classes[0]]
                print(f"INFO: Using discovered class: {{atom_classes[0]}}", file=sys.stderr)
            else:
                raise ValueError(f"No atom class found. Expected '{{expected_class_name}}' or any class ending with 'Atom'.")
        # Instantiate the atom
        atom = atom_class()
        # Test the call method with example parameters
        result = await atom.call(**{json.dumps(spec["example"])})
        # Validate result structure
        if not isinstance(result, dict):
            raise ValueError(f"Expected dict result, got {{type(result)}}")
        if "summary" not in result:
            raise ValueError("Result missing required 'summary' field")
        print(json.dumps({{"success": True, "result": result}}))
    except Exception as e:
        error_info = {{
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }}
        print(json.dumps(error_info))
if __name__ == "__main__":
    asyncio.run(test_tool())
"""
        )
        success, output = await self.sandbox.run(module_code)
        if success:
            try:
                result = json.loads(output)
                if result.get("success"):
                    return True, f"Test passed: {result['result']['summary']}"
                return False, f"Test failed: {result.get('error', 'Unknown error')}"
            except json.JSONDecodeError:
                return False, f"Invalid test output: {output}"
        else:
            return False, f"Sandbox execution failed: {output}"

    async def _register_atom(self, code: str, spec: dict[str, Any]) -> None:
        """Register the new atom in the system."""
        try:
            # Create atoms directory if it doesn't exist
            atoms_dir = Path("src/atoms")
            atoms_dir.mkdir(exist_ok=True)
            # Write code to file
            filename = f"auto_{spec['key']}.py"
            file_path = atoms_dir / filename
            file_path.write_text(code)
            logger.info(f"📁 Saved tool code to: {file_path}")
            # Store in neural memory
            if hasattr(self.store, "upsert"):
                await self.store.upsert(
                    content={
                        "key": spec["key"],
                        "name": spec["name"],
                        "description": spec["description"],
                        "signature": spec["signature"],
                        "source_file": str(file_path),
                        "source_code": code,
                        "auto_generated": True,
                    },
                    hierarchy_path=["tools", "auto"],
                    owner_plugin="auto_tools",
                )
                logger.info(f"🧠 Stored tool in neural memory: {spec['key']}")
        except Exception as e:
            logger.error(f"Error registering atom: {e}")
            raise
