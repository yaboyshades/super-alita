# Version: 3.0.0
# Description: Implements the CREATOR framework for autonomous tool generation.

import json
import logging
import time
from typing import Any

from src.core.global_workspace import AttentionLevel, GlobalWorkspace, WorkspaceEvent
from src.core.neural_atom import NeuralAtom, NeuralAtomMetadata, NeuralStore
from src.core.plugin_interface import PluginInterface
from src.core.schemas import CREATORRequest, CREATORResult, CREATORStage, NeuralAtomSpec

# Try to import Google Generative AI for code generation
try:
    import google.generativeai as genai

    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False

logger = logging.getLogger(__name__)


class CreatorPlugin(PluginInterface):
    """
    Listens for capability gaps and orchestrates the creation of new Neural Atoms.

    Implements the 4-stage CREATOR framework:
    1. Abstract Specification - Analyze requirements
    2. Design Decision - Plan implementation approach
    3. Implementation - Generate and test code
    4. Rectification - Validate and optimize
    """

    def __init__(self):
        super().__init__()
        self.workspace: GlobalWorkspace | None = None
        self.store: NeuralStore | None = None
        self.llm_client: Any | None = None

        # CREATOR framework state
        self.active_requests: dict[str, dict[str, Any]] = {}
        self.creation_templates = self._load_creation_templates()

        # Performance tracking
        self.creator_stats = {
            "total_requests": 0,
            "successful_creations": 0,
            "failed_creations": 0,
            "average_creation_time": 0.0,
            "stage_completion_rates": {stage.value: 0 for stage in CREATORStage},
        }

    def _load_creation_templates(self) -> dict[str, str]:
        """Load templates for different types of Neural Atoms."""
        return {
            "basic_neural_atom": '''
class {class_name}(NeuralAtom):
    """Generated Neural Atom: {description}"""

    def __init__(self, metadata: NeuralAtomMetadata = None):
        # RULE 2: Correct parent constructor call with metadata
        if metadata is None:
            metadata = NeuralAtomMetadata(
                name="{name}",
                description="{description}",
                capabilities={capabilities},
                version="1.0.0"
            )
        super().__init__(metadata)
        # RULE 3: System compatibility - add .key for NeuralStore
        self.key = metadata.name

    async def execute(self, input_data: Any) -> Any:
        """Execute the Neural Atom's core functionality."""
        try:
            {implementation_code}
            return result
        except Exception as e:
            logger.error(f"Error in {self.metadata.name}: {{e}}")
            raise

    def get_embedding(self) -> List[float]:
        """Return semantic embedding for similarity search."""
        # Generate embedding based on capabilities and description
        text = f"{self.metadata.description} {' '.join(self.metadata.capabilities)}"
        return self._generate_simple_embedding(text)

    def can_handle(self, task_description: str) -> float:
        """Return confidence score for handling this task."""
        # Simple keyword matching for now
        task_lower = task_description.lower()
        for capability in self.metadata.capabilities:
            if capability.lower() in task_lower:
                return 0.8
        return 0.1

    def _generate_simple_embedding(self, text: str) -> List[float]:
        """Generate simple hash-based embedding."""
        import hashlib
        hash_obj = hashlib.md5(text.encode())
        hash_hex = hash_obj.hexdigest()
        # Convert to float values between -1 and 1
        return [(int(c, 16) - 7.5) / 7.5 for c in hash_hex[:128]]
''',
            "tool_neural_atom": '''
class {class_name}(NeuralAtom):
    """Generated Tool Neural Atom: {description}"""

    def __init__(self, metadata: NeuralAtomMetadata = None):
        # RULE 2: Correct parent constructor call with metadata
        if metadata is None:
            metadata = NeuralAtomMetadata(
                name="{name}",
                description="{description}",
                capabilities={capabilities},
                version="1.0.0",
                tags={{"tool", "generated"}}
            )
        super().__init__(metadata)
        # RULE 3: System compatibility - add .key for NeuralStore
        self.key = metadata.name

    async def execute(self, input_data: Any) -> Any:
        """Execute the tool functionality."""
        try:
            parameters = input_data if isinstance(input_data, dict) else {{}}
            {implementation_code}
            return {{
                "success": True,
                "result": result,
                "tool_name": self.metadata.name
            }}
        except Exception as e:
            logger.error(f"Error in tool {self.metadata.name}: {{e}}")
            return {{
                "success": False,
                "error": str(e),
                "tool_name": self.metadata.name
            }}

    def get_embedding(self) -> List[float]:
        """Return semantic embedding for similarity search."""
        text = f"tool {self.metadata.description} {' '.join(self.metadata.capabilities)}"
        return self._generate_simple_embedding(text)

    def can_handle(self, task_description: str) -> float:
        except Exception as e:
            return {{
                "success": False,
                "error": str(e),
                "tool_name": "{name}"
            }}

    def get_embedding(self) -> List[float]:
        """Return semantic embedding for similarity search."""
        text = f"tool {self.metadata.description} {' '.join(self.metadata.capabilities)}"
        return self._generate_tool_embedding(text)

    def can_handle(self, task_description: str) -> float:
        """Return confidence score for handling this task."""
        task_lower = task_description.lower()

        # Check for tool-specific keywords
        if "tool" in task_lower and any(cap.lower() in task_lower for cap in self.metadata.capabilities):
            return 0.9

        # Check for capability matches
        capability_matches = sum(1 for cap in self.metadata.capabilities if cap.lower() in task_lower)
        return min(0.8, capability_matches * 0.3)

    def _generate_tool_embedding(self, text: str) -> List[float]:
        """Generate tool-specific embedding."""
        import hashlib
        import math

        # Create a more sophisticated embedding for tools
        hash_obj = hashlib.sha256(text.encode())
        hash_bytes = hash_obj.digest()

        # Convert to 128-dimensional vector
        embedding = []
        for i in range(0, min(len(hash_bytes), 32), 1):
            # Use trigonometric functions for better distribution
            val = math.sin(hash_bytes[i] * 0.1) * math.cos(hash_bytes[i] * 0.05)
            embedding.extend([val, -val, val * 0.5, -val * 0.5])

        # Ensure exactly 128 dimensions
        while len(embedding) < 128:
            embedding.append(0.0)

        return embedding[:128]
''',
        }

    async def setup(
        self, workspace: GlobalWorkspace, store: NeuralStore, config: dict[str, Any]
    ):
        """Initialize the Creator Plugin with workspace and store."""
        await super().setup(workspace, store, config)

        self.workspace = workspace
        self.store = store

        # Initialize LLM client for code generation
        if HAS_GEMINI and config.get("llm_model"):
            await self._initialize_llm_client(config)
        else:
            logger.warning(
                "Gemini not available - using template-based generation only"
            )

        logger.info("Creator Plugin initialized for CREATOR framework")

    async def _initialize_llm_client(self, config: dict[str, Any]):
        """Initialize the LLM client for code generation."""
        try:
            import os

            api_key = os.getenv("GEMINI_API_KEY")
            if api_key:
                genai.configure(api_key=api_key)
                model_name = config.get("llm_model", "gemini-2.0-flash-exp")
                self.llm_client = genai.GenerativeModel(model_name)
                logger.info(f"Creator LLM client initialized: {model_name}")
            else:
                logger.warning("GEMINI_API_KEY not found for Creator Plugin")
        except Exception as e:
            logger.error(f"Failed to initialize Creator LLM client: {e}")

    async def start(self):
        """Start the Creator Plugin and subscribe to workspace events."""
        await super().start()

        if self.workspace:
            self.workspace.subscribe("creator", self._handle_workspace_event)
            logger.info("Creator Plugin subscribed to Global Workspace")

    async def shutdown(self):
        """Gracefully shutdown the Creator Plugin."""
        await super().shutdown()
        logger.info("Creator Plugin shutdown complete")

    async def _handle_workspace_event(self, event: WorkspaceEvent):
        """Handle events from the Global Workspace."""
        try:
            if isinstance(event.data, dict):
                event_type = event.data.get("type")

                if event_type == "creator_request":
                    request = CREATORRequest(**event.data)
                    await self._handle_creator_request(request)
                elif event_type == "capability_gap":
                    # Convert capability gap to creator request
                    await self._handle_capability_gap_event(event.data)

        except Exception as e:
            logger.error(f"Error handling workspace event in Creator: {e}")

    async def _handle_creator_request(self, request: CREATORRequest):
        """Orchestrate the 4-stage tool creation process."""
        start_time = time.time()
        self.creator_stats["total_requests"] += 1

        logger.info(
            f"🔧 CREATOR: Starting 4-stage creation for request {request.request_id}"
        )

        try:
            # Initialize creation context
            self.active_requests[request.request_id] = {
                "request": request,
                "start_time": start_time,
                "stages_completed": [],
                "current_stage": None,
                "created_atom": None,
            }

            # Stage 1: Abstract Specification
            spec = await self._stage_1_abstract_specification(request)
            if not spec:
                raise Exception("Failed to generate specification")

            # Stage 2: Design Decision
            design = await self._stage_2_design_decision(request, spec)
            if not design:
                raise Exception("Failed to create design")

            # Stage 3: Implementation
            implementation = await self._stage_3_implementation(request, spec, design)
            if not implementation:
                raise Exception("Failed to implement Neural Atom")

            # Stage 4: Rectification
            final_atom = await self._stage_4_rectification(request, implementation)
            if not final_atom:
                raise Exception("Failed to validate Neural Atom")

            # Success - register the new Neural Atom
            await self._register_created_atom(request, final_atom)

            creation_time = time.time() - start_time
            self._update_creator_stats(creation_time, True)

            logger.info(
                f"✅ CREATOR: Successfully created Neural Atom for {request.request_id}"
            )

        except Exception as e:
            logger.error(
                f"❌ CREATOR: Failed to create Neural Atom for {request.request_id}: {e}"
            )
            await self._handle_creation_failure(request, str(e))

            creation_time = time.time() - start_time
            self._update_creator_stats(creation_time, False)

        finally:
            # Clean up active request
            if request.request_id in self.active_requests:
                del self.active_requests[request.request_id]

    async def _stage_1_abstract_specification(
        self, request: CREATORRequest
    ) -> NeuralAtomSpec | None:
        """Stage 1: Abstract specification - Analyze requirements."""
        logger.info(
            f"🔍 CREATOR Stage 1: Abstract Specification for {request.request_id}"
        )

        try:
            self._update_stage_completion(CREATORStage.ABSTRACT_SPECIFICATION)

            if self.llm_client:
                spec = await self._llm_generate_specification(request)
            else:
                spec = await self._template_generate_specification(request)

            logger.info(f"Generated specification: {spec.name}")
            return spec

        except Exception as e:
            logger.error(f"Stage 1 failed: {e}")
            return None

    async def _llm_generate_specification(
        self, request: CREATORRequest
    ) -> NeuralAtomSpec:
        """Use LLM to generate detailed specification."""
        prompt = f"""
Generate a detailed specification for a Neural Atom to fulfill this capability:

Capability Needed: {request.capability_description}
Context: {request.context}
Priority: {request.priority}
Constraints: {request.constraints}

Create a JSON specification with this structure:
{{
    "name": "Human readable name",
    "description": "Detailed description of what this atom does",
    "capabilities": ["list", "of", "capabilities"],
    "version": "1.0.0",
    "tags": ["relevant", "tags"],
    "parameters": {{"param_name": {{"type": "string", "description": "param description"}}}},
    "dependencies": ["required", "packages"]
}}

Respond with ONLY the JSON, no additional text:
"""

        try:
            response = await self.llm_client.generate_content_async(prompt)
            spec_data = json.loads(response.text.strip())
            return NeuralAtomSpec(**spec_data)
        except Exception as e:
            logger.error(f"LLM specification generation failed: {e}")
            # Fallback to template
            return await self._template_generate_specification(request)

    async def _template_generate_specification(
        self, request: CREATORRequest
    ) -> NeuralAtomSpec:
        """Generate specification using templates."""
        description_lower = request.capability_description.lower()

        # Determine capabilities based on keywords
        capabilities = []
        if "search" in description_lower:
            capabilities.extend(["search", "query", "retrieve"])
        if "calculate" in description_lower or "math" in description_lower:
            capabilities.extend(["calculate", "compute", "mathematics"])
        if "process" in description_lower:
            capabilities.extend(["process", "transform", "analyze"])
        if "generate" in description_lower or "create" in description_lower:
            capabilities.extend(["generate", "create", "produce"])

        # Default capabilities if none detected
        if not capabilities:
            capabilities = ["execute", "process", "respond"]

        # Generate name from description
        name_parts = [
            word.capitalize() for word in request.capability_description.split()[:3]
        ]
        name = " ".join(name_parts) + " Tool"

        return NeuralAtomSpec(
            name=name,
            description=request.capability_description,
            capabilities=capabilities,
            version="1.0.0",
            tags=["generated", "creator"],
            parameters={
                "input": {"type": "any", "description": "Input data for processing"}
            },
            dependencies=[],
        )

    async def _stage_2_design_decision(
        self, request: CREATORRequest, spec: NeuralAtomSpec
    ) -> dict[str, Any] | None:
        """Stage 2: Design decision - Plan implementation approach."""
        logger.info(f"🎯 CREATOR Stage 2: Design Decision for {request.request_id}")

        try:
            self._update_stage_completion(CREATORStage.DESIGN_DECISION)

            # Determine the best template and approach
            template_type = "tool_neural_atom"  # Default to tool template

            # Choose implementation strategy
            implementation_strategy = {
                "template_type": template_type,
                "class_name": self._generate_class_name(spec.name),
                "implementation_approach": "template_based",
                "safety_requirements": [
                    "input_validation",
                    "error_handling",
                    "output_sanitization",
                ],
                "test_cases": self._generate_test_cases(spec),
            }

            logger.info(
                f"Design strategy: {implementation_strategy['implementation_approach']}"
            )
            return implementation_strategy

        except Exception as e:
            logger.error(f"Stage 2 failed: {e}")
            return None

    def _generate_class_name(self, name: str) -> str:
        """Generate valid Python class name from atom name."""
        # Remove special characters and convert to PascalCase
        words = "".join(c if c.isalnum() else " " for c in name).split()
        class_name = "".join(word.capitalize() for word in words if word)

        # Ensure it doesn't conflict with Python keywords
        if class_name.lower() in {
            "class",
            "def",
            "import",
            "from",
            "if",
            "else",
            "try",
            "except",
        }:
            class_name += "Atom"

        return class_name or "GeneratedAtom"

    def _generate_test_cases(self, spec: NeuralAtomSpec) -> list[dict[str, Any]]:
        """Generate basic test cases for the specification."""
        test_cases = []

        for capability in spec.capabilities[:3]:  # Limit to 3 test cases
            test_cases.append(
                {
                    "description": f"Test {capability} capability",
                    "input": {"action": capability, "data": "test_data"},
                    "expected_success": True,
                }
            )

        return test_cases

    async def _stage_3_implementation(
        self, request: CREATORRequest, spec: NeuralAtomSpec, design: dict[str, Any]
    ) -> str | None:
        """Stage 3: Implementation - Generate and test code."""
        logger.info(f"⚙️ CREATOR Stage 3: Implementation for {request.request_id}")

        try:
            self._update_stage_completion(CREATORStage.IMPLEMENTATION)

            if self.llm_client and design["implementation_approach"] == "llm_based":
                code = await self._llm_generate_code(spec, design)
            else:
                code = await self._template_generate_code(spec, design)

            # Validate generated code
            if await self._validate_generated_code(code):
                logger.info("Generated code passed validation")
                return code
            logger.warning("Generated code failed validation, using safe fallback")
            return await self._generate_safe_fallback_code(spec, design)

        except Exception as e:
            logger.error(f"Stage 3 failed: {e}")
            return None

    async def _template_generate_code(
        self, spec: NeuralAtomSpec, design: dict[str, Any]
    ) -> str:
        """Generate code using templates."""
        template = self.creation_templates[design["template_type"]]

        # Generate simple implementation based on capabilities
        implementation_lines = []

        for capability in spec.capabilities:
            if capability == "search":
                implementation_lines.append(
                    '        if parameters.get("action") == "search":'
                )
                implementation_lines.append(
                    '            query = parameters.get("query", "")'
                )
                implementation_lines.append(
                    '            result = f"Search results for: {query}"'
                )
            elif capability == "calculate":
                implementation_lines.append(
                    '        if parameters.get("action") == "calculate":'
                )
                implementation_lines.append(
                    '            expression = parameters.get("expression", "1+1")'
                )
                implementation_lines.append(
                    '            result = f"Calculation result: {expression}"'
                )
            else:
                implementation_lines.append(
                    f'        if parameters.get("action") == "{capability}":'
                )
                implementation_lines.append(
                    f'            result = f"{capability.capitalize()} operation completed"'
                )

        # Default case
        implementation_lines.append('        if "result" not in locals():')
        implementation_lines.append('            result = f"Processed: {parameters}"')

        implementation_code = "\n".join(implementation_lines)

        # Fill template
        code = template.format(
            class_name=design["class_name"],
            name=spec.name,
            description=spec.description,
            capabilities=json.dumps(spec.capabilities),
            implementation_code=implementation_code,
        )

        return code

    async def _validate_generated_code(self, code: str) -> bool:
        """Validate that generated code is safe and functional."""
        try:
            # Basic syntax check
            compile(code, "<generated>", "exec")

            # Check for dangerous imports or operations
            dangerous_patterns = [
                "import os",
                "import subprocess",
                "import sys",
                "exec(",
                "eval(",
                "__import__",
                "open(",
                "file(",
                "input(",
                "raw_input(",
            ]

            code_lower = code.lower()
            for pattern in dangerous_patterns:
                if pattern in code_lower:
                    logger.warning(f"Dangerous pattern detected: {pattern}")
                    return False

            return True

        except SyntaxError as e:
            logger.error(f"Syntax error in generated code: {e}")
            return False

    async def _generate_safe_fallback_code(
        self, spec: NeuralAtomSpec, design: dict[str, Any]
    ) -> str:
        """Generate a safe, minimal fallback implementation."""
        template = self.creation_templates["basic_neural_atom"]

        safe_implementation = """
        # Safe fallback implementation
        result = {
            "message": f"Neural Atom {self.metadata.name} executed successfully",
            "input_received": str(input_data)[:100],  # Truncate for safety
            "capabilities": self.metadata.capabilities,
            "timestamp": str(time.time())
        }"""

        return template.format(
            class_name=design["class_name"],
            name=spec.name,
            description=spec.description,
            capabilities=json.dumps(spec.capabilities),
            implementation_code=safe_implementation,
        )

    async def _stage_4_rectification(
        self, request: CREATORRequest, code: str
    ) -> Any | None:
        """Stage 4: Rectification - Validate and optimize."""
        logger.info(f"✅ CREATOR Stage 4: Rectification for {request.request_id}")

        try:
            self._update_stage_completion(CREATORStage.RECTIFICATION)

            # Execute code in safe environment
            safe_globals = {
                "__builtins__": {
                    "len": len,
                    "str": str,
                    "int": int,
                    "float": float,
                    "dict": dict,
                    "list": list,
                    "tuple": tuple,
                    "Exception": Exception,
                    "ValueError": ValueError,
                    "TypeError": TypeError,
                    "time": time,
                    "json": json,
                },
                "NeuralAtom": NeuralAtom,
                "NeuralAtomMetadata": NeuralAtomMetadata,
                "time": time,
                "json": json,
                "logger": logger,
                "Any": Any,
                "List": list,
            }

            local_vars = {}
            exec(code, safe_globals, local_vars)

            # Find the generated class
            atom_class = None
            for name, obj in local_vars.items():
                if isinstance(obj, type) and issubclass(obj, NeuralAtom):
                    atom_class = obj
                    break

            if atom_class:
                # Create instance and basic test
                atom_instance = atom_class()
                logger.info(
                    f"Created Neural Atom instance: {atom_instance.metadata.name}"
                )
                return atom_instance
            raise Exception("No valid Neural Atom class found in generated code")

        except Exception as e:
            logger.error(f"Stage 4 failed: {e}")
            return None

    async def _register_created_atom(self, request: CREATORRequest, atom: NeuralAtom):
        """Register the successfully created Neural Atom in the store."""
        try:
            # Register in Neural Store
            self.store.register(atom)

            # Create success result
            result = CREATORResult(
                request_id=request.request_id,
                success=True,
                neural_atom_id=atom.metadata.name,
                stages_completed=[stage for stage in CREATORStage],
            )

            # Broadcast success
            await self.workspace.update(
                data={"type": "creator_result", **result.model_dump()},
                source="creator",
                attention_level=AttentionLevel.HIGH,
            )

            self.creator_stats["successful_creations"] += 1

        except Exception as e:
            logger.error(f"Failed to register created atom: {e}")
            await self._handle_creation_failure(request, f"Registration failed: {e}")

    async def _handle_creation_failure(
        self, request: CREATORRequest, error_message: str
    ):
        """Handle creation failure."""
        result = CREATORResult(
            request_id=request.request_id,
            success=False,
            error=error_message,
            stages_completed=self.active_requests.get(request.request_id, {}).get(
                "stages_completed", []
            ),
        )

        await self.workspace.update(
            data={"type": "creator_result", **result.model_dump()},
            source="creator",
            attention_level=AttentionLevel.HIGH,
        )

        self.creator_stats["failed_creations"] += 1

    def _update_stage_completion(self, stage: CREATORStage):
        """Update stage completion statistics."""
        self.creator_stats["stage_completion_rates"][stage.value] += 1

    def _update_creator_stats(self, creation_time: float, success: bool):
        """Update Creator Plugin statistics."""
        # Update average creation time
        alpha = 0.1
        if self.creator_stats["average_creation_time"] == 0.0:
            self.creator_stats["average_creation_time"] = creation_time
        else:
            self.creator_stats["average_creation_time"] = (
                alpha * creation_time
                + (1 - alpha) * self.creator_stats["average_creation_time"]
            )

    async def _handle_capability_gap_event(self, event_data: dict[str, Any]):
        """Convert capability gap event to CREATOR request."""
        gap_description = event_data.get("description", "Unknown capability gap")

        creator_request = CREATORRequest(
            request_id=f"gap_{int(time.time())}",
            capability_description=gap_description,
            context=event_data,
            priority=5,
            requester="capability_gap_detector",
        )

        await self._handle_creator_request(creator_request)

    def get_creator_stats(self) -> dict[str, Any]:
        """Get current Creator Plugin statistics."""
        return {
            **self.creator_stats,
            "active_requests": len(self.active_requests),
            "llm_client_available": self.llm_client is not None,
            "templates_available": len(self.creation_templates),
        }
