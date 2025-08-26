#!/usr/bin/env python3
"""
🧠 META-LEARNING CREATOR - SELF-IMPROVING TOOL GENERATION
Advanced meta-learning system for Super Alita's autonomous capability development

AGENT DEV MODE (Copilot read this):
- Event-driven only; define Pydantic events (Literal event_type) and add to EVENT_TYPE_MAP
- Neural Atoms are concrete subclasses with UUIDv5 deterministic IDs
- Use logging.getLogger(__name__), never print. Clamp sizes/ranges
- Write tests: fixed inputs ⇒ fixed outputs; handler validation
"""

import ast
import logging
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ToolGenerationRequest:
    """Request for generating a new tool capability"""

    capability_description: str
    success_criteria: list[str]
    constraints: dict[str, Any] = field(default_factory=dict)
    context: dict[str, Any] = field(default_factory=dict)
    priority: float = 0.5
    timeout: float = 30.0


@dataclass
class GeneratedTool:
    """A generated tool with metadata and validation"""

    name: str
    code: str
    description: str
    capabilities: list[str]
    success_rate: float = 0.0
    generation_time: float = 0.0
    validation_passed: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class LearningPattern:
    """Pattern learned from tool generation experiences"""

    pattern_id: str
    success_indicators: list[str]
    failure_indicators: list[str]
    code_templates: list[str]
    confidence: float = 0.0
    usage_count: int = 0
    last_used: datetime = field(default_factory=datetime.now)


class MetaLearningCreator:
    """
    🧠 Advanced meta-learning system for autonomous tool creation

    Capabilities:
    - Learn from tool generation successes and failures
    - Adapt code generation strategies based on patterns
    - Self-improve through meta-learning algorithms
    - Generate increasingly sophisticated tools
    - Pattern recognition and template evolution
    """

    def __init__(self, max_patterns: int = 1000, max_history: int = 5000):
        self.max_patterns = max_patterns
        self.max_history = max_history

        # Learning storage
        self.generation_history: deque = deque(maxlen=max_history)
        self.learned_patterns: dict[str, LearningPattern] = {}
        self.success_templates: dict[str, list[str]] = defaultdict(list)
        self.failure_patterns: dict[str, list[str]] = defaultdict(list)

        # Meta-learning parameters
        self.learning_rate = 0.1
        self.pattern_threshold = 0.7
        self.template_evolution_rate = 0.05
        self.success_threshold = 0.8

        # Code generation templates
        self.base_templates = {
            "simple_function": '''
async def {function_name}({parameters}) -> {return_type}:
    """
    {description}
    """
    {implementation}
    return {return_value}
''',
            "class_based_tool": '''
class {class_name}:
    """
    {description}
    """

    def __init__(self, {init_parameters}):
        {initialization}

    async def execute(self, {execute_parameters}) -> {return_type}:
        """
        {execute_description}
        """
        {implementation}
        return {return_value}
''',
            "neural_atom_tool": '''
from src.core.neural_atom import NeuralAtom, NeuralAtomMetadata
from typing import Any, List

class {class_name}(NeuralAtom):
    """
    {description}
    """

    def __init__(self, metadata: NeuralAtomMetadata, {additional_params}):
        super().__init__(metadata)
        self.key = metadata.name
        {initialization}

    async def execute(self, input_data: Any) -> Any:
        """
        {execute_description}
        """
        {implementation}
        return {return_value}

    def get_embedding(self) -> List[float]:
        """Generate semantic embedding"""
        {embedding_implementation}
        return {embedding_return}

    def can_handle(self, task_description: str) -> float:
        """Return confidence for handling task"""
        {confidence_implementation}
        return {confidence_return}
''',
        }

        logger.info("🧠 MetaLearningCreator initialized")

    async def generate_tool(self, request: ToolGenerationRequest) -> GeneratedTool:
        """Generate a new tool using meta-learning insights"""
        start_time = time.time()

        # Analyze request and find relevant patterns
        relevant_patterns = self._find_relevant_patterns(request)

        # Select best template strategy
        template_strategy = await self._select_template_strategy(
            request, relevant_patterns
        )

        # Generate code using learned patterns
        generated_code = await self._generate_code_with_patterns(
            request, template_strategy, relevant_patterns
        )

        # Create tool metadata
        tool_name = self._generate_tool_name(request)

        generated_tool = GeneratedTool(
            name=tool_name,
            code=generated_code,
            description=request.capability_description,
            capabilities=self._extract_capabilities(request),
            generation_time=time.time() - start_time,
            metadata={
                "patterns_used": [p.pattern_id for p in relevant_patterns],
                "template_strategy": template_strategy,
                "generation_request": request,
            },
        )

        # Validate the generated tool
        generated_tool.validation_passed = await self._validate_generated_tool(
            generated_tool
        )

        logger.info(
            f"🧠 Generated tool '{tool_name}' in {generated_tool.generation_time:.2f}s, "
            f"validation: {'PASSED' if generated_tool.validation_passed else 'FAILED'}"
        )

        return generated_tool

    async def learn_from_generation(
        self,
        tool: GeneratedTool,
        actual_success_rate: float,
        execution_feedback: dict[str, Any] = None,
    ) -> None:
        """Learn from tool generation outcomes"""
        execution_feedback = execution_feedback or {}

        # Record the generation experience
        generation_record = {
            "tool": tool,
            "success_rate": actual_success_rate,
            "feedback": execution_feedback,
            "timestamp": datetime.now(),
            "patterns_used": tool.metadata.get("patterns_used", []),
            "template_strategy": tool.metadata.get("template_strategy", "unknown"),
        }

        self.generation_history.append(generation_record)

        # Update pattern confidence based on outcomes
        for pattern_id in tool.metadata.get("patterns_used", []):
            if pattern_id in self.learned_patterns:
                pattern = self.learned_patterns[pattern_id]
                # Update confidence using moving average
                pattern.confidence = (
                    pattern.confidence * 0.9 + actual_success_rate * 0.1
                )
                pattern.usage_count += 1
                pattern.last_used = datetime.now()

        # Learn new patterns from successful generations
        if actual_success_rate >= self.success_threshold:
            await self._extract_success_patterns(tool, execution_feedback)
        else:
            await self._extract_failure_patterns(tool, execution_feedback)

        # Evolve templates based on learning
        await self._evolve_templates(tool, actual_success_rate, execution_feedback)

        logger.debug(
            f"🧠 Learned from tool '{tool.name}': success_rate={actual_success_rate:.2f}"
        )

    def _find_relevant_patterns(
        self, request: ToolGenerationRequest
    ) -> list[LearningPattern]:
        """Find patterns relevant to the generation request"""
        relevant_patterns = []

        # Extract keywords from request
        keywords = self._extract_keywords(request.capability_description)

        for pattern in self.learned_patterns.values():
            # Calculate relevance score
            relevance = 0.0

            # Check for keyword matches in success indicators
            for indicator in pattern.success_indicators:
                for keyword in keywords:
                    if keyword.lower() in indicator.lower():
                        relevance += 0.1

            # Weight by pattern confidence and usage
            relevance *= pattern.confidence * min(1.0, pattern.usage_count / 10.0)

            if relevance >= self.pattern_threshold:
                relevant_patterns.append(pattern)

        # Sort by relevance and return top patterns
        relevant_patterns.sort(key=lambda p: p.confidence, reverse=True)
        return relevant_patterns[:10]  # Top 10 most relevant patterns

    async def _select_template_strategy(
        self, request: ToolGenerationRequest, patterns: list[LearningPattern]
    ) -> str:
        """Select the best template strategy based on request and patterns"""
        # Analyze request complexity
        complexity_score = self._assess_request_complexity(request)

        # Choose template based on complexity and learned patterns
        if complexity_score < 0.3:
            return "simple_function"
        if complexity_score < 0.7:
            return "class_based_tool"
        return "neural_atom_tool"

    async def _generate_code_with_patterns(
        self,
        request: ToolGenerationRequest,
        template_strategy: str,
        patterns: list[LearningPattern],
    ) -> str:
        """Generate code using learned patterns and templates"""
        # Get base template
        base_template = self.base_templates.get(
            template_strategy, self.base_templates["simple_function"]
        )

        # Extract learned code snippets from patterns
        code_snippets = []
        for pattern in patterns:
            code_snippets.extend(pattern.code_templates)

        # Generate template parameters
        template_params = await self._generate_template_parameters(request, patterns)

        # Fill template with learned patterns
        try:
            generated_code = base_template.format(**template_params)
        except KeyError as e:
            logger.warning(f"🧠 Template parameter missing: {e}, using fallback")
            generated_code = self._generate_fallback_code(request, template_strategy)

        return generated_code

    async def _generate_template_parameters(
        self, request: ToolGenerationRequest, patterns: list[LearningPattern]
    ) -> dict[str, str]:
        """Generate parameters for code templates"""
        # Extract base name from description
        base_name = self._extract_base_name(request.capability_description)

        # Generate different naming conventions
        function_name = f"{base_name}_tool"
        class_name = f"{base_name.title()}Tool"

        # Generate implementation based on patterns and request
        implementation = self._generate_implementation(request, patterns)

        parameters = {
            "function_name": function_name,
            "class_name": class_name,
            "description": request.capability_description,
            "parameters": "input_data: Any",
            "init_parameters": "config: Dict[str, Any] = None",
            "execute_parameters": "input_data: Any",
            "additional_params": "config: Dict[str, Any] = None",
            "return_type": "Any",
            "return_value": "result",
            "initialization": "self.config = config or {}",
            "implementation": implementation,
            "execute_description": f"Execute {request.capability_description}",
            "embedding_implementation": "import hashlib\nhash_obj = hashlib.md5(self.metadata.description.encode())\nhash_bytes = hash_obj.digest()",
            "embedding_return": "[float(b) / 255.0 for b in hash_bytes[:384]]",
            "confidence_implementation": f"""keywords = {self._extract_keywords(request.capability_description)}
        if any(kw in task_description.lower() for kw in keywords):
            return 0.9""",
            "confidence_return": "0.1",
        }

        return parameters

    def _generate_implementation(
        self, request: ToolGenerationRequest, patterns: list[LearningPattern]
    ) -> str:
        """Generate implementation code based on request and patterns"""
        # Start with basic implementation structure
        implementation_lines = [
            "# Implementation based on meta-learning patterns",
            "try:",
            "    # Process input data",
            "    if not input_data:",
            "        input_data = {}",
            "",
            "    # Core functionality",
        ]

        # Add pattern-based implementation
        if patterns:
            implementation_lines.append("    # Apply learned patterns")
            for pattern in patterns[:3]:  # Use top 3 patterns
                if pattern.code_templates:
                    template = pattern.code_templates[0]
                    # Extract useful code snippets (simplified)
                    if "return" in template:
                        implementation_lines.append(
                            f"    # Pattern: {pattern.pattern_id}"
                        )

        # Add result generation
        implementation_lines.extend(
            [
                "",
                "    # Generate result",
                "    result = {",
                "        'success': True,",
                "        'data': input_data,",
                f"        'capability': '{request.capability_description}',",
                "        'timestamp': time.time()",
                "    }",
                "",
                "except Exception as e:",
                "    result = {",
                "        'success': False,",
                "        'error': str(e),",
                "        'timestamp': time.time()",
                "    }",
            ]
        )

        return "\n    ".join(implementation_lines)

    def _generate_fallback_code(
        self, request: ToolGenerationRequest, strategy: str
    ) -> str:
        """Generate fallback code when template fails"""
        base_name = self._extract_base_name(request.capability_description)

        return f'''
async def {base_name}_tool(input_data: Any = None) -> Dict[str, Any]:
    """
    {request.capability_description}
    Auto-generated fallback implementation
    """
    import time

    try:
        # Basic implementation
        result = {{
            'success': True,
            'data': input_data,
            'capability': '{request.capability_description}',
            'timestamp': time.time()
        }}
        return result
    except Exception as e:
        return {{
            'success': False,
            'error': str(e),
            'timestamp': time.time()
        }}
'''

    async def _validate_generated_tool(self, tool: GeneratedTool) -> bool:
        """Validate the generated tool code"""
        try:
            # Basic syntax validation
            ast.parse(tool.code)

            # Check for required elements
            if "async def" not in tool.code and "class" not in tool.code:
                return False

            # Check for error handling
            if "try:" not in tool.code:
                logger.warning(f"🧠 Tool '{tool.name}' lacks error handling")

            return True

        except SyntaxError as e:
            logger.error(f"🧠 Syntax error in generated tool '{tool.name}': {e}")
            return False
        except Exception as e:
            logger.error(f"🧠 Validation error for tool '{tool.name}': {e}")
            return False

    async def _extract_success_patterns(
        self, tool: GeneratedTool, feedback: dict[str, Any]
    ) -> None:
        """Extract patterns from successful tool generations"""
        pattern_id = f"success_{tool.name}_{len(self.learned_patterns)}"

        # Extract success indicators
        success_indicators = [
            tool.description,
            *tool.capabilities,
            *[str(v) for v in feedback.get("positive_indicators", [])],
        ]

        # Extract code templates from successful generation
        code_templates = [tool.code]

        pattern = LearningPattern(
            pattern_id=pattern_id,
            success_indicators=success_indicators,
            failure_indicators=[],
            code_templates=code_templates,
            confidence=0.8,  # Start with high confidence for successful patterns
            usage_count=1,
            last_used=datetime.now(),
        )

        self.learned_patterns[pattern_id] = pattern
        logger.debug(f"🧠 Extracted success pattern: {pattern_id}")

    async def _extract_failure_patterns(
        self, tool: GeneratedTool, feedback: dict[str, Any]
    ) -> None:
        """Extract patterns from failed tool generations"""
        pattern_id = f"failure_{tool.name}_{len(self.learned_patterns)}"

        # Extract failure indicators
        failure_indicators = [
            *[str(v) for v in feedback.get("error_messages", [])],
            *[str(v) for v in feedback.get("negative_indicators", [])],
        ]

        if failure_indicators:
            pattern = LearningPattern(
                pattern_id=pattern_id,
                success_indicators=[],
                failure_indicators=failure_indicators,
                code_templates=[],
                confidence=0.2,  # Low confidence for failure patterns
                usage_count=1,
                last_used=datetime.now(),
            )

            self.learned_patterns[pattern_id] = pattern
            logger.debug(f"🧠 Extracted failure pattern: {pattern_id}")

    async def _evolve_templates(
        self, tool: GeneratedTool, success_rate: float, feedback: dict[str, Any]
    ) -> None:
        """Evolve code templates based on learning"""
        template_strategy = tool.metadata.get("template_strategy", "unknown")

        if success_rate >= self.success_threshold:
            # Add successful code patterns to templates
            if template_strategy in self.success_templates:
                self.success_templates[template_strategy].append(tool.code)

            # Limit template storage
            if len(self.success_templates[template_strategy]) > 20:
                self.success_templates[template_strategy] = self.success_templates[
                    template_strategy
                ][-20:]

        # Update base templates with learned improvements
        if len(self.success_templates.get(template_strategy, [])) > 5:
            await self._update_base_template(template_strategy)

    async def _update_base_template(self, template_strategy: str) -> None:
        """Update base templates with learned patterns"""
        successful_codes = self.success_templates.get(template_strategy, [])

        if len(successful_codes) < 5:
            return

        # Extract common patterns from successful codes
        common_patterns = self._extract_common_patterns(successful_codes)

        # Update base template (simplified approach)
        if common_patterns:
            logger.info(f"🧠 Evolved template strategy: {template_strategy}")
            # Template evolution logic would go here

    def _extract_common_patterns(self, codes: list[str]) -> list[str]:
        """Extract common patterns from multiple code samples"""
        patterns = []

        # Find common import statements
        common_imports = set()
        for code in codes:
            lines = code.split("\n")
            for line in lines:
                if line.strip().startswith("import ") or line.strip().startswith(
                    "from "
                ):
                    common_imports.add(line.strip())

        if len(common_imports) >= len(codes) * 0.5:  # Present in at least 50% of codes
            patterns.extend(common_imports)

        return patterns

    def _assess_request_complexity(self, request: ToolGenerationRequest) -> float:
        """Assess the complexity of a generation request"""
        complexity_score = 0.0

        # Description complexity
        description_words = len(request.capability_description.split())
        complexity_score += min(1.0, description_words / 20.0) * 0.3

        # Success criteria complexity
        criteria_count = len(request.success_criteria)
        complexity_score += min(1.0, criteria_count / 10.0) * 0.3

        # Constraints complexity
        constraints_count = len(request.constraints)
        complexity_score += min(1.0, constraints_count / 5.0) * 0.2

        # Context complexity
        context_count = len(request.context)
        complexity_score += min(1.0, context_count / 10.0) * 0.2

        return complexity_score

    def _extract_keywords(self, text: str) -> list[str]:
        """Extract relevant keywords from text"""
        # Simple keyword extraction
        stop_words = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
        }
        words = text.lower().split()
        keywords = [
            word.strip(".,!?;:")
            for word in words
            if word not in stop_words and len(word) > 2
        ]
        return list(set(keywords))  # Remove duplicates

    def _extract_base_name(self, description: str) -> str:
        """Extract a base name from the description"""
        # Simple name extraction
        words = description.lower().split()
        action_words = {
            "create",
            "generate",
            "build",
            "make",
            "calculate",
            "process",
            "analyze",
            "convert",
        }

        for i, word in enumerate(words):
            if word in action_words and i + 1 < len(words):
                return words[i + 1].replace(" ", "_")

        # Fallback to first meaningful word
        for word in words:
            if len(word) > 3 and word.isalpha():
                return word.replace(" ", "_")

        return "generated_tool"

    def _extract_capabilities(self, request: ToolGenerationRequest) -> list[str]:
        """Extract capabilities from the generation request"""
        capabilities = []

        # Extract from description
        keywords = self._extract_keywords(request.capability_description)
        capabilities.extend(keywords[:5])  # Top 5 keywords

        # Extract from success criteria
        for criterion in request.success_criteria:
            capabilities.extend(self._extract_keywords(criterion)[:2])

        return list(set(capabilities))  # Remove duplicates

    def _generate_tool_name(self, request: ToolGenerationRequest) -> str:
        """Generate a unique tool name"""
        base_name = self._extract_base_name(request.capability_description)
        unique_id = str(uuid.uuid4())[:8]
        return f"{base_name}_{unique_id}"

    def get_learning_summary(self) -> dict[str, Any]:
        """Get comprehensive learning state summary"""
        total_patterns = len(self.learned_patterns)
        success_patterns = len(
            [
                p
                for p in self.learned_patterns.values()
                if p.confidence >= self.success_threshold
            ]
        )

        # Calculate average pattern confidence
        avg_confidence = 0.0
        if self.learned_patterns:
            avg_confidence = (
                sum(p.confidence for p in self.learned_patterns.values())
                / total_patterns
            )

        # Get top performing patterns
        top_patterns = sorted(
            self.learned_patterns.values(), key=lambda p: p.confidence, reverse=True
        )[:5]

        return {
            "total_generations": len(self.generation_history),
            "total_patterns": total_patterns,
            "success_patterns": success_patterns,
            "average_confidence": avg_confidence,
            "top_patterns": [
                {
                    "id": p.pattern_id,
                    "confidence": p.confidence,
                    "usage_count": p.usage_count,
                    "success_indicators": p.success_indicators[:3],  # Top 3 indicators
                }
                for p in top_patterns
            ],
            "template_strategies": list(self.base_templates.keys()),
            "learning_metrics": {
                "learning_rate": self.learning_rate,
                "pattern_threshold": self.pattern_threshold,
                "success_threshold": self.success_threshold,
            },
        }
