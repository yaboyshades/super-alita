"""Reasoning chain generation and verification for explainable code generation."""

import contextlib
import json
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

import structlog
import torch
from sentence_transformers import SentenceTransformer

if TYPE_CHECKING:  # pragma: no cover - import cycle guard
    from src.reug_runtime.llm_client import LLMClient

logger = structlog.get_logger(__name__)


class ReasoningStepType(Enum):
    """Types of reasoning steps in code generation."""

    PROBLEM_ANALYSIS = "problem_analysis"
    REQUIREMENT_DECOMPOSITION = "requirement_decomposition"
    SOLUTION_STRATEGY = "solution_strategy"
    IMPLEMENTATION_PLANNING = "implementation_planning"
    CODE_GENERATION = "code_generation"
    VALIDATION = "validation"
    OPTIMIZATION = "optimization"


@dataclass
class ReasoningStep:
    """A single step in the reasoning chain."""

    step_type: ReasoningStepType
    description: str
    reasoning: str
    confidence: float
    dependencies: list[int] = field(default_factory=list)
    verification_status: str | None = None
    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass
class ReasoningChain:
    """Complete reasoning chain for code generation."""

    steps: list[ReasoningStep]
    overall_confidence: float
    logical_consistency_score: float
    completeness_score: float
    generated_code: str | None = None
    generation_mode: str = "heuristic"
    metadata: dict[str, Any] = field(default_factory=dict)


class LogicalConsistencyChecker:
    """Checks logical consistency of reasoning steps."""

    def __init__(self):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Sentence transformer for semantic similarity
        self.sentence_model = SentenceTransformer("all-MiniLM-L6-v2")

        # Contradiction detection model (using NLI)
        self.nli_model_name = "microsoft/deberta-v3-base"
        self.nli_tokenizer = None
        self.nli_model = None

    async def check_step_consistency(
        self, current_step: ReasoningStep, previous_steps: list[ReasoningStep]
    ) -> dict[str, Any]:
        """Check consistency of current step with previous steps."""

        if not previous_steps:
            return {
                "is_consistent": True,
                "consistency_score": 1.0,
                "conflicts": [],
            }

        # Extract relevant previous steps
        relevant_steps = self._get_relevant_steps(current_step, previous_steps)

        # Check for logical contradictions
        contradictions = await self._detect_contradictions(
            current_step, relevant_steps
        )

        # Check for missing dependencies
        missing_deps = self._check_dependencies(current_step, previous_steps)

        # Calculate overall consistency score
        consistency_score = self._calculate_consistency_score(
            contradictions, missing_deps
        )

        return {
            "is_consistent": consistency_score > 0.7,
            "consistency_score": consistency_score,
            "conflicts": contradictions,
            "missing_dependencies": missing_deps,
        }

    def _get_relevant_steps(
        self, current_step: ReasoningStep, previous_steps: list[ReasoningStep]
    ) -> list[ReasoningStep]:
        """Get steps relevant to current step."""

        # Get dependency-specified steps
        relevant = []
        for dep_idx in current_step.dependencies:
            if 0 <= dep_idx < len(previous_steps):
                relevant.append(previous_steps[dep_idx])

        # If no explicit dependencies, use recent steps
        if not relevant and previous_steps:
            relevant = previous_steps[-3:]  # Last 3 steps

        return relevant

    async def _detect_contradictions(
        self, current_step: ReasoningStep, previous_steps: list[ReasoningStep]
    ) -> list[dict[str, Any]]:
        """Detect logical contradictions using NLI model."""

        contradictions = []

        # Check current step against each previous step
        for i, prev_step in enumerate(previous_steps):
            # Compute semantic similarity
            similarity = self._compute_semantic_similarity(
                current_step.reasoning, prev_step.reasoning
            )

            # If highly similar but different conclusions, potential contradiction
            if similarity > 0.7:
                # Check for contradiction using NLI (stub for now)
                is_contradiction = await self._check_nli_contradiction(
                    current_step.reasoning, prev_step.reasoning
                )

                if is_contradiction:
                    contradictions.append(
                        {
                            "step_index": i,
                            "type": "logical_contradiction",
                            "severity": (
                                "high" if similarity > 0.85 else "medium"
                            ),
                            "description": f"Contradicts step {i}",
                        }
                    )

        return contradictions

    def _compute_semantic_similarity(self, text1: str, text2: str) -> float:
        """Compute semantic similarity between two texts."""

        try:
            embeddings = self.sentence_model.encode([text1, text2])

            # Cosine similarity
            similarity = torch.cosine_similarity(
                torch.tensor(embeddings[0]), torch.tensor(embeddings[1]), dim=0
            ).item()

            return max(0.0, similarity)

        except Exception as e:
            logger.warning("Failed to compute semantic similarity", exc_info=e)
            return 0.0

    async def _check_nli_contradiction(
        self, premise: str, hypothesis: str
    ) -> bool:
        """Check if hypothesis contradicts premise using NLI."""

        # Stub implementation - would use actual NLI model
        # For now, use simple heuristics

        # Check for negation patterns
        negation_patterns = ["not", "never", "no", "cannot", "should not"]

        premise_has_neg = any(
            neg in premise.lower() for neg in negation_patterns
        )
        hypothesis_has_neg = any(
            neg in hypothesis.lower() for neg in negation_patterns
        )

        # Simple heuristic: if one has negation and the other doesn't, might be contradiction
        return premise_has_neg != hypothesis_has_neg

    def _check_dependencies(
        self, current_step: ReasoningStep, previous_steps: list[ReasoningStep]
    ) -> list[int]:
        """Check for missing dependencies."""

        missing_deps = []

        for dep_idx in current_step.dependencies:
            if dep_idx < 0 or dep_idx >= len(previous_steps):
                missing_deps.append(dep_idx)

        return missing_deps

    def _calculate_consistency_score(
        self, contradictions: list[dict[str, Any]], missing_deps: list[int]
    ) -> float:
        """Calculate overall consistency score."""

        # Start with perfect score
        score = 1.0

        # Deduct for contradictions
        for contradiction in contradictions:
            severity = contradiction.get("severity", "medium")
            if severity == "high":
                score -= 0.2
            elif severity == "medium":
                score -= 0.1
            else:
                score -= 0.05

        # Deduct for missing dependencies
        score -= len(missing_deps) * 0.1

        return max(0.0, score)


class FactualCorrectnessVerifier:
    """Verifies factual correctness of reasoning steps."""

    def __init__(self):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Knowledge base of programming facts (stub)
        self.knowledge_base = self._initialize_knowledge_base()

    async def verify_factual_correctness(
        self, reasoning_step: ReasoningStep
    ) -> dict[str, Any]:
        """Verify factual correctness of a reasoning step."""

        # Extract claims from reasoning
        claims = self._extract_claims(reasoning_step.reasoning)

        # Verify each claim
        verified_claims = []
        unverified_claims = []

        for claim in claims:
            is_verified = await self._verify_claim(claim)

            if is_verified:
                verified_claims.append(claim)
            else:
                unverified_claims.append(claim)

        # Calculate correctness score
        if not claims:
            correctness_score = 0.5  # Neutral if no claims
        else:
            correctness_score = len(verified_claims) / len(claims)

        return {
            "is_correct": correctness_score > 0.7,
            "correctness_score": correctness_score,
            "verified_claims": verified_claims,
            "unverified_claims": unverified_claims,
        }

    def _initialize_knowledge_base(self) -> dict[str, Any]:
        """Initialize programming knowledge base."""

        # Stub knowledge base with common programming facts
        return {
            "python_syntax": {
                "indentation": "Python uses indentation for code blocks",
                "dynamic_typing": "Python is dynamically typed",
                "list_mutable": "Python lists are mutable",
            },
            "algorithms": {
                "quicksort_complexity": "Quicksort has O(n log n) average complexity",
                "binary_search": "Binary search requires sorted array",
            },
            "best_practices": {
                "error_handling": "Use try-except for error handling",
                "type_hints": "Type hints improve code readability",
            },
        }

    def _extract_claims(self, reasoning: str) -> list[str]:
        """Extract factual claims from reasoning text."""

        # Simple heuristic: split by sentences and filter
        sentences = reasoning.split(".")

        claims = []
        for sentence in sentences:
            sentence = sentence.strip()

            # Filter for claim-like sentences
            if len(sentence) > 10 and any(
                keyword in sentence.lower()
                for keyword in [
                    "should",
                    "must",
                    "requires",
                    "is",
                    "are",
                    "has",
                ]
            ):
                claims.append(sentence)

        return claims

    async def _verify_claim(self, claim: str) -> bool:
        """Verify a single claim against knowledge base."""

        # Stub verification - check if claim keywords exist in knowledge base
        claim_lower = claim.lower()

        for category in self.knowledge_base.values():
            for fact in category.values():
                if any(word in claim_lower for word in fact.lower().split()):
                    return True

        # Default to unverified
        return False


class CompletenessAnalyzer:
    """Analyzes completeness of reasoning chain."""

    def __init__(self):
        # Required step types for complete reasoning
        self.required_steps = {
            ReasoningStepType.PROBLEM_ANALYSIS,
            ReasoningStepType.SOLUTION_STRATEGY,
            ReasoningStepType.IMPLEMENTATION_PLANNING,
            ReasoningStepType.CODE_GENERATION,
        }

    async def analyze_completeness(
        self, reasoning_chain: list[ReasoningStep]
    ) -> dict[str, Any]:
        """Analyze completeness of reasoning chain."""

        # Check for required step types
        present_types = {step.step_type for step in reasoning_chain}
        missing_types = self.required_steps - present_types

        # Check for logical flow
        flow_issues = self._check_logical_flow(reasoning_chain)

        # Calculate completeness score
        completeness_score = self._calculate_completeness_score(
            missing_types, flow_issues
        )

        return {
            "is_complete": completeness_score > 0.8,
            "completeness_score": completeness_score,
            "missing_step_types": [st.value for st in missing_types],
            "flow_issues": flow_issues,
        }

    def _check_logical_flow(
        self, reasoning_chain: list[ReasoningStep]
    ) -> list[str]:
        """Check for logical flow issues."""

        issues = []

        # Check if steps are in reasonable order
        expected_order = [
            ReasoningStepType.PROBLEM_ANALYSIS,
            ReasoningStepType.REQUIREMENT_DECOMPOSITION,
            ReasoningStepType.SOLUTION_STRATEGY,
            ReasoningStepType.IMPLEMENTATION_PLANNING,
            ReasoningStepType.CODE_GENERATION,
            ReasoningStepType.VALIDATION,
            ReasoningStepType.OPTIMIZATION,
        ]

        # Find positions of each step type
        type_positions = {}
        for i, step in enumerate(reasoning_chain):
            if step.step_type not in type_positions:
                type_positions[step.step_type] = i

        # Check order
        for i in range(len(expected_order) - 1):
            curr_type = expected_order[i]
            next_type = expected_order[i + 1]

            if curr_type in type_positions and next_type in type_positions:
                if type_positions[curr_type] > type_positions[next_type]:
                    issues.append(
                        f"{next_type.value} appears before {curr_type.value}"
                    )

        # Check for duplicate critical steps
        type_counts = {}
        for step in reasoning_chain:
            type_counts[step.step_type] = (
                type_counts.get(step.step_type, 0) + 1
            )

        for step_type, count in type_counts.items():
            if step_type in self.required_steps and count > 1:
                issues.append(
                    f"Multiple {step_type.value} steps (may indicate redundancy)"
                )

        return issues

    def _calculate_completeness_score(
        self, missing_types: set, flow_issues: list[str]
    ) -> float:
        """Calculate completeness score."""

        # Start with perfect score
        score = 1.0

        # Deduct for missing required steps
        score -= len(missing_types) * 0.15

        # Deduct for flow issues
        score -= len(flow_issues) * 0.05

        return max(0.0, score)


class ReasoningChainGenerator:
    """Generates reasoning chains for code generation tasks."""

    def __init__(self, config: dict[str, Any] = None):
        self.config = config or {}
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    async def generate_reasoning_chain(
        self, requirements: str, context: dict[str, Any] | None = None
    ) -> list[ReasoningStep]:
        """Generate reasoning chain for requirements."""

        context = context or {}
        steps = []

        # Step 1: Problem Analysis
        analysis_step = await self._generate_problem_analysis(
            requirements, context
        )
        steps.append(analysis_step)

        # Step 2: Requirement Decomposition
        decomp_step = await self._generate_requirement_decomposition(
            requirements, context, [0]
        )
        steps.append(decomp_step)

        # Step 3: Solution Strategy
        strategy_step = await self._generate_solution_strategy(
            requirements, context, [0, 1]
        )
        steps.append(strategy_step)

        # Step 4: Implementation Planning
        planning_step = await self._generate_implementation_planning(
            requirements, context, [1, 2]
        )
        steps.append(planning_step)

        # Step 5: Code Generation (placeholder)
        generation_step = ReasoningStep(
            step_type=ReasoningStepType.CODE_GENERATION,
            description="Generate code based on implementation plan",
            reasoning="Implement the planned solution in Python code",
            confidence=0.8,
            dependencies=[3],
        )
        steps.append(generation_step)

        return steps

    async def _generate_problem_analysis(
        self, requirements: str, context: dict[str, Any]
    ) -> ReasoningStep:
        """Generate problem analysis step."""

        # Analyze requirements
        key_concepts = self._extract_key_concepts(requirements)
        complexity = self._assess_complexity(requirements)

        reasoning = (
            f"Analyzing requirements: '{requirements[:100]}...'. "
            f"Key concepts identified: {', '.join(key_concepts[:3])}. "
            f"Estimated complexity: {complexity}."
        )

        return ReasoningStep(
            step_type=ReasoningStepType.PROBLEM_ANALYSIS,
            description="Analyze problem requirements and constraints",
            reasoning=reasoning,
            confidence=0.85,
            dependencies=[],
            evidence={"key_concepts": key_concepts, "complexity": complexity},
        )

    async def _generate_requirement_decomposition(
        self,
        requirements: str,
        context: dict[str, Any],
        dependencies: list[int],
    ) -> ReasoningStep:
        """Generate requirement decomposition step."""

        # Decompose into sub-requirements
        sub_requirements = self._decompose_requirements(requirements)

        reasoning = (
            f"Decomposing requirements into {len(sub_requirements)} sub-tasks: "
            + ", ".join(sub_requirements[:3])
        )

        return ReasoningStep(
            step_type=ReasoningStepType.REQUIREMENT_DECOMPOSITION,
            description="Break down requirements into manageable sub-tasks",
            reasoning=reasoning,
            confidence=0.8,
            dependencies=dependencies,
            evidence={"sub_requirements": sub_requirements},
        )

    async def _generate_solution_strategy(
        self,
        requirements: str,
        context: dict[str, Any],
        dependencies: list[int],
    ) -> ReasoningStep:
        """Generate solution strategy step."""

        # Determine appropriate approach
        strategy = self._determine_strategy(requirements, context)

        reasoning = (
            f"Solution strategy: {strategy['approach']}. "
            f"Rationale: {strategy['rationale']}"
        )

        return ReasoningStep(
            step_type=ReasoningStepType.SOLUTION_STRATEGY,
            description="Determine overall solution approach",
            reasoning=reasoning,
            confidence=0.82,
            dependencies=dependencies,
            evidence=strategy,
        )

    async def _generate_implementation_planning(
        self,
        requirements: str,
        context: dict[str, Any],
        dependencies: list[int],
    ) -> ReasoningStep:
        """Generate implementation planning step."""

        # Plan implementation details
        plan = self._create_implementation_plan(requirements, context)

        reasoning = (
            f"Implementation plan: {len(plan['steps'])} steps. "
            f"Primary components: {', '.join(plan['components'][:3])}"
        )

        return ReasoningStep(
            step_type=ReasoningStepType.IMPLEMENTATION_PLANNING,
            description="Plan implementation details and structure",
            reasoning=reasoning,
            confidence=0.78,
            dependencies=dependencies,
            evidence=plan,
        )

    def _extract_key_concepts(self, requirements: str) -> list[str]:
        """Extract key concepts from requirements."""

        # Simple keyword extraction (stub)
        words = requirements.lower().split()

        # Filter for meaningful words
        stopwords = {
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
        }
        keywords = [w for w in words if w not in stopwords and len(w) > 3]

        return keywords[:10]

    def _assess_complexity(self, requirements: str) -> str:
        """Assess complexity of requirements."""

        word_count = len(requirements.split())

        if word_count < 20:
            return "low"
        elif word_count < 50:
            return "medium"
        else:
            return "high"

    def _decompose_requirements(self, requirements: str) -> list[str]:
        """Decompose requirements into sub-tasks."""

        # Simple decomposition (stub)
        sentences = requirements.split(".")
        sub_reqs = [s.strip() for s in sentences if len(s.strip()) > 10]

        return sub_reqs[:5]

    def _determine_strategy(
        self, requirements: str, context: dict[str, Any]
    ) -> dict[str, Any]:
        """Determine solution strategy."""

        # Simple strategy selection (stub)
        if "class" in requirements.lower() or "object" in requirements.lower():
            approach = "object-oriented design"
            rationale = (
                "Requirements indicate need for structured data and behavior"
            )
        elif (
            "function" in requirements.lower()
            or "algorithm" in requirements.lower()
        ):
            approach = "functional decomposition"
            rationale = (
                "Requirements focus on specific operations and transformations"
            )
        else:
            approach = "procedural implementation"
            rationale = "Requirements suitable for step-by-step implementation"

        return {"approach": approach, "rationale": rationale, "patterns": []}

    def _create_implementation_plan(
        self, requirements: str, context: dict[str, Any]
    ) -> dict[str, Any]:
        """Create implementation plan."""

        # Simple planning (stub)
        components = self._identify_components(requirements)
        steps = [f"Implement {comp}" for comp in components]

        return {
            "components": components,
            "steps": steps,
            "dependencies": {},
            "estimated_complexity": "medium",
        }

    def _identify_components(self, requirements: str) -> list[str]:
        """Identify main components needed."""

        # Simple component identification (stub)
        components = []

        if "class" in requirements.lower():
            components.append("class definition")
        if "function" in requirements.lower():
            components.append("function implementation")
        if "data" in requirements.lower():
            components.append("data structure")
        if "test" in requirements.lower():
            components.append("test suite")

        if not components:
            components = ["main implementation"]

        return components


class ReasoningChainVerifier:
    """Main reasoning chain verifier with optional LLM-backed reasoning."""

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        *,
        llm_client: "LLMClient" | None = None,
    ) -> None:
        self.config = config or {}
        if llm_client is None:
            llm_client = self.config.get("llm_client")
        self.llm_client = llm_client

        auto_use_llm = bool(self.config.get("auto_use_llm", True))
        use_llm_flag = bool(self.config.get("use_llm", False))
        self.use_llm = bool(self.llm_client) and (auto_use_llm or use_llm_flag)
        self.allow_heuristic_fallback = bool(
            self.config.get("allow_heuristic_fallback", True)
        )
        self.llm_timeout = float(self.config.get("llm_timeout", 30.0))

        # Initialize components
        self.consistency_checker = LogicalConsistencyChecker()
        self.correctness_verifier = FactualCorrectnessVerifier()
        self.completeness_analyzer = CompletenessAnalyzer()
        self.chain_generator = ReasoningChainGenerator(config)

    async def generate_with_reasoning_chain(
        self, requirements: str, context: dict[str, Any] | None = None
    ) -> ReasoningChain:
        """Generate code with explicit reasoning chain."""

        start_time = time.time()
        context = context or {}
        generation_mode = "heuristic"
        fallback_used = False

        try:
            reasoning_steps: list[ReasoningStep] | None = None

            if self.use_llm and self.llm_client is not None:
                try:
                    reasoning_steps = await self._generate_reasoning_chain_llm(
                        requirements, context
                    )
                    generation_mode = "llm"
                except (
                    Exception
                ) as llm_error:  # pragma: no cover - depends on runtime
                    logger.warning(
                        "LLM reasoning generation failed; falling back to heuristics",
                        exc_info=llm_error,
                    )
                    if not self.allow_heuristic_fallback:
                        raise
                    fallback_used = True

            if reasoning_steps is None:
                reasoning_steps = (
                    await self.chain_generator.generate_reasoning_chain(
                        requirements, context
                    )
                )
                generation_mode = "heuristic"

            # Verify reasoning chain
            verification_result = await self._verify_reasoning_chain(
                reasoning_steps
            )

            # If verification passes, generate code
            if verification_result["is_valid"]:
                generated_code = await self._generate_code_from_reasoning(
                    reasoning_steps, requirements
                )
            else:
                generated_code = None
                logger.warning(
                    "Reasoning chain verification failed",
                    issues=verification_result.get("issues", []),
                )

            # Create reasoning chain result
            chain = ReasoningChain(
                steps=reasoning_steps,
                overall_confidence=verification_result["overall_confidence"],
                logical_consistency_score=verification_result[
                    "consistency_score"
                ],
                completeness_score=verification_result["completeness_score"],
                generated_code=generated_code,
                generation_mode=generation_mode,
            )
            chain.metadata.update(
                {
                    "generation_mode": generation_mode,
                    "llm_used": generation_mode == "llm",
                    "fallback_used": fallback_used,
                    "uses_heuristics": generation_mode != "llm",
                }
            )

            logger.info(
                "Reasoning chain generation completed",
                generation_time=time.time() - start_time,
                step_count=len(reasoning_steps),
                is_valid=verification_result["is_valid"],
                generation_mode=generation_mode,
            )

            return chain

        except Exception as e:
            logger.error("Reasoning chain generation failed", exc_info=e)

            if not self.allow_heuristic_fallback:
                raise

            # Return minimal chain
            return ReasoningChain(
                steps=[],
                overall_confidence=0.0,
                logical_consistency_score=0.0,
                completeness_score=0.0,
                generated_code=None,
                generation_mode=generation_mode,
                metadata={
                    "generation_mode": generation_mode,
                    "llm_used": False,
                    "fallback_used": True,
                    "uses_heuristics": True,
                },
            )

    async def _generate_reasoning_chain_llm(
        self, requirements: str, context: dict[str, Any]
    ) -> list[ReasoningStep]:
        if self.llm_client is None:
            raise RuntimeError(
                "LLM client not configured for reasoning chain generation"
            )

        system_prompt = self.config.get(
            "llm_prompt_preamble",
            (
                "You are a planning assistant that converts software requirements into "
                "structured reasoning steps. Respond strictly in JSON with a top-level "
                "'steps' array. Each step must include: step_type, description, reasoning,"
                " confidence (0-1), dependencies (indices), and evidence (object)."
            ),
        )
        input_payload = {
            "requirements": requirements.strip(),
            "context": context,
            "available_step_types": [st.value for st in ReasoningStepType],
        }

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    "Convert the following payload into a structured reasoning chain.\n"
                    + json.dumps(input_payload, indent=2)
                    + "\nReturn JSON only."
                ),
            },
        ]

        raw_response = await self._collect_llm_response(messages)
        steps_payload = self._parse_llm_reasoning_response(raw_response)

        reasoning_steps: list[ReasoningStep] = []
        for _idx, item in enumerate(steps_payload):
            step_type = self._resolve_step_type(item.get("step_type"))
            description = (
                item.get("description")
                or item.get("summary")
                or step_type.value
            )
            reasoning_text = (
                item.get("reasoning")
                or item.get("detail")
                or item.get("analysis")
                or description
            )
            confidence = self._coerce_confidence(item.get("confidence"))
            dependencies = self._coerce_dependencies(item.get("dependencies"))
            evidence = item.get("evidence")
            evidence_dict = evidence if isinstance(evidence, dict) else {}

            reasoning_steps.append(
                ReasoningStep(
                    step_type=step_type,
                    description=description,
                    reasoning=reasoning_text,
                    confidence=confidence,
                    dependencies=dependencies,
                    evidence=evidence_dict,
                )
            )

        if not reasoning_steps:
            raise ValueError(
                "LLM response did not include any reasoning steps"
            )

        return reasoning_steps

    async def _collect_llm_response(
        self, messages: list[dict[str, Any]]
    ) -> str:
        if self.llm_client is None:
            raise RuntimeError("LLM client not configured")

        response_chunks: list[str] = []
        async for chunk in self.llm_client.stream_chat(
            messages, timeout=self.llm_timeout
        ):
            if isinstance(chunk, dict):
                content = chunk.get("content")
                if not content and chunk.get("type") == "content":
                    content = chunk.get("content")
                if not content and chunk.get("type") == "tool_calls":
                    continue
            else:
                content = str(chunk)
            if content:
                response_chunks.append(str(content))

        response_text = "".join(response_chunks).strip()
        if not response_text:
            raise ValueError("LLM returned empty response")
        return response_text

    def _parse_llm_reasoning_response(self, text: str) -> list[dict[str, Any]]:
        candidates: list[Any] = []

        with contextlib.suppress(json.JSONDecodeError):
            candidates.append(json.loads(text))

        for snippet in self._extract_json_candidates(text):
            try:
                candidates.append(json.loads(snippet))
            except json.JSONDecodeError:
                continue

        for payload in candidates:
            if isinstance(payload, dict):
                steps = (
                    payload.get("steps")
                    or payload.get("reasoning_steps")
                    or payload.get("chain")
                )
            else:
                steps = payload if isinstance(payload, list) else None

            if isinstance(steps, list):
                return steps

        raise ValueError("Unable to parse reasoning steps from LLM response")

    def _extract_json_candidates(self, text: str) -> list[str]:
        candidates: list[str] = []
        fence_pattern = re.compile(
            r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", re.DOTALL
        )
        candidates.extend(
            match.group(1) for match in fence_pattern.finditer(text)
        )

        if not candidates:
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                candidates.append(text[start : end + 1])
        return candidates

    def _resolve_step_type(self, value: Any) -> ReasoningStepType:
        if not value:
            return ReasoningStepType.PROBLEM_ANALYSIS
        normalized = (
            str(value).strip().replace("-", "_").replace(" ", "_").upper()
        )
        for member in ReasoningStepType:
            if normalized == member.name:
                return member
            if normalized == member.value.upper().replace("-", "_"):
                return member
        return ReasoningStepType.SOLUTION_STRATEGY

    def _coerce_confidence(self, value: Any) -> float:
        try:
            confidence = float(value)
        except (TypeError, ValueError):
            confidence = 0.75
        return max(0.0, min(1.0, confidence))

    def _coerce_dependencies(self, value: Any) -> list[int]:
        if not isinstance(value, list):
            return []
        dependencies: list[int] = []
        for item in value:
            try:
                dependencies.append(int(item))
            except (TypeError, ValueError):
                continue
        return dependencies

    async def _verify_reasoning_chain(
        self, reasoning_steps: list[ReasoningStep]
    ) -> dict[str, Any]:
        """Verify complete reasoning chain."""

        # Check completeness
        completeness_result = (
            await self.completeness_analyzer.analyze_completeness(
                reasoning_steps
            )
        )

        # Check each step for consistency and correctness
        consistency_scores = []
        correctness_scores = []
        issues = []

        for i, step in enumerate(reasoning_steps):
            # Consistency check
            consistency = (
                await self.consistency_checker.check_step_consistency(
                    step, reasoning_steps[:i]
                )
            )
            consistency_scores.append(consistency["consistency_score"])

            if not consistency["is_consistent"]:
                issues.extend(consistency["conflicts"])

            # Correctness check
            correctness = (
                await self.correctness_verifier.verify_factual_correctness(
                    step
                )
            )
            correctness_scores.append(correctness["correctness_score"])

            # Update step verification status
            step.verification_status = (
                "verified"
                if consistency["is_consistent"] and correctness["is_correct"]
                else "needs_review"
            )

        # Calculate overall scores
        avg_consistency = (
            sum(consistency_scores) / len(consistency_scores)
            if consistency_scores
            else 0.0
        )
        avg_correctness = (
            sum(correctness_scores) / len(correctness_scores)
            if correctness_scores
            else 0.0
        )

        # Overall confidence
        overall_confidence = (
            avg_consistency * 0.3
            + avg_correctness * 0.3
            + completeness_result["completeness_score"] * 0.4
        )

        # Chain is valid if all metrics are above threshold
        is_valid = (
            avg_consistency > 0.7
            and avg_correctness > 0.6
            and completeness_result["is_complete"]
        )

        return {
            "is_valid": is_valid,
            "overall_confidence": overall_confidence,
            "consistency_score": avg_consistency,
            "correctness_score": avg_correctness,
            "completeness_score": completeness_result["completeness_score"],
            "issues": issues + completeness_result.get("flow_issues", []),
        }

    async def _generate_code_from_reasoning(
        self, reasoning_steps: list[ReasoningStep], requirements: str
    ) -> str:
        """Generate code based on verified reasoning chain."""

        # Extract implementation details from reasoning chain
        implementation_plans = [
            step
            for step in reasoning_steps
            if step.step_type == ReasoningStepType.IMPLEMENTATION_PLANNING
        ]

        if not implementation_plans:
            return self._generate_default_code(requirements)

        # Use implementation plan to guide code generation
        plan = implementation_plans[0]
        components = plan.evidence.get("components", [])

        # Generate code structure
        code_parts = []

        # Add header
        code_parts.append('"""Generated code based on requirements."""\n')

        # Generate components
        for component in components:
            if "class" in component.lower():
                code_parts.append(self._generate_class_code(requirements))
            elif "function" in component.lower():
                code_parts.append(self._generate_function_code(requirements))
            else:
                code_parts.append(self._generate_generic_code(requirements))

        return "\n\n".join(code_parts)

    def _generate_default_code(self, requirements: str) -> str:
        """Generate default code when no specific plan available."""

        return f'''"""
Implementation for: {requirements[:100]}
"""

def main():
    """Main implementation."""
    # TODO: Implement based on requirements
    pass

if __name__ == "__main__":
    main()
'''

    def _generate_class_code(self, requirements: str) -> str:
        """Generate class-based code."""

        return '''class Solution:
    """Solution class for the given requirements."""
    
    def __init__(self):
        """Initialize solution."""
        pass
    
    def solve(self):
        """Solve the problem."""
        # TODO: Implement solution
        pass
'''

    def _generate_function_code(self, requirements: str) -> str:
        """Generate function-based code."""

        return '''def solve_problem():
    """
    Solve the problem based on requirements.
    
    Returns:
        Solution result
    """
    # TODO: Implement solution
    pass
'''

    def _generate_generic_code(self, requirements: str) -> str:
        """Generate generic implementation code."""

        return '''# Implementation
# TODO: Complete implementation based on requirements

def process():
    """Process the task."""
    pass
'''
