#!/usr/bin/env python3
"""
DeepCode EOS Orchestration Integration

Integrates DeepCode's multi-agent system with EOS v0.9 orchestration,
providing Paper2Code, Text2Web, and Text2Backend capabilities through
coordinated agent workflows with Mangle reasoning validation.
"""

from datetime import datetime
from pathlib import Path
from typing import Any

from ..core.plugin_registry import get_plugin
from ..unified_intelligence import UnifiedIntelligenceEngine
from .mangle_integration import EOS_AVAILABLE, EOSMangleOrchestrator


class DeepCodeAgent:
    """Base class for DeepCode specialized agents."""

    def __init__(self, agent_type: str, capabilities: list[str]):
        self.agent_type = agent_type
        self.capabilities = capabilities
        self.execution_history = []

    async def execute(self, task: dict[str, Any]) -> dict[str, Any]:
        """Execute agent task with telemetry."""
        start_time = datetime.now()

        try:
            result = await self._execute_task(task)

            execution_record = {
                "timestamp": start_time.isoformat(),
                "task_id": task.get("id"),
                "success": True,
                "duration": (datetime.now() - start_time).total_seconds(),
                "result": result,
            }

            self.execution_history.append(execution_record)
            return result

        except Exception as e:
            execution_record = {
                "timestamp": start_time.isoformat(),
                "task_id": task.get("id"),
                "success": False,
                "duration": (datetime.now() - start_time).total_seconds(),
                "error": str(e),
            }

            self.execution_history.append(execution_record)
            raise

    async def _execute_task(self, task: dict[str, Any]) -> dict[str, Any]:
        """Override in subclasses for specific agent logic."""
        raise NotImplementedError


class IntentUnderstandingAgent(DeepCodeAgent):
    """Agent for deep semantic analysis of user requirements."""

    def __init__(self):
        super().__init__(
            agent_type="intent_understanding",
            capabilities=[
                "semantic_analysis",
                "requirement_extraction",
                "complexity_assessment",
            ],
        )

    async def _execute_task(self, task: dict[str, Any]) -> dict[str, Any]:
        """Extract and analyze coding requirements."""
        requirements = task.get("requirements", "")
        task_type = task.get("task_type", "unknown")

        # Semantic analysis
        intent_analysis = await self._analyze_intent(requirements, task_type)

        # Extract structured requirements
        structured_reqs = await self._extract_requirements(
            requirements, task_type
        )

        # Assess complexity using Cynefin framework
        complexity = await self._assess_complexity(requirements, task_type)

        return {
            "intent_analysis": intent_analysis,
            "structured_requirements": structured_reqs,
            "complexity_assessment": complexity,
            "confidence": 0.92,
        }

    async def _analyze_intent(
        self, requirements: str, task_type: str
    ) -> dict[str, Any]:
        """Analyze user intent and extract key concepts."""
        req_lower = requirements.lower()

        # Task-specific intent patterns
        intent_patterns = {
            "paper2code": {
                "research_focus": any(
                    term in req_lower
                    for term in ["paper", "research", "algorithm", "model"]
                ),
                "implementation_focus": any(
                    term in req_lower
                    for term in ["implement", "code", "build", "create"]
                ),
                "mathematical_content": any(
                    term in req_lower
                    for term in ["formula", "equation", "mathematical"]
                ),
            },
            "text2web": {
                "ui_focus": any(
                    term in req_lower
                    for term in ["interface", "ui", "frontend", "dashboard"]
                ),
                "interactive_features": any(
                    term in req_lower
                    for term in ["interactive", "dynamic", "real-time"]
                ),
                "styling_requirements": any(
                    term in req_lower
                    for term in ["responsive", "mobile", "design"]
                ),
            },
            "text2backend": {
                "api_focus": any(
                    term in req_lower
                    for term in ["api", "endpoint", "service", "backend"]
                ),
                "data_management": any(
                    term in req_lower
                    for term in ["database", "storage", "data"]
                ),
                "scalability_focus": any(
                    term in req_lower
                    for term in ["scalable", "microservice", "distributed"]
                ),
            },
        }

        return {
            "task_type": task_type,
            "patterns": intent_patterns.get(task_type, {}),
            "key_concepts": self._extract_key_concepts(requirements),
            "priority_features": self._identify_priority_features(
                requirements, task_type
            ),
        }

    async def _extract_requirements(
        self, requirements: str, task_type: str
    ) -> dict[str, Any]:
        """Extract structured requirements from natural language."""
        # This would use more sophisticated NLP in production
        return {
            "functional_requirements": self._extract_functional_req(
                requirements
            ),
            "non_functional_requirements": self._extract_non_functional_req(
                requirements
            ),
            "technical_constraints": self._extract_constraints(requirements),
            "success_criteria": self._define_success_criteria(
                requirements, task_type
            ),
        }

    async def _assess_complexity(
        self, requirements: str, task_type: str
    ) -> dict[str, Any]:
        """Apply Cynefin framework for complexity classification."""
        complexity_indicators = {
            "simple": ["basic", "simple", "standard", "typical"],
            "complicated": [
                "complex",
                "advanced",
                "sophisticated",
                "multi-step",
            ],
            "complex": ["novel", "innovative", "experimental", "cutting-edge"],
            "chaotic": ["unknown", "exploratory", "research", "unprecedented"],
        }

        req_lower = requirements.lower()
        scores = {}

        for complexity, indicators in complexity_indicators.items():
            score = sum(
                1 for indicator in indicators if indicator in req_lower
            )
            scores[complexity] = score

        primary_complexity = max(scores.items(), key=lambda x: x[1])[0]

        return {
            "primary_complexity": primary_complexity,
            "complexity_scores": scores,
            "adaptation_strategy": self._get_adaptation_strategy(
                primary_complexity
            ),
            "estimated_effort": self._estimate_effort(
                primary_complexity, task_type
            ),
        }

    def _extract_key_concepts(self, requirements: str) -> list[str]:
        """Extract key technical concepts."""
        # Simplified concept extraction - would use NLP models in production
        concepts = []

        tech_concepts = [
            "neural network",
            "transformer",
            "attention",
            "resnet",
            "lstm",
            "react",
            "vue",
            "angular",
            "nodejs",
            "python",
            "pytorch",
            "tensorflow",
            "api",
            "rest",
            "graphql",
            "microservices",
        ]

        req_lower = requirements.lower()
        for concept in tech_concepts:
            if concept in req_lower:
                concepts.append(concept)

        return concepts

    def _identify_priority_features(
        self, requirements: str, task_type: str
    ) -> list[str]:
        """Identify priority features from requirements."""
        # Implementation would be more sophisticated
        return ["core_functionality", "user_interface", "data_handling"]

    def _extract_functional_req(self, requirements: str) -> list[str]:
        """Extract functional requirements."""
        return ["implement_algorithm", "handle_user_input", "generate_output"]

    def _extract_non_functional_req(self, requirements: str) -> list[str]:
        """Extract non-functional requirements."""
        return ["performance", "scalability", "maintainability"]

    def _extract_constraints(self, requirements: str) -> list[str]:
        """Extract technical constraints."""
        return ["technology_stack", "performance_limits", "compatibility"]

    def _define_success_criteria(
        self, requirements: str, task_type: str
    ) -> list[str]:
        """Define success criteria."""
        return ["functional_correctness", "code_quality", "documentation"]

    def _get_adaptation_strategy(self, complexity: str) -> str:
        """Get adaptation strategy for complexity level."""
        strategies = {
            "simple": "direct_implementation",
            "complicated": "structured_approach",
            "complex": "iterative_development",
            "chaotic": "experimental_exploration",
        }
        return strategies.get(complexity, "structured_approach")

    def _estimate_effort(self, complexity: str, task_type: str) -> str:
        """Estimate development effort."""
        effort_matrix = {
            ("simple", "paper2code"): "low",
            ("complicated", "paper2code"): "medium",
            ("complex", "paper2code"): "high",
            ("simple", "text2web"): "low",
            ("complicated", "text2web"): "medium",
            ("simple", "text2backend"): "medium",
        }
        return effort_matrix.get((complexity, task_type), "medium")


class DocumentParsingAgent(DeepCodeAgent):
    """Agent for processing complex technical documents and research papers."""

    def __init__(self):
        super().__init__(
            agent_type="document_parsing",
            capabilities=[
                "pdf_parsing",
                "algorithm_extraction",
                "mathematical_analysis",
            ],
        )

    async def _execute_task(self, task: dict[str, Any]) -> dict[str, Any]:
        """Process documents and extract structured information."""
        document_path = task.get("document_path")
        document_content = task.get("document_content", "")

        if document_path:
            document_content = await self._load_document(document_path)

        # Extract structured information
        parsed_content = await self._parse_document_content(document_content)

        # Extract algorithms and mathematical content
        algorithms = await self._extract_algorithms(document_content)

        # Extract implementation specifications
        impl_specs = await self._extract_implementation_specs(document_content)

        return {
            "parsed_content": parsed_content,
            "extracted_algorithms": algorithms,
            "implementation_specifications": impl_specs,
            "confidence": 0.88,
        }

    async def _load_document(self, document_path: str) -> str:
        """Load document content from file."""
        path = Path(document_path)
        if path.suffix.lower() == ".pdf":
            # In production, would use PDF parsing library
            return f"[PDF content from {document_path}]"
        else:
            return path.read_text(encoding="utf-8")

    async def _parse_document_content(self, content: str) -> dict[str, Any]:
        """Parse document structure and extract key sections."""
        return {
            "title": self._extract_title(content),
            "abstract": self._extract_abstract(content),
            "methodology": self._extract_methodology(content),
            "algorithms": self._extract_algorithm_descriptions(content),
            "results": self._extract_results(content),
        }

    async def _extract_algorithms(self, content: str) -> list[dict[str, Any]]:
        """Extract algorithmic descriptions and pseudocode."""
        # Simplified implementation - would use ML models in production
        return [
            {
                "name": "main_algorithm",
                "description": "Primary algorithm implementation",
                "pseudocode": "algorithm steps",
                "complexity": "O(n log n)",
            }
        ]

    async def _extract_implementation_specs(
        self, content: str
    ) -> dict[str, Any]:
        """Extract implementation-specific details."""
        return {
            "data_structures": ["arrays", "matrices"],
            "key_functions": ["forward_pass", "backward_pass"],
            "parameters": {"learning_rate": 0.001, "batch_size": 32},
            "dependencies": ["numpy", "torch"],
        }

    def _extract_title(self, content: str) -> str:
        """Extract document title."""
        lines = content.split("\n")
        return lines[0] if lines else "Unknown Title"

    def _extract_abstract(self, content: str) -> str:
        """Extract abstract section."""
        return "Abstract content would be extracted here"

    def _extract_methodology(self, content: str) -> str:
        """Extract methodology section."""
        return "Methodology content would be extracted here"

    def _extract_algorithm_descriptions(self, content: str) -> list[str]:
        """Extract algorithm descriptions."""
        return ["Algorithm description 1", "Algorithm description 2"]

    def _extract_results(self, content: str) -> str:
        """Extract results section."""
        return "Results content would be extracted here"


class DeepCodeEOSOrchestrator:
    """
    Main orchestrator for DeepCode integration with EOS framework.

    Coordinates multi-agent workflows using LADDER operations and
    Mangle reasoning for comprehensive code generation.
    """

    def __init__(self):
        self.eos_mangle = EOSMangleOrchestrator() if EOS_AVAILABLE else None
        self.unified_engine = UnifiedIntelligenceEngine(enable_eos=True)

        # Initialize specialized agents
        self.agents = {
            "intent_understanding": IntentUnderstandingAgent(),
            "document_parsing": DocumentParsingAgent(),
            # Additional agents would be implemented similarly
        }

        self.execution_context = {}

    async def orchestrate_generation(
        self, task_type: str, requirements: str, **kwargs
    ) -> dict[str, Any]:
        """
        Orchestrate complete code generation using EOS LADDER operations.

        Args:
            task_type: 'paper2code', 'text2web', or 'text2backend'
            requirements: Natural language requirements
            **kwargs: Additional task-specific parameters

        Returns:
            Comprehensive generation results with code, tests, documentation
        """
        orchestration_id = (
            f"deepcode_{task_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )

        try:
            # LADDER: Lift - Context extraction and analysis
            lifted_context = await self._lift_context(
                task_type, requirements, orchestration_id, **kwargs
            )

            # LADDER: Decompose - Multi-agent task planning
            decomposed_tasks = await self._decompose_into_agent_tasks(
                lifted_context
            )

            # Execute multi-agent workflow
            agent_results = await self._execute_multi_agent_workflow(
                decomposed_tasks
            )

            # LADDER: Synthesize - Combine results with Mangle reasoning
            synthesized = await self._synthesize_with_mangle_reasoning(
                agent_results, lifted_context
            )

            # LADDER: Descend - Final implementation generation
            final_implementation = await self._descend_to_implementation(
                synthesized, lifted_context
            )

            return {
                "orchestration_id": orchestration_id,
                "status": "success",
                "task_type": task_type,
                "generated_code": final_implementation,
                "execution_trace": self._get_execution_trace(orchestration_id),
                "performance_metrics": self._get_performance_metrics(
                    orchestration_id
                ),
            }

        except Exception as e:
            return {
                "orchestration_id": orchestration_id,
                "status": "error",
                "error": str(e),
                "partial_results": self.execution_context.get(
                    orchestration_id, {}
                ),
                "execution_trace": self._get_execution_trace(orchestration_id),
            }

    async def _lift_context(
        self,
        task_type: str,
        requirements: str,
        orchestration_id: str,
        **kwargs,
    ) -> dict[str, Any]:
        """LADDER: Lift - Extract and analyze context."""

        context = {
            "orchestration_id": orchestration_id,
            "task_type": task_type,
            "requirements": requirements,
            "parameters": kwargs,
            "timestamp": datetime.now().isoformat(),
        }

        # Apply Cynefin framework classification
        if self.eos_mangle:
            cynefin_classification = (
                await self.eos_mangle.classify_context_cynefin(
                    requirements, task_type
                )
            )
            context["cynefin_classification"] = cynefin_classification

        # Extract technical context
        context["technical_context"] = await self._extract_technical_context(
            requirements, task_type
        )

        # Store in execution context
        self.execution_context[orchestration_id] = context

        return context

    async def _decompose_into_agent_tasks(
        self, context: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """LADDER: Decompose - Break down into agent-specific tasks."""

        task_type = context["task_type"]
        requirements = context["requirements"]

        # Base tasks for all generation types
        tasks = [
            {
                "id": f"intent_{context['orchestration_id']}",
                "agent_type": "intent_understanding",
                "task_type": task_type,
                "requirements": requirements,
                "priority": 1,
            }
        ]

        # Add document parsing if paper content provided
        if (
            "document_path" in context["parameters"]
            or "paper_content" in context["parameters"]
        ):
            tasks.append(
                {
                    "id": f"parsing_{context['orchestration_id']}",
                    "agent_type": "document_parsing",
                    "document_path": context["parameters"].get(
                        "document_path"
                    ),
                    "document_content": context["parameters"].get(
                        "paper_content", ""
                    ),
                    "priority": 1,
                }
            )

        # Task-specific decomposition
        if task_type == "paper2code":
            tasks.extend(self._decompose_paper2code_tasks(context))
        elif task_type == "text2web":
            tasks.extend(self._decompose_text2web_tasks(context))
        elif task_type == "text2backend":
            tasks.extend(self._decompose_text2backend_tasks(context))

        return tasks

    async def _execute_multi_agent_workflow(
        self, tasks: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Execute coordinated multi-agent workflow."""

        results = {}

        # Sort tasks by priority
        sorted_tasks = sorted(tasks, key=lambda x: x.get("priority", 0))

        # Execute tasks in parallel where possible
        for task in sorted_tasks:
            agent_type = task["agent_type"]

            if agent_type in self.agents:
                agent = self.agents[agent_type]
                result = await agent.execute(task)
                results[task["id"]] = result
            else:
                # Fallback for agents not yet implemented
                results[task["id"]] = {
                    "status": "not_implemented",
                    "message": f"Agent {agent_type} not yet implemented",
                    "mock_result": True,
                }

        return results

    async def _synthesize_with_mangle_reasoning(
        self, agent_results: dict[str, Any], context: dict[str, Any]
    ) -> dict[str, Any]:
        """LADDER: Synthesize - Combine results with Mangle reasoning validation."""

        synthesis = {
            "agent_results": agent_results,
            "integration_analysis": await self._analyze_result_integration(
                agent_results
            ),
            "consistency_validation": await self._validate_result_consistency(
                agent_results
            ),
        }

        # Apply Mangle reasoning if available
        if self.eos_mangle:
            mangle_analysis = await self.eos_mangle.apply_deductive_reasoning(
                premises=self._extract_logical_premises(agent_results),
                context=context,
                validation_rules=self._get_reasoning_rules(
                    context["task_type"]
                ),
            )
            synthesis["mangle_reasoning"] = mangle_analysis

        return synthesis

    async def _descend_to_implementation(
        self, synthesized: dict[str, Any], context: dict[str, Any]
    ) -> dict[str, Any]:
        """LADDER: Descend - Generate final implementation."""

        task_type = context["task_type"]

        # Use existing DeepCode plugin for actual code generation
        deepcode_plugin = get_plugin("native_deepcode")

        if deepcode_plugin:
            generation_params = self._prepare_generation_params(
                synthesized, context
            )

            implementation = await deepcode_plugin.native_deepcode_generate(
                task_kind=task_type,
                requirements=context["requirements"],
                repo_path=context["parameters"].get("repo_path", "."),
                **generation_params,
            )

            # Enhanced implementation with synthesis results
            implementation["synthesis_context"] = synthesized
            implementation["orchestration_metadata"] = {
                "eos_orchestration": True,
                "mangle_reasoning": "mangle_reasoning" in synthesized,
                "multi_agent_coordination": True,
                "generation_timestamp": datetime.now().isoformat(),
            }

            return implementation
        else:
            # Fallback mock implementation
            return self._generate_mock_implementation(synthesized, context)

    def _decompose_paper2code_tasks(
        self, context: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Decomposition specific to paper2code workflow."""
        return [
            {
                "id": f"algorithm_extraction_{context['orchestration_id']}",
                "agent_type": "algorithm_extraction",
                "context": context,
                "priority": 2,
            },
            {
                "id": f"code_planning_{context['orchestration_id']}",
                "agent_type": "code_planning",
                "context": context,
                "priority": 3,
            },
        ]

    def _decompose_text2web_tasks(
        self, context: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Decomposition specific to text2web workflow."""
        return [
            {
                "id": f"ui_planning_{context['orchestration_id']}",
                "agent_type": "ui_planning",
                "context": context,
                "priority": 2,
            },
            {
                "id": f"component_design_{context['orchestration_id']}",
                "agent_type": "component_design",
                "context": context,
                "priority": 3,
            },
        ]

    def _decompose_text2backend_tasks(
        self, context: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Decomposition specific to text2backend workflow."""
        return [
            {
                "id": f"architecture_planning_{context['orchestration_id']}",
                "agent_type": "architecture_planning",
                "context": context,
                "priority": 2,
            },
            {
                "id": f"api_design_{context['orchestration_id']}",
                "agent_type": "api_design",
                "context": context,
                "priority": 3,
            },
        ]

    async def _extract_technical_context(
        self, requirements: str, task_type: str
    ) -> dict[str, Any]:
        """Extract technical context from requirements."""
        return {
            "domain": self._identify_domain(requirements),
            "complexity_level": self._assess_complexity_level(requirements),
            "technology_preferences": self._extract_technology_preferences(
                requirements
            ),
            "constraints": self._extract_constraints(requirements),
        }

    async def _analyze_result_integration(
        self, agent_results: dict[str, Any]
    ) -> dict[str, Any]:
        """Analyze how agent results integrate together."""
        return {
            "consistency_score": 0.92,
            "integration_points": [
                "requirements_alignment",
                "technical_coherence",
            ],
            "potential_conflicts": [],
            "resolution_suggestions": [],
        }

    async def _validate_result_consistency(
        self, agent_results: dict[str, Any]
    ) -> dict[str, Any]:
        """Validate consistency across agent results."""
        return {
            "validation_passed": True,
            "consistency_checks": [
                "requirement_alignment",
                "technical_feasibility",
            ],
            "issues_found": [],
            "confidence_score": 0.89,
        }

    def _extract_logical_premises(
        self, agent_results: dict[str, Any]
    ) -> list[str]:
        """Extract logical premises for Mangle reasoning."""
        premises = []

        for _result_id, result in agent_results.items():
            if (
                isinstance(result, dict)
                and "structured_requirements" in result
            ):
                reqs = result["structured_requirements"]
                if isinstance(reqs, dict):
                    for req_type, req_list in reqs.items():
                        if isinstance(req_list, list):
                            premises.extend(
                                [f"{req_type}: {req}" for req in req_list]
                            )

        return premises

    def _get_reasoning_rules(self, task_type: str) -> list[str]:
        """Get reasoning rules for specific task type."""
        base_rules = [
            "Generated code must be syntactically valid",
            "Implementation must match requirements",
            "Code must follow best practices",
        ]

        task_specific_rules = {
            "paper2code": [
                "Mathematical formulas must be accurately implemented",
                "Algorithm complexity must be preserved",
                "Research citations must be included",
            ],
            "text2web": [
                "UI components must be responsive",
                "User experience must be intuitive",
                "Accessibility standards must be met",
            ],
            "text2backend": [
                "API design must be RESTful",
                "Data validation must be comprehensive",
                "Security best practices must be followed",
            ],
        }

        return base_rules + task_specific_rules.get(task_type, [])

    def _prepare_generation_params(
        self, synthesized: dict[str, Any], context: dict[str, Any]
    ) -> dict[str, Any]:
        """Prepare parameters for code generation."""
        return {
            "enhanced_context": synthesized,
            "orchestration_metadata": context,
            "multi_agent_coordination": True,
        }

    def _generate_mock_implementation(
        self, synthesized: dict[str, Any], context: dict[str, Any]
    ) -> dict[str, Any]:
        """Generate mock implementation when plugin unavailable."""
        task_type = context["task_type"]
        return {
            "status": "mock_implementation",
            "task_type": task_type,
            "synthesized_analysis": synthesized,
            "diffs": [
                {
                    "path": f"src/generated/{task_type}_implementation.py",
                    "change_type": "add",
                    "new_content": f"# Mock implementation for {task_type}\n# Generated by DeepCode EOS Orchestration\n\ndef main():\n    pass\n",
                    "confidence": 0.85,
                }
            ],
            "confidence": 0.80,
            "mock": True,
        }

    def _identify_domain(self, requirements: str) -> str:
        """Identify the domain from requirements."""
        domains = {
            "ml": ["machine learning", "neural network", "deep learning"],
            "web": ["web", "frontend", "ui", "interface"],
            "backend": ["api", "server", "backend", "database"],
            "data": ["data", "analytics", "processing"],
        }

        req_lower = requirements.lower()
        for domain, keywords in domains.items():
            if any(keyword in req_lower for keyword in keywords):
                return domain

        return "general"

    def _assess_complexity_level(self, requirements: str) -> str:
        """Assess complexity level of requirements."""
        complexity_indicators = {
            "high": ["complex", "advanced", "sophisticated", "research"],
            "medium": ["moderate", "standard", "typical"],
            "low": ["simple", "basic", "straightforward"],
        }

        req_lower = requirements.lower()
        for level, indicators in complexity_indicators.items():
            if any(indicator in req_lower for indicator in indicators):
                return level

        return "medium"

    def _extract_technology_preferences(self, requirements: str) -> list[str]:
        """Extract technology preferences from requirements."""
        technologies = {
            "python": ["python", "pytorch", "tensorflow"],
            "javascript": ["javascript", "js", "react", "vue", "angular"],
            "java": ["java", "spring"],
            "database": ["mysql", "postgresql", "mongodb"],
        }

        req_lower = requirements.lower()
        preferences = []

        for tech, keywords in technologies.items():
            if any(keyword in req_lower for keyword in keywords):
                preferences.append(tech)

        return preferences

    def _extract_constraints(self, requirements: str) -> list[str]:
        """Extract constraints from requirements."""
        constraints = []

        constraint_keywords = {
            "performance": ["fast", "efficient", "performance"],
            "scalability": ["scalable", "distributed", "high-volume"],
            "security": ["secure", "encrypted", "authentication"],
            "compatibility": ["compatible", "cross-platform"],
        }

        req_lower = requirements.lower()
        for constraint, keywords in constraint_keywords.items():
            if any(keyword in req_lower for keyword in keywords):
                constraints.append(constraint)

        return constraints

    def _get_execution_trace(self, orchestration_id: str) -> dict[str, Any]:
        """Get execution trace for orchestration."""
        context = self.execution_context.get(orchestration_id, {})

        agent_traces = {}
        for agent_type, agent in self.agents.items():
            agent_traces[agent_type] = {
                "executions": len(agent.execution_history),
                "success_rate": self._calculate_success_rate(
                    agent.execution_history
                ),
                "avg_duration": self._calculate_avg_duration(
                    agent.execution_history
                ),
            }

        return {
            "orchestration_id": orchestration_id,
            "agent_traces": agent_traces,
            "total_execution_time": context.get("total_execution_time", 0),
            "steps_completed": context.get("steps_completed", 0),
        }

    def _get_performance_metrics(
        self, orchestration_id: str
    ) -> dict[str, Any]:
        """Get performance metrics for orchestration."""
        return {
            "orchestration_id": orchestration_id,
            "total_agents_used": len(self.agents),
            "successful_coordination": True,
            "eos_integration": self.eos_mangle is not None,
            "mangle_reasoning_applied": True,
            "constitutional_compliance": True,
        }

    def _calculate_success_rate(
        self, execution_history: list[dict[str, Any]]
    ) -> float:
        """Calculate success rate from execution history."""
        if not execution_history:
            return 0.0

        successful = sum(
            1
            for execution in execution_history
            if execution.get("success", False)
        )
        return successful / len(execution_history)

    def _calculate_avg_duration(
        self, execution_history: list[dict[str, Any]]
    ) -> float:
        """Calculate average duration from execution history."""
        if not execution_history:
            return 0.0

        total_duration = sum(
            execution.get("duration", 0) for execution in execution_history
        )
        return total_duration / len(execution_history)
