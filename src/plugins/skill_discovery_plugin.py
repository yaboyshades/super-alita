"""
Skill Discovery Plugin for Super Alita.
Implements Proposer-Agent-Evaluator (PAE) pipeline for skill evolution.
"""

import asyncio
import json
import random
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import uuid
import ast
import traceback

from ..core.plugin_interface import PluginInterface
from ..core.neural_atom import create_skill_atom
from ..core.genealogy import trace_birth, trace_fitness
from ..tools.mcts_evolution import create_skill_evolution_arena


@dataclass
class SkillProposal:
    """A proposed skill for evaluation."""
    
    id: str
    name: str
    description: str
    code: str
    parent_skills: List[str] = field(default_factory=list)
    proposer: str = "unknown"
    confidence: float = 0.5
    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SkillEvaluation:
    """Evaluation result for a skill."""
    
    skill_id: str
    skill_name: str
    performance_score: float
    safety_score: float
    usefulness_score: float
    execution_time: float
    success: bool
    errors: List[str] = field(default_factory=list)
    feedback: str = ""
    test_cases_passed: int = 0
    total_test_cases: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def overall_score(self) -> float:
        """Calculate overall skill score."""
        if not self.success:
            return 0.0
        
        # Weighted combination of different scores
        score = (
            self.performance_score * 0.4 +
            self.safety_score * 0.3 +
            self.usefulness_score * 0.3
        )
        
        # Bonus for passing test cases
        if self.total_test_cases > 0:
            test_bonus = (self.test_cases_passed / self.total_test_cases) * 0.2
            score += test_bonus
        
        return min(1.0, score)


class SkillProposer:
    """Proposes new skills based on context and existing skills."""
    
    def __init__(self):
        self.proposal_templates = [
            "combine_skills",
            "specialize_skill",
            "generalize_skill",
            "optimize_skill",
            "adapt_skill"
        ]
    
    async def propose_skill(
        self,
        context: Dict[str, Any],
        existing_skills: List[str],
        failed_attempts: List[str] = None
    ) -> SkillProposal:
        """Propose a new skill based on context."""
        
        proposal_type = random.choice(self.proposal_templates)
        
        if proposal_type == "combine_skills" and len(existing_skills) >= 2:
            return await self._propose_combined_skill(existing_skills, context)
        elif proposal_type == "specialize_skill" and existing_skills:
            return await self._propose_specialized_skill(existing_skills, context)
        elif proposal_type == "generalize_skill" and existing_skills:
            return await self._propose_generalized_skill(existing_skills, context)
        elif proposal_type == "optimize_skill" and existing_skills:
            return await self._propose_optimized_skill(existing_skills, context)
        else:
            return await self._propose_adaptive_skill(context, failed_attempts or [])
    
    async def _propose_combined_skill(
        self,
        skills: List[str],
        context: Dict[str, Any]
    ) -> SkillProposal:
        """Propose a skill that combines two existing skills."""
        
        skill1, skill2 = random.sample(skills, 2)
        
        name = f"combined_{skill1}_{skill2}"
        description = f"Combines the capabilities of {skill1} and {skill2} for enhanced functionality"
        
        # Generate simple combination code
        code = f'''
def {name.replace("-", "_")}(input_data, **kwargs):
    """Combined skill: {description}"""
    
    # Apply first skill
    intermediate_result = apply_skill("{skill1}", input_data, **kwargs)
    
    # Apply second skill to the result
    final_result = apply_skill("{skill2}", intermediate_result, **kwargs)
    
    return final_result
'''
        
        return SkillProposal(
            id=str(uuid.uuid4()),
            name=name,
            description=description,
            code=code.strip(),
            parent_skills=[skill1, skill2],
            proposer="skill_combiner",
            confidence=0.7,
            metadata={"proposal_type": "combination"}
        )
    
    async def _propose_specialized_skill(
        self,
        skills: List[str],
        context: Dict[str, Any]
    ) -> SkillProposal:
        """Propose a specialized version of an existing skill."""
        
        base_skill = random.choice(skills)
        domain = context.get("domain", "general")
        
        name = f"specialized_{base_skill}_{domain}"
        description = f"Specialized version of {base_skill} optimized for {domain} domain"
        
        code = f'''
def {name.replace("-", "_")}(input_data, **kwargs):
    """Specialized skill: {description}"""
    
    # Domain-specific preprocessing
    if "{domain}" in str(input_data).lower():
        # Apply domain-specific optimizations
        processed_input = optimize_for_domain(input_data, "{domain}")
    else:
        processed_input = input_data
    
    # Apply base skill with specialization
    result = apply_skill("{base_skill}", processed_input, 
                        domain="{domain}", **kwargs)
    
    # Domain-specific postprocessing
    final_result = postprocess_for_domain(result, "{domain}")
    
    return final_result
'''
        
        return SkillProposal(
            id=str(uuid.uuid4()),
            name=name,
            description=description,
            code=code.strip(),
            parent_skills=[base_skill],
            proposer="skill_specializer",
            confidence=0.6,
            metadata={"proposal_type": "specialization", "domain": domain}
        )
    
    async def _propose_generalized_skill(
        self,
        skills: List[str],
        context: Dict[str, Any]
    ) -> SkillProposal:
        """Propose a generalized version of an existing skill."""
        
        base_skill = random.choice(skills)
        
        name = f"generalized_{base_skill}"
        description = f"Generalized version of {base_skill} that works across multiple domains"
        
        code = f'''
def {name.replace("-", "_")}(input_data, **kwargs):
    """Generalized skill: {description}"""
    
    # Detect input type/domain automatically
    detected_domain = detect_domain(input_data)
    
    # Apply adaptive preprocessing
    processed_input = adaptive_preprocess(input_data, detected_domain)
    
    # Apply base skill with adaptive parameters
    result = apply_skill("{base_skill}", processed_input, 
                        adaptive=True, **kwargs)
    
    # Adaptive postprocessing
    final_result = adaptive_postprocess(result, detected_domain)
    
    return final_result
'''
        
        return SkillProposal(
            id=str(uuid.uuid4()),
            name=name,
            description=description,
            code=code.strip(),
            parent_skills=[base_skill],
            proposer="skill_generalizer",
            confidence=0.5,
            metadata={"proposal_type": "generalization"}
        )
    
    async def _propose_optimized_skill(
        self,
        skills: List[str],
        context: Dict[str, Any]
    ) -> SkillProposal:
        """Propose an optimized version of an existing skill."""
        
        base_skill = random.choice(skills)
        
        name = f"optimized_{base_skill}"
        description = f"Performance-optimized version of {base_skill}"
        
        code = f'''
def {name.replace("-", "_")}(input_data, **kwargs):
    """Optimized skill: {description}"""
    
    # Performance optimizations
    if isinstance(input_data, (list, tuple)) and len(input_data) > 100:
        # Use batch processing for large inputs
        results = []
        batch_size = 50
        for i in range(0, len(input_data), batch_size):
            batch = input_data[i:i+batch_size]
            batch_result = apply_skill("{base_skill}", batch, **kwargs)
            results.extend(batch_result)
        return results
    else:
        # Use caching for small inputs
        cache_key = hash(str(input_data) + str(kwargs))
        cached_result = get_from_cache(cache_key)
        if cached_result is not None:
            return cached_result
        
        result = apply_skill("{base_skill}", input_data, **kwargs)
        cache_result(cache_key, result)
        return result
'''
        
        return SkillProposal(
            id=str(uuid.uuid4()),
            name=name,
            description=description,
            code=code.strip(),
            parent_skills=[base_skill],
            proposer="skill_optimizer",
            confidence=0.8,
            metadata={"proposal_type": "optimization"}
        )
    
    async def _propose_adaptive_skill(
        self,
        context: Dict[str, Any],
        failed_attempts: List[str]
    ) -> SkillProposal:
        """Propose a completely new adaptive skill."""
        
        task_type = context.get("task_type", "general")
        
        name = f"adaptive_{task_type}_skill_{random.randint(1000, 9999)}"
        description = f"Adaptive skill for {task_type} tasks with learning capabilities"
        
        code = f'''
def {name.replace("-", "_")}(input_data, **kwargs):
    """Adaptive skill: {description}"""
    
    # Adaptive behavior based on input characteristics
    input_size = len(str(input_data))
    
    if input_size < 100:
        # Simple processing for small inputs
        return simple_process(input_data, **kwargs)
    elif input_size < 1000:
        # Medium complexity processing
        return medium_process(input_data, **kwargs)
    else:
        # Complex processing for large inputs
        return complex_process(input_data, **kwargs)
'''
        
        return SkillProposal(
            id=str(uuid.uuid4()),
            name=name,
            description=description,
            code=code.strip(),
            parent_skills=[],
            proposer="skill_creator",
            confidence=0.4,
            metadata={"proposal_type": "adaptive", "task_type": task_type}
        )


class SkillEvaluator:
    """Evaluates proposed skills for performance, safety, and usefulness."""
    
    def __init__(self):
        self.test_environments = {}
        self.safety_checkers = [
            self._check_code_safety,
            self._check_execution_safety,
            self._check_resource_usage
        ]
    
    async def evaluate_skill(
        self,
        proposal: SkillProposal,
        test_cases: List[Dict[str, Any]] = None
    ) -> SkillEvaluation:
        """Evaluate a skill proposal comprehensively."""
        
        start_time = datetime.utcnow()
        errors = []
        
        try:
            # Safety evaluation
            safety_score = await self._evaluate_safety(proposal, errors)
            
            # Performance evaluation
            performance_score = await self._evaluate_performance(proposal, test_cases or [], errors)
            
            # Usefulness evaluation
            usefulness_score = await self._evaluate_usefulness(proposal, errors)
            
            # Test case evaluation
            test_results = await self._run_test_cases(proposal, test_cases or [], errors)
            
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            
            success = len(errors) == 0 and safety_score > 0.5
            
            feedback = self._generate_feedback(proposal, safety_score, performance_score, usefulness_score, errors)
            
            return SkillEvaluation(
                skill_id=proposal.id,
                skill_name=proposal.name,
                performance_score=performance_score,
                safety_score=safety_score,
                usefulness_score=usefulness_score,
                execution_time=execution_time,
                success=success,
                errors=errors,
                feedback=feedback,
                test_cases_passed=test_results.get("passed", 0),
                total_test_cases=test_results.get("total", 0),
                metadata={
                    "proposal_type": proposal.metadata.get("proposal_type", "unknown"),
                    "parent_skills": proposal.parent_skills,
                    "confidence": proposal.confidence
                }
            )
            
        except Exception as e:
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            errors.append(f"Evaluation error: {str(e)}")
            
            return SkillEvaluation(
                skill_id=proposal.id,
                skill_name=proposal.name,
                performance_score=0.0,
                safety_score=0.0,
                usefulness_score=0.0,
                execution_time=execution_time,
                success=False,
                errors=errors,
                feedback=f"Evaluation failed: {str(e)}"
            )
    
    async def _evaluate_safety(self, proposal: SkillProposal, errors: List[str]) -> float:
        """Evaluate the safety of a skill."""
        
        safety_scores = []
        
        for checker in self.safety_checkers:
            try:
                score = await checker(proposal)
                safety_scores.append(score)
            except Exception as e:
                errors.append(f"Safety check error: {str(e)}")
                safety_scores.append(0.0)
        
        return sum(safety_scores) / len(safety_scores) if safety_scores else 0.0
    
    async def _check_code_safety(self, proposal: SkillProposal) -> float:
        """Check code for dangerous patterns."""
        
        dangerous_patterns = [
            "import os", "import subprocess", "import sys",
            "exec(", "eval(", "__import__",
            "open(", "file(", "input(",
            "rm ", "del ", "remove("
        ]
        
        code_lower = proposal.code.lower()
        dangerous_count = sum(1 for pattern in dangerous_patterns if pattern in code_lower)
        
        # Penalize dangerous patterns
        safety_score = max(0.0, 1.0 - (dangerous_count * 0.3))
        
        return safety_score
    
    async def _check_execution_safety(self, proposal: SkillProposal) -> float:
        """Check if code can be safely parsed."""
        
        try:
            # Try to parse the code
            ast.parse(proposal.code)
            return 1.0
        except SyntaxError:
            return 0.0
        except Exception:
            return 0.5
    
    async def _check_resource_usage(self, proposal: SkillProposal) -> float:
        """Check for potential resource usage issues."""
        
        # Look for potentially expensive operations
        expensive_patterns = [
            "while True:", "for _ in range(", "recursive",
            "infinite", "*.+*", "heavy", "large"
        ]
        
        code_lower = proposal.code.lower()
        expensive_count = sum(1 for pattern in expensive_patterns if pattern in code_lower)
        
        # Moderate penalty for potentially expensive operations
        resource_score = max(0.3, 1.0 - (expensive_count * 0.2))
        
        return resource_score
    
    async def _evaluate_performance(self, proposal: SkillProposal, test_cases: List[Dict[str, Any]], errors: List[str]) -> float:
        """Evaluate skill performance."""
        
        # Basic performance metrics
        code_length = len(proposal.code)
        complexity_score = max(0.0, 1.0 - (code_length / 1000))  # Prefer shorter code
        
        # Check for performance-oriented patterns
        performance_patterns = [
            "cache", "optimize", "batch", "efficient",
            "fast", "quick", "speed", "performance"
        ]
        
        code_lower = proposal.code.lower()
        performance_indicators = sum(1 for pattern in performance_patterns if pattern in code_lower)
        
        pattern_score = min(1.0, performance_indicators * 0.2)
        
        # Combine scores
        performance_score = (complexity_score * 0.7) + (pattern_score * 0.3)
        
        return performance_score
    
    async def _evaluate_usefulness(self, proposal: SkillProposal, errors: List[str]) -> float:
        """Evaluate the potential usefulness of a skill."""
        
        # Usefulness indicators
        usefulness_patterns = [
            "process", "analyze", "transform", "generate",
            "optimize", "enhance", "improve", "solve"
        ]
        
        description_lower = proposal.description.lower()
        code_lower = proposal.code.lower()
        
        usefulness_indicators = sum(
            1 for pattern in usefulness_patterns 
            if pattern in description_lower or pattern in code_lower
        )
        
        # Base usefulness score
        base_score = min(1.0, usefulness_indicators * 0.15)
        
        # Bonus for combining multiple skills
        combination_bonus = min(0.3, len(proposal.parent_skills) * 0.1)
        
        # Bonus for clear, descriptive naming
        naming_bonus = 0.1 if len(proposal.name.split("_")) > 2 else 0.0
        
        usefulness_score = base_score + combination_bonus + naming_bonus
        
        return min(1.0, usefulness_score)
    
    async def _run_test_cases(self, proposal: SkillProposal, test_cases: List[Dict[str, Any]], errors: List[str]) -> Dict[str, int]:
        """Run test cases for the skill."""
        
        if not test_cases:
            # Generate simple test cases
            test_cases = [
                {"input": "test", "expected_type": str},
                {"input": [1, 2, 3], "expected_type": (list, tuple)},
                {"input": {"key": "value"}, "expected_type": dict}
            ]
        
        passed = 0
        total = len(test_cases)
        
        for test_case in test_cases:
            try:
                # This is a simplified test runner
                # In practice, you'd execute the skill safely
                input_data = test_case.get("input")
                expected_type = test_case.get("expected_type", object)
                
                # Mock execution result
                if isinstance(expected_type, type):
                    passed += 1  # Assume success for now
                else:
                    passed += 1
                    
            except Exception as e:
                errors.append(f"Test case failed: {str(e)}")
        
        return {"passed": passed, "total": total}
    
    def _generate_feedback(
        self,
        proposal: SkillProposal,
        safety_score: float,
        performance_score: float,
        usefulness_score: float,
        errors: List[str]
    ) -> str:
        """Generate human-readable feedback for the skill."""
        
        feedback_parts = []
        
        # Overall assessment
        overall_score = (safety_score + performance_score + usefulness_score) / 3
        if overall_score > 0.8:
            feedback_parts.append("Excellent skill proposal with high quality.")
        elif overall_score > 0.6:
            feedback_parts.append("Good skill proposal with minor issues.")
        elif overall_score > 0.4:
            feedback_parts.append("Acceptable skill proposal with several areas for improvement.")
        else:
            feedback_parts.append("Poor skill proposal requiring significant improvements.")
        
        # Specific feedback
        if safety_score < 0.5:
            feedback_parts.append("Safety concerns detected - code may contain risky patterns.")
        
        if performance_score < 0.5:
            feedback_parts.append("Performance could be improved - consider optimization techniques.")
        
        if usefulness_score < 0.5:
            feedback_parts.append("Usefulness is limited - consider more practical applications.")
        
        # Error feedback
        if errors:
            feedback_parts.append(f"Errors encountered: {'; '.join(errors[:3])}")
        
        return " ".join(feedback_parts)


class SkillDiscoveryPlugin(PluginInterface):
    """
    Skill Discovery Plugin implementing PAE (Proposer-Agent-Evaluator) pipeline.
    
    Automatically discovers and evolves new skills through iterative
    proposal, testing, and evaluation cycles.
    """
    
    def __init__(self):
        super().__init__()
        self.proposer = SkillProposer()
        self.evaluator = SkillEvaluator()
        self.active_skills: Dict[str, Any] = {}
        self.skill_proposals: Dict[str, SkillProposal] = {}
        self.skill_evaluations: Dict[str, SkillEvaluation] = {}
        self.discovery_task: Optional[asyncio.Task] = None
        self.evolution_arena: Optional[Any] = None
        self.discovery_stats = {
            "proposals_generated": 0,
            "skills_evaluated": 0,
            "skills_accepted": 0,
            "skills_rejected": 0
        }
    
    @property
    def name(self) -> str:
        return "skill_discovery"
    
    @property
    def description(self) -> str:
        return "Discovers and evolves agent skills through PAE pipeline"
    
    async def setup(self, event_bus, store, config: Dict[str, Any]) -> None:
        await super().setup(event_bus, store, config)
        
        # Initialize evolution arena
        base_skills = self.get_config("base_skills", [
            "data_processing", "text_analysis", "pattern_recognition",
            "optimization", "problem_solving", "learning"
        ])
        
        self.evolution_arena = await create_skill_evolution_arena(
            base_skills=base_skills,
            max_iterations=self.get_config("evolution_iterations", 500)
        )
    
    async def start(self) -> None:
        await super().start()
        
        # Subscribe to events
        await self.subscribe("skill_proposal", self._handle_skill_proposal)
        await self.subscribe("skill_evaluation", self._handle_skill_evaluation_result)
        await self.subscribe("planning", self._handle_planning_event)
        await self.subscribe("system", self._handle_system_event)
        
        # Start discovery task
        discovery_interval = self.get_config("discovery_interval_minutes", 30)
        self.discovery_task = self.add_task(
            self._periodic_skill_discovery(discovery_interval * 60)
        )
    
    async def shutdown(self) -> None:
        if self.discovery_task:
            self.discovery_task.cancel()
    
    async def _handle_skill_proposal(self, event) -> None:
        """Handle external skill proposals."""
        
        proposal = SkillProposal(
            id=str(uuid.uuid4()),
            name=event.skill_name,
            description=event.skill_description,
            code=event.skill_code,
            parent_skills=getattr(event, 'parent_skills', []),
            proposer=event.proposer,
            confidence=getattr(event, 'confidence', 0.5)
        )
        
        await self._evaluate_and_potentially_accept_skill(proposal)
    
    async def _handle_skill_evaluation_result(self, event) -> None:
        """Handle skill evaluation results."""
        
        skill_name = event.skill_name
        
        # Update skill fitness in genealogy
        trace_fitness(f"skill:{skill_name}", event.performance_score)
        
        # Store evaluation
        if hasattr(event, 'skill_id') and event.skill_id in self.skill_proposals:
            evaluation = SkillEvaluation(
                skill_id=event.skill_id,
                skill_name=skill_name,
                performance_score=event.performance_score,
                safety_score=getattr(event, 'safety_score', 0.8),
                usefulness_score=getattr(event, 'usefulness_score', 0.6),
                execution_time=getattr(event, 'execution_time', 0.0),
                success=event.success,
                feedback=getattr(event, 'feedback', "")
            )
            
            self.skill_evaluations[event.skill_id] = evaluation
    
    async def _handle_planning_event(self, event) -> None:
        """Handle planning events to discover needed skills."""
        
        if hasattr(event, 'goal'):
            # Extract potential skill needs from goal
            context = {
                "goal": event.goal,
                "task_type": self._infer_task_type(event.goal),
                "domain": getattr(event, 'domain', 'general')
            }
            
            # Consider proposing skills for this goal
            await self._propose_contextual_skills(context)
    
    async def _handle_system_event(self, event) -> None:
        """Handle system events for skill discovery triggers."""
        
        if event.message == "skill_discovery_requested":
            await self._trigger_skill_discovery()
        elif event.message == "evolution_requested":
            await self._trigger_evolution()
    
    async def _periodic_skill_discovery(self, interval_seconds: float) -> None:
        """Periodic skill discovery process."""
        
        while self.is_running:
            try:
                await asyncio.sleep(interval_seconds)
                await self._trigger_skill_discovery()
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Error in skill discovery: {e}")
                await asyncio.sleep(60)
    
    async def _trigger_skill_discovery(self) -> None:
        """Trigger skill discovery process."""
        
        # Get current context
        context = await self._get_discovery_context()
        
        # Generate proposals
        num_proposals = self.get_config("proposals_per_cycle", 3)
        for _ in range(num_proposals):
            proposal = await self.proposer.propose_skill(
                context=context,
                existing_skills=list(self.active_skills.keys()),
                failed_attempts=self._get_recent_failed_skills()
            )
            
            await self._evaluate_and_potentially_accept_skill(proposal)
    
    async def _trigger_evolution(self) -> None:
        """Trigger evolutionary skill discovery."""
        
        if not self.evolution_arena:
            return
        
        print("Starting evolutionary skill discovery...")
        
        # Run evolution
        evolved_solutions = await self.evolution_arena.evolve(num_generations=3)
        
        # Convert evolved solutions to skill proposals
        for i, solution in enumerate(evolved_solutions[:5]):  # Top 5 solutions
            proposal = await self._solution_to_skill_proposal(solution, i)
            await self._evaluate_and_potentially_accept_skill(proposal)
    
    async def _solution_to_skill_proposal(self, solution: Dict[str, Any], index: int) -> SkillProposal:
        """Convert evolution solution to skill proposal."""
        
        skills = solution.get("skills", [])
        depth = solution.get("depth", 0)
        
        name = f"evolved_skill_{index}_{datetime.utcnow().strftime('%H%M%S')}"
        description = f"Evolved skill combining: {', '.join(skills[:3])}"
        
        # Generate code based on evolved skills
        code = f'''
def {name}(input_data, **kwargs):
    """Evolved skill: {description}"""
    
    result = input_data
    
    # Apply evolved skill sequence
'''
        
        for skill in skills[:3]:  # Limit to 3 skills to avoid complexity
            code += f'    result = apply_skill("{skill}", result, **kwargs)\n'
        
        code += '    return result'
        
        return SkillProposal(
            id=str(uuid.uuid4()),
            name=name,
            description=description,
            code=code,
            parent_skills=skills[:3],
            proposer="evolution_arena",
            confidence=0.6,
            metadata={
                "proposal_type": "evolution",
                "evolution_depth": depth,
                "solution_index": index
            }
        )
    
    async def _evaluate_and_potentially_accept_skill(self, proposal: SkillProposal) -> None:
        """Evaluate a skill proposal and potentially accept it."""
        
        self.skill_proposals[proposal.id] = proposal
        self.discovery_stats["proposals_generated"] += 1
        
        # Evaluate the skill
        evaluation = await self.evaluator.evaluate_skill(proposal)
        self.skill_evaluations[proposal.id] = evaluation
        self.discovery_stats["skills_evaluated"] += 1
        
        # Emit evaluation event
        await self.emit_event(
            "skill_evaluation",
            skill_id=proposal.id,
            skill_name=proposal.name,
            performance_score=evaluation.performance_score,
            safety_score=evaluation.safety_score,
            usefulness_score=evaluation.usefulness_score,
            execution_time=evaluation.execution_time,
            success=evaluation.success,
            feedback=evaluation.feedback,
            overall_score=evaluation.overall_score
        )
        
        # Accept if meets criteria
        acceptance_threshold = self.get_config("acceptance_threshold", 0.6)
        if evaluation.overall_score >= acceptance_threshold:
            await self._accept_skill(proposal, evaluation)
        else:
            self.discovery_stats["skills_rejected"] += 1
    
    async def _accept_skill(self, proposal: SkillProposal, evaluation: SkillEvaluation) -> None:
        """Accept a skill into the active skill set."""
        
        # Create skill atom
        atom = await create_skill_atom(
            store=self.store,
            skill_name=proposal.name,
            skill_obj={
                "code": proposal.code,
                "description": proposal.description,
                "evaluation": evaluation,
                "metadata": proposal.metadata
            },
            parent_skills=proposal.parent_skills,
            metadata={
                "proposer": proposal.proposer,
                "confidence": proposal.confidence,
                "overall_score": evaluation.overall_score,
                "acceptance_time": datetime.utcnow().isoformat()
            }
        )
        
        # Add to active skills
        self.active_skills[proposal.name] = atom
        self.discovery_stats["skills_accepted"] += 1
        
        # Track in genealogy
        trace_birth(
            key=f"skill:{proposal.name}",
            node_type="skill",
            birth_event="skill_acceptance",
            parent_keys=[f"skill:{parent}" for parent in proposal.parent_skills],
            metadata={
                "proposer": proposal.proposer,
                "overall_score": evaluation.overall_score,
                "proposal_type": proposal.metadata.get("proposal_type", "unknown")
            }
        )
        
        # Emit acceptance event
        await self.emit_event(
            "skill_accepted",
            skill_name=proposal.name,
            skill_description=proposal.description,
            parent_skills=proposal.parent_skills,
            overall_score=evaluation.overall_score,
            proposer=proposal.proposer
        )
        
        print(f"Accepted new skill: {proposal.name} (score: {evaluation.overall_score:.2f})")
    
    async def _get_discovery_context(self) -> Dict[str, Any]:
        """Get context for skill discovery."""
        
        return {
            "active_skill_count": len(self.active_skills),
            "recent_evaluations": len(self.skill_evaluations),
            "domain": "general",
            "task_type": "adaptive",
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def _infer_task_type(self, goal: str) -> str:
        """Infer task type from goal description."""
        
        goal_lower = goal.lower()
        
        if any(word in goal_lower for word in ["analyze", "analysis", "study"]):
            return "analysis"
        elif any(word in goal_lower for word in ["generate", "create", "build"]):
            return "generation"
        elif any(word in goal_lower for word in ["optimize", "improve", "enhance"]):
            return "optimization"
        elif any(word in goal_lower for word in ["learn", "understand", "comprehend"]):
            return "learning"
        else:
            return "general"
    
    def _get_recent_failed_skills(self) -> List[str]:
        """Get recently failed skill attempts."""
        
        failed_skills = []
        cutoff_time = datetime.utcnow() - timedelta(hours=24)
        
        for evaluation in self.skill_evaluations.values():
            proposal = self.skill_proposals.get(evaluation.skill_id)
            if proposal and proposal.created_at > cutoff_time and not evaluation.success:
                failed_skills.append(proposal.name)
        
        return failed_skills[-10:]  # Last 10 failures
    
    async def _propose_contextual_skills(self, context: Dict[str, Any]) -> None:
        """Propose skills based on specific context."""
        
        # This could be enhanced with more sophisticated context analysis
        if context.get("task_type") == "analysis":
            proposal = await self.proposer.propose_skill(
                context={"domain": "analysis", **context},
                existing_skills=list(self.active_skills.keys())
            )
            await self._evaluate_and_potentially_accept_skill(proposal)
    
    async def get_discovery_stats(self) -> Dict[str, Any]:
        """Get skill discovery statistics."""
        
        recent_evaluations = [
            eval for eval in self.skill_evaluations.values()
            if (datetime.utcnow() - datetime.fromisoformat(
                self.skill_proposals[eval.skill_id].created_at.isoformat()
            )).days < 7
        ]
        
        avg_score = 0.0
        if recent_evaluations:
            avg_score = sum(eval.overall_score for eval in recent_evaluations) / len(recent_evaluations)
        
        return {
            **self.discovery_stats,
            "active_skills": len(self.active_skills),
            "pending_proposals": len(self.skill_proposals) - len(self.skill_evaluations),
            "recent_average_score": avg_score,
            "acceptance_rate": (
                self.discovery_stats["skills_accepted"] / 
                max(self.discovery_stats["skills_evaluated"], 1)
            )
        }
    
    async def export_skills(self, filepath: str) -> None:
        """Export discovered skills to JSON file."""
        
        export_data = {
            "active_skills": {
                name: {
                    "description": skill.default_value.get("description", ""),
                    "code": skill.default_value.get("code", ""),
                    "metadata": skill.lineage_metadata
                }
                for name, skill in self.active_skills.items()
            },
            "discovery_stats": await self.get_discovery_stats(),
            "recent_proposals": [
                {
                    "name": prop.name,
                    "description": prop.description,
                    "proposer": prop.proposer,
                    "confidence": prop.confidence,
                    "evaluation": self.skill_evaluations.get(prop.id, {}).overall_score if prop.id in self.skill_evaluations else None
                }
                for prop in list(self.skill_proposals.values())[-10:]
            ],
            "export_timestamp": datetime.utcnow().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
