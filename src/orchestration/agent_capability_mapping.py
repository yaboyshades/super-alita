"""
Agent Capability Mapping for Solo Developer Multi-Agent Orchestration

Maps agent capabilities to tasks and provides intelligent routing
based on capability matching and performance history.
"""

import logging
from datetime import UTC, datetime
from typing import Any, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum


logger = logging.getLogger(__name__)


class CapabilityType(Enum):
    """Types of agent capabilities"""

    TECHNICAL_SKILL = "technical_skill"
    DOMAIN_KNOWLEDGE = "domain_knowledge"
    TOOL_USAGE = "tool_usage"
    PROGRAMMING_LANGUAGE = "programming_language"
    FRAMEWORK = "framework"
    PLATFORM = "platform"
    SECURITY_EXPERTISE = "security_expertise"
    TESTING_CAPABILITY = "testing_capability"


class ProficiencyLevel(Enum):
    """Proficiency levels for capabilities"""

    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


@dataclass
class Capability:
    """Individual capability definition"""

    capability_id: str
    name: str
    description: str
    capability_type: CapabilityType
    required_for_tasks: List[str] = field(default_factory=list)
    related_capabilities: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentCapabilityProfile:
    """Agent's capability profile with proficiency levels"""

    agent_id: str
    capabilities: Dict[str, ProficiencyLevel] = field(
        default_factory=dict
    )  # capability_id -> proficiency
    capability_scores: Dict[str, float] = field(
        default_factory=dict
    )  # capability_id -> performance score (0-1)
    learned_capabilities: Set[str] = field(
        default_factory=set
    )  # Capabilities learned over time
    improving_capabilities: Set[str] = field(
        default_factory=set
    )  # Currently improving capabilities
    last_updated: datetime = field(default_factory=lambda: datetime.now(UTC))
    total_tasks_completed: int = 0
    specialization_focus: Optional[CapabilityType] = None


@dataclass
class TaskRequirement:
    """Task capability requirements"""

    task_id: str
    task_type: str
    required_capabilities: List[str]  # Must have these
    preferred_capabilities: List[str] = field(default_factory=list)  # Nice to have
    minimum_proficiency: Dict[str, ProficiencyLevel] = field(
        default_factory=dict
    )  # capability_id -> min level
    capability_weights: Dict[str, float] = field(
        default_factory=dict
    )  # Importance weights
    complexity_score: float = 1.0  # 0.1 to 5.0
    description: str = ""


@dataclass
class CapabilityMatch:
    """Match result between agent capabilities and task requirements"""

    agent_id: str
    task_id: str
    overall_score: float  # 0-1
    required_capabilities_met: float  # Percentage of required capabilities met
    preferred_capabilities_met: float  # Percentage of preferred capabilities met
    capability_scores: Dict[str, float] = field(
        default_factory=dict
    )  # Individual capability scores
    missing_capabilities: List[str] = field(default_factory=list)
    weak_capabilities: List[str] = field(
        default_factory=list
    )  # Below required proficiency
    strengths: List[str] = field(default_factory=list)  # Strong capability matches
    confidence: float = 0.5
    recommendation: str = ""


class AgentCapabilityMapping:
    """System for mapping agent capabilities to tasks"""

    def __init__(self, event_bus=None):
        self.event_bus = event_bus
        self.capabilities: Dict[str, Capability] = {}
        self.agent_profiles: Dict[str, AgentCapabilityProfile] = {}
        self.task_requirements: Dict[str, TaskRequirement] = {}
        self.capability_performance_history: Dict[
            str, List[Tuple[datetime, str, bool, float]]
        ] = {}  # capability_id -> [(timestamp, agent_id, success, score)]

        # Initialize default capabilities
        self._initialize_default_capabilities()

    def _initialize_default_capabilities(self):
        """Initialize default capability definitions"""
        default_capabilities = [
            # Programming Languages
            Capability(
                capability_id="python_programming",
                name="Python Programming",
                description="Ability to write, debug, and optimize Python code",
                capability_type=CapabilityType.PROGRAMMING_LANGUAGE,
                required_for_tasks=["feature_development", "bug_fix", "testing"],
                keywords=["python", "py", "script", "code", "programming"],
            ),
            Capability(
                capability_id="javascript_programming",
                name="JavaScript Programming",
                description="Ability to write JavaScript/TypeScript code",
                capability_type=CapabilityType.PROGRAMMING_LANGUAGE,
                required_for_tasks=["frontend_development", "web_development"],
                keywords=["javascript", "js", "typescript", "ts", "node", "web"],
            ),
            # Security Capabilities
            Capability(
                capability_id="security_analysis",
                name="Security Analysis",
                description="Ability to identify and analyze security vulnerabilities",
                capability_type=CapabilityType.SECURITY_EXPERTISE,
                required_for_tasks=[
                    "security_scan",
                    "code_review",
                    "vulnerability_assessment",
                ],
                keywords=["security", "vulnerability", "cve", "owasp", "penetration"],
            ),
            Capability(
                capability_id="secure_coding",
                name="Secure Coding Practices",
                description="Knowledge of secure coding practices and standards",
                capability_type=CapabilityType.SECURITY_EXPERTISE,
                required_for_tasks=["secure_development", "code_review"],
                related_capabilities=["security_analysis"],
                keywords=[
                    "secure",
                    "sanitization",
                    "encryption",
                    "auth",
                    "authorization",
                ],
            ),
            # Testing Capabilities
            Capability(
                capability_id="unit_testing",
                name="Unit Testing",
                description="Ability to write and maintain unit tests",
                capability_type=CapabilityType.TESTING_CAPABILITY,
                required_for_tasks=["testing", "tdd"],
                keywords=["unittest", "pytest", "test", "mock", "assert"],
            ),
            Capability(
                capability_id="integration_testing",
                name="Integration Testing",
                description="Ability to design and implement integration tests",
                capability_type=CapabilityType.TESTING_CAPABILITY,
                required_for_tasks=["testing", "qa"],
                related_capabilities=["unit_testing"],
                keywords=["integration", "api_testing", "system_test", "e2e"],
            ),
            # Architecture & Design
            Capability(
                capability_id="system_architecture",
                name="System Architecture Design",
                description="Ability to design scalable system architectures",
                capability_type=CapabilityType.TECHNICAL_SKILL,
                required_for_tasks=["architecture_design", "system_design"],
                keywords=[
                    "architecture",
                    "design",
                    "scalability",
                    "patterns",
                    "microservices",
                ],
            ),
            Capability(
                capability_id="api_design",
                name="API Design",
                description="Ability to design REST APIs and GraphQL schemas",
                capability_type=CapabilityType.TECHNICAL_SKILL,
                required_for_tasks=["api_development", "backend_development"],
                keywords=["api", "rest", "graphql", "openapi", "swagger"],
            ),
            # Documentation
            Capability(
                capability_id="technical_documentation",
                name="Technical Documentation",
                description="Ability to write clear technical documentation",
                capability_type=CapabilityType.TECHNICAL_SKILL,
                required_for_tasks=["documentation", "api_documentation"],
                keywords=[
                    "documentation",
                    "readme",
                    "docs",
                    "markdown",
                    "technical_writing",
                ],
            ),
            # Performance & Optimization
            Capability(
                capability_id="performance_optimization",
                name="Performance Optimization",
                description="Ability to analyze and optimize application performance",
                capability_type=CapabilityType.TECHNICAL_SKILL,
                required_for_tasks=["performance_optimization", "code_optimization"],
                keywords=[
                    "performance",
                    "optimization",
                    "profiling",
                    "benchmarking",
                    "speed",
                ],
            ),
            # Tools & Frameworks
            Capability(
                capability_id="git_version_control",
                name="Git Version Control",
                description="Proficiency with Git version control system",
                capability_type=CapabilityType.TOOL_USAGE,
                required_for_tasks=["code_review", "collaboration"],
                keywords=["git", "version_control", "branch", "merge", "pull_request"],
            ),
            Capability(
                capability_id="docker_containerization",
                name="Docker Containerization",
                description="Ability to containerize applications with Docker",
                capability_type=CapabilityType.PLATFORM,
                required_for_tasks=["deployment", "devops"],
                keywords=["docker", "container", "dockerfile", "image", "deployment"],
            ),
        ]

        for capability in default_capabilities:
            self.capabilities[capability.capability_id] = capability

        logger.info(f"Initialized {len(default_capabilities)} default capabilities")

    def register_capability(self, capability: Capability):
        """Register a new capability"""
        self.capabilities[capability.capability_id] = capability
        logger.info(f"Registered capability: {capability.name}")

    def register_agent_capability(
        self,
        agent_id: str,
        capability_id: str,
        proficiency: ProficiencyLevel,
        initial_score: float = 0.5,
    ):
        """Register an agent's capability"""
        if agent_id not in self.agent_profiles:
            self.agent_profiles[agent_id] = AgentCapabilityProfile(agent_id=agent_id)

        profile = self.agent_profiles[agent_id]
        profile.capabilities[capability_id] = proficiency
        profile.capability_scores[capability_id] = initial_score
        profile.last_updated = datetime.now(UTC)

        logger.info(
            f"Registered capability {capability_id} for agent {agent_id} at {proficiency.value} level"
        )

    def update_agent_capability_score(
        self,
        agent_id: str,
        capability_id: str,
        performance_score: float,
        task_success: bool,
    ):
        """Update agent's capability score based on task performance"""
        if agent_id not in self.agent_profiles:
            return

        profile = self.agent_profiles[agent_id]

        if capability_id not in profile.capability_scores:
            profile.capability_scores[capability_id] = 0.5

        # Exponential moving average update
        alpha = 0.1
        current_score = profile.capability_scores[capability_id]
        new_score = alpha * performance_score + (1 - alpha) * current_score
        profile.capability_scores[capability_id] = max(0.0, min(1.0, new_score))

        # Track performance history
        if capability_id not in self.capability_performance_history:
            self.capability_performance_history[capability_id] = []

        self.capability_performance_history[capability_id].append(
            (datetime.now(UTC), agent_id, task_success, performance_score)
        )

        # Keep only recent history (last 100 entries)
        if len(self.capability_performance_history[capability_id]) > 100:
            self.capability_performance_history[capability_id] = (
                self.capability_performance_history[capability_id][-100:]
            )

        # Check if capability is improving
        if len(self.capability_performance_history[capability_id]) >= 5:
            recent_scores = [
                score
                for _, aid, _, score in self.capability_performance_history[
                    capability_id
                ][-5:]
                if aid == agent_id
            ]
            if len(recent_scores) >= 3:
                trend = (recent_scores[-1] - recent_scores[0]) / len(recent_scores)
                if trend > 0.05:  # Improving trend
                    profile.improving_capabilities.add(capability_id)
                else:
                    profile.improving_capabilities.discard(capability_id)

        profile.last_updated = datetime.now(UTC)
        logger.debug(
            f"Updated capability score for {agent_id}/{capability_id}: {new_score:.3f}"
        )

    def define_task_requirements(
        self,
        task_id: str,
        task_type: str,
        required_capabilities: List[str],
        preferred_capabilities: Optional[List[str]] = None,
        minimum_proficiency: Optional[Dict[str, ProficiencyLevel]] = None,
        capability_weights: Optional[Dict[str, float]] = None,
        complexity_score: float = 1.0,
        description: str = "",
    ):
        """Define capability requirements for a task"""
        requirements = TaskRequirement(
            task_id=task_id,
            task_type=task_type,
            required_capabilities=required_capabilities,
            preferred_capabilities=preferred_capabilities or [],
            minimum_proficiency=minimum_proficiency or {},
            capability_weights=capability_weights or {},
            complexity_score=complexity_score,
            description=description,
        )

        self.task_requirements[task_id] = requirements
        logger.info(
            f"Defined requirements for task {task_id}: {len(required_capabilities)} required capabilities"
        )

    def auto_detect_task_requirements(
        self, task_description: str, task_type: str
    ) -> List[str]:
        """Automatically detect required capabilities from task description"""
        detected_capabilities = []
        description_lower = task_description.lower()

        for capability_id, capability in self.capabilities.items():
            # Check if any keywords match
            for keyword in capability.keywords:
                if keyword.lower() in description_lower:
                    detected_capabilities.append(capability_id)
                    break

            # Check if capability is required for this task type
            if task_type in capability.required_for_tasks:
                detected_capabilities.append(capability_id)

        # Remove duplicates while preserving order
        unique_capabilities = []
        seen = set()
        for cap in detected_capabilities:
            if cap not in seen:
                unique_capabilities.append(cap)
                seen.add(cap)

        logger.info(
            f"Auto-detected {len(unique_capabilities)} capabilities for task: {unique_capabilities}"
        )
        return unique_capabilities

    def match_agents_to_task(self, task_id: str) -> List[CapabilityMatch]:
        """Find agents that match task requirements"""
        if task_id not in self.task_requirements:
            logger.error(f"Task requirements not found for {task_id}")
            return []

        requirements = self.task_requirements[task_id]
        matches = []

        for agent_id, profile in self.agent_profiles.items():
            match = self._calculate_capability_match(agent_id, profile, requirements)
            matches.append(match)

        # Sort by overall score descending
        matches.sort(key=lambda x: x.overall_score, reverse=True)

        logger.info(f"Found {len(matches)} agent matches for task {task_id}")
        return matches

    def _calculate_capability_match(
        self,
        agent_id: str,
        profile: AgentCapabilityProfile,
        requirements: TaskRequirement,
    ) -> CapabilityMatch:
        """Calculate how well an agent matches task requirements"""
        required_caps = requirements.required_capabilities
        preferred_caps = requirements.preferred_capabilities
        all_caps = required_caps + preferred_caps

        capability_scores = {}
        missing_capabilities = []
        weak_capabilities = []
        strengths = []

        # Score required capabilities
        required_score = 0.0
        required_met = 0

        for capability_id in required_caps:
            if capability_id in profile.capabilities:
                # Agent has this capability
                proficiency = profile.capabilities[capability_id]
                performance_score = profile.capability_scores.get(capability_id, 0.5)

                # Check minimum proficiency requirement
                min_proficiency = requirements.minimum_proficiency.get(capability_id)
                proficiency_met = True

                if min_proficiency:
                    proficiency_levels = [
                        ProficiencyLevel.BEGINNER,
                        ProficiencyLevel.INTERMEDIATE,
                        ProficiencyLevel.ADVANCED,
                        ProficiencyLevel.EXPERT,
                    ]
                    agent_level_idx = proficiency_levels.index(proficiency)
                    min_level_idx = proficiency_levels.index(min_proficiency)
                    proficiency_met = agent_level_idx >= min_level_idx

                if proficiency_met:
                    # Calculate capability score based on proficiency and performance
                    proficiency_score = (
                        proficiency_levels.index(proficiency) + 1
                    ) / len(proficiency_levels)
                    final_score = (proficiency_score * 0.6) + (performance_score * 0.4)

                    capability_scores[capability_id] = final_score
                    required_score += final_score
                    required_met += 1

                    if final_score > 0.8:
                        strengths.append(capability_id)
                else:
                    weak_capabilities.append(capability_id)
                    capability_scores[capability_id] = (
                        0.2  # Low score for insufficient proficiency
                    )
            else:
                missing_capabilities.append(capability_id)
                capability_scores[capability_id] = 0.0

        # Score preferred capabilities
        preferred_score = 0.0
        preferred_met = 0

        for capability_id in preferred_caps:
            if capability_id in profile.capabilities:
                proficiency = profile.capabilities[capability_id]
                performance_score = profile.capability_scores.get(capability_id, 0.5)

                proficiency_levels = [
                    ProficiencyLevel.BEGINNER,
                    ProficiencyLevel.INTERMEDIATE,
                    ProficiencyLevel.ADVANCED,
                    ProficiencyLevel.EXPERT,
                ]
                proficiency_score = (proficiency_levels.index(proficiency) + 1) / len(
                    proficiency_levels
                )
                final_score = (proficiency_score * 0.6) + (performance_score * 0.4)

                capability_scores[capability_id] = final_score
                preferred_score += final_score
                preferred_met += 1

                if final_score > 0.8:
                    strengths.append(capability_id)

        # Calculate overall score
        required_weight = 0.8
        preferred_weight = 0.2

        required_percentage = (
            required_met / len(required_caps) if required_caps else 1.0
        )
        preferred_percentage = (
            preferred_met / len(preferred_caps) if preferred_caps else 1.0
        )

        normalized_required_score = (
            (required_score / len(required_caps)) if required_caps else 0.0
        )
        normalized_preferred_score = (
            (preferred_score / len(preferred_caps)) if preferred_caps else 0.0
        )

        overall_score = (required_weight * normalized_required_score) + (
            preferred_weight * normalized_preferred_score
        )

        # Apply complexity penalty if agent is not experienced enough
        if requirements.complexity_score > 2.0 and profile.total_tasks_completed < 10:
            overall_score *= (
                0.8  # Reduce score for inexperienced agents on complex tasks
            )

        # Calculate confidence based on amount of data
        confidence = min(0.9, 0.3 + (profile.total_tasks_completed * 0.02))

        # Generate recommendation
        recommendation = self._generate_match_recommendation(
            required_percentage,
            preferred_percentage,
            overall_score,
            missing_capabilities,
            weak_capabilities,
            strengths,
        )

        return CapabilityMatch(
            agent_id=agent_id,
            task_id=requirements.task_id,
            overall_score=overall_score,
            required_capabilities_met=required_percentage,
            preferred_capabilities_met=preferred_percentage,
            capability_scores=capability_scores,
            missing_capabilities=missing_capabilities,
            weak_capabilities=weak_capabilities,
            strengths=strengths,
            confidence=confidence,
            recommendation=recommendation,
        )

    def _generate_match_recommendation(
        self,
        required_met: float,
        preferred_met: float,
        overall_score: float,
        missing: List[str],
        weak: List[str],
        strengths: List[str],
    ) -> str:
        """Generate recommendation based on match analysis"""
        if overall_score >= 0.9:
            return "Excellent match - highly recommended"
        elif overall_score >= 0.75:
            return "Good match - recommended"
        elif overall_score >= 0.6:
            if missing:
                return f"Fair match - missing capabilities: {', '.join(missing[:3])}"
            elif weak:
                return f"Fair match - needs improvement in: {', '.join(weak[:3])}"
            else:
                return "Fair match - consider for less critical tasks"
        elif overall_score >= 0.4:
            return "Poor match - significant gaps in required capabilities"
        else:
            return "Not recommended - lacks essential capabilities"

    def get_agent_capability_profile(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed capability profile for an agent"""
        if agent_id not in self.agent_profiles:
            return None

        profile = self.agent_profiles[agent_id]

        # Calculate capability statistics
        capability_details = {}
        for capability_id, proficiency in profile.capabilities.items():
            capability = self.capabilities.get(capability_id)
            performance_score = profile.capability_scores.get(capability_id, 0.5)

            capability_details[capability_id] = {
                "name": capability.name if capability else capability_id,
                "proficiency": proficiency.value,
                "performance_score": performance_score,
                "improving": capability_id in profile.improving_capabilities,
                "recently_learned": capability_id in profile.learned_capabilities,
                "capability_type": (
                    capability.capability_type.value if capability else "unknown"
                ),
            }

        # Find specialization areas
        capability_types = {}
        for capability_id, proficiency in profile.capabilities.items():
            capability = self.capabilities.get(capability_id)
            if capability:
                cap_type = capability.capability_type.value
                if cap_type not in capability_types:
                    capability_types[cap_type] = []
                capability_types[cap_type].append(
                    {
                        "capability_id": capability_id,
                        "proficiency": proficiency.value,
                        "score": profile.capability_scores.get(capability_id, 0.5),
                    }
                )

        # Calculate strengths and areas for improvement
        strengths = []
        improvements_needed = []

        for capability_id, score in profile.capability_scores.items():
            capability = self.capabilities.get(capability_id)
            if capability:
                if score > 0.8:
                    strengths.append(capability.name)
                elif score < 0.4:
                    improvements_needed.append(capability.name)

        return {
            "agent_id": agent_id,
            "total_capabilities": len(profile.capabilities),
            "specialization_focus": (
                profile.specialization_focus.value
                if profile.specialization_focus
                else None
            ),
            "total_tasks_completed": profile.total_tasks_completed,
            "capabilities": capability_details,
            "capability_types": capability_types,
            "strengths": strengths[:5],  # Top 5
            "improvements_needed": improvements_needed[:5],  # Top 5
            "currently_improving": list(profile.improving_capabilities),
            "recently_learned": list(profile.learned_capabilities),
            "last_updated": profile.last_updated.isoformat(),
        }

    def get_capability_analytics(self) -> Dict[str, Any]:
        """Get analytics on capability usage and performance"""
        capability_stats = {}

        for capability_id, capability in self.capabilities.items():
            agents_with_capability = []
            total_performance = 0.0
            performance_count = 0

            for agent_id, profile in self.agent_profiles.items():
                if capability_id in profile.capabilities:
                    agents_with_capability.append(
                        {
                            "agent_id": agent_id,
                            "proficiency": profile.capabilities[capability_id].value,
                            "performance_score": profile.capability_scores.get(
                                capability_id, 0.5
                            ),
                        }
                    )
                    total_performance += profile.capability_scores.get(
                        capability_id, 0.5
                    )
                    performance_count += 1

            # Calculate performance history stats
            history = self.capability_performance_history.get(capability_id, [])
            success_rate = 0.0
            if history:
                successes = sum(1 for _, _, success, _ in history if success)
                success_rate = successes / len(history)

            capability_stats[capability_id] = {
                "name": capability.name,
                "type": capability.capability_type.value,
                "agents_count": len(agents_with_capability),
                "average_performance": total_performance / max(performance_count, 1),
                "success_rate": success_rate,
                "total_usage": len(history),
                "agents_with_capability": agents_with_capability,
                "required_for_tasks": capability.required_for_tasks,
            }

        # Find most/least common capabilities
        capability_counts = [
            (cid, stats["agents_count"]) for cid, stats in capability_stats.items()
        ]
        capability_counts.sort(key=lambda x: x[1], reverse=True)

        most_common = capability_counts[:5] if capability_counts else []
        least_common = capability_counts[-5:] if len(capability_counts) >= 5 else []

        return {
            "total_capabilities": len(capability_stats),
            "total_agents": len(self.agent_profiles),
            "capability_details": capability_stats,
            "most_common_capabilities": [
                {"capability_id": cid, "agent_count": count}
                for cid, count in most_common
            ],
            "least_common_capabilities": [
                {"capability_id": cid, "agent_count": count}
                for cid, count in least_common
            ],
            "capability_gaps": self._identify_capability_gaps(),
        }

    def _identify_capability_gaps(self) -> List[Dict[str, Any]]:
        """Identify capability gaps in the agent system"""
        gaps = []

        for capability_id, capability in self.capabilities.items():
            agents_with_cap = sum(
                1
                for profile in self.agent_profiles.values()
                if capability_id in profile.capabilities
            )

            if agents_with_cap == 0:
                gaps.append(
                    {
                        "capability_id": capability_id,
                        "capability_name": capability.name,
                        "severity": "critical",
                        "description": f"No agents have {capability.name} capability",
                        "required_for_tasks": capability.required_for_tasks,
                    }
                )
            elif agents_with_cap == 1:
                gaps.append(
                    {
                        "capability_id": capability_id,
                        "capability_name": capability.name,
                        "severity": "high",
                        "description": f"Only one agent has {capability.name} capability",
                        "required_for_tasks": capability.required_for_tasks,
                    }
                )
            elif agents_with_cap <= 2 and len(capability.required_for_tasks) > 0:
                gaps.append(
                    {
                        "capability_id": capability_id,
                        "capability_name": capability.name,
                        "severity": "medium",
                        "description": f"Limited agents ({agents_with_cap}) have {capability.name} capability",
                        "required_for_tasks": capability.required_for_tasks,
                    }
                )

        return gaps

    async def suggest_capability_improvements(
        self, agent_id: str
    ) -> List[Dict[str, Any]]:
        """Suggest capability improvements for an agent"""
        if agent_id not in self.agent_profiles:
            return []

        profile = self.agent_profiles[agent_id]
        suggestions = []

        # Find capabilities with low performance scores
        for capability_id, score in profile.capability_scores.items():
            if score < 0.6:
                capability = self.capabilities.get(capability_id)
                if capability:
                    suggestions.append(
                        {
                            "type": "improve_existing",
                            "capability_id": capability_id,
                            "capability_name": capability.name,
                            "current_score": score,
                            "priority": "high" if score < 0.4 else "medium",
                            "recommendation": f"Focus on improving {capability.name} through practice and training",
                        }
                    )

        # Find missing capabilities that are commonly required
        all_required_capabilities = set()
        for req in self.task_requirements.values():
            all_required_capabilities.update(req.required_capabilities)

        for capability_id in all_required_capabilities:
            if capability_id not in profile.capabilities:
                capability = self.capabilities.get(capability_id)
                if capability:
                    # Check how often this capability is needed
                    usage_count = sum(
                        1
                        for req in self.task_requirements.values()
                        if capability_id in req.required_capabilities
                    )

                    if usage_count >= 2:  # Required by multiple tasks
                        suggestions.append(
                            {
                                "type": "learn_new",
                                "capability_id": capability_id,
                                "capability_name": capability.name,
                                "usage_frequency": usage_count,
                                "priority": "high" if usage_count >= 5 else "medium",
                                "recommendation": f"Consider learning {capability.name} - required by {usage_count} task types",
                            }
                        )

        # Sort suggestions by priority
        priority_order = {"high": 3, "medium": 2, "low": 1}
        suggestions.sort(
            key=lambda x: priority_order.get(x["priority"], 0), reverse=True
        )

        return suggestions[:10]  # Return top 10 suggestions
