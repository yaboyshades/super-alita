from __future__ import annotations

import asyncio
import logging
import random
import time
import uuid
from datetime import datetime
from typing import Any

from cortex.planner.interfaces import KG, Bandit, Orchestrator, TodoStore
from cortex.todo.models import Evidence, ExitCriteria, LadderStage, Todo, TodoStatus

logger = logging.getLogger(__name__)


class EnhancedLadderPlanner:
    """Enhanced LADDER planner with multi-armed bandit learning and advanced task decomposition.

    Key improvements:
    - Advanced task decomposition strategies
    - Multi-armed bandit tool selection with ε-greedy algorithm
    - Energy-based task prioritization
    - Shadow/Active execution modes
    - Knowledge base integration for learning
    """

    def __init__(
        self,
        kg: KG,
        bandit: Bandit,
        store: TodoStore,
        orchestrator: Orchestrator,
        mode: str = "shadow",
    ):
        self.kg = kg
        self.bandit = bandit
        self.store = store
        self.orch = orchestrator
        self.mode = mode  # "shadow" or "active"
        self.bandit_stats: dict[str, dict[str, Any]] = {}
        self.knowledge_base: dict[str, Any] = {}
        self.exploration_rate = 0.1  # ε for ε-greedy bandit

    async def plan_from_user_event(self, user_event: Any) -> Todo:
        """Entry point: create a root TODO and run enhanced LADDER stages."""
        root = self._localize(user_event)
        await self._emit(
            "todo.created", root.id, {"title": root.title, "mode": self.mode}
        )
        self.store.upsert(root)
        await self._enhanced_ladder(root)
        return root

    async def _enhanced_ladder(self, root: Todo) -> None:
        """Enhanced LADDER implementation with energy-based prioritization."""
        logger.info(
            f"🎯 Starting enhanced LADDER for: {root.title} (mode: {self.mode})"
        )

        # L -> A
        self._advance_stage(root, LadderStage.ASSESS)
        root = self._enhanced_assess(root)

        # A -> D1
        self._advance_stage(root, LadderStage.DECOMPOSE)
        children = await self._enhanced_decompose(root)
        root = self.store.get(root.id)  # Refresh after decomposition

        # D1 -> D2
        self._advance_stage(root, LadderStage.DECIDE)
        children = self._enhanced_decide(children)

        # D2 -> E
        self._advance_stage(root, LadderStage.EXECUTE)
        await self._enhanced_execute(root, children)

        # E -> R
        self._advance_stage(root, LadderStage.REVIEW)
        await self._enhanced_review(root, children)

    # ===== L =====
    def _localize(self, user_event: Any) -> Todo:
        """L: Enhanced localization with energy estimation."""
        title = user_event.payload.get("query", "user task")
        desc = user_event.payload.get("context", "")

        # Estimate initial energy based on task complexity
        estimated_energy = self._estimate_task_energy(title, desc)

        t = Todo(
            title=title,
            description=desc,
            stage=LadderStage.LOCALIZE,
            energy=estimated_energy,
            exit_criteria=[ExitCriteria(description="Measurable outcome defined")],
            evidence=[
                Evidence(
                    kind="note",
                    summary=f"Initial energy estimate: {estimated_energy}",
                    score=0.8,
                )
            ],
        )
        return t

    def _estimate_task_energy(self, title: str, desc: str) -> float:
        """Estimate task energy based on complexity indicators."""
        text = f"{title} {desc}".lower()

        # Base energy
        energy = 1.0

        # Complexity indicators
        if any(word in text for word in ["test", "format", "lint"]):
            energy *= 0.5  # Simple tasks
        elif any(word in text for word in ["build", "deploy", "configure"]):
            energy *= 1.5  # Medium tasks
        elif any(word in text for word in ["implement", "create", "develop"]):
            energy *= 2.0  # Complex tasks
        elif any(word in text for word in ["refactor", "migrate", "optimize"]):
            energy *= 3.0  # Very complex tasks

        # Scale by description length (more detail = more complexity)
        if len(desc) > 100:
            energy *= 1.2

        return min(energy, 5.0)  # Cap at 5.0

    # ===== A =====
    def _enhanced_assess(self, t: Todo) -> Todo:
        """A: Enhanced assessment with knowledge base integration."""
        logger.info(f"🔍 Enhanced assessment for: {t.title}")

        # Get context from knowledge graph
        ctx = self.kg.get_context_for_title(t.title)

        # Check knowledge base for similar tasks
        similar_tasks = self._find_similar_tasks(t.title)

        # Calculate confidence based on historical performance
        confidence = self._calculate_confidence(t.title, similar_tasks)

        # Enhanced energy calculation
        kg_energy = self.kg.compute_energy_for_title(t.title)
        final_energy = (t.energy + kg_energy) / 2.0 if kg_energy > 0 else t.energy

        updated_data = {
            "energy": final_energy,
            "confidence": confidence,
        }

        if ctx and len(ctx) > 10:
            updated_data["description"] = (
                t.description or ""
            ) + f"\n\n[KG Context]: {ctx[:500]}"

        if similar_tasks:
            similar_summary = ", ".join([task["title"] for task in similar_tasks[:3]])
            updated_data["evidence"] = t.evidence + [
                Evidence(
                    kind="note",
                    summary=f"Similar tasks found: {similar_summary}",
                    score=confidence,
                )
            ]

        updated_t = t.model_copy(update=updated_data)
        self.store.upsert(updated_t)

        self._emit_sync(
            "plan.assessed",
            updated_t.id,
            {
                "energy": updated_t.energy,
                "confidence": updated_t.confidence,
                "similar_tasks": len(similar_tasks),
            },
        )

        return updated_t

    def _find_similar_tasks(self, title: str) -> list[dict[str, Any]]:
        """Find similar tasks in knowledge base."""
        similar = []
        title_words = set(title.lower().split())

        for task_id, task_data in self.knowledge_base.items():
            if "title" in task_data:
                task_words = set(task_data["title"].lower().split())
                similarity = len(title_words & task_words) / len(
                    title_words | task_words
                )
                if similarity > 0.3:  # 30% similarity threshold
                    similar.append(
                        {
                            "id": task_id,
                            "title": task_data["title"],
                            "similarity": similarity,
                            "result": task_data.get("result", "unknown"),
                        }
                    )

        return sorted(similar, key=lambda x: x["similarity"], reverse=True)

    def _calculate_confidence(
        self, title: str, similar_tasks: list[dict[str, Any]]
    ) -> float:
        """Calculate confidence based on historical performance."""
        if not similar_tasks:
            return 0.5  # Base confidence

        # Weight by similarity and success rate
        total_weight = 0.0
        weighted_success = 0.0

        for task in similar_tasks:
            weight = task["similarity"]
            success = 1.0 if "success" in str(task.get("result", "")).lower() else 0.0
            total_weight += weight
            weighted_success += weight * success

        return min(weighted_success / total_weight if total_weight > 0 else 0.5, 0.95)

    # ===== D1 =====
    async def _enhanced_decompose(self, root: Todo) -> list[Todo]:
        """Enhanced decomposition with task-specific strategies."""
        logger.info(f"🧩 Enhanced decomposition for: {root.title}")

        # Select decomposition strategy based on task type
        strategy = self._select_decomposition_strategy(root)
        children = await strategy(root)

        if not children:
            # Fallback to default decomposition
            children = await self._decompose_default(root)

        # Store children and update root
        child_ids = []
        for child in children:
            self.store.upsert(child)
            child_ids.append(child.id)

        updated_root = root.model_copy(
            update={
                "children_ids": root.children_ids + child_ids,
                "updated_at": datetime.now(),
            }
        )
        self.store.upsert(updated_root)

        logger.info(f"📝 Created {len(children)} children for todo {root.id}")
        self._emit_sync(
            "plan.decomposed",
            updated_root.id,
            {"children_count": len(children), "strategy": strategy.__name__},
        )

        return children

    def _select_decomposition_strategy(self, task: Todo):
        """Select appropriate decomposition strategy based on task content."""
        title_lower = task.title.lower()

        strategies = {
            "test": self._decompose_test_task,
            "format": self._decompose_format_task,
            "lint": self._decompose_lint_task,
            "build": self._decompose_build_task,
            "deploy": self._decompose_deploy_task,
            "setup": self._decompose_setup_task,
        }

        for keyword, strategy in strategies.items():
            if keyword in title_lower:
                return strategy

        return self._decompose_default

    async def _decompose_test_task(self, task: Todo) -> list[Todo]:
        """Decompose testing tasks into logical steps."""
        base_id = int(time.time())
        return [
            Todo(
                id=str(uuid.uuid4()),
                title="Prepare test environment",
                description="Set up test dependencies and configuration",
                energy=0.5,
                parent_id=task.id,
                tool_hint="pytest_setup",
                priority=self._calculate_priority(0.5, 0),
            ),
            Todo(
                id=str(uuid.uuid4()),
                title="Run unit tests with coverage",
                description="Execute test suite and generate coverage report",
                energy=1.5,
                parent_id=task.id,
                tool_hint="pytest",
                priority=self._calculate_priority(1.5, 0),
            ),
            Todo(
                id=str(uuid.uuid4()),
                title="Analyze test results",
                description="Review test outcomes and coverage metrics",
                energy=0.5,
                parent_id=task.id,
                tool_hint="coverage_report",
                priority=self._calculate_priority(0.5, 0),
            ),
        ]

    async def _decompose_format_task(self, task: Todo) -> list[Todo]:
        """Decompose code formatting tasks."""
        return [
            Todo(
                id=str(uuid.uuid4()),
                title="Check code format",
                description="Analyze current code formatting issues",
                energy=0.3,
                parent_id=task.id,
                tool_hint="black_check",
                priority=self._calculate_priority(0.3, 0),
            ),
            Todo(
                id=str(uuid.uuid4()),
                title="Apply code formatting",
                description="Format code with Black formatter",
                energy=0.5,
                parent_id=task.id,
                tool_hint="black",
                priority=self._calculate_priority(0.5, 0),
            ),
            Todo(
                id=str(uuid.uuid4()),
                title="Verify formatting",
                description="Confirm all code meets formatting standards",
                energy=0.2,
                parent_id=task.id,
                tool_hint="black_verify",
                priority=self._calculate_priority(0.2, 0),
            ),
        ]

    async def _decompose_lint_task(self, task: Todo) -> list[Todo]:
        """Decompose linting tasks."""
        return [
            Todo(
                id=str(uuid.uuid4()),
                title="Run static analysis",
                description="Perform static code analysis with Ruff",
                energy=0.8,
                parent_id=task.id,
                tool_hint="ruff",
                priority=self._calculate_priority(0.8, 0),
            ),
            Todo(
                id=str(uuid.uuid4()),
                title="Fix auto-fixable issues",
                description="Apply automatic fixes for common issues",
                energy=0.5,
                parent_id=task.id,
                tool_hint="ruff_fix",
                priority=self._calculate_priority(0.5, 0),
            ),
        ]

    async def _decompose_build_task(self, task: Todo) -> list[Todo]:
        """Decompose build tasks."""
        return [
            Todo(
                id=str(uuid.uuid4()),
                title="Install dependencies",
                description="Install required packages and dependencies",
                energy=1.0,
                parent_id=task.id,
                tool_hint="pip_install",
                priority=self._calculate_priority(1.0, 0),
            ),
            Todo(
                id=str(uuid.uuid4()),
                title="Build package",
                description="Compile and build the package",
                energy=1.5,
                parent_id=task.id,
                tool_hint="build",
                priority=self._calculate_priority(1.5, 0),
            ),
            Todo(
                id=str(uuid.uuid4()),
                title="Validate build",
                description="Test and validate the build output",
                energy=0.8,
                parent_id=task.id,
                tool_hint="build_test",
                priority=self._calculate_priority(0.8, 0),
            ),
        ]

    async def _decompose_deploy_task(self, task: Todo) -> list[Todo]:
        """Decompose deployment tasks."""
        return [
            Todo(
                id=str(uuid.uuid4()),
                title="Pre-deployment checks",
                description="Verify system state and prerequisites",
                energy=0.8,
                parent_id=task.id,
                tool_hint="deploy_check",
                priority=self._calculate_priority(0.8, 0),
            ),
            Todo(
                id=str(uuid.uuid4()),
                title="Deploy to staging",
                description="Deploy to staging environment for testing",
                energy=2.0,
                parent_id=task.id,
                tool_hint="deploy_staging",
                priority=self._calculate_priority(2.0, 0),
            ),
            Todo(
                id=str(uuid.uuid4()),
                title="Production deployment",
                description="Deploy to production environment",
                energy=2.5,
                parent_id=task.id,
                tool_hint="deploy_prod",
                priority=self._calculate_priority(2.5, 0),
            ),
        ]

    async def _decompose_setup_task(self, task: Todo) -> list[Todo]:
        """Decompose environment setup tasks."""
        return [
            Todo(
                id=str(uuid.uuid4()),
                title="Create virtual environment",
                description="Set up isolated Python environment",
                energy=0.5,
                parent_id=task.id,
                tool_hint="venv_create",
                priority=self._calculate_priority(0.5, 0),
            ),
            Todo(
                id=str(uuid.uuid4()),
                title="Install base dependencies",
                description="Install core project dependencies",
                energy=1.0,
                parent_id=task.id,
                tool_hint="pip_install_base",
                priority=self._calculate_priority(1.0, 0),
            ),
            Todo(
                id=str(uuid.uuid4()),
                title="Configure development tools",
                description="Set up linting, formatting, and testing tools",
                energy=1.2,
                parent_id=task.id,
                tool_hint="dev_tools_setup",
                priority=self._calculate_priority(1.2, 0),
            ),
        ]

    async def _decompose_default(self, task: Todo) -> list[Todo]:
        """Default decomposition for unknown task types."""
        return [
            Todo(
                id=str(uuid.uuid4()),
                title=f"Analyze: {task.title}",
                description="Analyze task requirements and dependencies",
                energy=task.energy * 0.3,
                parent_id=task.id,
                tool_hint="analyze",
                priority=self._calculate_priority(task.energy * 0.3, 0),
            ),
            Todo(
                id=str(uuid.uuid4()),
                title=f"Execute: {task.title}",
                description="Perform the main task execution",
                energy=task.energy * 0.6,
                parent_id=task.id,
                tool_hint="execute",
                priority=self._calculate_priority(task.energy * 0.6, 0),
            ),
            Todo(
                id=str(uuid.uuid4()),
                title=f"Verify: {task.title}",
                description="Verify task completion and quality",
                energy=task.energy * 0.1,
                parent_id=task.id,
                tool_hint="verify",
                priority=self._calculate_priority(task.energy * 0.1, 0),
            ),
        ]

    def _calculate_priority(self, energy: float, unmet_deps: int) -> float:
        """Calculate task priority based on energy and dependencies."""
        # Lower energy tasks get higher priority (easier to complete)
        # Tasks with fewer unmet dependencies get higher priority
        energy_factor = 1.0 / (energy + 0.1)  # Avoid division by zero
        dependency_factor = 1.0 / (unmet_deps + 1.0)
        return energy_factor * 0.7 + dependency_factor * 0.3

    # ===== D2 =====
    def _enhanced_decide(self, children: list[Todo]) -> list[Todo]:
        """Enhanced decision making with multi-armed bandit tool selection."""
        logger.info(f"🎯 Enhanced decision making for {len(children)} children")

        updated_children = []

        for child in children:
            # Calculate dependencies
            unmet_deps = len(
                [
                    d
                    for d in child.depends_on
                    if self.store.get(d).status != TodoStatus.DONE
                ]
            )

            # Enhanced tool selection using bandit
            tool_hint = self._select_tool_bandit(child)

            # Update priority based on current state
            priority = self._calculate_priority(child.energy, unmet_deps)

            # Create updated child
            updated_child = child.model_copy(
                update={
                    "tool_hint": tool_hint,
                    "priority": priority,
                    "updated_at": datetime.now(),
                }
            )

            updated_children.append(updated_child)
            self.store.upsert(updated_child)

            self._emit_sync(
                "plan.decided",
                updated_child.id,
                {
                    "priority": updated_child.priority,
                    "tool_hint": updated_child.tool_hint,
                    "unmet_deps": unmet_deps,
                    "energy": updated_child.energy,
                },
            )

        return updated_children

    def _select_tool_bandit(self, task: Todo) -> str:
        """Select tool using ε-greedy multi-armed bandit algorithm."""
        # Get available tools for this task type
        available_tools = self._get_available_tools(task)

        if not available_tools:
            return "default_tool"

        # Initialize bandit stats for new tools
        for tool in available_tools:
            if tool not in self.bandit_stats:
                self.bandit_stats[tool] = {
                    "wins": 0,
                    "attempts": 0,
                    "rewards": [],
                }

        # ε-greedy selection
        if len(available_tools) > 1 and random.random() < self.exploration_rate:
            # Exploration: choose random tool
            selected = random.choice(available_tools)
            logger.info(f"🎲 Exploration: selected {selected} for {task.title}")
        else:
            # Exploitation: choose best tool based on success rate
            best_tool = None
            best_success_rate = -1

            for tool in available_tools:
                stats = self.bandit_stats[tool]
                if stats["attempts"] > 0:
                    success_rate = stats["wins"] / stats["attempts"]
                    if success_rate > best_success_rate:
                        best_success_rate = success_rate
                        best_tool = tool

            selected = best_tool or available_tools[0]
            logger.info(
                f"🎯 Exploitation: selected {selected} (rate: {best_success_rate:.2f}) for {task.title}"
            )

        return selected

    def _get_available_tools(self, task: Todo) -> list[str]:
        """Get available tools based on task characteristics."""
        tools = []

        # Add tool_hint if available
        if task.tool_hint:
            tools.append(task.tool_hint)

        # Add tools based on task type
        title_lower = task.title.lower()

        if "test" in title_lower:
            tools.extend(["pytest", "unittest", "coverage"])
        elif "format" in title_lower:
            tools.extend(["black", "autopep8"])
        elif "lint" in title_lower:
            tools.extend(["ruff", "flake8", "pylint"])
        elif "build" in title_lower:
            tools.extend(["pip", "setuptools", "wheel"])
        else:
            tools.extend(["default_tool", "manual_execution"])

        return list(set(tools))  # Remove duplicates

    # ===== E =====
    async def _enhanced_execute(self, root: Todo, children: list[Todo]) -> None:
        """Enhanced execution with energy-based prioritization and learning."""
        logger.info(
            f"🚀 Enhanced execution for {len(children)} children (mode: {self.mode})"
        )

        # Sort children by priority (higher priority first)
        sorted_children = sorted(children, key=lambda x: x.priority, reverse=True)

        execution_results = []

        for child in sorted_children:
            # Skip blocked tasks
            unmet_deps = [
                d
                for d in child.depends_on
                if self.store.get(d).status != TodoStatus.DONE
            ]
            if unmet_deps:
                logger.info(
                    f"⏸️  Skipping blocked task: {child.title} (deps: {unmet_deps})"
                )
                continue

            # Execute the task
            result = await self._execute_single_task(child, root)
            execution_results.append(result)

            # Update task status based on result
            status = (
                TodoStatus.DONE if result.get("success", False) else TodoStatus.PENDING
            )

            updated_child = child.model_copy(
                update={
                    "status": status,
                    "evidence": child.evidence
                    + [
                        Evidence(
                            kind="log",
                            summary=f"[{self.mode.upper()}] tool={result.get('tool_used')} "
                            f"result={str(result.get('output', ''))[:100]}",
                            score=1.0 if result.get("success", False) else 0.0,
                        )
                    ],
                    "updated_at": datetime.now(),
                }
            )

            self.store.upsert(updated_child)

            # Update bandit statistics
            self._update_bandit_stats(child.tool_hint or "default", result)

            self._emit_sync(
                "plan.executed",
                updated_child.id,
                {
                    "tool": result.get("tool_used"),
                    "success": result.get("success", False),
                    "mode": self.mode,
                    "energy": child.energy,
                },
            )

        logger.info(f"✅ Completed execution of {len(execution_results)} tasks")

    async def _execute_single_task(self, task: Todo, root: Todo) -> dict[str, Any]:
        """Execute a single task with the selected tool."""
        tool = task.tool_hint or "default_tool"
        context = f"root={root.title}, current={task.title}, energy={task.energy}"

        logger.info(f"🔧 Executing {task.title} with {tool} (mode: {self.mode})")

        if self.mode == "shadow":
            # Shadow mode: simulate execution
            await asyncio.sleep(0.1)  # Simulate work
            success_prob = 0.8 + (
                task.confidence * 0.2
            )  # Higher confidence = higher success
            success = random.random() < success_prob

            return {
                "success": success,
                "output": f"Shadow execution: {task.description[:100]}",
                "tool_used": tool,
                "simulated": True,
                "execution_time": 0.1,
            }
        else:
            # Active mode: actual execution
            try:
                start_time = time.time()
                result = await self.orch.execute_action(
                    tool, task, context, shadow=False
                )
                execution_time = time.time() - start_time

                return {
                    "success": True,
                    "output": result,
                    "tool_used": tool,
                    "simulated": False,
                    "execution_time": execution_time,
                }
            except Exception as e:
                logger.error(f"❌ Execution failed for {task.title}: {e}")
                return {
                    "success": False,
                    "error": str(e),
                    "tool_used": tool,
                    "simulated": False,
                }

    def _update_bandit_stats(self, tool: str, result: dict[str, Any]) -> None:
        """Update bandit statistics based on execution result."""
        if tool not in self.bandit_stats:
            self.bandit_stats[tool] = {
                "wins": 0,
                "attempts": 0,
                "rewards": [],
            }

        self.bandit_stats[tool]["attempts"] += 1
        if result.get("success", False):
            self.bandit_stats[tool]["wins"] += 1

        logger.debug(
            f"📊 Updated bandit stats for {tool}: "
            f"{self.bandit_stats[tool]['wins']}/{self.bandit_stats[tool]['attempts']}"
        )

    def _record_bandit_reward(self, tool: str, reward: float) -> None:
        """Store reward history and update bandit with consolidation penalty."""
        if tool not in self.bandit_stats:
            self.bandit_stats[tool] = {
                "wins": 0,
                "attempts": 0,
                "rewards": [],
            }

        rewards = self.bandit_stats[tool].setdefault("rewards", [])
        rewards.append(reward)

        penalty = self._calculate_consolidation_penalty(rewards)
        adjusted = max(reward - penalty, 0.0)
        self.bandit.update(tool, adjusted)

        logger.debug(
            f"📉 Bandit reward update for {tool}: reward={reward:.2f} penalty={penalty:.2f}"
        )

    def _calculate_consolidation_penalty(self, rewards: list[float]) -> float:
        """Calculate penalty based on recent reward history to prevent dominance."""
        if not rewards:
            return 0.0
        recent = rewards[-5:]
        avg_recent = sum(recent) / len(recent)
        return 0.1 * avg_recent

    # ===== R =====
    async def _enhanced_review(self, root: Todo, children: list[Todo]) -> None:
        """Enhanced review with learning and knowledge base updates."""
        logger.info(
            f"📋 Enhanced review for {root.title} with {len(children)} children"
        )

        total_reward = 0.0
        successful_tasks = 0

        for child in children:
            # Get latest version from store
            current_child = self.store.get(child.id)

            # Calculate reward based on success and impact
            success_proxy = 1.0 if current_child.status == TodoStatus.DONE else 0.0
            metric_delta = self.kg.estimate_metric_delta(child.title)

            # Enhanced reward calculation
            base_reward = success_proxy * metric_delta
            energy_bonus = (1.0 / child.energy) * 0.1  # Bonus for efficient tasks
            confidence_factor = child.confidence * 0.2

            final_reward = base_reward + energy_bonus + confidence_factor
            total_reward += final_reward

            if current_child.status == TodoStatus.DONE:
                successful_tasks += 1

            # Update tool performance in bandit with consolidation penalty
            if child.tool_hint:
                self._record_bandit_reward(child.tool_hint, final_reward)
                self.kg.write_decision(child.tool_hint, child.id, final_reward)

            # Update knowledge base
            self._update_knowledge_base(
                child,
                {
                    "success": current_child.status == TodoStatus.DONE,
                    "reward": final_reward,
                    "tool_used": child.tool_hint,
                },
            )

            self._emit_sync(
                "plan.reviewed",
                child.id,
                {
                    "reward": final_reward,
                    "metric_delta": metric_delta,
                    "success": current_child.status == TodoStatus.DONE,
                    "energy_bonus": energy_bonus,
                },
            )

        # Check if all children are done and update root status
        completion_rate = successful_tasks / len(children) if children else 0.0

        if completion_rate >= 0.8:  # 80% completion threshold
            updated_root = root.model_copy(
                update={
                    "status": TodoStatus.DONE,
                    "updated_at": datetime.now(),
                    "evidence": root.evidence
                    + [
                        Evidence(
                            kind="metric",
                            summary=f"Completed with {completion_rate:.1%} success rate",
                            score=completion_rate,
                        )
                    ],
                }
            )
            self.store.upsert(updated_root)

        # Log learning insights
        self._log_learning_insights(root, children, total_reward, completion_rate)

        await self._emit(
            "plan.completed",
            root.id,
            {
                "aggregate_reward": total_reward,
                "completion_rate": completion_rate,
                "successful_tasks": successful_tasks,
                "total_tasks": len(children),
                "mode": self.mode,
            },
        )

    def _update_knowledge_base(self, task: Todo, result: dict[str, Any]) -> None:
        """Update knowledge base with task execution results."""
        self.knowledge_base[task.id] = {
            "title": task.title,
            "description": task.description,
            "energy": task.energy,
            "priority": task.priority,
            "tool_used": task.tool_hint,
            "result": result,
            "timestamp": time.time(),
            "success": result.get("success", False),
            "reward": result.get("reward", 0.0),
        }

        logger.debug(f"💾 Updated knowledge base for task: {task.title}")

    def _log_learning_insights(
        self,
        root: Todo,
        children: list[Todo],
        total_reward: float,
        completion_rate: float,
    ) -> None:
        """Log insights gained from this planning cycle."""
        logger.info(f"🧠 Learning Insights for {root.title}:")
        logger.info(f"   Completion Rate: {completion_rate:.1%}")
        logger.info(f"   Total Reward: {total_reward:.2f}")
        logger.info(
            f"   Avg Energy per Task: {sum(c.energy for c in children) / len(children):.2f}"
        )

        # Log top performing tools
        tool_performance = {}
        for tool, stats in self.bandit_stats.items():
            if stats["attempts"] > 0:
                success_rate = stats["wins"] / stats["attempts"]
                tool_performance[tool] = success_rate

        if tool_performance:
            best_tools = sorted(
                tool_performance.items(), key=lambda x: x[1], reverse=True
            )[:3]
            logger.info(
                f"   Top Tools: {', '.join([f'{tool} ({rate:.1%})' for tool, rate in best_tools])}"
            )

    # ----- Helper methods -----
    def _advance_stage(self, t: Todo, stage: LadderStage) -> None:
        """Advance task to the next LADDER stage."""
        updated_t = t.model_copy(update={"stage": stage, "updated_at": datetime.now()})
        self.store.upsert(updated_t)
        self._emit_sync("todo.stage_advanced", t.id, {"stage": stage.value})

    async def _emit(
        self, kind: str, todo_id: str | None, payload: dict[str, Any]
    ) -> None:
        """Emit asynchronous event."""
        await self.orch.event_bus.emit(kind, todo_id=todo_id, payload=payload)

    def _emit_sync(
        self, kind: str, todo_id: str | None, payload: dict[str, Any]
    ) -> None:
        """Emit synchronous event."""
        self.orch.event_bus.emit_sync(kind, todo_id=todo_id, payload=payload)

    # ----- Public API methods -----
    def set_mode(self, mode: str) -> None:
        """Set execution mode (shadow/active)."""
        if mode not in ["shadow", "active"]:
            raise ValueError("Mode must be 'shadow' or 'active'")
        self.mode = mode
        logger.info(f"🔄 Planner mode set to: {mode}")

    def get_bandit_stats(self) -> dict[str, dict[str, Any]]:
        """Get current bandit statistics."""
        return self.bandit_stats.copy()

    def get_knowledge_base_summary(self) -> dict[str, Any]:
        """Get summary of knowledge base."""
        if not self.knowledge_base:
            return {"size": 0, "insights": "No data available"}

        successful_tasks = sum(
            1 for task in self.knowledge_base.values() if task.get("success", False)
        )
        total_tasks = len(self.knowledge_base)
        avg_reward = (
            sum(task.get("reward", 0.0) for task in self.knowledge_base.values())
            / total_tasks
        )

        return {
            "size": total_tasks,
            "success_rate": successful_tasks / total_tasks if total_tasks > 0 else 0.0,
            "average_reward": avg_reward,
            "mode": self.mode,
        }


# Backward compatibility alias
LadderPlanner = EnhancedLadderPlanner
