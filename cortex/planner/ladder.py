from __future__ import annotations

import logging
import uuid
from datetime import datetime

from cortex.planner.interfaces import KG, Bandit, Orchestrator, TodoStore
from cortex.todo.models import Evidence, ExitCriteria, LadderStage, Todo, TodoStatus

logger = logging.getLogger(__name__)


class LadderPlanner:
    """Implements LADDER: Localize → Assess → Decompose → Decide → Execute → Review."""

    def __init__(
        self, kg: KG, bandit: Bandit, store: TodoStore, orchestrator: Orchestrator
    ):
        self.kg = kg
        self.bandit = bandit
        self.store = store
        self.orch = orchestrator

    async def plan_from_user_event(self, user_event) -> Todo:
        """Entry point: create a root TODO and run LADDER stages."""
        root = self._localize(user_event)
        await self._emit("todo.created", root.id, {"title": root.title})
        self.store.upsert(root)
        await self._ladder(root)
        return root

    async def _ladder(self, root: Todo) -> None:
        self._advance_stage(root, LadderStage.ASSESS)
        self._assess(root)

        self._advance_stage(root, LadderStage.DECOMPOSE)
        children = self._decompose(root)
        # Get the updated root with children_ids after decomposition
        root = self.store.get(root.id)

        self._advance_stage(root, LadderStage.DECIDE)
        children = self._decide(children)

        self._advance_stage(root, LadderStage.EXECUTE)
        await self._execute(root, children)

        self._advance_stage(root, LadderStage.REVIEW)
        await self._review(root, children)

    # ===== L =====
    def _localize(self, user_event) -> Todo:
        """L: Localize the user request into a concrete Todo."""
        title = user_event.payload.get("query", "user task")
        desc = user_event.payload.get("context")
        t = Todo(
            title=title,
            description=(desc or ""),
            stage=LadderStage.LOCALIZE,
            exit_criteria=[ExitCriteria(description="Measurable outcome defined")],
        )
        return t

    # ===== A =====
    def _assess(self, t: Todo) -> Todo:
        """A: Assess energy, confidence, dependencies."""
        ctx = self.kg.get_context_for_title(t.title)

        # Update t with context and assessment if context exists
        updated_data = {}
        if ctx and len(ctx) > 10:
            updated_data["description"] = (
                t.description or ""
            ) + f"\n\n[context]: {ctx[:1000]}"

        # Create updated todo with assessment
        updated_t = t.model_copy(update=updated_data)

        self._emit_sync(
            "plan.assessed",
            updated_t.id,
            {"energy": updated_t.energy, "confidence": updated_t.confidence},
        )
        return updated_t

    # ===== D1 =====
    def _decompose(self, root: Todo) -> list[Todo]:
        """Decompose a Todo into smaller subtasks."""
        logger.info(f"🧩 Decomposing todo: {root.title}")
        children = []
        child_ids = []

        # Basic decomposition: split into 3 example subtasks
        for i in range(3):
            st = Todo(
                id=str(uuid.uuid4()),
                title=f"Subtask {i + 1} of {root.title}",
                description=f"Part {i + 1} of the main task",
                status=TodoStatus.PENDING,
                parent_id=root.id,
                created_at=datetime.now(),
                priority=root.priority,
            )

            children.append(st)
            child_ids.append(st.id)
            self.store.upsert(st)

        # Update root with child_ids using immutable model pattern
        updated_root = root.model_copy(
            update={
                "children_ids": root.children_ids + child_ids,
                "updated_at": datetime.now(),
            }
        )
        self.store.upsert(updated_root)

        logger.info(f"📝 Created {len(children)} children for todo {root.id}")
        logger.info(f"🔗 Updated root children_ids: {updated_root.children_ids}")

        # Emit the decomposed event
        self._emit_sync(
            "plan.decomposed", updated_root.id, {"children_count": len(children)}
        )

        return children

    # ===== D2 =====
    def _decide(self, children: list[Todo]) -> list[Todo]:
        """D2: Decision about tools, priorities."""
        updated_children = []

        for st in children:
            unmet = len(
                [
                    d
                    for d in st.depends_on
                    if self.store.get(d).status != TodoStatus.DONE
                ]
            )
            base_energy = st.energy or 0.7
            tool_hint = self.bandit.select_tool(
                context={"title": st.title, "energy": base_energy}
            )

            # Update the child with tool hint (immutable pattern)
            updated_st = st.model_copy(update={"tool_hint": tool_hint})
            updated_children.append(updated_st)
            self.store.upsert(updated_st)  # Persist the update

            self._emit_sync(
                "plan.decided",
                updated_st.id,
                {
                    "priority": updated_st.priority,
                    "tool_hint": updated_st.tool_hint,
                    "unmet_deps": unmet,
                },
            )

        return updated_children

    def _priority_score(self, energy: float, unmet_deps: int) -> float:
        return energy * 0.8 + (1.0 / (1 + unmet_deps)) * 0.2

    # ===== E =====
    async def _execute(self, root: Todo, children: list[Todo]) -> None:
        """E: Execute leaf tasks; skip blocked tasks."""
        for st in children:
            # Skip blocked tasks
            if any(self.store.get(d).status != TodoStatus.DONE for d in st.depends_on):
                continue

            # Execute the task
            tool = st.tool_hint or self.bandit.select_tool(context={"title": st.title})
            ctx = f"root={root.title}, current={st.title}"
            result = await self.orch.execute_action(tool, st, ctx, shadow=True)

            # Create updated todo with execution results and DONE status
            updated_st = st.model_copy(
                update={
                    "status": TodoStatus.DONE,  # Mark as DONE
                    "evidence": st.evidence
                    + [
                        Evidence(
                            kind="log",
                            summary=f"[SHADOW] tool={tool} result={str(result)[:200]}",
                        )
                    ],
                    "updated_at": datetime.now(),
                }
            )

            # Persist the status update
            self.store.upsert(updated_st)

            self._emit_sync(
                "plan.executed", updated_st.id, {"tool": tool, "shadow": True}
            )

    # ===== R =====
    async def _review(self, root: Todo, children: list[Todo]) -> None:
        """R: Review progress and assign rewards."""
        total_reward = 0.0

        for st in children:
            # Get latest version from store to check current status
            current_st = self.store.get(st.id)
            proxy = 1.0 if current_st.status == TodoStatus.DONE else 0.0
            delta = self.kg.estimate_metric_delta(st.title)
            reward = proxy * delta
            total_reward += reward

            # Update tool performance
            self.bandit.update(st.tool_hint or "unknown", reward)
            self.kg.write_decision(st.tool_hint or "unknown", st.id, reward)
            self._emit_sync(
                "plan.reviewed", st.id, {"reward": reward, "metric_delta": delta}
            )

        # Check if all children are done and mark root as done
        all_children_done = all(
            self.store.get(child.id).status == TodoStatus.DONE for child in children
        )

        if (
            all_children_done and children
        ):  # Only if there are children and all are done
            updated_root = root.model_copy(
                update={"status": TodoStatus.DONE, "updated_at": datetime.now()}
            )
            self.store.upsert(updated_root)

        await self._emit("plan.completed", root.id, {"aggregate_reward": total_reward})

    # ----- helpers -----
    def _advance_stage(self, t: Todo, stage: LadderStage) -> None:
        t.stage = stage
        self.store.upsert(t)
        self._emit_sync("todo.stage_advanced", t.id, {"stage": stage.value})

    async def _emit(self, kind: str, todo_id: str | None, payload: dict) -> None:
        await self.orch.event_bus.emit(kind, todo_id=todo_id, payload=payload)

    def _emit_sync(self, kind: str, todo_id: str | None, payload: dict) -> None:
        self.orch.event_bus.emit_sync(kind, todo_id=todo_id, payload=payload)
