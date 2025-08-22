"""
Closed-Loop Plan Executor for Super Alita 2.0
Transforms fire-and-forget planning into stateful, recoverable execution
"""

import asyncio
import json
import logging
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any

from src.core.events import AtomGapEvent
from src.core.neural_atom import NeuralAtomMetadata, NeuralStore, TextualMemoryAtom
from src.core.secure_executor import get_tool_registry

logger = logging.getLogger(__name__)


@dataclass
class Step:
    """Single step in an execution plan."""

    tool: str
    params: dict[str, Any]
    status: str = "pending"  # pending, running, success, failed
    result: Any | None = None
    error: str | None = None
    retries: int = 0
    start_time: float | None = None
    end_time: float | None = None


class PlanExecutor:
    """
    Closed-loop plan executor that tracks execution state and handles failures.
    """

    def __init__(self, event_bus, store: NeuralStore, llm_client=None):
        self.event_bus = event_bus
        self.store = store
        self.llm = llm_client

        # Active plan tracking
        self.active_plans: dict[str, list[Step]] = {}
        self.result_waiters: dict[str, asyncio.Event] = {}

        # Subscribe to tool results to complete the execution loop
        asyncio.create_task(self._setup_subscriptions())

        logger.info("PlanExecutor initialized - ready for closed-loop execution")

    async def _setup_subscriptions(self):
        """Set up event subscriptions for tool results."""
        await self.event_bus.subscribe("tool_result", self._handle_tool_result)

    async def execute_plan(
        self, plan_id: str, session_id: str, goal: str, tools_needed: list[str]
    ) -> str:
        """Execute a plan with closed-loop tracking and recovery."""
        try:
            # Generate execution plan
            plan = self._create_plan(goal, tools_needed)
            self.active_plans[plan_id] = plan

            logger.info(f"🎯 Starting plan execution: {plan_id} with {len(plan)} steps")

            # Persist plan for recovery
            await self._persist_plan(plan_id, goal, plan)

            # Execute steps sequentially
            for step_idx, step in enumerate(plan):
                await self._execute_step(plan_id, session_id, step_idx, step)

                # Stop on failure for critical errors
                if (
                    step.status == "failed"
                    and step.error == "Service offline - no retry needed"
                ):
                    logger.warning("Stopping plan execution due to service offline")
                    break

            # Generate summary
            summary = await self._summarize(plan_id, goal, plan)

            # Clean up
            await self._cleanup_plan(plan_id)

            return summary

        except Exception as e:
            logger.error(f"Plan execution failed: {e}")
            return f"❌ Plan execution failed: {e!s}"

    def _create_plan(self, goal: str, tools_needed: list[str]) -> list[Step]:
        """Create execution plan from goal and tools."""
        plan = []

        for tool in tools_needed:
            if tool == "web_agent":
                # Extract search query from goal
                query = self._extract_query(goal)
                plan.append(
                    Step(
                        tool="web_agent",
                        params={"query": query, "web_k": 5, "github_k": 3},
                    )
                )
            elif tool == "code_runner":
                plan.append(
                    Step(
                        tool="code_runner",
                        params={"code": f"# Code for: {goal}", "language": "python"},
                    )
                )
            elif tool == "memory_manager":
                # Memory manager expects action and content parameters
                # Extract the actual content to save from the goal
                content = self._extract_memory_content(goal)
                plan.append(
                    Step(
                        tool="memory_manager",
                        params={"action": "save", "content": content},
                    )
                )
            elif tool == "web_search":
                # Legacy web_search tool name - normalize to web_agent
                query = self._extract_query(goal)
                plan.append(
                    Step(
                        tool="web_agent",
                        params={"query": query, "web_k": 5, "github_k": 3},
                    )
                )
            else:
                plan.append(Step(tool=tool, params={"input": goal}))

        return plan

    def _extract_query(self, goal: str) -> str:
        """Extract clean search query from goal."""
        if goal.startswith("/"):
            parts = goal.lstrip("/").split(maxsplit=1)
            return parts[1].strip() if len(parts) > 1 else parts[0].strip()

        goal_lower = goal.lower()
        if goal_lower.startswith("search for "):
            return goal[11:].strip()
        if goal_lower.startswith("find "):
            return goal[5:].strip()

        return goal.strip()

    def _extract_memory_content(self, goal: str) -> str:
        """Extract content to save from memory-related goal."""
        goal_lower = goal.lower()

        # Handle "save that to memory" - refers to previous context
        if "save that" in goal_lower or "remember that" in goal_lower:
            # For now, return a placeholder that the memory manager can handle
            # In a more sophisticated system, this would access conversation context
            return "Previous search results or conversation content"

        # Handle explicit save commands
        if goal_lower.startswith("save ") or goal_lower.startswith("remember "):
            return goal[goal_lower.index(" ") + 1 :].strip()

        # Default: use the goal as-is
        return goal.strip()

    async def _execute_step(
        self, plan_id: str, session_id: str, step_idx: int, step: Step
    ):
        """Execute single step with retry and error handling."""
        # Idempotency check - avoid re-executing completed or running steps
        if step.status != "pending":
            logger.debug(
                "Step %s already processed (status: %s)", step_idx + 1, step.status
            )
            return

        # 1️⃣ Gap detection: Check if tool exists in registry
        registry = get_tool_registry()
        known_tools = [
            "web_agent",
            "web_search",
            "memory_manager",
            "calculator",
            "self_reflection",
        ]  # Core system tools
        all_available_tools = set(known_tools + registry.list_tools())

        if step.tool not in all_available_tools:
            logger.info(f"🔍 Gap detected: Tool '{step.tool}' not found in registry")

            # Emit AtomGapEvent to trigger CREATOR
            gap_event = AtomGapEvent(
                source_plugin="plan_executor",
                missing_tool=step.tool,
                description=f"Tool needed for: {step.params}",
                session_id=session_id,
                conversation_id=session_id,  # Use session_id as conversation_id
                gap_id=str(uuid.uuid4()),
            )

            await self.event_bus.publish(gap_event)
            logger.info(f"📢 Emitted AtomGapEvent for tool: {step.tool}")

            # 2️⃣ Return early to pause execution while gap is being filled
            step.status = "gap_detected"
            step.error = f"Tool '{step.tool}' not available - gap creation in progress"

            # Wait a moment for CREATOR to work, then retry
            await asyncio.sleep(2)  # Give CREATOR time to work

            # Check if tool is now available after CREATOR work
            updated_registry = get_tool_registry()
            if step.tool in updated_registry.list_tools():
                logger.info(f"✅ Tool '{step.tool}' now available after gap fill")
                step.status = "pending"  # Reset to retry
            else:
                step.status = "failed"
                step.error = f"Tool '{step.tool}' could not be created"
                logger.error(f"❌ Failed to create tool: {step.tool}")
                return

        max_retries = 3

        while step.retries < max_retries:
            try:
                step.status = "running"
                step.start_time = time.time()

                logger.info(
                    f"⚡ Executing step {step_idx + 1}: {step.tool} with {step.params}"
                )

                # Dispatch tool call
                await self._dispatch_tool(step, session_id, plan_id, step_idx)

                # Wait for result with timeout
                result = await self._await_result(
                    session_id, plan_id, step_idx, timeout=60
                )

                # Fail-fast policy: Don't retry network connectivity issues
                if (
                    isinstance(result, dict)
                    and result.get("error") == "Perplexica offline"
                ):
                    step.result = result
                    step.status = "failed"
                    step.end_time = time.time()
                    step.error = "Service offline - no retry needed"
                    logger.warning(
                        f"❌ Step {step_idx + 1} failed due to service offline - stopping retries"
                    )
                    break

                step.result = result
                step.status = "success"
                step.end_time = time.time()

                logger.info(f"✅ Step {step_idx + 1} completed successfully")
                break

            except TimeoutError:
                step.error = "Timeout waiting for result"
                step.status = "failed"
                logger.warning(f"Step {step_idx + 1} timed out")

            except Exception as e:
                step.error = str(e)
                step.status = "failed"
                logger.error(f"Step {step_idx + 1} failed: {e}")

            step.retries += 1
            if step.retries < max_retries:
                wait_time = 2**step.retries  # Exponential backoff
                logger.info(
                    f"Retrying step {step_idx + 1} in {wait_time}s (attempt {step.retries + 1})"
                )
                await asyncio.sleep(wait_time)
            else:
                logger.error(f"Step {step_idx + 1} failed after {max_retries} attempts")
                step.end_time = time.time()
                raise Exception(f"Step execution failed: {step.error}")

    async def _dispatch_tool(
        self, step: Step, session_id: str, plan_id: str, step_idx: int
    ):
        """Dispatch tool call with tracking metadata."""
        # Create result waiter
        result_key = f"{plan_id}_{step_idx}"
        self.result_waiters[result_key] = asyncio.Event()

        # Map tool names to event types and ensure correct parameter schemas
        if step.tool == "web_agent" or step.tool == "web_search":
            # Web agent uses web_search event with query parameter
            # Normalize both web_agent and web_search (legacy) to the same handling
            from src.core.events import WebSearchEvent

            # Extract query from either "query" or "input" parameter (defensive)
            query = step.params.get("query") or step.params.get("input") or ""

            tool_event = WebSearchEvent(
                source_plugin="planner",
                session_id=session_id,
                query=query,
                web_k=step.params.get("web_k", 5),
                github_k=step.params.get("github_k", 3),
            )
        else:
            # Other tools use generic tool_call events
            from src.core.events import ToolCallEvent

            call_id = f"plan_{int(time.time() * 1000)}_{step.tool}"
            tool_event = ToolCallEvent(
                source_plugin="planner",
                tool_name=step.tool,
                parameters=step.params,
                conversation_id=session_id,
                session_id=session_id,
                tool_call_id=call_id,
            )

        await self.event_bus.publish(tool_event)
        logger.debug(f"🚀 Dispatched {step.tool} tool call")

    async def _await_result(
        self, session_id: str, plan_id: str, step_idx: int, timeout: int = 60
    ):
        """Wait for tool result with timeout."""
        result_key = f"{plan_id}_{step_idx}"

        try:
            await asyncio.wait_for(
                self.result_waiters[result_key].wait(), timeout=timeout
            )

            # Retrieve result from store using get method
            result_atom = self.store.get(f"result_{result_key}")
            if result_atom:
                return result_atom.value
            raise Exception("Result not found in store")

        except TimeoutError:
            raise TimeoutError(f"Tool result timeout after {timeout}s") from None
        finally:
            # Clean up waiter
            if result_key in self.result_waiters:
                del self.result_waiters[result_key]

    async def _handle_tool_result(self, event):
        """Handle incoming tool results - fixed to match by conversation_id."""
        try:
            data = event.model_dump() if hasattr(event, "model_dump") else event

            conversation_id = data.get("conversation_id")
            result = data.get("result")
            success = data.get("success", True)

            logger.info(
                f"🔍 DEBUG: Received tool result - conversation_id: {conversation_id}"
            )
            logger.info(
                f"🔍 DEBUG: Current waiters: {list(self.result_waiters.keys())}"
            )

            if not conversation_id:
                logger.warning("Tool result event missing conversation_id")
                return

            # Find waiting step by conversation_id
            waiting_key = None
            for key in self.result_waiters.keys():
                logger.info(
                    f"🔍 DEBUG: Checking key {key} against conversation_id {conversation_id}"
                )
                # Key format: plan_{session_id}_{counter}_{step_idx}
                if f"plan_{conversation_id}_" in key:
                    waiting_key = key
                    break

            if not waiting_key:
                logger.debug(
                    f"No waiting step found for conversation_id: {conversation_id}"
                )
                return

            # Store result using TextualMemoryAtom (following Sacred Rules)
            result_metadata = NeuralAtomMetadata(
                name=f"result_{waiting_key}",
                description=f"Execution result for {waiting_key}",
                capabilities=["result", "storage"],
                version="1.0.0",
            )
            result_atom = TextualMemoryAtom(
                metadata=result_metadata,
                content=str(result),  # Convert result to string for storage
                embedding_client=None,
            )
            self.store.register(result_atom)

            # Signal waiter
            self.result_waiters[waiting_key].set()

            logger.info(
                f"📊 Received and matched result for {waiting_key} (success={success})"
            )

        except Exception as e:
            logger.error(f"Error handling tool result: {e}")

    async def _summarize(self, plan_id: str, goal: str, plan: list[Step]) -> str:
        """Generate final summary of execution."""
        successful_steps = [s for s in plan if s.status == "success"]
        failed_steps = [s for s in plan if s.status == "failed"]

        # Check for service offline errors first
        offline_steps = [
            s
            for s in failed_steps
            if s.result
            and isinstance(s.result, dict)
            and s.result.get("error") == "Perplexica offline"
        ]
        if offline_steps:
            return (
                f"❌ **Service Offline**: Cannot complete '{goal}' because Perplexica is not reachable.\n\n"
                f"**Quick Fix**: Start Perplexica with `docker compose up -d` in the Perplexica directory, then try again.\n\n"
                f"**Status**: {len(failed_steps)} step(s) failed due to connectivity issues."
            )

        if not successful_steps:
            return f"❌ Goal failed: {goal}. All {len(plan)} steps failed."

        if self.llm:
            return await self._llm_summarize(goal, plan)
        return await self._rule_summarize(goal, successful_steps, failed_steps)

    async def _llm_summarize(self, goal: str, plan: list[Step]) -> str:
        """Use LLM to create intelligent summary."""
        try:
            plan_data = [
                {"step": i + 1, "tool": s.tool, "status": s.status, "result": s.result}
                for i, s in enumerate(plan)
            ]

            prompt = f"""
Goal: {goal}
Execution Results: {json.dumps(plan_data, indent=2)}

Create a concise summary (max 150 words) of what was accomplished.
Focus on the key findings or outcomes, not the technical steps.
"""

            response = await self.llm.generate_content_async(prompt)
            return response.text.strip()

        except Exception as e:
            logger.warning(f"LLM summary failed: {e}, using rule-based summary")
            successful_steps = [s for s in plan if s.status == "success"]
            failed_steps = [s for s in plan if s.status == "failed"]
            return await self._rule_summarize(goal, successful_steps, failed_steps)

    async def _rule_summarize(
        self, goal: str, successful_steps: list[Step], failed_steps: list[Step]
    ) -> str:
        """Create rule-based summary with actual result content."""
        summary_parts = [f"✅ **Goal Completed**: {goal}"]

        if successful_steps:
            summary_parts.append(f"📋 **Completed**: {len(successful_steps)} steps")

            # Add detailed result content
            for i, step in enumerate(successful_steps[:2], 1):  # Top 2 results
                if isinstance(step.result, dict):
                    # Handle web search results
                    if "web" in step.result and "github" in step.result:
                        web_results = step.result.get("web", [])
                        github_results = step.result.get("github", [])

                        summary_parts.append(
                            f"\n🌐 **Web Results** ({len(web_results)} found):"
                        )
                        for idx, hit in enumerate(web_results[:3], 1):  # Top 3 web hits
                            title = hit.get("title", "No title")[:80]
                            url = hit.get("url", "")
                            snippet = hit.get("snippet", "")[:100]
                            summary_parts.append(
                                f"  {idx}. **{title}**\n     {snippet}...\n     🔗 {url}"
                            )

                        summary_parts.append(
                            f"\n💻 **GitHub Results** ({len(github_results)} found):"
                        )
                        for idx, hit in enumerate(
                            github_results[:3], 1
                        ):  # Top 3 GitHub hits
                            title = hit.get("title", hit.get("full_name", "Unknown"))
                            url = hit.get("url", "")
                            description = hit.get(
                                "snippet", hit.get("description", "")
                            )[:100]
                            stars = hit.get("stars", 0)
                            language = hit.get("language", "")
                            summary_parts.append(
                                f"  {idx}. **{title}** ({language}, ⭐{stars})\n     {description}...\n     🔗 {url}"
                            )

                    # Handle other result types
                    elif "summary" in step.result:
                        summary_parts.append(f"{i}. {step.result['summary']}")

        if failed_steps:
            summary_parts.append(f"⚠️ **Failed**: {len(failed_steps)} steps")

        return "\n".join(summary_parts)

    async def _persist_plan(self, plan_id: str, goal: str, plan: list[Step]):
        """Persist plan for recovery using TextualMemoryAtom (following Sacred Rules)."""
        plan_data = {
            "goal": goal,
            "steps": [asdict(s) for s in plan],
            "created_at": datetime.now().isoformat(),
            "status": "running",
        }

        plan_metadata = NeuralAtomMetadata(
            name=plan_id,
            description=f"Execution plan for: {goal[:100]}...",
            capabilities=["plan", "storage", "recovery"],
            version="1.0.0",
        )

        plan_atom = TextualMemoryAtom(
            metadata=plan_metadata,
            content=str(plan_data),  # Convert plan to string for storage
            embedding_client=None,
        )
        self.store.register(plan_atom)

    async def _cleanup_plan(self, plan_id: str):
        """Clean up completed plan."""
        if plan_id in self.active_plans:
            del self.active_plans[plan_id]

        # Remove result waiters (should be empty by now)
        keys_to_remove = [
            k for k in self.result_waiters.keys() if k.startswith(plan_id)
        ]
        for key in keys_to_remove:
            del self.result_waiters[key]

        logger.debug(f"Cleaned up plan {plan_id}")

    def get_stats(self) -> dict[str, Any]:
        """Get executor statistics."""
        return {
            "active_plans": len(self.active_plans),
            "pending_results": len(self.result_waiters),
            "total_steps": sum(len(plan) for plan in self.active_plans.values()),
        }
