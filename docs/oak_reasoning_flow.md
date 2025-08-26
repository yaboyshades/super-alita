# OaK Reasoning Flow Architecture

This document outlines the integration of the OaK (Options and Knowledge) system as a "Tactical Layer" within the agent's reasoning flow. This new architecture creates a three-layer model for decision-making:

1.  **Strategic Layer:** A high-level planner that decomposes large goals into smaller, more manageable sub-goals.
2.  **Tactical Layer (OaK):** A system that receives sub-goals and selects or learns "options" (sequences of actions) to achieve them.
3.  **Operational Layer:** The low-level action execution system that processes primitive tool calls.

## Event Flow

The new reasoning flow is orchestrated through a series of events on the system's event bus.

```
[User Goal]
    |
    v
+-----------------------+      (goal_received)      +------------------------+
|   Conversation Plugin | ------------------------> | Strategic Planner      |
| (e.g. LLMPlanner)     |                           | (planner_plugin.py)    |
+-----------------------+                           +------------------------+
                                                              |
                                                              v (subgoal_defined)
+-----------------------+      (tool_call_request)    +------------------------+
| Action Executor       | <------------------------ | OaK Tactical Layer     |
| (e.g. WebAgentAtom)   |                           | (oak_coordinator.py &   |
+-----------------------+                           |  option_executor_plugin.py) |
                                                    +------------------------+
```

1.  A user's high-level goal is received and processed, resulting in a `goal_received` event.
2.  The **Strategic Planner** (`planner_plugin.py`) listens for `goal_received` events. Instead of creating a detailed plan, it decomposes the goal into one or more sub-goals and emits a `subgoal_defined` event for each.
3.  The **OaK Coordinator** (`oak_coordinator.py`) listens for `subgoal_defined` events. This is the entry point to the Tactical Layer.
4.  Upon receiving a sub-goal, the coordinator invokes the OaK **Planning Engine**, which selects the best "option" to achieve the sub-goal. It emits an `oak.plan_proposed` event containing the selected option.
5.  A new **Option Executor** plugin (`option_executor_plugin.py`) listens for `oak.plan_proposed` events. It acts as a bridge between the abstract OaK system and the agent's concrete tools. It translates the selected option into one or more primitive `tool_call_request` events.
6.  The **Action Executor** layer processes these `tool_call_request` events as usual.

## Event Schema: `subgoal_defined`

This new event is used to communicate a sub-goal from the Strategic Layer to the Tactical Layer.

**Event Name:** `subgoal_defined`

**Payload Schema (JSON):**

```json
{
  "event_id": "str (UUID)",
  "timestamp": "str (ISO 8601)",
  "source": "str (e.g., 'planner_plugin')",
  "subgoal": {
    "description": "str (Natural language description of the sub-goal)",
    "parent_goal_id": "str (ID of the original, high-level goal)",
    "subgoal_id": "str (Unique ID for this sub-goal)"
  }
}
```

-   `description`: A clear, actionable description of the task for the OaK system (e.g., "Find the current weather in San Francisco").
-   `parent_goal_id`: Links the sub-goal back to the user's original request, allowing for context and tracking.
-   `subgoal_id`: A unique identifier for this specific sub-goal.

## The Option-to-Action Bridge

A critical piece of this integration is translating the abstract "options" from the OaK system into concrete tool calls. The current OaK implementation uses reinforcement learning to train options, which are represented as neural network policies that output abstract integer actions.

To bridge this gap without a major re-architecture of the OaK core, we will introduce a simple, hardcoded mapping. The new `OptionExecutor` plugin will contain a dictionary that maps `option_id` strings to `tool_call_request` structures.

**Example Mapping:**

```python
OPTION_TO_ACTION_MAPPING = {
    "option-web-search": {
        "tool_name": "web_agent",
        "parameters": {"query": "{subgoal_description}"}
    },
    "option-write-file": {
        "tool_name": "file_manager",
        "parameters": {"action": "write", "path": "/path/to/file.txt", "content": "{subgoal_description}"}
    }
}
```

When the `OptionExecutor` receives a plan with a selected option, it will:
1.  Look up the `option_id` in this mapping.
2.  Extract the tool name and parameter schema.
3.  Populate the parameters using information from the `subgoal_defined` event (e.g., substituting the sub-goal's description into the query parameter).
4.  Emit the fully-formed `tool_call_request` event.

This approach provides a functional integration while isolating the OaK system's internal complexity. Future work could involve making this mapping more dynamic or data-driven.
