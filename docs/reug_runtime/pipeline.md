# REUG Runtime Chat Stream Pipeline

The diagram below traces a single streamed turn through the REUG runtime, starting from the HTTP router and following the reasoning/acting loop until the final answer is emitted. It highlights how reasoning chunks, tool execution, telemetry, and SSE frames are coordinated.

```mermaid
sequenceDiagram
    participant Client
    participant Router as router.chat_stream
    participant Loop as execute_turn
    participant Orchestrator
    participant Model as llm_model.stream_chat
    participant Registry as ability_registry
    participant EventBus
    participant KG as kg adapter
    participant Streamer as sse_transformer

    Client->>Router: POST /v1/chat/stream (message, session_id)
    Router->>Loop: execute_turn(user_msg, session_id, state...)
    Loop->>EventBus: emit TaskStarted
    Loop->>KG: retrieve_relevant_context()/get_goal_for_session
    Note over Loop,KG: Optional goal/context hydration for reasoning seed
    Loop->>Model: stream_chat(messages, tools)
    Model-->>Loop: LLMChunk ("Thinking... <tool_call>...")
    Loop->>Streamer: yield {type: LLMChunk}
    Streamer-->>Client: event: content\ndata: {"content": "<tool_call>..."}

    loop Reasoning / Acting cycle (max_tool_calls)
        Loop->>Orchestrator: _acting_step(tool_calls)
        Orchestrator->>EventBus: emit AbilityCalled (span_id)
        Orchestrator->>Registry: execute(tool, args)
        Registry-->>Orchestrator: result payload
        Orchestrator->>EventBus: emit AbilitySucceeded / AbilityFailed
        Orchestrator-->>Loop: tool_messages for conversation state
        Loop->>Loop: inject <tool_result tool="…"> JSON </tool_result>
        Loop->>Streamer: yield {type: AbilitySucceeded}
        Streamer-->>Client: event: tool_result\ndata: {"type": "AbilitySucceeded", ...}
        Loop->>Model: stream_chat(messages + tool_result)
        Model-->>Loop: LLMChunk ("<final_answer>{...}</final_answer>")
        Loop->>Streamer: yield {type: LLMChunk}
        Streamer-->>Client: event: content\ndata: {"content": "<final_answer>..."}
    end

    Loop->>EventBus: emit LoopAlignmentTelemetry (atoms, bonds, energy, TODO, bandit, reward)
    Loop->>KG: create_atom("final_answer", ...)
    Loop->>EventBus: emit KnowledgeAtomCreated / KnowledgeBondCreated
    Loop->>EventBus: emit TaskSucceeded (final answer payload)
    Loop->>Streamer: yield {type: TaskSucceeded}
    Streamer-->>Client: event: done\ndata: {"type": "TaskSucceeded", "data": {"content": ...}}
```

* **Reasoning loop** — continues until no tool calls remain or `max_tool_calls` is reached.
* **Tool result injection** — `<tool_result>` blocks are appended to the assistant history so the model can ground the next reasoning step.
* **Telemetry alignment** — `LoopAlignmentTelemetry` captures atoms, bonds, energy propagation, TODO pressure, bandit readiness, and reward signals before bonds are persisted.
* **Final emission** — the stream concludes with `TaskSucceeded`, ensuring downstream consumers receive both the `<final_answer>` frame and the structured completion event.
