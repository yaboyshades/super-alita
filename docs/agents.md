# Super Alita – Agents Registry (Living Document)

> Status: LIVING • Source of truth for agents, abilities, plugins, and session stitching.
> Updated automatically by `.github/workflows/update-agents-md.yml` after each PR merge.

- Last Updated: <!-- AGENTS:LAST_UPDATED -->2025-08-27T22:48:02Z
- Current Release: <!-- AGENTS:RELEASE -->master

---

## 0. Quick Links
- Health: `/healthz` • Telemetry: `/metrics` • Streaming: `${API_PREFIX}/v1/chat/stream`
- EventBus: `file://` (dev) or `redis://` (prod)
- Session Ledger: `.alita/sessions/ledger.json` (auto‑maintained)

---

## 1. Agents (Top‑Level)
<!-- AGENTS:START -->
| Agent | Kind | Entrypoint | Abilities (count) | Plugins (count) | Owner(s) | Stability | Notes |
|---:|----|----|----|----|----|----|----|
| super-alita | runtime | src/main.py | 1 | 56 | @owners | beta |  |
<!-- AGENTS:END -->

### 1.1 Ownership & Contacts
<!-- AGENTS:OWNERS_START -->
| Component | CODEOWNERS | Slack | Escalation |
|---:|----|----|----|
<!-- AGENTS:OWNERS_END -->

---

## 2. Abilities
> Contract‑first tools that the runtime can call (dynamic registry supported).

<!-- ABILITIES:START -->
| Ability | Module | Signature | Guardrails | Telemetry Events | Notes |
|---:|----|----|----|----|----|
| _utcnow | src/abilities/gemini_codegen_ability.py | (…) | yes | Ability* events |  |
<!-- ABILITIES:END -->

---

## 3. Plugins
> Pluggable modules (planner, memory, search, MCP, etc.)

<!-- PLUGINS:START -->
| Plugin | Module | Capabilities | Config Keys | Health Check | Notes |
|---:|----|----|----|----|----|
| autonomy_tracker | src/plugins/autonomy_tracker.py | (…) | ENV_* | function() => ok |  |
| planner_plugin_v2 | src/plugins/planner_plugin_v2.py | (…) | ENV_* | function() => ok |  |
| memory_manager_plugin_unified | src/plugins/memory_manager_plugin_unified.py | (…) | ENV_* | function() => ok |  |
| self_reflection_plugin | src/plugins/self_reflection_plugin.py | Enumerate all available tools and plugins, parameters.get("requested_capability", ""), {e}", exc_info=True) | ENV_* | function() => ok |  |
| semantic_fsm_plugin | src/plugins/semantic_fsm_plugin.py | (…) | ENV_* | function() => ok |  |
| flowise_adapter_plugin | src/plugins/flowise_adapter_plugin.py | (…) | ENV_* | function() => ok |  |
| atom_creator_plugin | src/plugins/atom_creator_plugin.py | (…) | ENV_* | function() => ok |  |
| compose_plugin | src/plugins/compose_plugin.py | atom.tool | ENV_* | function() => ok |  |
| pythonic_preprocessor_plugin | src/plugins/pythonic_preprocessor_plugin.py | (…) | ENV_* | function() => ok |  |
| creator_plugin | src/plugins/creator_plugin.py | (…) | ENV_* | function() => ok |  |
| conversation_plugin | src/plugins/conversation_plugin.py | **, 🧠 **Cognitive Architecture**: I use a plugin-based system with neural atoms for reactive state management | ENV_* | function() => ok |  |
| puter_plugin | src/plugins/puter_plugin.py | "cloud_storage", "process_execution", "file_io" | ENV_* | function() => ok |  |
| atom_executor_plugin | src/plugins/atom_executor_plugin.py | (…) | ENV_* | function() => ok |  |
| core_utils_plugin | src/plugins/core_utils_plugin.py | (…) | ENV_* | function() => ok |  |
| semantic_memory_plugin | src/plugins/semantic_memory_plugin.py | "memory", "storage", "retrieval", "memory", "storage", "retrieval", "semantic_search" | ENV_* | function() => ok |  |
| auto_tools_plugin | src/plugins/auto_tools_plugin.py | (…) | ENV_* | function() => ok |  |
| perplexica_search_plugin | src/plugins/perplexica_search_plugin.py | (…) | ENV_* | function() => ok |  |
| tool_lifecycle_plugin | src/plugins/tool_lifecycle_plugin.py | (…) | ENV_* | function() => ok |  |
| deepcode_generator_plugin | src/plugins/deepcode_generator_plugin.py | (…) | ENV_* | function() => ok |  |
| deepcode_puter_bridge_plugin | src/plugins/deepcode_puter_bridge_plugin.py | (…) | ENV_* | function() => ok |  |
| calculator_plugin | src/plugins/calculator_plugin.py | (…) | ENV_* | function() => ok |  |
| cortex_adapter_plugin | src/plugins/cortex_adapter_plugin.py | (…) | ENV_* | function() => ok |  |
| openai_agent_plugin | src/plugins/openai_agent_plugin.py | (…) | ENV_* | function() => ok |  |
| enhanced_pythonic_preprocessor_plugin | src/plugins/enhanced_pythonic_preprocessor_plugin.py | (…) | ENV_* | function() => ok |  |
| llm_planner_plugin | src/plugins/llm_planner_plugin.py | (…) | ENV_* | function() => ok |  |
| dify_adapter_plugin | src/plugins/dify_adapter_plugin.py | (…) | ENV_* | function() => ok |  |
| predictive_world_model_plugin | src/plugins/predictive_world_model_plugin.py | (…) | ENV_* | function() => ok |  |
| ladder_aog_plugin | src/plugins/ladder_aog_plugin.py | (…) | ENV_* | function() => ok |  |
| tool_executor_plugin_unified | src/plugins/tool_executor_plugin_unified.py | (…) | ENV_* | function() => ok |  |
| tool_executor_plugin | src/plugins/tool_executor_plugin.py | (…) | ENV_* | function() => ok |  |
| skill_discovery_plugin | src/plugins/skill_discovery_plugin.py | (…) | ENV_* | function() => ok |  |
| plugin_interface | src/plugins/plugin_interface.py | (…) | ENV_* | function() => ok |  |
| atom_tools_plugin | src/plugins/atom_tools_plugin.py | (…) | ENV_* | function() => ok |  |
| memory_manager_plugin_clean | src/plugins/memory_manager_plugin_clean.py | "storage", "recall", "memory" | ENV_* | function() => ok |  |
| enhanced_protocol_plugin | src/plugins/enhanced_protocol_plugin.py | (…) | ENV_* | function() => ok |  |
| deepcode_orchestrator_plugin | src/plugins/deepcode_orchestrator_plugin.py | (…) | ENV_* | function() => ok |  |
| core_utils_plugin_dynamic | src/plugins/core_utils_plugin_dynamic.py | ", len(self._capabilities)), # Tool not in our discovered capabilities, dict[str, Callable, {name}" | ENV_* | function() => ok |  |
| creator_plugin_unified | src/plugins/creator_plugin_unified.py | = "calculate":, Capability Needed: {request.capability_description}, [, capabilities = ["execute", "process", "respond", capabilities,, if capability == "search":, if capability.lower() in task_lower:, json.dumps(spec.capabilities),, {capabilities}, | ENV_* | function() => ok |  |
| option_executor_plugin | src/plugins/option_executor_plugin.py | (…) | ENV_* | function() => ok |  |
| brainstorm_plugin | src/plugins/brainstorm_plugin.py | atom.tool | ENV_* | function() => ok |  |
| adaptive_neural_atom_plugin | src/plugins/adaptive_neural_atom_plugin.py | (…) | ENV_* | function() => ok |  |
| meta_learning_creator_plugin | src/plugins/meta_learning_creator_plugin.py | (…) | ENV_* | function() => ok |  |
| memory_manager_plugin | src/plugins/memory_manager_plugin.py | "storage", "recall", "memory" | ENV_* | function() => ok |  |
| llm_planner_plugin_unified | src/plugins/llm_planner_plugin_unified.py | {', '.join(atom_info['capabilities' | ENV_* | function() => ok |  |
| self_heal_plugin | src/plugins/self_heal_plugin.py | (…) | ENV_* | function() => ok |  |
| knowledge_gap_detector | src/plugins/knowledge_gap_detector.py | (…) | ENV_* | function() => ok |  |
| planner_plugin | src/plugins/planner_plugin.py | (…) | ENV_* | function() => ok |  |
| event_bus_plugin | src/plugins/event_bus_plugin.py | (…) | ENV_* | function() => ok |  |
| system_introspection_plugin | src/plugins/system_introspection_plugin.py | (…) | ENV_* | function() => ok |  |
| subproblem_manager | src/plugins/oak_core/subproblem_manager.py | (…) | ENV_* | function() => ok |  |
| feature_discovery | src/plugins/oak_core/feature_discovery.py | (…) | ENV_* | function() => ok |  |
| planning_engine | src/plugins/oak_core/planning_engine.py | (…) | ENV_* | function() => ok |  |
| curation_manager | src/plugins/oak_core/curation_manager.py | (…) | ENV_* | function() => ok |  |
| prediction_engine | src/plugins/oak_core/prediction_engine.py | (…) | ENV_* | function() => ok |  |
| coordinator | src/plugins/oak_core/coordinator.py | (…) | ENV_* | function() => ok |  |
| option_trainer | src/plugins/oak_core/option_trainer.py | (…) | ENV_* | function() => ok |  |
<!-- PLUGINS:END -->

---

## 4. Runtime Surfaces
- **HTTP**: FastAPI (`app.py` / `src/main.py`) — `/healthz`, `${API_PREFIX}/v1/chat/stream`
- **Eventing**: EventBus (file/Redis), MCP telemetry broadcaster
- **Sandbox**: `src/sandbox/exec_sandbox.py`
- **VS Code** (optional): extension client (gRPC when wired)

---

## 5. Session Stitching (Cross‑Session Context)
The updater maintains a session ledger for continuity across “AI sessions” and human sessions.

**Ledger:** `.alita/sessions/ledger.json`
```json
{
  "series": [
    {
      "series_id": "2025W34-streaming-router-hardening",
      "prs": [123, 129, 131],
      "branches": ["feat/streaming-hardening", "hotfix/disconnect"],
      "session_notes": [
        {"ts": "2025-08-24T18:27Z", "summary": "Tool synthesis path stabilized"},
        {"ts": "2025-08-25T03:04Z", "summary": "Disconnect test added"}
      ]
    }
  ]
}

Index (recent):

<!-- SESSIONS:START -->
- **series-202534** · PRs: [62, 63, 64, 65, 66, 69, 70, 73, 74, 76, 77, 78, 83, 89, 90, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 121, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 155, 156, 157, 158, 159, 160, 164, 166, 167, 168, 171, 172, 174, 175, 176, 178]
<!-- SESSIONS:END -->Add a line to .alita/sessions/notes/*.md to seed context for the next session; the ledger links it back here.

---

6. Changelog (Auto‑appended per PR)

<!-- CHANGELOG:START -->
- 2025-08-27T22:48:02Z #178 Merge pull request #178 from yaboyshades/codex/enhance-plugin-capability-inspection (owner: @yaboyshades)
<!-- CHANGELOG:START -->
- 2025-08-27T22:46:50Z #176 Merge pull request #176 from yaboyshades/codex/add-stress-test-for-event-bus-dcjzj9 (owner: @yaboyshades)
