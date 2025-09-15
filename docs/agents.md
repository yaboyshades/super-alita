# Super Alita – Agents Registry (Living Document)

> Status: LIVING • Source of truth for agents, abilities, plugins, and session stitching.
> Updated automatically by `.githu---

## 6. Session Ledger Details

The updater maintains a session ledger for continuity across "AI sessions" and human sessions.

**Ledger:** `.alita/sessions/ledger.json`

```jsonws/update-agents-md.yml` after each PR merge.

- Last Updated: <!-- AGENTS:LAST_UPDATED -->2025-09-15T22:00:05Z
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
| super-alita | runtime | src/main.py | 7 | 62 | @owners | beta |  |
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
| __init__ | src/abilities/base_ability.py | (…) | unknown | Ability* events |  |
| __init__ | src/abilities/unified_registry.py | (…) | yes | Ability* events |  |
| __init__ | src/abilities/simple_mangle_ability.py | (…) | unknown | Ability* events |  |
| __init__ | src/abilities/deepconf_ability.py | (…) | yes | Ability* events |  |
| get_available_queries | src/abilities/mangle_reasoning_ability.py | (…) | yes | Ability* events |  |
| _utcnow | src/abilities/gemini_codegen_ability.py | (…) | yes | Ability* events |  |
| __init__ | src/abilities/mangle/mangle_ability.py | (…) | yes | Ability* events |  |
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
| mangle_plugin | src/plugins/mangle_plugin.py | (…) | ENV_* | function() => ok |  |
| llm_planner_plugin | src/plugins/llm_planner_plugin.py | (…) | ENV_* | function() => ok |  |
| dify_adapter_plugin | src/plugins/dify_adapter_plugin.py | (…) | ENV_* | function() => ok |  |
| native_deepcode_plugin | src/plugins/native_deepcode_plugin.py | """Factory function to create {task_kind} capability""", """Generated capability for {task_kind}""", create_{task_kind}_capability() | ENV_* | function() => ok |  |
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
| core_utils_plugin_dynamic | src/plugins/core_utils_plugin_dynamic.py | ", len(self._capabilities), # Tool not in our discovered capabilities, dict[str, Callable, {name}" | ENV_* | function() => ok |  |
| autogen_creator_plugin | src/plugins/autogen_creator_plugin.py | (…) | ENV_* | function() => ok |  |
| creator_plugin_unified | src/plugins/creator_plugin_unified.py | = "calculate":, Capability Needed: {request.capability_description}, [, capabilities = ["execute", "process", "respond", capabilities,, if capability == "search":, if capability.lower() in task_lower:, json.dumps(spec.capabilities),, {capabilities}, | ENV_* | function() => ok |  |
| option_executor_plugin | src/plugins/option_executor_plugin.py | (…) | ENV_* | function() => ok |  |
| brainstorm_plugin | src/plugins/brainstorm_plugin.py | atom.tool | ENV_* | function() => ok |  |
| adaptive_neural_atom_plugin | src/plugins/adaptive_neural_atom_plugin.py | (…) | ENV_* | function() => ok |  |
| meta_learning_creator_plugin | src/plugins/meta_learning_creator_plugin.py | (…) | ENV_* | function() => ok |  |
| native_perplexica_plugin | src/plugins/native_perplexica_plugin.py | (…) | ENV_* | function() => ok |  |
| memory_manager_plugin | src/plugins/memory_manager_plugin.py | "storage", "recall", "memory" | ENV_* | function() => ok |  |
| mcp_adapter_plugin | src/plugins/mcp_adapter_plugin.py | (…) | ENV_* | function() => ok |  |
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
| __init__ | src/plugins/telemetry_pipeline/__init__.py | (…) | ENV_* | function() => ok |  |
<!-- PLUGINS:END -->

---

## 4. Runtime Surfaces

- **HTTP**: FastAPI (`app.py` / `src/main.py`) — `/healthz`, `${API_PREFIX}/v1/chat/stream`
- **Eventing**: EventBus (file/Redis), MCP telemetry broadcaster
- **Sandbox**: `src/sandbox/exec_sandbox.py`
- **VS Code** (optional): extension client (gRPC when wired)

---

## 4.1 GitHub Copilot Agent Mode Development Tools

GitHub Copilot's Agent Mode leverages a comprehensive suite of development tools and VS Code extensions for enhanced code quality, testing, and productivity:

### Code Quality & Formatting Tools

- **Ruff**: Integrated fast Python linter for real-time code analysis and style enforcement
- **Black**: Automated Python code formatting for consistent code styling
- **isort**: Import statement organization and sorting
- **Pylint**: Additional static code analysis for comprehensive error detection
- **mypy**: Type checking for Python code quality assurance

### Testing & Validation Framework

- **pytest**: Primary testing framework with advanced plugin ecosystem
- **pytest-cov**: Code coverage analysis and reporting
- **pytest-asyncio**: Async/await testing support for modern Python applications
- **pytest-xdist**: Parallel test execution for faster feedback loops
- **unittest**: Built-in Python testing framework support

### Essential VS Code Extensions

- **Python Extension Pack**: Core Python development environment
- **GitHub Copilot**: AI-powered code completion and suggestions
- **GitHub Copilot Chat**: Interactive AI assistance for development tasks
- **Python Docstring Generator**: Automated docstring creation with templates
- **autoDocstring**: Enhanced documentation generation with multiple formats
- **Error Lens**: Inline display of errors, warnings, and info messages
- **GitLens**: Advanced Git integration with blame, history, and branch visualization
- **Python Snippets**: Pre-built code snippets for rapid development
- **Bracket Pair Colorizer**: Enhanced code readability with colored brackets
- **indent-rainbow**: Visual indentation guides for Python code

### Documentation & Productivity Extensions

- **Markdown All in One**: Enhanced Markdown editing with preview and shortcuts
- **TODO Highlight**: Task and comment highlighting for better code organization
- **Thunder Client**: REST API testing directly within VS Code
- **YAML**: Language support for configuration files
- **Python Test Explorer**: Visual test discovery and execution interface
- **Code Spell Checker**: Spell checking for comments and documentation

### GitHub Copilot Agent Mode Integration

The agent mode enhances development workflow through:

- **Intelligent Tool Selection**: Automatically chooses appropriate linting and formatting tools based on project context
- **Real-time Code Analysis**: Runs ruff and black formatting on code suggestions before presenting them
- **Test-Driven Development**: Integrates pytest for automated test generation and validation
- **Context-Aware Assistance**: Uses extension capabilities to provide more accurate code suggestions
- **Documentation Generation**: Leverages docstring generators and snippet libraries for comprehensive code documentation
- **Error Prevention**: Proactively suggests fixes using Error Lens and static analysis tools

---

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
- **series-202537** · PRs: [219, 220, 221, 222, 223, 224, 225, 227, 228, 229, 230, 235]
- **series-202536** · PRs: [212, 213]
- **series-202535** · PRs: [200, 201, 202, 203, 204, 207]
- **series-202534** · PRs: [62, 63, 64, 65, 66, 69, 70, 73, 74, 76, 77, 78, 83, 89, 90, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 121, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 155, 156, 157, 158, 159, 160, 164, 166, 167, 168, 171, 196]
<!-- SESSIONS:END -->Add a line to .alita/sessions/notes/*.md to seed context for the next session; the ledger links it back here.

---

6. Changelog (Auto‑appended per PR)

<!-- CHANGELOG:START -->
- 2025-09-15T22:00:05Z #235 Merge pull request #235 from yaboyshades/codex/review-cli-commands-and-scaffolding-hooks (owner: @yaboyshades)
<!-- CHANGELOG:START -->
- 2025-09-15T21:41:20Z #230 Merge pull request #230 from yaboyshades/codex/compare-and-update-template-files (owner: @yaboyshades)
