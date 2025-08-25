# Super Alita – Agents Registry (Living Document)

> Status: LIVING • Source of truth for agents, abilities, plugins, and session stitching.
> Updated automatically by `.github/workflows/update-agents-md.yml` after each PR merge.

- Last Updated: <!-- AGENTS:LAST_UPDATED -->pending<!-- AGENTS:LAST_UPDATED -->
- Current Release: <!-- AGENTS:RELEASE -->unreleased<!-- AGENTS:RELEASE -->

---

## 0. Quick Links
- Health: `/healthz` • Telemetry: `/metrics` • Streaming: `/v1/chat/stream`
- EventBus: `file://` (dev) or `redis://` (prod)
- Session Ledger: `.alita/sessions/ledger.json` (auto‑maintained)

---

## 1. Agents (Top‑Level)
<!-- AGENTS:START -->
<!-- The updater fills this table by scanning src/abilities, src/plugins, src/reug_runtime, mcp_* trees -->

| Agent | Kind | Entrypoint | Abilities (count) | Plugins (count) | Owner(s) | Stability | Notes |
|------:|------|------------|-------------------:|-----------------:|----------|-----------|-------|
| (scanning…) | | | | | | | |

<!-- AGENTS:END -->

### 1.1 Ownership & Contacts
<!-- AGENTS:OWNERS_START -->
| Component | CODEOWNERS | Slack | Escalation |
|-----------|------------|-------|------------|
| (populated by updater) | | | |
<!-- AGENTS:OWNERS_END -->

---

## 2. Abilities
> Contract‑first tools that the runtime can call (dynamic registry supported).

<!-- ABILITIES:START -->
| Ability | Module | Signature | Guardrails | Telemetry Events | Notes |
|--------:|--------|-----------|------------|------------------|-------|
| (scanning…) | | | | | |
<!-- ABILITIES:END -->

---

## 3. Plugins
> Pluggable modules (planner, memory, search, MCP, etc.)

<!-- PLUGINS:START -->
| Plugin | Module | Capabilities | Config Keys | Health Check | Notes |
|-------:|--------|--------------|-------------|--------------|-------|
| (scanning…) | | | | | |
<!-- PLUGINS:END -->

---

## 4. Runtime Surfaces
- **HTTP**: FastAPI (`app.py` / `src/main.py`) — `/healthz`, `/v1/chat/stream`
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

<!-- SESSIONS:START -->(populated by updater)


<!-- SESSIONS:END -->Add a line to .alita/sessions/notes/*.md to seed context for the next session; the ledger links it back here.

---

6. Changelog (Auto‑appended per PR)

<!-- CHANGELOG:START -->(auto‑generated entries appear here; newest first)


<!-- CHANGELOG:END -->> Format (Conventional Commits style): - 2025-08-25 #134 feat(router): add client-disconnect handling (owner: @alice)



---

7. Runbooks

Cold start: make deps && make run

Runtime Tests: pytest -q tests/runtime

Telemetry parity: ensure MCPTelemetryBroadcaster set with MCP_URL and MCP_TOKEN.

---

8. SLOs & Alerts (MVP)

Uptime > 99.0% • P95 < 2s • Error rate < 0.5%

Circuit‑breaker trips < 2/day • Stream disconnects auto‑recover < 1s

---

9. To‑Watch (Open Items)

MCP telemetry transport (live) • VS Code gRPC wiring • Disconnect e2e test • Capabilities live detection


---
