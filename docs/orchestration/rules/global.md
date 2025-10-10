# Global Orchestration Guardrails

These guardrails apply to every agent orchestrated through the REUG runtime.

- Maintain the closed-loop cognitive model: Event → Atom/Bond → Energy → TODO → Bandit → Reward.
- Always emit telemetry for `STATE_TRANSITION`, `TaskStarted`, `TaskSucceeded`, and `TaskFailed` events.
- Enforce sandboxed execution paths (`src/sandbox/exec_sandbox.py`) for untrusted code.
- Escalate to the reliability review workflow if latency, retries, or schema bypass counts exceed policy.
