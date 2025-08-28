# Automation & Continuous Improvement Layer

This document summarizes how GitHub Actions integrate with the Super-Alita tri-layer (Strategic / Tactical / Operational) architecture.

## Goals
- Provide guardrails (safety-lint, config validation)
- Prevent prompt regressions
- Automate adaptation loops (bandit prior refresh, distillation)
- Enable reproducible experiments
- Supply cost & performance transparency

## Workflow Map

| Workflow | Trigger | Purpose | Blocking |
|----------|---------|---------|----------|
| CI | PR, push | Lint, test, coverage | Yes |
| Safety & Policy Lint | PR (prompts/config) | Security + prompt hygiene | Yes |
| Prompt & Strategy Regression | PR (prompts/strategies) | Detect quality drift | Yes (recommend) |
| Bandit Priors Refresh | 6h schedule | Update strategy weights | No (PR created) |
| Nightly Distillation | Nightly | Summarize episodic memory | No |
| Cost & Usage Report | Daily | Visibility of spend | No |
| Experiment Runner | Manual | Controlled experiments | No |
| Docs Sync | Push (architecture/diagrams) | Architecture transparency | No |

## Data Artifacts
- evaluation/results/*.json
- evaluation/summary.json
- memory/distilled/LATEST.md
- reports/cost.md
- config/strategies.json (updated via PR)

## Safety Considerations
- Synthetic placeholders in scripts must be replaced with real fetchers that scrub PII.
- Secrets restricted to specific workflows; no environment-wide exposures.

## Extending
Add new evaluation dimension: create test_cases JSON, update harness to compute metric, plug aggregator.

## Next Steps
Replace placeholder model & data fetch logic with actual service clients.
Decide which workflows are required status checks (CI, safety-lint, prompt-regression recommended).
Introduce JSON Schema validation for tool manifests & strategies.
Implement real RL/bandit update logic (e.g., Thompson Sampling) in recompute script.
Add hallucination / factuality test cases to evaluation harness.
Add caching & concurrency limits (matrix size) once stable.