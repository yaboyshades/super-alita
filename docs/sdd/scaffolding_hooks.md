# SDD Scaffolding Integration Guide

## Scope
This guide explains how automation or project scaffolding scripts can drive the Specification-Driven Development (SDD) workflow that lives under `src/sdd/`. It highlights the callable surfaces for the **specify → plan → tasks** phases, notes how to extend the pipeline, and documents how to enforce constitutional validation gates.

## Pipeline Surfaces
### ConstitutionalSDDPipeline
The base async pipeline (`ConstitutionalSDDPipeline`) provides the following features:
- Writes artifacts into a workspace `specs/` tree.
- Exposes `specify`, `plan`, and `tasks` coroutines.
- Each coroutine accepts a Pydantic request model and returns the matching response model, which includes generated files, metadata, and compliance scores.
- Constitutional scoring is executed only when the request sets `constitutional_gates=True` (default).
- Success is marked via `compliance_threshold_met` once the `0.75` threshold is satisfied.
- The scorer instance is shared across phases, so downstream calls inherit the same gate threshold.

**Hook it when:**
- A script wants to run the SDD phases in-process without Mangle reasoning.
- You need direct access to generated markdown artifacts (`spec.md`, `implementation-plan.md`, `tasks.md`) and compliance metadata before triggering other tooling.

**Invocation pattern:**
```python
from pathlib import Path
import asyncio

from src.sdd import ConstitutionalSDDPipeline, SpecifyRequest

pipeline = ConstitutionalSDDPipeline(Path.cwd())
request = SpecifyRequest(user_input="Build streaming chat", context={"priority": "high"})
response = asyncio.run(pipeline.specify(request))
if not response.compliance_threshold_met:
    raise SystemExit("Specification failed constitutional gates")
```

### EnhancedSDDFramework
`EnhancedSDDFramework` subclasses the base pipeline. It adds Mangle analysis, materializes inline specifications, and exposes extra helpers such as `ask_question` and `validate_constitutional_compliance`. Use this framework when scaffolding requires richer analysis, reuse detection, or when you need to publish results over FastAPI or CLI surfaces. The enhanced responses embed extra data inside `analysis_results`. This allows downstream scripts to decide whether to inject additional compliance tasks or store traceability metadata.

### FastAPI Router
`create_sdd_router()` wires the enhanced framework into FastAPI. Scripts that spin up bespoke tooling servers can mount this router directly and inherit the same request/response contracts. This keeps the CLI, HTTP, and in-process flows consistent.

### Models and Validators
All request/response schemas live in `src/sdd/models.py`; each request exposes a `constitutional_gates` boolean that defaults to `True`. Scripts that generate raw specifications or plans can pass those blobs directly through `PlanRequest.specification` or `TasksRequest.plan`, letting the framework persist them before gating. For pre-flight checks or custom scoring, `SDDValidator` offers async helpers that mirror the constitutional expectations (user stories, acceptance criteria, dependency coverage, etc.).

### Configuration Surface
`SDDConfig`, `DEFAULT_SDD_COMMANDS`, and `CONSTITUTIONAL_SDD_INTEGRATION` describe phase ordering, templates, validation rules, and suggested pre/post hooks. Scripts can read or extend these structures to register additional gates, swap templates, or align with internal naming. Because the config declares which constitutional gates belong to each phase, scaffolding code can auto-populate UI hints or generate TODO items when a gate fails.

## CLI Surfaces
Two CLIs wrap the same pipeline:

- `python -m src.sdd.cli ...` uses `argparse` and the base pipeline. Commands (`specify`, `plan`, `tasks`, `validate`) accept `--workspace` roots and an optional `--no-gates` switch. Omit `--no-gates` so scripts keep constitutional checks enabled.
- `python -m src.sdd.sdd_cli ...` is Click-based and calls the enhanced framework. It also surfaces knowledge-graph helpers (`ask`, `validate`, `trace`, `analyze`, etc.). Each command runs the async framework via `asyncio.run(...)`, so shell scripts can rely on synchronous command execution.

Use the thin CLI when you just need markdown artifacts and compliance scores; use the enhanced CLI when you want Mangle analysis or to export additional reports without embedding Python.

## Hook Points for Scaffolding Scripts
1. **Direct Python integration** – Instantiate `ConstitutionalSDDPipeline` or `EnhancedSDDFramework` with a workspace root. Call the async coroutines via `asyncio.run` (or integrate them into an existing event loop). Capture response fields to route artifacts, show compliance feedback, or block progression when `compliance_threshold_met` is `False`.
2. **HTTP automation** – Mount `create_sdd_router()` inside an automation FastAPI app and POST the same request payloads your scripts would send to the CLI. This is useful when scaffolding needs to serve other clients or run distributed jobs.
3. **CLI delegation** – Shell-based scaffolding can invoke `python -m src.sdd.cli ...` for minimal dependencies. Parse stdout for the generated file paths and compliance scores. Prefer the CLI flags over reimplementing parameter parsing.
4. **Config-driven extensions** – Load `DEFAULT_SDD_COMMANDS` and merge additional command definitions so your scaffolding pipeline can advertise custom gates or templates while reusing the same execution primitives.
5. **Custom validators** – Compose `SDDValidator` checks with the pipeline responses to run lightweight linting before or after the constitutional scorer. This is handy when scaffolding wants to enforce stricter project rules (e.g., minimum number of dependencies listed).

## Triggering Constitutional Checks and Gates
To ensure gates fire for every phase:
1. Build each request with `constitutional_gates=True` (the default), or mirror the CLI default by skipping `--no-gates`.
2. Submit the request to the chosen surface (Python, CLI, or HTTP).
3. Inspect the response’s `overall_compliance_score`, `compliance_threshold_met`, and `constitutional_compliance` map. Treat any `False` threshold result as a blocker and surface the associated article violations to the user.
4. Optionally run `EnhancedSDDFramework.validate_constitutional_compliance()` or `SDDValidator` to gather deeper diagnostics before retrying the phase.
5. Prevent downstream phases from running until prior phases return `compliance_threshold_met=True`. The default phase dependencies (`specify → plan → tasks`) already enforce this order; scaffolding scripts should preserve it when queuing jobs.

## Recommended Orchestration Flow
1. **Specification** – Collect raw requirements, call `specify`, and store `spec.md`. If compliance fails, emit feedback derived from the violation list and prompt for edits before re-running the phase.
2. **Planning** – Feed the generated `spec.md` back through `plan`. For ad-hoc content, pass it via `PlanRequest.specification` and let the framework materialize the file. Stop if the plan’s compliance threshold is missed.
3. **Tasking** – Call `tasks` with the plan path (or raw content). Enforce the gate and surface the structured `TaskBreakdown` objects to downstream systems (e.g., ticket generators).
4. **Optional quality passes** – When using the enhanced framework, call `analyze_code_quality` or `validate_constitutional_compliance` to produce supplemental gating reports for documentation or dashboards.

Following this sequence ensures that every scaffolded project inherits the constitutional SDD guardrails regardless of whether you call the pipeline via Python, HTTP, or the provided CLIs.
