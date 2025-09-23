"""Generate or validate .github/copilot-instructions.md and companion manifest.

This script is intentionally small and deterministic: it contains a canonical
markdown blob (the agent instructions) and a compact manifest builder used by
CI to detect drift. The script exposes --write (write files), --check (verify
files match expectations), --print (emit markdown) and --ps1 (emit a PS1
bootstrap snippet).
"""

from __future__ import annotations

import argparse
import json
import textwrap
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MD_PATH = ROOT / ".github" / "copilot-instructions.md"
MANIFEST_PATH = ROOT / ".github" / "copilot-instructions.json"
REPO_NAME = ROOT.name
LAST_REVIEWED = "2025-09-18"


INSTRUCTIONS_MD = (
    textwrap.dedent(
        """
    # High-Impact Context
    - Super Alita is a Python-first orchestration platform that fuses Spec-Kit planning, unified intelligence reasoning, and sandboxed execution.
    - Success demands constitutional compliance (>=75% article score) and unified scoring (`scripts/unified_sdd_mangle.py`) before code lands.
    - Treat this doc as the agent contract: CI fails if sections drift, and generator scripts keep it synchronized.

    ## Architecture Overview
    - Entrypoints: `python -m src.main` (runtime orchestration), `uvicorn app:app --reload --port 8080` (FastAPI dev server).
    - Major pillars:
      - **Runtime** (`src/core/`): sandboxed Python code execution, telemetry, event schemas (see `docs/specs/unified_orchestration_p0_event_schema_spec.md`).
      - **Spec-Kit** (`src/spec_kit/`): planning engine, templates under `templates/spec_kit/`.
      - **Contracts** (`src/contracts/`): constitutional validators, templates under `templates/contracts/`.
      - **Reasoning** (`src/reasoning/`): unified intelligence, templates under `templates/reasoning/`.
      - **Code Reasoning** (`src/code_reasoning/`): unified intelligence, templates under `templates/reasoning/`.
      - **Unified SDD Mangle** (`src/unified_sdd_mangle/`): unified intelligence, templates under `templates/reasoning/`.
      - **Orchestration Core** (`src/orchestration/`): reliability manager, observability rails, event schemas (see `docs/specs/unified_orchestration_p0_event_schema_spec.md`).
      - **Abilities & Plugins** (`src/abilities/`, `src/plugins/`): capability adapters (e.g., Mangle) registered via tool factories.
    - Supporting services: Redis optional (integration tests), SQLite facts DB for `unified_sdd_mangle`, gRPC consensus service under `src/consensus_grpc/`.
    - Observability: structured events through `src/orchestration/observability.py`, telemetry collector in `src/core/telemetry/collector.py`.

    ## Developer Workflows
    - Bootstrap (PowerShell):
      ```powershell
      python -m venv .venv
      . .venv/Scripts/Activate.ps1
      pip install -r requirements.txt -c constraints.txt
      ```
    - Bootstrap (POSIX):
      ```bash
      python -m venv .venv
      source .venv/bin/activate
      pip install -r requirements.txt -c constraints.txt
      ```
    - Unified reasoning sweep:
      ```bash
      python scripts/unified_sdd_mangle.py --repo . --spec .spec --db .ai/facts.sqlite --report .ai/report.json --advice .ai/advice.json --workflow refactor
      jq '.' .ai/advice.json || python -c "import json,sys;print(json.dumps(json.load(open('.ai/advice.json')), indent=2))"
      ```
    - Golden checks: `pytest -q`, `ruff check src tests`, `mypy --strict src core`, `black . -l 88`.
    - CI mirrors makefile-less commands; GitHub workflows live in `.github/workflows/` (see `ci.yml`, `ci-globkit.yml`, `copilot_instructions_drift.yml`).

    ## Project Conventions
    - Python 3.11+, 4-space indent, double quotes, explicit type hints. Keep functions <=50 LOC, pure when possible.
    - Dynamic code execution goes through `src/sandbox/exec_sandbox.py`; subprocesses via `src/core/proc.py` (no `shell=True`). YAML via `src/core/yaml_utils.py`.
    - Tests mirror `src/` under `tests/`; integration markers exist (`-m integration_redis`). Target >=70% coverage for new work.
    - Naming: snake_case modules/functions, PascalCase classes, UPPER_SNAKE constants. Files placed under domain-specific package (e.g., `src/sdd/session/`).
    - Update `AGENTS.md` when new abilities or agents are introduced; update templates when changing contracts.

    ## Configuration & Secrets
    - `.env.example` documents core env vars (LLM tokens, Redis URL, mode flags). Copy to `.env`; never commit secrets.
    - Runtime precedence: environment vars > `.env` > defaults in `src/core/settings.py`.
    - `SUPER_ALITA_MODE` controls planner behaviour (`shadow`, `act`, `batch`). `LLM_RETRY_*` parameters tune reliability manager.
    - Secrets must flow through `src/orchestration/event_sanitizer.py` before logging; redact before persisting to `artifacts/`.

    ## Module & Service Map
    - `src/sdd/`: Spec Kit models, routers, constitutional pipeline, session factory.
    - `src/unified_intelligence/`: orchestrator bridge, workflow detector, golden fixtures, code reasoning engine (`code_reasoning/`).
    - `src/orchestration/`: reliability manager, observability, unified orchestrator.
    - `src/abilities/mangle/`: adapters and validators for the Mangle reasoning engine.
    - `src/core/`: settings, telemetry, sandbox guardrails, env helpers.
    - `scripts/`: utilities (`unified_sdd_mangle.py`, `generate_copilot_instructions.py`, smoke scripts).
    - `tests/`: mirrors domains (`tests/contract/`, `tests/runtime/`, `tests/unified_intelligence.py`, etc.).

    ## Common Tasks (Recipes)
    ### Add an SDD endpoint
    1. Define request/response model in `src/sdd/models.py`.
    2. Add route logic in `src/sdd/router.py` with constitutional validation.
    3. Update templates under `templates/sdd/` if new artifacts emitted.
    4. Tests: `pytest -q -k "sdd"`; ensure `scripts/unified_sdd_mangle.py` advice is not `block`.

    ### Introduce a new ability module
    1. Create `src/abilities/<ability>/<ability>_ability.py` with sandbox-safe execution.
    2. Wire validators in `src/abilities/<ability>/` and register in tool factory / `AGENTS.md`.
    3. Add tests under `tests/abilities/` and run `pytest -q -k ability`.
    4. Regenerate unified advice to confirm no missing specs/tests.

    ### Adjust unified orchestrator scoring
    1. Modify scoring in `src/unified_intelligence/orchestrator.py` or workflow detector thresholds.
    2. Update documentation in `docs/specs/unified_orchestration_p0_event_schema_spec.md` if contract shifts.
    3. Tests: `pytest -q -k unified_intelligence`; run `scripts/unified_sdd_mangle.py --workflow debug` to verify fused score expectations.

    ### Ship a spec-driven feature
    1. Run Spec-Kit CLI: `python -m src.sdd.sdd_cli specify "<feature>"` -> plan -> tasks.
    2. Implement code under `src/sdd/` or relevant domain, using generated templates.
    3. Ensure templates/tests in `templates/sdd/` and `tests/contract/` updated.
    4. Run `pytest -q` and unified reasoning sweep; ensure decision >= revise.

    ### Update constitutional gate logic
    1. Touch `src/contracts/gates/common_gates.py` (or relevant gate) with new rule.
    2. Document in `docs/constitution_update_checklist.md` and update `memory/sdd/constitutional_sdd_framework.md` if needed.
    3. Tests: `pytest -q -k contracts`; run targeted runtime tests (`tests/runtime/test_router_result_cap.py`).

    ## Failure Mode Library
    - **`scripts/unified_sdd_mangle.py` returns `spec_present: false`** -> Point `--spec` at actual Spec Kit directory or ensure `.spec/` populated.
    - **`pytest` import errors in integration suites** -> Missing optional deps (Redis, gRPC stubs); skip via markers or install extras.
    - **`ruff` reports `F401` across generated templates** -> Update templates to use imports or mark with `# noqa: F401` where intentional.
    - **`mypy --strict` fails on new modules** -> Add explicit type hints and `TypedDict`/`Protocol` definitions; avoid Any leakage.
    - **`uvicorn` crashes due to missing env** -> Copy `.env.example` and export required keys (`ALITA_RUNTIME_HOST`, LLM credentials).
    - **`copilot-instructions` drift in CI** -> Run `python scripts/generate_copilot_instructions.py --write` and commit regenerated files.

    ## Actually…
    - Local dev rarely uses full Docker stack; prefer direct `uvicorn`/`python -m src.main` invocations.
    - Unified intelligence relies on SQLite facts from `unified_sdd_mangle.py`; skipping the sweep causes false positives in constitutional gates.
    - Mutation and CFG gates live in `.vscode/copilot-middleware/`; they may fail on Windows without WSL—run in POSIX shell if possible.
    - `requirements-test.txt` is optional; default installs cover most suites. Use when property/integration tests fail.

    ## Glossary & Acronyms
    - **SDD**: Spec-Driven Development pipeline (spec -> plan -> tasks).
    - **UIQ**: Unified Intelligence (reasoning orchestrator, `src/unified_intelligence/`).
    - **Mangle**: Integrated reasoning engine for code analysis abilities.
    - **Constitutional Gate**: Compliance scoring enforcing >=75% alignment.
    - **Advice JSON**: Output of `unified_sdd_mangle.py` summarizing rule hits and decisions.
    """
    ).strip()
    + "\n"
)


def build_manifest(generated_at: str | None = None) -> dict:
    if generated_at is None:
        generated_at = datetime.now(UTC).isoformat()
    return {
        "repo": REPO_NAME,
        "services": [
            {"name": "orchestrator", "path": "src/orchestration", "language": "python"},
            {"name": "sdd-api", "path": "src/sdd", "language": "python"},
            {
                "name": "unified-intelligence",
                "path": "src/unified_intelligence",
                "language": "python",
            },
        ],
        "scripts": [
            {"name": "run-runtime", "cmd": "python -m src.main"},
            {"name": "dev-server", "cmd": "uvicorn app:app --reload --port 8080"},
            {"name": "tests", "cmd": "pytest -q"},
            {"name": "lint", "cmd": "ruff check src tests"},
        ],
        "env": [
            {"name": "SUPER_ALITA_MODE", "required": False, "default": "shadow"},
            {
                "name": "ALITA_RUNTIME_HOST",
                "required": False,
                "default": "http://127.0.0.1:8080",
            },
            {"name": "LLM_RETRY_MULTIPLIER", "required": False, "default": "1.0"},
        ],
        "recipes": [
            "add_sdd_endpoint",
            "introduce_ability_module",
            "adjust_unified_orchestrator_scoring",
            "ship_spec_driven_feature",
            "update_constitutional_gate",
        ],
        "generated_at": generated_at,
        "schema_version": "1.0",
        "last_reviewed": LAST_REVIEWED,
    }


def write_files() -> None:
    MD_PATH.parent.mkdir(parents=True, exist_ok=True)
    MD_PATH.write_text(INSTRUCTIONS_MD, encoding="utf-8")
    manifest = build_manifest()
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {MD_PATH}")
    print(f"Wrote {MANIFEST_PATH}")


def check_files() -> None:
    errors: list[str] = []
    if not MD_PATH.exists():
        errors.append("instructions missing")
    else:
        current_md = MD_PATH.read_text(encoding="utf-8")
        if current_md != INSTRUCTIONS_MD:
            errors.append("instructions drift")
    if not MANIFEST_PATH.exists():
        errors.append("manifest missing")
    else:
        current_manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        expected = build_manifest(current_manifest.get("generated_at"))

        def without_ts(payload: dict) -> dict:
            return {k: v for k, v in payload.items() if k != "generated_at"}

        if without_ts(current_manifest) != without_ts(expected):
            errors.append("manifest drift")
    if errors:
        raise SystemExit("; ".join(errors))
    print("copilot instructions OK")


def emit_ps1_snippet() -> None:
    snippet = "\n".join(
        [
            "# PowerShell quick reference",
            "python -m venv .venv",
            ". .venv/Scripts/Activate.ps1",
            "pip install -r requirements.txt -c constraints.txt",
            "python scripts/unified_sdd_mangle.py --repo . --spec .spec --db .ai/facts.sqlite --report .ai/report.json --advice .ai/advice.json --workflow refactor",
        ]
    )
    print(snippet)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate or validate copilot instructions"
    )
    parser.add_argument(
        "--write", action="store_true", help="Write markdown and manifest"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate markdown and manifest match expectations",
    )
    parser.add_argument("--print", action="store_true", help="Print markdown to stdout")
    parser.add_argument(
        "--ps1", action="store_true", help="Emit PowerShell bootstrap snippet"
    )
    args = parser.parse_args()

    ran = False
    if args.write:
        write_files()
        ran = True
    if args.check:
        check_files()
        ran = True
    if args.print:
        print(INSTRUCTIONS_MD, end="")
        ran = True
    if args.ps1:
        emit_ps1_snippet()
        ran = True
    if not ran:
        parser.print_help()


if __name__ == "__main__":
    main()
