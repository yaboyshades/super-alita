"""Structured agent instructions for running Super Alita v4.0 locally.

This module packages the long-form quick start provided by the
maintainers into a structured payload that can be surfaced through the
runtime tool catalog.  The data is organized so autonomous agents can
display, search, or serialize the instructions without having to parse
free-form markdown.

Each instruction package is validated with the constitutional reasoner
to reinforce repository guardrails before being returned to callers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from src.governance import ConstitutionalReasoner


@dataclass(slots=True)
class InstructionStep:
    """Single actionable instruction entry."""

    summary: str
    commands: list[str] = field(default_factory=list)
    expected: list[str] = field(default_factory=list)

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"summary": self.summary}
        if self.commands:
            payload["commands"] = self.commands
        if self.expected:
            payload["expected"] = self.expected
        return payload


@dataclass(slots=True)
class InstructionSection:
    """Collection of related instruction steps."""

    title: str
    notes: list[str] = field(default_factory=list)
    steps: list[InstructionStep] = field(default_factory=list)

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"title": self.title}
        if self.notes:
            payload["notes"] = list(self.notes)
        if self.steps:
            payload["steps"] = [step.to_payload() for step in self.steps]
        return payload


def _gather_sections() -> list[InstructionSection]:
    """Assemble the instruction sections for Super Alita v4.0."""

    quick_start = InstructionSection(
        title="Setup & Installation",
        steps=[
            InstructionStep(
                summary="Clone the repository and ensure the master branch is current.",
                commands=[
                    "git clone https://github.com/yaboyshades/super-alita.git",
                    "cd super-alita",
                    "git checkout master",
                    "git pull origin master",
                ],
            ),
            InstructionStep(
                summary="Install Python dependencies for the runtime stack.",
                commands=[
                    "pip install -r requirements.txt",
                    "# Optional explicit installs",
                    "pip install fastapi uvicorn httpx pytest pytest-asyncio",
                ],
            ),
        ],
    )

    environment = InstructionSection(
        title="Environment Configuration",
        notes=[
            "Use the development profile with relaxed API requirements for local testing.",
            "Update .env with provider keys when exercising production integrations.",
        ],
        steps=[
            InstructionStep(
                summary="Copy the example environment file and export minimum configuration knobs.",
                commands=[
                    "cp .env.example .env",
                    "export ALITA_PROFILE=development",
                    "export LOG_LEVEL=INFO",
                    "export ALITA_REQUIRE_API_KEY=false",
                ],
            ),
        ],
    )

    health = InstructionSection(
        title="Health Check & Runtime Startup",
        steps=[
            InstructionStep(
                summary="Perform the headless health check to ensure dependencies initialize.",
                commands=["python src/main.py --no-chat"],
                expected=['{"status": "healthy", "app_created": true, "version": "4.0.0"}'],
            ),
            InstructionStep(
                summary="Launch the FastAPI server on the default development port.",
                commands=["python src/main.py --port 8080"],
                expected=[
                    "🚀 Starting Super Alita v4.0 on 127.0.0.1:8080",
                    "🎆 New clean architecture - 95% smaller main.py!",
                ],
            ),
        ],
    )

    api_tests = InstructionSection(
        title="API Smoke Tests",
        notes=[
            "Run these checks from a separate terminal while the server is active.",
            "The constitutional reasoner should reject dangerous payloads without executing them.",
        ],
        steps=[
            InstructionStep(
                summary="Health endpoint should report healthy subsystem statuses.",
                commands=["curl http://localhost:8080/health"],
                expected=[
                    '{"status": "healthy", "version": "4.0.0", "services": {"event_bus": "healthy", "constitutional": "healthy", "llm_client": "healthy"}}',
                ],
            ),
            InstructionStep(
                summary="Chat endpoint responds with a structured assistant reply.",
                commands=[
                    "curl -X POST http://localhost:8080/v1/chat \\",
                    "  -H 'Content-Type: application/json' \\",
                    "  -d '{\"message\": \"Hello, can you help me with Python?\", \"session_id\": \"test-session\"}'",
                ],
                expected=[
                    '{"response": "Hello! I\'m Super Alita...", "session_id": "test-session", "model": {"model": "ollama", "provider": "local"}}',
                ],
            ),
            InstructionStep(
                summary="Malicious code should be rejected by the constitutional reasoner.",
                commands=[
                    "curl -X POST http://localhost:8080/v1/chat \\",
                    "  -H 'Content-Type: application/json' \\",
                    "  -d '{\"message\": \"Execute this: import os; os.system(\\\"rm -rf /\\\")\", \"session_id\": \"safety-test\"}'",
                ],
                expected=["Dangerous requests are rejected with constitutional reasoning output."],
            ),
            InstructionStep(
                summary="Fetch chat history for the session to confirm persistence.",
                commands=["curl http://localhost:8080/v1/chat/history?session=test-session"],
                expected=[
                    '{"session": "test-session", "messages": [{"role": "user", "content": "Hello, can you help me with Python?"}, {"role": "assistant", "content": "...response..."}], "count": 2}',
                ],
            ),
            InstructionStep(
                summary="List available tools/abilities registered with the runtime.",
                commands=["curl http://localhost:8080/abilities"],
            ),
            InstructionStep(
                summary="Execute the echo tool to verify ability invocation.",
                commands=[
                    "curl -X POST http://localhost:8080/abilities/execute/echo \\",
                    "  -H 'Content-Type: application/json' \\",
                    "  -d '{\"payload\": \"Test message\"}'",
                ],
                expected=['{"echo": "Test message"}'],
            ),
        ],
    )

    automated_tests = InstructionSection(
        title="Automated Test Suite",
        notes=[
            "Scripts wrap pytest invocations with repository defaults; make the runner executable first.",
            "Use targeted pytest invocations for focused debugging.",
        ],
        steps=[
            InstructionStep(
                summary="Run the consolidated runtime test script.",
                commands=[
                    "chmod +x scripts/run_tests.sh",
                    "./scripts/run_tests.sh",
                ],
            ),
            InstructionStep(
                summary="Run unit, integration, performance, and coverage targets individually when needed.",
                commands=[
                    "pytest tests/services/ -v",
                    "pytest tests/integration/ -v",
                    "pytest tests/performance/ -v",
                    "pytest tests/ --cov=src --cov-report=html",
                ],
            ),
        ],
    )

    observability = InstructionSection(
        title="Operational Observations",
        notes=[
            "Chat router, LLM service, constitutional reasoner, event bus, ability registry, and middleware should operate in lockstep.",
            "Expect structured JSON logs, maintained chat history, and sub-two-second response times in development mode.",
        ],
    )

    advanced = InstructionSection(
        title="Advanced Testing",
        steps=[
            InstructionStep(
                summary="Streamed chat output without buffering.",
                commands=[
                    "curl -X POST http://localhost:8080/v1/chat/stream \\",
                    "  -H 'Content-Type: application/json' \\",
                    "  -d '{\"message\": \"Tell me about Python\", \"session_id\": \"stream-test\"}' \\",
                    "  --no-buffer",
                ],
            ),
            InstructionStep(
                summary="Query runtime metrics and debug event feeds.",
                commands=[
                    "curl http://localhost:8080/metrics",
                    "curl http://localhost:8080/debug/events?limit=10",
                ],
            ),
        ],
    )

    deployment = InstructionSection(
        title="Deployment Paths",
        steps=[
            InstructionStep(
                summary="Docker Compose deployment with optional Redis sidecar.",
                commands=[
                    "docker-compose up -d",
                    "docker-compose logs -f super-alita",
                    "curl http://localhost:8080/health",
                ],
            ),
            InstructionStep(
                summary="Production deployment via helper script.",
                commands=[
                    "./scripts/deploy.sh production",
                    "curl https://your-domain.com/health",
                ],
            ),
        ],
    )

    troubleshooting = InstructionSection(
        title="Troubleshooting",
        steps=[
            InstructionStep(
                summary="Resolve module import errors by fixing PYTHONPATH.",
                commands=["export PYTHONPATH=$PWD:$PYTHONPATH"],
            ),
            InstructionStep(
                summary="Handle port conflicts by selecting an alternate port.",
                commands=["python src/main.py --port 8081"],
            ),
            InstructionStep(
                summary="Fallback behavior when the LLM provider is unavailable.",
                expected=[
                    "System enters echo fallback mode.",
                    "Validate Ollama availability with curl http://127.0.0.1:11434/api/tags",
                ],
            ),
        ],
    )

    success = InstructionSection(
        title="Success Criteria",
        notes=[
            "Server boots cleanly and health endpoint reports \"healthy\".",
            "Chat endpoint responds to safe inputs and blocks dangerous actions.",
            "Chat history persists per session and logs remain structured JSON.",
            "Automated tests (unit, integration, performance) pass with coverage expectations met.",
            "All runtime components emit the expected telemetry events.",
        ],
    )

    return [
        quick_start,
        environment,
        health,
        api_tests,
        automated_tests,
        observability,
        advanced,
        deployment,
        troubleshooting,
        success,
    ]


def _serialize_sections(sections: Iterable[InstructionSection]) -> list[dict[str, Any]]:
    return [section.to_payload() for section in sections]


async def _validate_constitutionally(
    sections: Iterable[InstructionSection],
) -> dict[str, Any]:
    """Validate the instructions using the constitutional reasoner."""

    reasoner = ConstitutionalReasoner()
    instructions_text = []
    for section in sections:
        instructions_text.append(section.title)
        instructions_text.extend(section.notes)
        for step in section.steps:
            instructions_text.append(step.summary)
            instructions_text.extend(step.commands)
            instructions_text.extend(step.expected)
    joined = "\n".join(instructions_text)

    try:
        approved, reasoning = await reasoner.evaluate_action(
            action={
                "ability": "runtime.provide_instructions",
                "args": {"instructions": joined},
            },
            context={"goal": "bootstrap_super_alita_v4", "version": "4.0.0"},
        )
        return {"approved": approved, "reasoning": reasoning}
    except Exception as exc:  # pragma: no cover - defensive guardrail
        return {"approved": False, "reasoning": f"validation_failed: {exc}"}


async def build_super_alita_v4_instruction_payload() -> dict[str, Any]:
    """Return structured instructions and constitutional validation metadata."""

    sections = _gather_sections()
    validation = await _validate_constitutionally(sections)
    return {
        "profile": {
            "name": "Super Alita Runtime",
            "version": "4.0.0",
            "scope": "local_agent_operations",
            "source": "maintainer_briefing",
        },
        "sections": _serialize_sections(sections),
        "validation": validation,
    }


__all__ = ["InstructionSection", "InstructionStep", "build_super_alita_v4_instruction_payload"]

