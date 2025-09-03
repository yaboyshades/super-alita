"""Task planning ability.

Lightweight decomposition of a user objective into a small ordered list of
atomic, tool-oriented steps. Designed to be dependency-light and resilient:

Strategy:
1. If an OpenAI API key is available (env OPENAI_API_KEY) attempt a structured
   JSON plan using a constrained prompt.
2. Parse JSON strictly; on any failure fall back to heuristic planning.
3. Heuristic planning performs simple clause extraction and action verb
   normalization to produce up to max_steps steps.

External Contract (registered as tool_id = "task_planner"):
Input:
    {
        "prompt": str,
        "max_steps": int (default 6)
    }
Output:
  {
    "steps": [ {"id": int, "action": str, "rationale": str }, ...],
    "summary": str,
    "source": "llm" | "heuristic" | "fallback"
  }

This module intentionally avoids pulling in heavy agent frameworks so it can be
swapped later without changing external tool schemas.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

_ACTION_VERBS = [
    "analyze",
    "research",
    "collect",
    "design",
    "plan",
    "implement",
    "test",
    "validate",
    "summarize",
    "document",
    "review",
]


def _clean_text(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip())


def _heuristic_steps(prompt: str, max_steps: int) -> list[dict[str, object]]:
    # Split on punctuation or newlines into candidate clauses
    raw_clauses = re.split(r"[\n.;]+", prompt)
    clauses = [c.strip() for c in raw_clauses if c.strip()]
    if not clauses:
        clauses = [prompt]

    # If only one clause, attempt to synthesize sub-actions via verbs
    if len(clauses) == 1:
        clause = clauses[0]
        generated: list[str] = []
        for verb in _ACTION_VERBS:
            if verb in clause.lower():
                generated.append(f"{verb} details")
        if not generated:
            # Generic scaffold
            generated = [
                "clarify requirements",
                "identify resources",
                "draft solution",
                "review & refine",
            ]
        clauses = generated

    steps: list[dict[str, object]] = []
    for idx, clause in enumerate(clauses[:max_steps]):
        action = _clean_text(clause)
        if not action:
            continue
        # Ensure action starts with a verb-ish token
        token = action.split()[0].lower()
        if token not in _ACTION_VERBS:
            # prepend an inferred verb
            action = f"execute {action}"
        steps.append(
            {
                "id": idx + 1,
                "action": action,
                "rationale": f"Step derives from clause {idx + 1} of prompt",
            }
        )
    if not steps:
        steps = [
            {
                "id": 1,
                "action": _clean_text(prompt) or "perform task",
                "rationale": "Fallback single step",
            }
        ]
    return steps


def _call_openai_structured(prompt: str, max_steps: int) -> dict[str, Any]:
    """Attempt an OpenAI call returning a structured JSON plan.

    Raises on any error so caller can fall back.
    """

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")

    from openai import OpenAI  # type: ignore

    client = OpenAI(api_key=api_key)
    planning_prompt = (
        "You are a precise planning assistant. Decompose the objective into "
        f"at most {max_steps} atomic, execution-oriented steps. "
        "Each step must be JSON-ready and concise.\n"
        f"Objective: {prompt}\n"
        "Return ONLY minified JSON: {'steps':[{'id':1,'action':'a',"
        "'rationale':'r'}],'summary':'...'}"
    )

    resp = client.chat.completions.create(
        model=os.getenv("PLANNER_MODEL", "gpt-4o-mini"),
        messages=[{"role": "user", "content": planning_prompt}],
        temperature=0.2,
        max_tokens=600,
    )
    content = resp.choices[0].message.content  # type: ignore[attr-defined]
    if not content:
        raise RuntimeError("Empty LLM response")
    # Strip code fences if present
    content = re.sub(r"^```json|```$", "", content.strip(), flags=re.MULTILINE)
    plan = json.loads(content)
    if not isinstance(plan, dict) or "steps" not in plan:
        raise ValueError("Malformed plan JSON")
    return plan


def plan_task(prompt: str, max_steps: int = 6) -> dict[str, Any]:
    """Public planning function used by tool executor.

    Always returns a well-formed dictionary adhering to contract.
    """

    prompt = prompt.strip()
    if not prompt:
        return {
            "steps": [
                {
                    "id": 1,
                    "action": "clarify objective",
                    "rationale": "No objective provided",
                }
            ],
            "summary": "No objective provided",
            "source": "fallback",
        }

    # Try LLM path first
    if os.getenv("OPENAI_API_KEY"):
        try:
            plan = _call_openai_structured(prompt, max_steps)
            # Basic normalization
            steps_field = plan.get("steps")
            if not isinstance(steps_field, list):  # pragma: no cover
                raise ValueError("steps not list")
            steps: list[dict[str, Any]] = []
            for i, raw in enumerate(steps_field[:max_steps]):
                if not isinstance(raw, dict):
                    continue
                action = _clean_text(str(raw.get("action", "")))
                if not action:
                    continue
                rationale = _clean_text(str(raw.get("rationale", ""))) or ""
                steps.append({"id": i + 1, "action": action, "rationale": rationale})
            if not steps:
                raise ValueError("No valid steps parsed")
            summary = _clean_text(str(plan.get("summary", ""))) or action[:80]
            return {"steps": steps, "summary": summary, "source": "llm"}
        except Exception:  # noqa: BLE001
            # Fall through to heuristic
            pass

    steps = _heuristic_steps(prompt, max_steps)
    summary = _clean_text(steps[0]["action"]) if steps else prompt[:80]
    return {"steps": steps, "summary": summary, "source": "heuristic"}


__all__ = ["plan_task"]
