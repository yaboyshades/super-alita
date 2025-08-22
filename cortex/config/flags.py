from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    try:
        return int(v) if v is not None else default
    except Exception:
        return default


def _env_float(name: str, default: float) -> float:
    v = os.getenv(name)
    try:
        return float(v) if v is not None else default
    except Exception:
        return default


@dataclass(frozen=True)
class PlannerFlags:
    """Feature flags controlling planner behavior."""

    use_ladder_router: bool = _env_bool("CORTEX_USE_LADDER_ROUTER", True)
    ladder_llm_decompose: bool = _env_bool("CORTEX_LADDER_LLM_DECOMPOSE", False)
    ladder_shadow_mode: bool = _env_bool("CORTEX_LADDER_SHADOW_MODE", True)
    ladder_active_inference: bool = _env_bool(
        "CORTEX_LADDER_ACTIVE_INFERENCE", False
    )


@dataclass(frozen=True)
class LeanRAGFlags:
    """Flags for hierarchical KG aggregation and LCA retrieval."""

    enable: bool = _env_bool("CORTEX_LEANRAG_ENABLE", True)
    max_depth: int = _env_int("CORTEX_LEANRAG_MAX_DEPTH", 3)
    min_cluster_size: int = _env_int("CORTEX_LEANRAG_MIN_CLUSTER", 8)
    gmm_k_init: int = _env_int("CORTEX_LEANRAG_GMM_K", 8)
    link_threshold: float = _env_float("CORTEX_LEANRAG_LINK_THRESH", 0.08)
    seeds_k: int = _env_int("CORTEX_LEANRAG_SEEDS_K", 5)
    brief_max_nodes: int = _env_int("CORTEX_LEANRAG_BRIEF_MAXN", 64)


@dataclass(frozen=True)
class PromptFlags:
    """Prompt building & reminder injection."""

    enable_prompt_builder: bool = _env_bool("CORTEX_PROMPT_ENABLE", True)
    enable_reminders: bool = _env_bool("CORTEX_PROMPT_REMINDERS", True)
    max_history_tokens: int = _env_int("CORTEX_PROMPT_MAX_HISTORY_TOKENS", 6000)
    max_context_chars: int = _env_int("CORTEX_PROMPT_MAX_CONTEXT_CHARS", 35000)
    reinforce_keywords: tuple = field(
        default=("IMPORTANT", "VERY IMPORTANT", "NEVER", "ALWAYS")
    )
    # When to inject JIT reminders
    remind_if_todo_empty: bool = _env_bool("CORTEX_REMIND_TODO_EMPTY", True)
    remind_if_no_tools_used_steps: int = _env_int("CORTEX_REMIND_NO_TOOLS_STEPS", 5)
    remind_if_stalled_secs: int = _env_int("CORTEX_REMIND_STALLED_SECS", 120)


@dataclass(frozen=True)
class SubAgentFlags:
    """Sub-agent orchestration settings."""

    enable_subagents: bool = _env_bool("CORTEX_SUBAGENTS_ENABLE", True)
    summary_max_chars: int = _env_int("CORTEX_SUBAGENTS_SUMMARY_MAX_CHARS", 8000)
    inherit_history: bool = _env_bool(
        "CORTEX_SUBAGENTS_INHERIT_HISTORY", False
    )  # must remain False for isolation
    max_steps: int = _env_int("CORTEX_SUBAGENTS_MAX_STEPS", 12)
    allow_tools: bool = _env_bool("CORTEX_SUBAGENTS_ALLOW_TOOLS", True)
    # Optional safety fence
    deny_shell: bool = _env_bool("CORTEX_SUBAGENTS_DENY_SHELL", True)
    deny_net: bool = _env_bool("CORTEX_SUBAGENTS_DENY_NET", True)


FLAGS = PlannerFlags()
LEANRAG = LeanRAGFlags()
PROMPT = PromptFlags()
SUBAGENT = SubAgentFlags()
