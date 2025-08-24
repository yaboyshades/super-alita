#!/usr/bin/env python3
from __future__ import annotations
import os
from dataclasses import dataclass, field
from typing import List, Dict, Any

# ---- Version stamps & required fields ---------------------------------------
PROMPT_VERSION = os.getenv("PROMPT_VERSION", "2.1.0")
SCHEMA_VERSION = os.getenv("SCHEMA_VERSION", "alg_extraction_v1.2")

REQUIRED_FIELDS_ALG = [
    "algorithms", "components", "hyperparameters", "equations",
    "implementation_plan", "sources"
]

# Deterministic ordering rules help diff-based CI/regression validation
# (and align with “deterministic” engineering discipline) .
ORDERING_POLICY = {
    "algorithms": "appearance_order",  # or "alphabetical"
    "components": "appearance_order",
    "hyperparameters": "alphabetical"
}

# Keyword profiles are centralized to cut duplication/ drift (DRY) .
KEYWORD_PROFILES: Dict[str, List[str]] = {
    "algorithm_extraction": ["method", "model", "architecture", "training", "evaluation", "ablation", "discussion", "analysis", "result"],
    "concept_analysis": ["definition", "intuition", "limitation", "future work"],
}

# Compression directive thresholds (readability vs completeness)
MAX_HPARAMS_FULL_LIST = int(os.getenv("MAX_HPARAMS_FULL_LIST", "100"))

@dataclass
class RetrievalFallbackPolicy:
    condition: str = 'segments_returned < requested OR total_chars < 4000'
    broaden_keywords: List[str] = field(default_factory=lambda: ["training","evaluation","ablation","result","discussion","analysis"])
    max_rounds: int = 2
    per_round_segment_cap: int = int(os.getenv("RETRIEVAL_SEGMENT_CAP", "8"))

@dataclass
class ExecutionGuards:
    # Token/segment ceilings per phase to prevent runaway loops
    segment_pull_ceiling: int = int(os.getenv("SEGMENT_PULL_CEILING", "24"))
    # Minimal mode yields diff-friendly outputs (omit narratives)
    minimal_mode: bool = os.getenv("CODEGEN_MINIMAL", os.getenv("DEEPCODE_MINIMAL","0")) not in ("0", "", "false", "False")
    # Pagination/read ledger is always on to avoid skipping later chunks.
    track_read_ledger: bool = True

BASE_POLICY_BLOCK = lambda: f"""
SYSTEM_POLICY:
  prompt_version: {PROMPT_VERSION}
  schema_version: {SCHEMA_VERSION}
  retrieval_mode: segmented
  fallback_policy:
    condition: "segments_returned < requested OR total_chars < 4000"
    action: "broaden keywords and re-run once"
  unknown_field_policy: "Use literal 'UNKNOWN' and add to missing_fields[] with reason"
  math_fidelity:
    latex: "Preserve LaTeX verbatim within fenced math blocks"
    ordering: "Do not reorder fractions/equations; preserve source ordering"
  pagination:
    enforce_read_ledger: true
  repository_download_guard:
    integrity_checks: ["non_empty_dir","has_README","has_build_descriptor","has_license"]
    status_on_missing: "partial"
  path_containment:
    write_root: "./deepcode_lab/"
    disallow_parent_escape: true
  output_namespacing:
    root_key: "alg_extraction_v1"
  ordering:
    algorithms: "{ORDERING_POLICY['algorithms']}"
    components: "{ORDERING_POLICY['components']}"
    hyperparameters: "{ORDERING_POLICY['hyperparameters']}"
  completeness_target:
    min_algorithms: 1
    min_equations: 3
    min_components: 3
  ceiling_controls:
    segment_pull_ceiling: {ExecutionGuards().segment_pull_ceiling}
"""

def build_role_extension(role: str) -> str:
    if role == "algorithm_extraction":
        keywords = ", ".join(KEYWORD_PROFILES["algorithm_extraction"])
        return f"""
ROLE_EXTENSION:
  name: algorithm_extraction
  keyword_profile: [{keywords}]
  dedup_policy: "de-duplicate identical hyperparameters across tables; record duplicates_found[] with merge rationale"
  large_list_compression:
    max_hparams_full_list: {MAX_HPARAMS_FULL_LIST}
    also_include_stats: ["count","categories"]
"""
    return f"ROLE_EXTENSION:\n  name: {role}\n"

def build_validation_footer() -> str:
    # Downstream CI reads this; model must fill missing/unknown arrays.
    return """
VALIDATION_AND_TELEMETRY_REQUIREMENTS:
  include_schema_echo_block: true
  include_validation_summary: true   # {missing_required_fields:[], unknown_fields:[]}
  include_confidence_and_sources: true
  include_telemetry_envelope: true   # {prompt_version, extraction_duration_ms, segments_used, retrieval_rounds}
  include_ambiguity_register: true
  include_dependency_graph: true
  include_repro_hooks: true          # deterministic_elements, non_deterministic_risks
  include_plan_traceability: true
"""

def build_header(role: str) -> str:
    return BASE_POLICY_BLOCK() + "\n" + build_role_extension(role) + "\n" + build_validation_footer()
