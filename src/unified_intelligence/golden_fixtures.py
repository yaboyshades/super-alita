"""
Golden Test Fixtures for Unified Intelligence Layer

Contains sample payloads and expected outputs for testing
the hardened orchestrator. These fixtures validate the
contract compliance, score fusion math, and decision logic.
"""

import json
from typing import Any

# Sample Request Payloads
GOLDEN_REQUESTS = {
    "new_feature_json_export": {
        "request_id": "req_9f3",
        "ts": "2025-09-18T14:05:00Z",
        "intent_text": "Create a new feature to add JSON export " "to the analyzer.",
        "code_refs": ["src/analyzer/__init__.py", "tests/test_analyzer.py"],
        "context": {},
    },
    "refactor_complex_function": {
        "request_id": "req_abc",
        "ts": "2025-09-18T15:30:00Z",
        "intent_text": "Refactor the complex data processing function "
        "to improve readability.",
        "code_refs": ["src/processor/data_processor.py"],
        "context": {"complexity_threshold": 10},
    },
    "debug_api_issue": {
        "request_id": "req_xyz",
        "ts": "2025-09-18T16:45:00Z",
        "intent_text": "Debug the API endpoint that's returning 500 errors.",
        "code_refs": ["src/api/endpoints.py", "tests/test_api.py"],
        "context": {"error_code": 500},
    },
}


# Expected Component Results
GOLDEN_WORKFLOW_RESULTS = {
    "new_feature_json_export": {
        "label": "new_feature",
        "confidence": 0.82,
        "features": ["imperative", "feature_verb"],
        "errors": [],
    },
    "refactor_complex_function": {
        "label": "refactor",
        "confidence": 0.75,
        "features": ["refactor_verb", "complexity_mention"],
        "errors": [],
    },
    "debug_api_issue": {
        "label": "debug",
        "confidence": 0.88,
        "features": ["debug_verb", "error_mention"],
        "errors": [],
    },
}


GOLDEN_MANGLE_RESULTS = {
    "new_feature_json_export": {
        "ok": True,
        "facts": [
            {"type": "function", "name": "analyze", "tests": 0},
            {"type": "class", "name": "Analyzer", "methods": 5},
        ],
        "metrics": {"complexity": 0.62, "coverage_gap": 0.7},
        "findings": [
            {"severity": "med", "note": "Analyzer lacks export boundary tests"}
        ],
        "confidence": 0.77,
        "errors": [],
    },
    "refactor_complex_function": {
        "ok": True,
        "facts": [{"type": "function", "name": "process_data", "complexity": 15}],
        "metrics": {"complexity": 0.85, "coverage_gap": 0.3},
        "findings": [
            {"severity": "high", "note": "Function exceeds " "complexity threshold"}
        ],
        "confidence": 0.82,
        "errors": [],
    },
    "debug_api_issue": {
        "ok": True,
        "facts": [{"type": "function", "name": "api_endpoint", "error_paths": 3}],
        "metrics": {"complexity": 0.45, "coverage_gap": 0.8},
        "findings": [
            {"severity": "high", "note": "Missing error handling " "in API endpoint"}
        ],
        "confidence": 0.79,
        "errors": [],
    },
}


GOLDEN_CONSTITUTION_RESULTS = {
    "new_feature_json_export": {
        "ok": True,
        "article_scores": {
            "library_first": 0.9,
            "test_first": 0.55,
            "simplicity_gate": 0.8,
            "integration_first": 0.7,
            "clarity_unambiguity": 0.85,
            "counterfactual_justification": 0.75,
            "documentation_driven": 0.6,
            "template_driven": 0.65,
            "cli_interface": 0.8,
        },
        "overall": 0.68,
        "infractions": [
            {"article": "test_first", "severity": "med", "note": "No pretests"}
        ],
        "confidence": 0.83,
        "errors": [],
    },
    "refactor_complex_function": {
        "ok": True,
        "article_scores": {
            "library_first": 0.8,
            "test_first": 0.7,
            "simplicity_gate": 0.4,
            "integration_first": 0.75,
            "clarity_unambiguity": 0.8,
            "counterfactual_justification": 0.6,
            "documentation_driven": 0.7,
            "template_driven": 0.55,
            "cli_interface": 0.8,
        },
        "overall": 0.62,
        "infractions": [
            {
                "article": "simplicity_gate",
                "severity": "high",
                "note": "High complexity",
            }
        ],
        "confidence": 0.85,
        "errors": [],
    },
    "debug_api_issue": {
        "ok": True,
        "article_scores": {
            "library_first": 0.85,
            "test_first": 0.6,
            "simplicity_gate": 0.75,
            "integration_first": 0.8,
            "clarity_unambiguity": 0.7,
            "counterfactual_justification": 0.8,
            "documentation_driven": 0.75,
            "template_driven": 0.7,
            "cli_interface": 0.8,
        },
        "overall": 0.71,
        "infractions": [
            {
                "article": "test_first",
                "severity": "med",
                "note": "Limited error path tests",
            }
        ],
        "confidence": 0.87,
        "errors": [],
    },
}


GOLDEN_COPILOT_RESULTS = {
    "new_feature_json_export": {
        "ok": True,
        "templates_applied": ["feature_spec_template"],
        "guidance": [
            {"section": "requirements", "text": "Define clear acceptance criteria"},
            {"section": "testing", "text": "Plan test strategy before implementation"},
        ],
        "errors": [],
    },
    "refactor_complex_function": {
        "ok": True,
        "templates_applied": ["refactor_template"],
        "guidance": [
            {"section": "simplicity", "text": "Break down complex function"},
            {"section": "testing", "text": "Ensure tests cover refactored code"},
        ],
        "errors": [],
    },
    "debug_api_issue": {
        "ok": True,
        "templates_applied": ["debug_template"],
        "guidance": [
            {"section": "error_handling", "text": "Add comprehensive error handling"},
            {"section": "testing", "text": "Add tests for error scenarios"},
        ],
        "errors": [],
    },
}


# Expected Fused Results (with FusionConfig defaults)
GOLDEN_FUSED_RESULTS = {
    "new_feature_json_export": {
        "ok": True,
        "decision": "revise",
        "reasons": [
            "Test-first article below threshold",
            "Coverage gap is high; add export boundary tests",
        ],
        "recommendations": [
            {
                "action": "Add tests for JSON export boundary",
                "rationale": "Raise test_first & reduce coverage_gap",
                "refs": ["tests/test_export_json.py"],
            },
            {
                "action": "Introduce small adapter instead of " "custom serializer",
                "rationale": "Library-first improves simplicity_gate",
                "refs": [],
            },
        ],
        "scores": {
            "fused": 0.66,
            "contributors": {"mangle": 0.63, "constitution": 0.75, "workflow": 0.82},
        },
        "telemetry": {
            "request_id": "req_9f3",
            "components": ["workflow", "mangle", "constitution", "copilot"],
            "timings_ms": {
                "total": 214,
                "mangle": 92,
                "constitution": 61,
                "copilot": 45,
                "workflow": 16,
            },
            "version": "1",
        },
        "errors": [],
    },
    "refactor_complex_function": {
        "ok": True,
        "decision": "revise",
        "reasons": [
            "Simplicity gate article below threshold",
            "Function exceeds complexity threshold",
        ],
        "recommendations": [
            {
                "action": "Break down complex data processing function",
                "rationale": "Improve simplicity_gate score",
                "refs": ["src/processor/data_processor.py"],
            }
        ],
        "scores": {
            "fused": 0.58,
            "contributors": {"mangle": 0.72, "constitution": 0.65, "workflow": 0.75},
        },
        "telemetry": {
            "request_id": "req_abc",
            "components": ["workflow", "mangle", "constitution", "copilot"],
            "timings_ms": {
                "total": 198,
                "mangle": 85,
                "constitution": 58,
                "copilot": 42,
                "workflow": 13,
            },
            "version": "1",
        },
        "errors": [],
    },
    "debug_api_issue": {
        "ok": True,
        "decision": "proceed",
        "reasons": ["Analysis completed successfully"],
        "recommendations": [
            {
                "action": "Add comprehensive error handling to API endpoint",
                "rationale": "Improve reliability and test coverage",
                "refs": ["src/api/endpoints.py"],
            }
        ],
        "scores": {
            "fused": 0.72,
            "contributors": {"mangle": 0.68, "constitution": 0.78, "workflow": 0.88},
        },
        "telemetry": {
            "request_id": "req_xyz",
            "components": ["workflow", "mangle", "constitution", "copilot"],
            "timings_ms": {
                "total": 205,
                "mangle": 88,
                "constitution": 59,
                "copilot": 46,
                "workflow": 12,
            },
            "version": "1",
        },
        "errors": [],
    },
}


# Code Analysis Results
GOLDEN_CODE_ANALYSIS_RESULTS = {
    "new_feature_json_export": {
        "ok": True,
        "repo_path": ".",
        "total_files": 12,
        "total_symbols": 45,
        "findings": {
            "untested_function": [
                {
                    "rule_name": "untested_function",
                    "symbol": "src.analyzer.analyze",
                    "file": "src/analyzer/__init__.py",
                    "complexity": 0.6,
                }
            ],
            "orphan_complex": [],
            "cycle": [],
            "hot_path": [],
            "reinvention_json": [],
        },
        "summary": {
            "untested_function": 1,
            "orphan_complex": 0,
            "cycle": 0,
            "hot_path": 0,
            "reinvention_json": 0,
        },
        "analysis_time": 0.15,
        "confidence": 0.75,
        "errors": [],
    },
    "refactor_complex_function": {
        "ok": True,
        "repo_path": ".",
        "total_files": 8,
        "total_symbols": 32,
        "findings": {
            "untested_function": [],
            "orphan_complex": [
                {
                    "rule_name": "orphan_complex",
                    "symbol": "src.processor.data_processor.process_data",
                    "file": "src/processor/data_processor.py",
                    "complexity": 0.8,
                }
            ],
            "cycle": [],
            "hot_path": [
                {
                    "rule_name": "hot_path",
                    "symbol": "src.processor.data_processor.process_data",
                    "file": "src/processor/data_processor.py",
                    "complexity": 0.8,
                    "indegree": 5,
                }
            ],
            "reinvention_json": [],
        },
        "summary": {
            "untested_function": 0,
            "orphan_complex": 1,
            "cycle": 0,
            "hot_path": 1,
            "reinvention_json": 0,
        },
        "analysis_time": 0.12,
        "confidence": 0.82,
        "errors": [],
    },
    "debug_api_issue": {
        "ok": True,
        "repo_path": ".",
        "total_files": 15,
        "total_symbols": 67,
        "findings": {
            "untested_function": [
                {
                    "rule_name": "untested_function",
                    "symbol": "src.api.endpoints.handle_request",
                    "file": "src/api/endpoints.py",
                    "complexity": 0.5,
                }
            ],
            "orphan_complex": [],
            "cycle": [],
            "hot_path": [],
            "reinvention_json": [],
        },
        "summary": {
            "untested_function": 1,
            "orphan_complex": 0,
            "cycle": 0,
            "hot_path": 0,
            "reinvention_json": 0,
        },
        "analysis_time": 0.18,
        "confidence": 0.78,
        "errors": [],
    },
}


# Expected Results with Code Analysis Fusion
GOLDEN_FUSED_RESULTS_WITH_ANALYSIS = {
    "new_feature_json_export": {
        "ok": True,
        "decision": "revise",
        "reasons": [
            "Test-first article below threshold",
            "Coverage gap is high; add export boundary tests",
            "Untested function detected in analyzer",
        ],
        "recommendations": [
            {
                "action": "Add tests for JSON export boundary",
                "rationale": "Raise test_first & reduce coverage_gap",
                "refs": ["tests/test_export_json.py"],
            },
            {
                "action": "Introduce small adapter instead of " "custom serializer",
                "rationale": "Library-first improves simplicity_gate",
                "refs": [],
            },
            {
                "action": "Test the 'analyze' function in the analyzer",
                "rationale": "Increase coverage for untested functions",
                "refs": ["src/analyzer/__init__.py"],
            },
        ],
        "scores": {
            "fused": 0.64,
            "contributors": {"mangle": 0.63, "constitution": 0.75, "workflow": 0.82},
        },
        "telemetry": {
            "request_id": "req_9f3",
            "components": ["workflow", "mangle", "constitution", "copilot"],
            "timings_ms": {
                "total": 214,
                "mangle": 92,
                "constitution": 61,
                "copilot": 45,
                "workflow": 16,
            },
            "version": "1",
        },
        "errors": [],
    },
    "refactor_complex_function": {
        "ok": True,
        "decision": "revise",
        "reasons": [
            "Simplicity gate article below threshold",
            "Function exceeds complexity threshold",
            "Orphan complex detected in data processor",
            "Hot path detected in data processor",
        ],
        "recommendations": [
            {
                "action": "Break down complex data processing function",
                "rationale": "Improve simplicity_gate score",
                "refs": ["src/processor/data_processor.py"],
            }
        ],
        "scores": {
            "fused": 0.57,
            "contributors": {"mangle": 0.72, "constitution": 0.65, "workflow": 0.75},
        },
        "telemetry": {
            "request_id": "req_abc",
            "components": ["workflow", "mangle", "constitution", "copilot"],
            "timings_ms": {
                "total": 198,
                "mangle": 85,
                "constitution": 58,
                "copilot": 42,
                "workflow": 13,
            },
            "version": "1",
        },
        "errors": [],
    },
    "debug_api_issue": {
        "ok": True,
        "decision": "proceed",
        "reasons": ["Analysis completed successfully"],
        "recommendations": [
            {
                "action": "Add comprehensive error handling to API endpoint",
                "rationale": "Improve reliability and test coverage",
                "refs": ["src/api/endpoints.py"],
            }
        ],
        "scores": {
            "fused": 0.71,
            "contributors": {"mangle": 0.68, "constitution": 0.78, "workflow": 0.88},
        },
        "telemetry": {
            "request_id": "req_xyz",
            "components": ["workflow", "mangle", "constitution", "copilot"],
            "timings_ms": {
                "total": 205,
                "mangle": 88,
                "constitution": 59,
                "copilot": 46,
                "workflow": 12,
            },
            "version": "1",
        },
        "errors": [],
    },
}


# Chaos Test Cases (component failures)
CHAOS_TESTS = {
    "mangle_unavailable": {
        "request": GOLDEN_REQUESTS["new_feature_json_export"],
        "component_failures": ["mangle"],
        "expected_decision": "revise",  # Should still work but with
        # reduced confidence
        "expected_fused_range": [0.55, 0.75],
    },
    "constitution_low_confidence": {
        "request": GOLDEN_REQUESTS["refactor_complex_function"],
        "constitution_confidence": 0.3,  # Below threshold
        "expected_decision": "revise",
        "expected_reasons_contain": ["Constitutional infractions"],
    },
    "workflow_unclear": {
        "request": {
            "request_id": "req_unclear",
            "ts": "2025-09-18T17:00:00Z",
            "intent_text": "I need to do something with the code",
            "code_refs": [],
            "context": {},
        },
        "expected_decision": "revise",
        "expected_workflow_confidence": 0.0,
    },
}


def load_golden_fixtures() -> dict[str, Any]:
    """Load all golden fixtures for validation."""
    return {
        "version": "1",
        "requests": GOLDEN_REQUESTS,
        "workflow_results": GOLDEN_WORKFLOW_RESULTS,
        "mangle_results": GOLDEN_MANGLE_RESULTS,
        "constitution_results": GOLDEN_CONSTITUTION_RESULTS,
        "code_analysis_results": GOLDEN_CODE_ANALYSIS_RESULTS,
        "copilot_results": GOLDEN_COPILOT_RESULTS,
        "fused_results": GOLDEN_FUSED_RESULTS,
        "chaos_tests": CHAOS_TESTS,
    }


def get_golden_fixture(scenario: str) -> dict[str, Any]:
    """Get complete golden fixture for a test scenario."""
    if scenario not in GOLDEN_REQUESTS:
        raise ValueError(f"Unknown scenario: {scenario}")

    return {
        "request": GOLDEN_REQUESTS[scenario],
        "workflow": GOLDEN_WORKFLOW_RESULTS[scenario],
        "mangle": GOLDEN_MANGLE_RESULTS[scenario],
        "constitution": GOLDEN_CONSTITUTION_RESULTS[scenario],
        "code_analysis": GOLDEN_CODE_ANALYSIS_RESULTS[scenario],
        "copilot": GOLDEN_COPILOT_RESULTS[scenario],
        "expected_advice": GOLDEN_FUSED_RESULTS[scenario],
    }


def save_fixtures_to_file(filepath: str):
    """Save all fixtures to a JSON file for external testing."""
    fixtures = {
        "version": "1",
        "requests": GOLDEN_REQUESTS,
        "workflow_results": GOLDEN_WORKFLOW_RESULTS,
        "mangle_results": GOLDEN_MANGLE_RESULTS,
        "constitution_results": GOLDEN_CONSTITUTION_RESULTS,
        "code_analysis_results": GOLDEN_CODE_ANALYSIS_RESULTS,
        "copilot_results": GOLDEN_COPILOT_RESULTS,
        "fused_results": GOLDEN_FUSED_RESULTS,
        "chaos_tests": CHAOS_TESTS,
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(fixtures, f, indent=2)


if __name__ == "__main__":
    # Save fixtures for testing
    save_fixtures_to_file("src/unified_intelligence/test_fixtures.json")
    print("Golden fixtures saved to test_fixtures.json")
