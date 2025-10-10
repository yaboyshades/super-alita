#!/usr/bin/env python3
"""
SARIF emitter for Spec Kit + CDIL
Generates SARIF reports for API differences and contract violations
"""
import json
import sys
from datetime import datetime
from typing import Any


def create_sarif_report(
    spec_differences: list[dict[str, Any]],
    contract_violations: list[dict[str, Any]],
    signature_changes: list[dict[str, Any]],
    semver_issues: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Create a SARIF report from spec differences and contract violations.
    """
    # SARIF template
    sarif = {
        "$schema": (
            "https://raw.githubusercontent.com/oasis-tcs/sarif-spec/"
            "master/Schemata/sarif-schema-2.1.0.json"
        ),
        "version": "2.1.0",
        "runs": [
            {
                "tool": {
                    "driver": {
                        "name": "SpecKit+CDIL",
                        "informationUri": (
                            "https://github.com/super-alita/" "spec-kit-cdil"
                        ),
                        "version": "1.0.0",
                        "rules": [
                            {
                                "id": "api-drift/breaking",
                                "name": "BreakingAPIDrift",
                                "shortDescription": {
                                    "text": (
                                        "Breaking API change detected without "
                                        "major version bump"
                                    )
                                },
                                "fullDescription": {
                                    "text": (
                                        "A breaking API change was detected "
                                        "but the specification version was "
                                        "not bumped to a major version."
                                    )
                                },
                                "defaultConfiguration": {"level": "error"},
                                "helpUri": (
                                    "https://github.com/super-alita/"
                                    "spec-kit-cdil#semver-policy"
                                ),
                            },
                            {
                                "id": "api-drift/additive",
                                "name": "AdditiveAPIDrift",
                                "shortDescription": {
                                    "text": (
                                        "Additive API change detected without "
                                        "minor version bump"
                                    )
                                },
                                "fullDescription": {
                                    "text": (
                                        "An additive API change was detected "
                                        "but the specification version was "
                                        "not bumped to a minor version."
                                    )
                                },
                                "defaultConfiguration": {"level": "warning"},
                                "helpUri": (
                                    "https://github.com/super-alita/"
                                    "spec-kit-cdil#semver-policy"
                                ),
                            },
                            {
                                "id": "contract/postcondition-fail",
                                "name": "PostconditionViolation",
                                "shortDescription": {
                                    "text": ("Function postcondition violated")
                                },
                                "fullDescription": {
                                    "text": (
                                        "The implementation does not satisfy "
                                        "the postconditions defined in the "
                                        "specification."
                                    )
                                },
                                "defaultConfiguration": {"level": "error"},
                                "helpUri": (
                                    "https://github.com/super-alita/"
                                    "spec-kit-cdil#contract-enforcement"
                                ),
                            },
                            {
                                "id": "contract/missing-generator",
                                "name": "MissingGenerator",
                                "shortDescription": {
                                    "text": (
                                        "Specification missing generator "
                                        "mapping"
                                    )
                                },
                                "fullDescription": {
                                    "text": (
                                        "The specification is missing "
                                        "generator mappings for composite "
                                        "types used in parameters or return "
                                        "values."
                                    )
                                },
                                "defaultConfiguration": {"level": "warning"},
                                "helpUri": (
                                    "https://github.com/super-alita/"
                                    "spec-kit-cdil#spec-linting"
                                ),
                            },
                        ],
                    }
                },
                "results": [],
                "invocations": [
                    {
                        "executionSuccessful": True,
                        "startTimeUtc": datetime.utcnow().isoformat() + "Z",
                    }
                ],
            }
        ],
    }

    # Add spec differences (breaking changes)
    for diff in spec_differences:
        desc = diff.get("description", "unspecified change")
        sarif["runs"][0]["results"].append(
            {
                "ruleId": "api-drift/breaking",
                "level": "error",
                "message": {
                    "text": (
                        f"Specification {diff.get('spec_id', 'unknown')} has "
                        f"breaking change: {desc}"
                    )
                },
                "locations": [
                    {
                        "physicalLocation": {
                            "artifactLocation": {
                                "uri": diff.get("file", "unknown")
                            },
                            "region": {"startLine": diff.get("line", 1)},
                        }
                    }
                ],
                "properties": {
                    "specId": diff.get("spec_id", "unknown"),
                    "semverRequired": "major",
                    "doc": diff.get("doc", ""),
                },
            }
        )

    # Add signature changes
    for change in signature_changes:
        desc = change.get("description", "unspecified change")
        change_type = change.get("type", "additive")
        rule_id = (
            "api-drift/breaking"
            if change_type == "breaking"
            else "api-drift/additive"
        )
        level = "error" if change_type == "breaking" else "warning"
        semver_required = "major" if change_type == "breaking" else "minor"

        sarif["runs"][0]["results"].append(
            {
                "ruleId": rule_id,
                "level": level,
                "message": {
                    "text": (
                        f"API signature {change.get('function', 'unknown')} has "
                        f"{change_type} change: {desc}"
                    )
                },
                "locations": [
                    {
                        "physicalLocation": {
                            "artifactLocation": {
                                "uri": change.get("file", "unknown")
                            },
                            "region": {"startLine": change.get("line", 1)},
                        }
                    }
                ],
                "properties": {
                    "function": change.get("function", "unknown"),
                    "semverRequired": semver_required,
                    "doc": change.get("doc", ""),
                },
            }
        )

    # Add contract violations
    for violation in contract_violations:
        desc = violation.get("description", "unspecified violation")
        func = violation.get("function", "unknown")
        sarif["runs"][0]["results"].append(
            {
                "ruleId": "contract/postcondition-fail",
                "level": "error",
                "message": {"text": (f"Contract violation in {func}: {desc}")},
                "locations": [
                    {
                        "physicalLocation": {
                            "artifactLocation": {
                                "uri": violation.get("file", "unknown")
                            },
                            "region": {"startLine": violation.get("line", 1)},
                        }
                    }
                ],
                "properties": {
                    "function": func,
                    "expected": violation.get("expected", ""),
                    "actual": violation.get("actual", ""),
                    "doc": violation.get("doc", ""),
                },
            }
        )

    # Add semver issues
    for issue in semver_issues:
        desc = issue.get("description", "unspecified semver issue")
        rule_id = (
            "api-drift/breaking"
            if issue.get("severity") == "error"
            else "api-drift/additive"
        )
        sarif["runs"][0]["results"].append(
            {
                "ruleId": rule_id,
                "level": issue.get("severity", "warning"),
                "message": {"text": desc},
                "locations": [
                    {
                        "physicalLocation": {
                            "artifactLocation": {
                                "uri": issue.get("file", "unknown")
                            },
                            "region": {"startLine": issue.get("line", 1)},
                        }
                    }
                ],
                "properties": {
                    "specId": issue.get("spec_id", "unknown"),
                    "semverRequired": issue.get("required_bump", "patch"),
                    "semverActual": issue.get("actual_bump", "patch"),
                },
            }
        )

    return sarif


def generate_pr_comment(
    spec_differences: list[dict[str, Any]],
    contract_violations: list[dict[str, Any]],
    signature_changes: list[dict[str, Any]],
    semver_issues: list[dict[str, Any]],
) -> str:
    """
    Generate a PR comment summarizing the changes.
    """
    comment = [
        "## Spec Kit + CDIL Analysis Report",
        "",
        "### Summary of Changes",
        "",
    ]

    spec_count = len(spec_differences)
    sig_count = len(signature_changes)
    violation_count = len(contract_violations)
    semver_count = len(semver_issues)

    if spec_differences:
        comment.append(f"- **{spec_count}** specification changes detected")

    if signature_changes:
        comment.append(f"- **{sig_count}** API signature changes detected")

    if contract_violations:
        comment.append(f"- **{violation_count}** contract violations detected")

    if semver_issues:
        comment.append(f"- **{semver_count}** semver policy issues detected")

    no_changes = not any(
        [
            spec_differences,
            signature_changes,
            contract_violations,
            semver_issues,
        ]
    )

    if no_changes:
        comment.append("- No significant changes detected")
        return "\n".join(comment)

    comment.append("")

    # Add spec changes details
    if spec_differences:
        comment.extend(["### Specification Changes", ""])
        for diff in spec_differences:
            spec_id = diff.get("spec_id", "unknown")
            desc = diff.get("description", "unspecified change")
            comment.append(f"- `{spec_id}`: {desc}")
        comment.append("")

    # Add signature changes details
    if signature_changes:
        comment.extend(["### API Signature Changes", ""])
        for change in signature_changes:
            func = change.get("function", "unknown")
            desc = change.get("description", "unspecified change")
            change_type = change.get("type", "additive")
            comment.append(f"- `{func}` ({change_type}): {desc}")
        comment.append("")

    # Add contract violations details
    if contract_violations:
        comment.extend(["### Contract Violations", ""])
        for violation in contract_violations:
            func = violation.get("function", "unknown")
            desc = violation.get("description", "unspecified violation")
            comment.append(f"- `{func}`: {desc}")
        comment.append("")

    # Add semver issues details
    if semver_issues:
        comment.extend(["### SemVer Policy Issues", ""])
        for issue in semver_issues:
            spec_id = issue.get("spec_id", "unknown")
            desc = issue.get("description", "unspecified semver issue")
            severity = issue.get("severity", "warning")
            comment.append(f"- `{spec_id}` ({severity}): {desc}")
        comment.append("")

    # Add recommendations
    comment.extend(["### Recommendations", ""])

    if spec_differences or signature_changes:
        comment.append(
            "- Review specification changes and ensure they are intentional"
        )

    if signature_changes:
        comment.append("- Verify API changes align with specification updates")

    if contract_violations:
        comment.append("- Fix contract violations before merging")

    if semver_issues:
        comment.append("- Address SemVer policy issues")

    if any([spec_differences, signature_changes]):
        comment.append("- Consider if a SemVer bump is required")
        comment.append("- Update documentation if needed")

    comment.append("")
    comment.append("> Generated by Spec Kit + CDIL integration")

    return "\n".join(comment)


def main():
    """
    Main entry point for the SARIF emitter.
    """
    # In a real implementation, these would come from analysis tools
    spec_differences = []
    contract_violations = []
    signature_changes = []
    semver_issues = []

    # Parse command line arguments
    if len(sys.argv) > 1:
        # Try to load from JSON file
        try:
            with open(sys.argv[1]) as f:
                data = json.load(f)
                spec_differences = data.get("spec_differences", [])
                contract_violations = data.get("contract_violations", [])
                signature_changes = data.get("signature_changes", [])
                semver_issues = data.get("semver_issues", [])
        except Exception as e:
            print(f"Warning: Could not load data from {sys.argv[1]}: {e}")

    # Generate SARIF report
    sarif = create_sarif_report(
        spec_differences, contract_violations, signature_changes, semver_issues
    )

    # Output SARIF to stdout
    print(json.dumps(sarif, indent=2))

    # Generate PR comment
    comment = generate_pr_comment(
        spec_differences, contract_violations, signature_changes, semver_issues
    )

    # Output comment to a file with UTF-8 encoding
    with open("pr_comment.md", "w", encoding="utf-8") as f:
        f.write(comment)

    print("SARIF report generated to stdout")
    print("PR comment generated to pr_comment.md")


if __name__ == "__main__":
    main()
