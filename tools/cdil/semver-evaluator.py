#!/usr/bin/env python3
"""
SemVer evaluator for Spec Kit + CDIL
Analyzes API changes and recommends appropriate version bumps
"""
import json
import sys
from typing import Any


def analyze_api_changes(
    spec_diff: dict[str, Any],
    signature_changes: list[dict[str, Any]]
) -> tuple[str, str]:
    """
    Analyze API changes and determine required SemVer bump.
    
    Args:
        spec_diff: Specification differences
        signature_changes: API signature changes
        
    Returns:
        Tuple of (required_bump, reason)
    """
    # Check for breaking changes
    breaking_changes = []
    
    # Check signature changes for breaking changes
    for change in signature_changes:
        change_type = change.get('type', 'additive')
        if change_type == 'breaking':
            breaking_changes.append(change)
    
    if breaking_changes:
        count = len(breaking_changes)
        return 'major', f"Breaking changes detected: {count} breaking changes"
    
    # Check for additive changes
    additive_changes = []
    for change in signature_changes:
        change_type = change.get('type', 'additive')
        if change_type == 'additive':
            additive_changes.append(change)
    
    if additive_changes:
        count = len(additive_changes)
        return 'minor', f"Additive changes detected: {count} new APIs"
    
    # Check for documentation/generator changes
    has_doc_changes = (
        spec_diff.get('description_changes') or 
        spec_diff.get('generator_changes')
    )
    if has_doc_changes:
        return 'patch', "Documentation or generator changes only"
    
    # No significant changes
    return 'patch', "No significant changes detected"


def evaluate_semver_compliance(
    current_version: str,
    new_version: str,
    required_bump: str
) -> dict[str, Any]:
    """
    Evaluate if the version bump complies with SemVer requirements.
    
    Args:
        current_version: Current version string
        new_version: New version string
        required_bump: Required bump level (major, minor, patch)
        
    Returns:
        Evaluation result with compliance status
    """
    try:
        # Parse versions
        current_parts = [int(x) for x in current_version.split('.')]
        new_parts = [int(x) for x in new_version.split('.')]
        
        if len(current_parts) < 3 or len(new_parts) < 3:
            return {
                "compliant": False,
                "reason": "Invalid version format",
                "required": required_bump
            }
        
        current_major, current_minor, current_patch = current_parts[:3]
        new_major, new_minor, new_patch = new_parts[:3]
        
        # Determine actual bump level
        if new_major > current_major:
            actual_bump = 'major'
        elif new_minor > current_minor:
            actual_bump = 'minor'
        elif new_patch > current_patch:
            actual_bump = 'patch'
        else:
            actual_bump = 'none'
        
        # Check compliance
        if required_bump == 'major':
            compliant = actual_bump == 'major'
        elif required_bump == 'minor':
            compliant = actual_bump in ['major', 'minor']
        else:  # patch
            compliant = True  # Any bump is acceptable for patch-level changes
        
        return {
            "compliant": compliant,
            "reason": f"Required: {required_bump}, Actual: {actual_bump}",
            "required": required_bump,
            "actual": actual_bump,
            "current_version": current_version,
            "new_version": new_version
        }
    except Exception as e:
        return {
            "compliant": False,
            "reason": f"Error parsing versions: {e}",
            "required": required_bump
        }


def main():
    """
    Main entry point for the SemVer evaluator.
    """
    usage = (
        "python semver-evaluator.py <changes-file> "
        "[current-version] [new-version]"
    )
    
    if len(sys.argv) < 2:
        print(f"Usage: {usage}")
        sys.exit(1)
    
    changes_file = sys.argv[1]
    
    # Load changes data
    try:
        with open(changes_file) as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading changes file: {e}")
        sys.exit(1)
    
    spec_diff = data.get('spec_diff', {})
    signature_changes = data.get('signature_changes', [])
    
    # Analyze changes
    required_bump, reason = analyze_api_changes(spec_diff, signature_changes)
    
    print(f"Required SemVer bump: {required_bump}")
    print(f"Reason: {reason}")
    
    # If versions provided, check compliance
    if len(sys.argv) >= 4:
        current_version = sys.argv[2]
        new_version = sys.argv[3]
        
        compliance = evaluate_semver_compliance(
            current_version, 
            new_version, 
            required_bump
        )
        
        status = '✅ PASS' if compliance['compliant'] else '❌ FAIL'
        print(f"Compliance: {status}")
        print(f"Details: {compliance['reason']}")
        
        # Output as JSON for SARIF integration
        print(json.dumps(compliance, indent=2))
    
    return {
        "required_bump": required_bump,
        "reason": reason
    }


if __name__ == "__main__":
    main()