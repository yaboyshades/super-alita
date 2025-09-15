#!/usr/bin/env python3
"""
Validate SDD artifacts against constitutional framework.
This script validates the specific SDD files for the Advanced Debugging and
Performance Optimization epic.
"""

import asyncio
import json
import sys
from pathlib import Path

from src.sdd.validators import SDDValidator


async def validate_sdd_epic():
    """Validate the SDD epic files."""
    validator = SDDValidator(constitutional_threshold=0.75)
    
    # File paths
    spec_file = Path("sdd/specs/advanced_debugging_and_perf.yaml")
    plan_file = Path("sdd/plans/advanced_debugging_and_perf.yaml")
    tasks_file = Path("sdd/tasks/advanced_debugging_and_perf.yaml")
    
    results = {
        "spec": None,
        "plan": None,
        "tasks": None,
        "overall_score": 0.0,
        "passed": False
    }
    
    # Validate spec
    if spec_file.exists():
        spec_content = spec_file.read_text()
        results["spec"] = await validator.validate_specification(spec_content)
        print(f"Spec validation score: {results['spec']['overall_score']:.3f}")
    else:
        print("Spec file not found!")
        return results
    
    # Validate plan
    if plan_file.exists():
        plan_content = plan_file.read_text()
        results["plan"] = await validator.validate_plan(plan_content)
        print(f"Plan validation score: {results['plan']['overall_score']:.3f}")
    else:
        print("Plan file not found!")
        return results
    
    # Validate tasks
    if tasks_file.exists():
        tasks_content = tasks_file.read_text()
        results["tasks"] = await validator.validate_tasks(tasks_content)
        print(f"Tasks validation score: {results['tasks']['overall_score']:.3f}")
    else:
        print("Tasks file not found!")
        return results
    
    # Calculate overall score
    scores = [
        results["spec"]["overall_score"],
        results["plan"]["overall_score"],
        results["tasks"]["overall_score"]
    ]
    results["overall_score"] = sum(scores) / len(scores)
    results["passed"] = results["overall_score"] >= 0.75
    
    print(f"\nOverall SDD Epic Score: {results['overall_score']:.3f}")
    print(f"Constitutional Threshold Met: {results['passed']}")
    
    return results


if __name__ == "__main__":
    results = asyncio.run(validate_sdd_epic())
    
    # Save results as JSON artifact
    artifacts_dir = Path("sdd/artifacts/advanced_debugging_and_perf")
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    
    with open(artifacts_dir / "validation.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Exit with error code if validation failed
    if not results["passed"]:
        print(
            f"\nValidation failed! Score {results['overall_score']:.3f} "
            "is below threshold 0.75"
        )
        sys.exit(1)
    else:
        print(
            f"\nValidation passed! Score {results['overall_score']:.3f} "
            "meets threshold 0.75"
        )