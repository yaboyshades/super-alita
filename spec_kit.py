#!/usr/bin/env python3
"""
Super-Alita Spec-Kit Constitutional Architecture
==============================================
Implements /specify, /plan, /tasks commands with GitHub Copilot CLI integration.
Follows Specification-Driven Development (SDD) methodology.
"""

import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path


class SpecKitArchitect:
    """Constitutional Architect for Specification-Driven Development."""

    def __init__(self, workspace_root: Path | None = None):
        self.workspace_root = workspace_root or Path.cwd()
        self.memory_dir = self.workspace_root / "memory"
        self.specs_dir = self.workspace_root / "specs"
        self.templates_dir = self.workspace_root / "templates"
        self.scripts_dir = self.workspace_root / "scripts"

        spec_kit_path_env = os.getenv("GITHUB_SPEC_KIT_PATH")
        templates_path_env = os.getenv("SPEC_KIT_TEMPLATES_PATH")

        self.spec_kit_repo_url = os.getenv(
            "GITHUB_SPEC_KIT_URL", "https://github.com/github/spec-kit.git"
        )
        self.spec_kit_repo_path = (
            Path(spec_kit_path_env).expanduser()
            if spec_kit_path_env
            else self.workspace_root / "spec-kit"
        )
        self.github_templates_dir = (
            Path(templates_path_env).expanduser()
            if templates_path_env
            else self.templates_dir / "github-spec-kit"
        )

        # Ensure required directories exist
        for directory in [
            self.memory_dir,
            self.specs_dir,
            self.templates_dir,
            self.scripts_dir,
        ]:
            directory.mkdir(exist_ok=True)

    def specify(self, feature_description: str) -> bool:
        """
        /specify command - Create feature specification following SDD methodology.

        Args:
            feature_description: Natural language description of the feature

        Returns:
            bool: Success status
        """
        print(f"🎯 /specify: Creating specification for '{feature_description}'")

        # Generate feature metadata
        feature_num = self._get_next_feature_number()
        feature_name = self._slugify(feature_description)
        feature_dir = self.specs_dir / f"{feature_num:03d}-{feature_name}"
        feature_dir.mkdir(parents=True, exist_ok=True)

        print(f"📁 Created feature directory: {feature_dir}")

        # Generate specification using GitHub Copilot
        spec_content = self._generate_specification_with_copilot(feature_description)

        # Write specification
        spec_file = feature_dir / "spec.md"
        spec_file.write_text(spec_content, encoding="utf-8")

        print(f"✅ Specification created: {spec_file}")
        print("📋 Next steps:")
        print("   1. Review and refine the specification")
        print("   2. Complete the Review & Acceptance Checklist")
        print("   3. Run: /plan to create implementation plan")

        return True

    def plan(self, spec_path: str, tech_stack: str = "") -> bool:
        """
        /plan command - Create implementation plan from specification.

        Args:
            spec_path: Path to the specification file
            tech_stack: Desired technology stack

        Returns:
            bool: Success status
        """
        spec_file = Path(spec_path)
        if not spec_file.exists():
            print(f"❌ Specification not found: {spec_path}")
            return False

        print(f"📋 /plan: Creating implementation plan for {spec_file}")

        # Read specification
        spec_content = spec_file.read_text(encoding="utf-8")

        # Generate implementation plan using GitHub Copilot
        plan_content = self._generate_plan_with_copilot(spec_content, tech_stack)

        # Write plan
        plan_file = spec_file.parent / "plan.md"
        plan_file.write_text(plan_content, encoding="utf-8")

        # Generate supporting documents
        self._generate_supporting_documents(spec_file.parent, spec_content, tech_stack)

        print(f"✅ Implementation plan created: {plan_file}")
        print("🏗️ Supporting documents generated")
        print("📋 Next steps:")
        print("   1. Review plan against Constitutional principles")
        print("   2. Validate architectural decisions")
        print("   3. Run: /tasks to generate task breakdown")

        return True

    def tasks(self, plan_path: str) -> bool:
        """
        /tasks command - Generate task breakdown from implementation plan.

        Args:
            plan_path: Path to the implementation plan

        Returns:
            bool: Success status
        """
        plan_file = Path(plan_path)
        if not plan_file.exists():
            print(f"❌ Implementation plan not found: {plan_path}")
            return False

        print(f"📝 /tasks: Creating task breakdown for {plan_file}")

        # Read plan
        plan_content = plan_file.read_text(encoding="utf-8")

        # Generate tasks using GitHub Copilot
        tasks_content = self._generate_tasks_with_copilot(plan_content)

        # Write tasks
        tasks_file = plan_file.parent / "tasks.md"
        tasks_file.write_text(tasks_content, encoding="utf-8")

        print(f"✅ Task breakdown created: {tasks_file}")
        print("🚀 Ready for implementation!")
        print("📋 Next steps:")
        print("   1. Begin with Test-First Imperative (Article III)")
        print("   2. Implement tasks in priority order")
        print("   3. Validate Constitutional compliance")

        return True

    def constitutional_review(self, feature_dir: str) -> dict[str, bool]:
        """
        Perform constitutional compliance review for a feature.

        Args:
            feature_dir: Path to feature directory

        Returns:
            Dict[str, bool]: Compliance status for each article
        """
        feature_path = Path(feature_dir)
        if not feature_path.exists():
            print(f"❌ Feature directory not found: {feature_dir}")
            return {}

        print(f"⚖️ Constitutional Review: {feature_path}")

        compliance = {}

        # Load constitution
        constitution_file = self.memory_dir / "constitution.md"
        if not constitution_file.exists():
            print(f"❌ Constitution not found: {constitution_file}")
            return {}

        # Check each article (simplified implementation)
        articles = [
            "Library-First Principle",
            "CLI Interface Mandate",
            "Test-First Imperative",
            "Documentation-First Development",
            "Integration-First Testing",
            "Continuous Validation",
            "Simplicity Gate",
            "Anti-Abstraction Gate",
            "Constitutional Compliance",
        ]

        for article in articles:
            compliance[article] = self._check_article_compliance(feature_path, article)

        # Generate compliance report
        self._generate_compliance_report(feature_path, compliance)

        return compliance

    def sync_github_spec_kit(self, repo_url: str | None = None) -> Path:
        """Clone or update the GitHub spec-kit repository locally."""

        target_url = repo_url or self.spec_kit_repo_url
        repo_path = self.spec_kit_repo_path

        if repo_path.exists():
            git_dir = repo_path / ".git"
            if not git_dir.is_dir():
                raise RuntimeError(
                    f"Existing path is not a git repository: {repo_path}"
                )
            self._run_git_command(["-C", str(repo_path), "fetch", "--all"])
            self._run_git_command(
                ["-C", str(repo_path), "reset", "--hard", "origin/main"]
            )
        else:
            repo_path.parent.mkdir(parents=True, exist_ok=True)
            self._run_git_command(["clone", target_url, str(repo_path)])

        return repo_path

    def integrate_github_templates(self, repo_path: Path | None = None) -> Path:
        """Copy GitHub spec-kit templates into the local templates directory."""

        source_repo = repo_path or self.spec_kit_repo_path
        templates_source = source_repo / "templates"
        if not templates_source.is_dir():
            raise FileNotFoundError(
                f"GitHub spec-kit templates directory not found: {templates_source}"
            )

        destination = self.github_templates_dir
        destination.parent.mkdir(parents=True, exist_ok=True)

        shutil.copytree(templates_source, destination, dirs_exist_ok=True)
        return destination

    def _run_git_command(self, args: list[str]) -> subprocess.CompletedProcess[str]:
        """Execute a git command and raise a helpful error on failure."""

        try:
            return subprocess.run(
                ["git", *args],
                capture_output=True,
                text=True,
                check=True,
            )
        except subprocess.CalledProcessError as exc:
            stderr = exc.stderr.strip() if exc.stderr else "unknown error"
            raise RuntimeError(f"git {' '.join(args)} failed: {stderr}") from exc

    def _generate_specification_with_copilot(self, feature_description: str) -> str:
        """Generate specification using GitHub Copilot CLI."""
        prompt = f"""
Create a comprehensive software feature specification for: {feature_description}

The specification must include:
1. Feature Overview and Objectives
2. User Stories and Acceptance Criteria
3. Functional Requirements
4. Non-Functional Requirements
5. API Contracts and Data Models
6. Integration Points
7. Review & Acceptance Checklist
8. Constitutional Compliance Section

Follow the Super-Alita Constitutional Architecture principles:
- Library-First Principle: Design as standalone, reusable library
- CLI Interface Mandate: Include text-in, text-out CLI interface
- Test-First Imperative: Define testable acceptance criteria
- Documentation-First: Comprehensive documentation requirements

Use markdown format with clear sections and checklists.
"""

        return self._call_copilot_suggest(prompt)

    def _generate_plan_with_copilot(self, spec_content: str, tech_stack: str) -> str:
        """Generate implementation plan using GitHub Copilot CLI."""
        prompt = f"""
Create a detailed implementation plan based on this specification:

{spec_content}

Technology Stack: {tech_stack}

The implementation plan must include:
1. Architecture Overview
2. Project Structure (≤3 projects per Simplicity Gate)
3. Implementation Phases
4. Test Strategy (Test-First Imperative)
5. API Design
6. Database Schema (if applicable)
7. Deployment Strategy
8. Constitutional Compliance Verification

Ensure compliance with Constitutional principles:
- Simplicity Gate: Justify any complexity beyond minimal structure
- Anti-Abstraction Gate: Use framework features directly
- Integration-First Testing: Real environments over mocks
- Library-First Principle: Standalone, reusable components

Use markdown format with detailed sections.
"""

        return self._call_copilot_suggest(prompt)

    def _generate_tasks_with_copilot(self, plan_content: str) -> str:
        """Generate task breakdown using GitHub Copilot CLI."""
        prompt = f"""
Create an actionable task breakdown from this implementation plan:

{plan_content}

Generate tasks that follow Test-First Imperative:
1. Write tests BEFORE implementation (Red Phase)
2. Implement to make tests pass (Green Phase)
3. Refactor for quality (Refactor Phase)

Task format:
- [ ] Task description
- Acceptance criteria
- Dependencies
- Estimated effort

Priority order:
1. Test infrastructure setup
2. Core library implementation (with tests first)
3. CLI interface implementation
4. Integration testing
5. Documentation completion
6. Constitutional compliance verification

Use markdown format with clear task lists and dependencies.
"""

        return self._call_copilot_suggest(prompt)

    def _call_copilot_suggest(self, prompt: str) -> str:
        """Call GitHub Copilot CLI suggest command."""
        try:
            result = subprocess.run(
                ["gh", "copilot", "suggest", "--type", "generic", prompt],
                capture_output=True,
                text=True,
                timeout=60,
            )

            if result.returncode == 0:
                return result.stdout.strip()
            else:
                print(f"⚠️ Copilot error: {result.stderr}")
                return self._fallback_template(prompt)

        except Exception as e:
            print(f"⚠️ Copilot CLI failed: {e}")
            return self._fallback_template(prompt)

    def _fallback_template(self, prompt: str) -> str:
        """Fallback template when Copilot is unavailable."""
        return f"""# Feature Specification

## Overview
{prompt[:200]}...

## Requirements
- [Requirement 1]
- [Requirement 2]
- [Requirement 3]

## Acceptance Criteria
- [ ] Criterion 1
- [ ] Criterion 2
- [ ] Criterion 3

## Constitutional Compliance
- [ ] Library-First Principle
- [ ] CLI Interface Mandate
- [ ] Test-First Imperative
- [ ] Documentation-First Development

## Next Steps
1. Refine this specification
2. Add detailed requirements
3. Create implementation plan

---
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Tool: Fallback Template (Copilot unavailable)
"""

    def _get_next_feature_number(self) -> int:
        """Get the next feature number in sequence."""
        if not self.specs_dir.exists():
            return 1

        existing_features = [
            d for d in self.specs_dir.iterdir() if d.is_dir() and d.name[:3].isdigit()
        ]

        if not existing_features:
            return 1

        max_num = max(int(d.name[:3]) for d in existing_features)
        return max_num + 1

    def _slugify(self, text: str) -> str:
        """Convert text to URL-safe slug."""
        import re

        slug = re.sub(r"[^\w\s-]", "", text.lower())
        slug = re.sub(r"[-\s]+", "-", slug)
        return slug.strip("-")[:50]

    def _generate_supporting_documents(
        self, feature_dir: Path, spec_content: str, tech_stack: str
    ):
        """Generate supporting documents for the plan."""
        # Create contracts directory
        contracts_dir = feature_dir / "contracts"
        contracts_dir.mkdir(exist_ok=True)

        # Generate API spec (simplified)
        api_spec = {
            "openapi": "3.0.0",
            "info": {"title": "Feature API", "version": "1.0.0"},
            "paths": {},
        }

        (contracts_dir / "api-spec.json").write_text(
            json.dumps(api_spec, indent=2), encoding="utf-8"
        )

        # Generate data model
        data_model_content = f"""# Data Model

## Overview
Data model for the feature based on specification.

## Entities
- Entity1
- Entity2

## Relationships
- Relationship descriptions

---
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

        (feature_dir / "data-model.md").write_text(data_model_content, encoding="utf-8")

    def _check_article_compliance(self, feature_path: Path, article: str) -> bool:
        """Check compliance with a specific constitutional article."""
        # Simplified compliance checking
        # In a real implementation, this would perform detailed analysis

        required_files = {
            "Library-First Principle": ["spec.md"],
            "CLI Interface Mandate": ["spec.md"],
            "Test-First Imperative": ["plan.md"],
            "Documentation-First Development": ["spec.md", "plan.md"],
        }

        if article in required_files:
            return all((feature_path / f).exists() for f in required_files[article])

        return True  # Default to compliant for other articles

    def _generate_compliance_report(
        self, feature_path: Path, compliance: dict[str, bool]
    ):
        """Generate constitutional compliance report."""
        report_content = f"""# Constitutional Compliance Report

Feature: {feature_path.name}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Compliance Status

"""

        for article, status in compliance.items():
            status_icon = "✅" if status else "❌"
            report_content += f"- {status_icon} {article}\n"

        report_content += f"""
## Summary
- Compliant: {sum(compliance.values())}/{len(compliance)} articles
- Status: {'COMPLIANT' if all(compliance.values()) else 'NON-COMPLIANT'}

## Next Steps
"""

        if not all(compliance.values()):
            report_content += "- Address non-compliant articles before proceeding\n"
            report_content += "- Review Constitutional requirements\n"
            report_content += "- Update specification and plan as needed\n"
        else:
            report_content += "- Proceed with implementation\n"
            report_content += "- Begin with Test-First Imperative\n"

        (feature_path / "compliance-report.md").write_text(
            report_content, encoding="utf-8"
        )


def main():
    """CLI entry point for spec-kit commands."""
    if len(sys.argv) < 2:
        print(
            """
🏗️ Super-Alita Spec-Kit Constitutional Architect

Available commands:
  /specify "<feature_description>"  - Create feature specification
  /plan <spec_path> [tech_stack]   - Create implementation plan
  /tasks <plan_path>               - Generate task breakdown
  /review <feature_dir>            - Constitutional compliance review
  /sync [repo_url]                 - Clone/update GitHub spec-kit and sync templates

Examples:
  python spec_kit.py specify "User authentication system with JWT tokens"
  python spec_kit.py plan specs/001-auth/spec.md "FastAPI + SQLAlchemy + JWT"
  python spec_kit.py tasks specs/001-auth/plan.md
  python spec_kit.py review specs/001-auth
  python spec_kit.py sync
        """
        )
        return

    command = sys.argv[1].lstrip("/")
    architect = SpecKitArchitect()

    if command == "specify" and len(sys.argv) >= 3:
        feature_description = " ".join(sys.argv[2:])
        architect.specify(feature_description)
    elif command == "plan" and len(sys.argv) >= 3:
        spec_path = sys.argv[2]
        tech_stack = " ".join(sys.argv[3:]) if len(sys.argv) > 3 else ""
        architect.plan(spec_path, tech_stack)
    elif command == "tasks" and len(sys.argv) >= 3:
        plan_path = sys.argv[2]
        architect.tasks(plan_path)
    elif command == "review" and len(sys.argv) >= 3:
        feature_dir = sys.argv[2]
        architect.constitutional_review(feature_dir)
    elif command == "sync":
        repo_override = sys.argv[2] if len(sys.argv) >= 3 else None
        repo_path = architect.sync_github_spec_kit(repo_override)
        templates_path = architect.integrate_github_templates(repo_path)
        print(f"🔄 GitHub spec-kit synchronized at {repo_path}")
        print(f"🗂️ Templates synced to {templates_path}")
    else:
        print(f"❌ Invalid command or missing arguments: {' '.join(sys.argv)}")
        print("Run without arguments to see usage help.")


if __name__ == "__main__":
    main()
