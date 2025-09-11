#!/usr/bin/env python3
"""
AI CLI Detection Script for Super-Alita
=======================================
Detects available AI CLI tools and recommends the best integration approach.
"""

import shutil
import subprocess


class AICliDetector:
    """Detects and evaluates available AI CLI tools."""

    def __init__(self):
        self.available_tools = {}
        self.recommendations = []

    def check_tool_availability(self) -> dict[str, bool]:
        """Check which AI CLI tools are available."""
        tools_to_check = {
            "gh": "GitHub CLI (for Copilot integration)",
            "copilot": "GitHub Copilot CLI",
            "gemini": "Google Gemini CLI",
            "openai": "OpenAI CLI",
            "code": "VS Code CLI",
            "cursor": "Cursor CLI",
            "aider": "Aider AI coding assistant",
        }

        results = {}
        for tool, description in tools_to_check.items():
            available = shutil.which(tool) is not None
            results[tool] = {
                "available": available,
                "description": description,
                "path": shutil.which(tool) if available else None,
            }

            if available:
                print(f"✅ {tool}: {description}")
                print(f"   Path: {shutil.which(tool)}")

                # Get version info if possible
                try:
                    version_result = subprocess.run(
                        [tool, "--version"], capture_output=True, text=True, timeout=5
                    )
                    if version_result.returncode == 0:
                        version = version_result.stdout.strip().split("\n")[0]
                        print(f"   Version: {version}")
                        results[tool]["version"] = version
                except Exception:
                    print("   Version: Could not determine")
            else:
                print(f"❌ {tool}: Not available")

        return results

    def check_copilot_features(self) -> dict[str, bool]:
        """Check GitHub Copilot specific features."""
        copilot_features = {}

        if shutil.which("gh"):
            try:
                # Check if Copilot extension is installed
                result = subprocess.run(
                    ["gh", "extension", "list"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )

                if "github/gh-copilot" in result.stdout:
                    copilot_features["gh_copilot_extension"] = True
                    print("✅ GitHub Copilot CLI extension installed")
                else:
                    copilot_features["gh_copilot_extension"] = False
                    print("❌ GitHub Copilot CLI extension not installed")
                    print("   Install with: gh extension install github/gh-copilot")

            except Exception as e:
                print(f"❌ Could not check GitHub Copilot extension: {e}")
                copilot_features["gh_copilot_extension"] = False

        return copilot_features

    def generate_recommendations(self, available_tools: dict[str, dict]) -> list[str]:
        """Generate recommendations based on available tools."""
        recommendations = []

        # Check for VS Code integration (you're using VS Code)
        if available_tools.get("code", {}).get("available"):
            recommendations.append(
                {
                    "priority": 1,
                    "tool": "VS Code + GitHub Copilot",
                    "reason": "Best integration with your current VS Code workflow",
                    "setup": [
                        "Install GitHub Copilot extension in VS Code",
                        "Install gh CLI: winget install GitHub.cli",
                        "Install Copilot CLI: gh extension install github/gh-copilot",
                        "Authenticate: gh auth login",
                    ],
                }
            )

        # Check for GitHub CLI + Copilot
        if available_tools.get("gh", {}).get("available"):
            recommendations.append(
                {
                    "priority": 2,
                    "tool": "GitHub CLI + Copilot",
                    "reason": "Command-line integration for Super-Alita workflows",
                    "setup": [
                        "gh extension install github/gh-copilot",
                        'gh copilot suggest "create a python function"',
                        'gh copilot explain "complex code snippet"',
                    ],
                }
            )

        # Check for OpenAI CLI
        if available_tools.get("openai", {}).get("available"):
            recommendations.append(
                {
                    "priority": 3,
                    "tool": "OpenAI CLI",
                    "reason": "Direct OpenAI API access",
                    "setup": [
                        "Set OPENAI_API_KEY environment variable",
                        "Test with: openai api completions.create",
                    ],
                }
            )

        return recommendations

    def create_super_alita_cli_integration(self) -> str:
        """Generate code for Super-Alita CLI integration."""
        return """#!/usr/bin/env python3
'''
Super-Alita AI CLI Integration
=============================
Integrates AI CLI tools with Super-Alita workflow following Specify methodology.
'''

import subprocess
import sys
from pathlib import Path
from typing import Optional, List

class SuperAlitaAI:
    def __init__(self):
        self.workspace_root = Path.cwd()
        self.specs_dir = self.workspace_root / "specs"
        self.templates_dir = self.workspace_root / "templates"
        self.memory_dir = self.workspace_root / "memory"

    def specify(self, prompt: str) -> bool:
        '''Equivalent to /specify command - create feature specification.'''
        print(f"🎯 Creating specification for: {prompt}")

        # Create feature directory
        feature_num = self._get_next_feature_number()
        feature_name = self._slugify(prompt)
        feature_dir = self.specs_dir / f"{feature_num:03d}-{feature_name}"
        feature_dir.mkdir(parents=True, exist_ok=True)

        # Use AI to generate specification
        spec_content = self._generate_with_ai(
            f"Create a detailed software specification for: {prompt}",
            template="spec-template.md"
        )

        (feature_dir / "spec.md").write_text(spec_content)
        print(f"✅ Specification created: {feature_dir}/spec.md")
        return True

    def plan(self, spec_path: str, tech_stack: str = "") -> bool:
        '''Equivalent to /plan command - create implementation plan.'''
        spec_file = Path(spec_path)
        if not spec_file.exists():
            print(f"❌ Specification not found: {spec_path}")
            return False

        print(f"📋 Creating implementation plan for: {spec_file}")

        spec_content = spec_file.read_text()
        plan_prompt = f"Create implementation plan for:\\n{spec_content}"

        if tech_stack:
            plan_prompt += f"\\nTech stack: {tech_stack}"

        plan_content = self._generate_with_ai(plan_prompt, template="plan-template.md")

        plan_file = spec_file.parent / "plan.md"
        plan_file.write_text(plan_content)
        print(f"✅ Implementation plan created: {plan_file}")
        return True

    def tasks(self, plan_path: str) -> bool:
        '''Equivalent to /tasks command - break down into tasks.'''
        plan_file = Path(plan_path)
        if not plan_file.exists():
            print(f"❌ Plan not found: {plan_path}")
            return False

        print(f"📝 Creating task breakdown for: {plan_file}")

        plan_content = plan_file.read_text()
        tasks_prompt = f"Break down into specific tasks:\\n{plan_content}"

        tasks_content = self._generate_with_ai(tasks_prompt, template="tasks-template.md")

        tasks_file = plan_file.parent / "tasks.md"
        tasks_file.write_text(tasks_content)
        print(f"✅ Task breakdown created: {tasks_file}")
        return True

    def _generate_with_ai(self, prompt: str, template: str = "") -> str:
        '''Generate content using available AI CLI.'''
        # Try GitHub Copilot first
        if self._has_github_copilot():
            return self._generate_with_copilot(prompt, template)

        # Try OpenAI CLI
        elif self._has_openai_cli():
            return self._generate_with_openai(prompt, template)

        # Fallback to template
        else:
            return self._generate_from_template(prompt, template)

    def _has_github_copilot(self) -> bool:
        '''Check if GitHub Copilot CLI is available.'''
        try:
            result = subprocess.run(['gh', 'copilot', '--help'],
                                  capture_output=True, timeout=5)
            return result.returncode == 0
        except:
            return False

    def _generate_with_copilot(self, prompt: str, template: str) -> str:
        '''Generate using GitHub Copilot CLI.'''
        try:
            result = subprocess.run([
                'gh', 'copilot', 'suggest',
                f"Generate {template} content for: {prompt}"
            ], capture_output=True, text=True, timeout=30)

            if result.returncode == 0:
                return result.stdout
            else:
                print(f"⚠️ Copilot error: {result.stderr}")
                return self._generate_from_template(prompt, template)
        except Exception as e:
            print(f"⚠️ Copilot failed: {e}")
            return self._generate_from_template(prompt, template)

def main():
    '''CLI entry point for Super-Alita AI commands.'''
    if len(sys.argv) < 2:
        print("Usage: python super_alita_ai.py [specify|plan|tasks] <args>")
        return

    command = sys.argv[1]
    ai = SuperAlitaAI()

    if command == "specify" and len(sys.argv) >= 3:
        prompt = " ".join(sys.argv[2:])
        ai.specify(prompt)
    elif command == "plan" and len(sys.argv) >= 3:
        spec_path = sys.argv[2]
        tech_stack = " ".join(sys.argv[3:]) if len(sys.argv) > 3 else ""
        ai.plan(spec_path, tech_stack)
    elif command == "tasks" and len(sys.argv) >= 3:
        plan_path = sys.argv[2]
        ai.tasks(plan_path)
    else:
        print("Invalid command or arguments")

if __name__ == "__main__":
    main()
"""


def main():
    """Main detection and recommendation function."""
    print("🔍 Super-Alita AI CLI Detection")
    print("=" * 50)

    detector = AICliDetector()

    # Check available tools
    print("\n📋 Checking Available AI CLI Tools:")
    available_tools = detector.check_tool_availability()

    # Check Copilot specific features
    print("\n🤖 Checking GitHub Copilot Features:")
    copilot_features = detector.check_copilot_features()

    # Generate recommendations
    print("\n💡 Recommendations:")
    recommendations = detector.generate_recommendations(available_tools)

    if recommendations:
        for i, rec in enumerate(recommendations, 1):
            print(f"\n{i}. {rec['tool']} (Priority {rec['priority']})")
            print(f"   Reason: {rec['reason']}")
            print("   Setup steps:")
            for step in rec["setup"]:
                print(f"     • {step}")
    else:
        print("❌ No AI CLI tools detected.")
        print("\n📦 Recommended installations:")
        print("   • GitHub CLI: winget install GitHub.cli")
        print("   • OpenAI CLI: pip install openai")
        print("   • Aider: pip install aider-chat")

    # Generate Super-Alita integration code
    print("\n🚀 Creating Super-Alita AI CLI Integration...")
    integration_code = detector.create_super_alita_cli_integration()

    with open("super_alita_ai.py", "w") as f:
        f.write(integration_code)

    print("✅ Created: super_alita_ai.py")
    print("\n📖 Usage examples:")
    print("   python super_alita_ai.py specify 'Create user authentication system'")
    print(
        "   python super_alita_ai.py plan specs/001-auth/spec.md 'FastAPI + JWT + SQLAlchemy'"
    )
    print("   python super_alita_ai.py tasks specs/001-auth/plan.md")


if __name__ == "__main__":
    main()
