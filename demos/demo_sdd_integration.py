#!/usr/bin/env python3
"""
Demo: GitHub Copilot SDD Integration

This demonstrates how the SDD-enhanced GitHub Copilot integration provides
specification-driven development guidance for different workflow patterns.
"""

import os
import sys

sys.path.append(os.path.dirname(__file__))

from copilot_sdd_seamless import (
    analyze_constitutional_compliance,
    copilot_sdd_enhance,
    detect_sdd_workflow_pattern,
    setup_sdd_integration,
)


def demo_sdd_workflow_patterns():
    """Demonstrate SDD workflow pattern detection and guidance."""
    print("🎯 SDD Workflow Pattern Detection Demo")
    print("=" * 50)

    # Demo scenarios covering different SDD patterns
    demo_scenarios = [
        {
            "scenario": "New Feature Specification",
            "input": "/new_feature Real-time collaborative document editing",
            "expected_pattern": "new_feature",
        },
        {
            "scenario": "Implementation Planning",
            "input": "/generate_plan WebSocket-based document synchronization with conflict resolution",
            "expected_pattern": "generate_plan",
        },
        {
            "scenario": "Constitutional Review",
            "input": "Is this code following constitutional principles?",
            "expected_pattern": "constitutional",
        },
        {
            "scenario": "Test-First Guidance",
            "input": "How should I implement TDD for this API endpoint?",
            "expected_pattern": "test_first",
        },
        {
            "scenario": "Library-First Design",
            "input": "Should I create a library for document parsing functionality?",
            "expected_pattern": "library_first",
        },
        {
            "scenario": "Simplicity Review",
            "input": "This code is getting complex, how can I simplify it?",
            "expected_pattern": "simplicity",
        },
        {
            "scenario": "CLI Interface Design",
            "input": "How do I add a command line interface to this library?",
            "expected_pattern": "cli_interface",
        },
        {
            "scenario": "Integration Testing",
            "input": "Setting up integration tests with real database",
            "expected_pattern": "integration_first",
        },
        {
            "scenario": "General Development",
            "input": "How do I optimize this function for better performance?",
            "expected_pattern": "general",
        },
    ]

    for scenario in demo_scenarios:
        print(f"\n📋 {scenario['scenario']}")
        print(f"Input: {scenario['input']}")

        detected_pattern = detect_sdd_workflow_pattern(scenario["input"])
        print(f"Detected Pattern: {detected_pattern}")
        print(f"Expected Pattern: {scenario['expected_pattern']}")

        if detected_pattern == scenario["expected_pattern"]:
            print("✅ Pattern detection: CORRECT")
        else:
            print("❌ Pattern detection: INCORRECT")

        # Show first few lines of guidance
        guidance = copilot_sdd_enhance(scenario["input"])
        lines = guidance.split("\n")
        print("Guidance preview:")
        for line in lines[:4]:
            if line.strip():
                print(f"  {line}")
        print("  [... more guidance ...]")
        print("-" * 30)


def demo_constitutional_analysis():
    """Demonstrate constitutional compliance analysis."""
    print("\n🏛️ Constitutional Compliance Analysis Demo")
    print("=" * 50)

    # Example code snippets for analysis
    test_codes = [
        {
            "name": "Well-designed Library Function",
            "code": '''
import click
import requests
from typing import Dict, List

def fetch_user_data(api_key: str, user_id: str) -> Dict:
    """Fetch user data from API."""
    response = requests.get(f"/api/users/{user_id}",
                          headers={"Authorization": f"Bearer {api_key}"})
    return response.json()

@click.command()
@click.option('--api-key', required=True)
@click.option('--user-id', required=True)
def main(api_key: str, user_id: str):
    """CLI interface for user data fetching."""
    data = fetch_user_data(api_key, user_id)
    click.echo(data)

if __name__ == "__main__":
    main()
''',
        },
        {
            "name": "Test-First Example",
            "code": '''
import pytest
from unittest.mock import Mock

def test_user_authentication():
    """Test user authentication logic."""
    auth_service = Mock()
    auth_service.authenticate.return_value = True

    result = authenticate_user("test@example.com", "password")
    assert result is True

def test_invalid_credentials():
    """Test authentication with invalid credentials."""
    auth_service = Mock()
    auth_service.authenticate.return_value = False

    result = authenticate_user("test@example.com", "wrong_password")
    assert result is False
''',
        },
        {
            "name": "Over-Abstracted Code (Anti-Pattern)",
            "code": """
from abc import ABC, abstractmethod

class AbstractUserFactory(ABC):
    @abstractmethod
    def create_user(self) -> 'AbstractUser':
        pass

class UserFactoryImpl(AbstractUserFactory):
    def create_user(self) -> 'User':
        return User()

class AbstractUserWrapper:
    def __init__(self, user_factory: AbstractUserFactory):
        self._factory = user_factory
        self._user = None

    def get_user_adapter(self) -> 'UserAdapter':
        if not self._user:
            self._user = self._factory.create_user()
        return UserAdapter(self._user)
""",
        },
    ]

    for test_case in test_codes:
        print(f"\n📝 {test_case['name']}")
        print("-" * 30)

        analysis = analyze_constitutional_compliance(test_case["code"])
        print(analysis)


def demo_full_sdd_workflow():
    """Demonstrate a complete SDD workflow."""
    print("\n🚀 Complete SDD Workflow Demo")
    print("=" * 40)

    workflow_steps = [
        {
            "phase": "Specification",
            "input": "/new_feature User notification system with multiple delivery channels",
            "description": "Define WHAT and WHY, avoid HOW",
        },
        {
            "phase": "Planning",
            "input": "/generate_plan Email, SMS, and push notification delivery with retry logic",
            "description": "Constitutional validation and implementation planning",
        },
        {
            "phase": "Library Design",
            "input": "Should I create a notification library with CLI interface?",
            "description": "Library-first and CLI-first principles",
        },
        {
            "phase": "Test Strategy",
            "input": "How do I implement test-first development for notification delivery?",
            "description": "Test-first imperative and integration testing",
        },
        {
            "phase": "Constitutional Review",
            "input": "Check constitutional compliance for notification system",
            "description": "Final constitutional validation",
        },
    ]

    for step in workflow_steps:
        print(f"\n🎯 Phase: {step['phase']}")
        print(f"Description: {step['description']}")
        print(f"Input: {step['input']}")
        print()

        guidance = copilot_sdd_enhance(step["input"])

        # Show key highlights from guidance
        lines = guidance.split("\n")
        key_lines = []
        for line in lines:
            if any(marker in line for marker in ["**", "•", "1.", "2.", "3.", "💡"]):
                key_lines.append(line)
                if len(key_lines) >= 5:  # Limit output
                    break

        print("Key Guidance:")
        for line in key_lines:
            print(f"  {line}")

        print("  [... detailed SDD guidance continues ...]")
        print()


def main():
    """Run the SDD integration demo."""
    print("🎉 GitHub Copilot SDD Integration Demo")
    print("=====================================")
    print()

    # Setup integration
    print("Setting up SDD integration...")
    setup_success = setup_sdd_integration()

    if not setup_success:
        print("❌ Setup failed!")
        return

    print("\n" + "=" * 60)

    # Run demos
    try:
        demo_sdd_workflow_patterns()
        demo_constitutional_analysis()
        demo_full_sdd_workflow()

        print("\n" + "=" * 60)
        print("🎉 SDD Integration Demo Complete!")
        print()
        print("📋 What this demonstrates:")
        print("• Automatic SDD workflow pattern detection")
        print("• Constitutional compliance analysis")
        print("• Template-driven LLM constraint guidance")
        print("• Specification-driven development methodology")
        print("• Phase gate enforcement")
        print()
        print("🚀 Integration Usage:")
        print("```python")
        print("from copilot_sdd_seamless import copilot_sdd_enhance")
        print("enhanced = copilot_sdd_enhance('/new_feature Authentication')")
        print("```")
        print()
        print("💡 Every GitHub Copilot interaction now includes:")
        print("• SDD constitutional guidance")
        print("• Workflow-specific recommendations")
        print("• Template-driven quality constraints")
        print("• Specification-driven development principles")

    except Exception as e:
        print(f"❌ Demo error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
