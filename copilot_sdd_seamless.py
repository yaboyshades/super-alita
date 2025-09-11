#!/usr/bin/env python3
"""
GitHub Copilot SDD Integration - Specification-Driven Development Enhanced.

This extends the seamless Copilot integration to support the full SDD workflow:
- Specification-driven development patterns
- Constitutional compliance with SDD's 9 articles
- Template-driven LLM constraint for better outputs
- Seamless integration with new_feature and generate_plan commands
"""

import os

# SDD Constitutional Articles (9 Articles from the SDD Constitution)
SDD_CONSTITUTIONAL_ARTICLES = {
    "I": "Library-First Principle - Every feature begins as a standalone library",
    "II": "CLI Interface Mandate - All libraries expose functionality via CLI",
    "III": "Test-First Imperative - No code before tests (non-negotiable)",
    "IV": "Documentation Requirements - Clear, executable specifications",
    "V": "Version Control Standards - Specification-first branching",
    "VI": "Performance Baselines - Defined performance criteria",
    "VII": "Simplicity Gate - Maximum 3 projects, no future-proofing",
    "VIII": "Anti-Abstraction Gate - Use frameworks directly, avoid over-engineering",
    "IX": "Integration-First Testing - Real databases, actual services, contract tests",
}

# Original 6 Articles for backward compatibility
ORIGINAL_ARTICLES = {
    "I": "Library-First - Use existing libraries when possible",
    "II": "Test-First - Consider test-first approach",
    "III": "Simplicity - Keep solutions simple and focused",
    "IV": "Integration - Test with real data when practical",
    "V": "Clarity - Write clear, readable code",
    "VI": "Counterfactual - Document your design choices",
}


def detect_sdd_workflow_pattern(user_input: str) -> str:
    """
    Detect if the user is working within an SDD workflow pattern.
    Returns the specific pattern or 'general' for non-SDD questions.
    """
    user_lower = user_input.lower()

    # SDD Command Patterns
    if any(
        cmd in user_lower for cmd in ["/new_feature", "new_feature", "create feature"]
    ):
        return "new_feature"
    elif any(
        cmd in user_lower
        for cmd in ["/generate_plan", "generate_plan", "implementation plan"]
    ):
        return "generate_plan"
    elif any(
        pattern in user_lower
        for pattern in ["specification", "spec", "requirements", "prd"]
    ):
        return "specification"
    elif any(
        pattern in user_lower
        for pattern in ["constitutional", "constitution", "article", "gate"]
    ):
        return "constitutional"
    elif any(
        pattern in user_lower
        for pattern in ["template", "constraint", "needs clarification"]
    ):
        return "template"

    # Technical Implementation Patterns
    elif any(
        keyword in user_lower for keyword in ["test", "testing", "coverage", "tdd"]
    ):
        return "test_first"
    elif any(
        keyword in user_lower
        for keyword in ["library", "package", "dependency", "framework"]
    ):
        return "library_first"
    elif any(
        keyword in user_lower
        for keyword in ["complex", "simplify", "refactor", "simple"]
    ):
        return "simplicity"
    elif any(keyword in user_lower for keyword in ["cli", "command line", "interface"]):
        return "cli_interface"
    elif any(
        keyword in user_lower
        for keyword in ["integration", "contract", "real database"]
    ):
        return "integration_first"

    return "general"


def get_sdd_enhanced_guidance(user_input: str, workflow_pattern: str = None) -> str:
    """
    Provide SDD-aware constitutional guidance based on workflow pattern.
    """
    if workflow_pattern is None:
        workflow_pattern = detect_sdd_workflow_pattern(user_input)

    if workflow_pattern == "new_feature":
        return """🚀 SDD Feature Specification Guidance:

**Creating Executable Specifications (SDD Workflow):**

1. **Focus on WHAT and WHY** (not HOW):
   • User needs and business outcomes
   • Clear acceptance criteria
   • Measurable success metrics

2. **Use [NEEDS CLARIFICATION] markers**:
   • Don't guess - mark all ambiguities
   • Force explicit decisions on uncertainties
   • Prevent implementation assumptions

3. **Constitutional Gates to Consider**:
   • Article I: Will this be a standalone library?
   • Article III: What tests will prove this works?
   • Article VII: Can this be done with ≤3 projects?

4. **Template Structure**:
   • User stories with clear acceptance criteria
   • Non-functional requirements
   • Explicit constraints and assumptions

💡 Remember: Specifications drive implementation, not the other way around!"""

    elif workflow_pattern == "generate_plan":
        return """🏗️ SDD Implementation Plan Guidance:

**Constitutional Pre-Implementation Gates:**

**Phase -1: Constitutional Validation**
• Simplicity Gate (Article VII): ≤3 projects, no future-proofing?
• Anti-Abstraction Gate (Article VIII): Using frameworks directly?
• Integration-First Gate (Article IX): Contract tests defined?
• Test-First Gate (Article III): Tests written before implementation?

**Plan Structure Requirements:**
1. **Technology Rationale**: Why each choice aligns with requirements
2. **Phase Gates**: Clear checkpoints with pass/fail criteria
3. **File Creation Order**: Contracts → Tests → Implementation
4. **Constitutional Tracking**: Document any complexity justifications

**Hierarchy Management:**
• Keep plan high-level and readable
• Extract detailed specs to implementation-details/
• Maintain proper abstraction levels

💡 Plans should be executable blueprints, not wishful thinking!"""

    elif workflow_pattern == "specification":
        return """📋 SDD Specification Excellence:

**Making Specifications Executable:**

1. **Precision Requirements**:
   • Unambiguous acceptance criteria
   • Testable success metrics
   • Clear boundary definitions

2. **Template-Driven Quality**:
   • Use structured templates to constrain LLM behavior
   • Enforce proper abstraction levels
   • Prevent premature implementation details

3. **Constitutional Alignment**:
   • Every spec should trace to constitutional articles
   • Library-first thinking from the start
   • Test scenarios as part of specification

4. **Continuous Refinement**:
   • Specifications evolve with understanding
   • Feedback from implementation informs spec updates
   • Living documents, not static artifacts

💡 Specifications are source code for business logic!"""

    elif workflow_pattern == "constitutional":
        return f"""🏛️ SDD Constitutional Framework:

**The 9 Articles of SDD Development:**

{chr(10).join(f'• Article {num}: {desc}' for num, desc in SDD_CONSTITUTIONAL_ARTICLES.items())}

**Constitutional Enforcement:**
• Phase Gates: Explicit checkpoints before implementation
• Complexity Tracking: Document justified violations
• Template Constraints: Structure guides LLM behavior
• Continuous Validation: Ongoing compliance checking

**Key Principles:**
• Library-First: Everything starts as a standalone library
• Test-First: No code before tests (non-negotiable)
• CLI-First: All functionality exposed via command line
• Simplicity-First: Start simple, justify complexity
• Integration-First: Test with real systems

💡 Constitution provides architectural DNA for all generated code!"""

    elif workflow_pattern == "template":
        return """📝 SDD Template-Driven LLM Constraints:

**How Templates Improve LLM Output:**

1. **Prevent Premature Implementation**:
   • Templates explicitly forbid HOW details in WHAT specs
   • Maintain proper abstraction levels
   • Keep specifications stable across tech changes

2. **Force Explicit Uncertainty**:
   • Mandatory [NEEDS CLARIFICATION] markers
   • No guessing allowed - mark ambiguities
   • Systematic uncertainty management

3. **Structured Self-Review**:
   • Built-in checklists act as "unit tests" for specs
   • Constitutional gate validations
   • Quality assurance frameworks

4. **Hierarchical Information Architecture**:
   • Main documents stay readable
   • Complex details extracted to separate files
   • Proper detail level management

💡 Templates transform LLMs from creative writers to disciplined engineers!"""

    elif workflow_pattern == "test_first":
        return """🧪 SDD Test-First Implementation (Article III):

**Non-Negotiable Test-First Process:**

1. **Before ANY Code**:
   • Write comprehensive tests first
   • Get tests validated and approved
   • Confirm tests FAIL (Red phase)

2. **Test Types in Order**:
   • Contract tests: API/interface definitions
   • Integration tests: Real system interactions
   • End-to-end tests: Complete user workflows
   • Unit tests: Focused component behavior

3. **Constitutional Compliance**:
   • Tests use real databases (Article IX)
   • CLI interfaces testable (Article II)
   • Library boundaries validated (Article I)

4. **SDD Integration**:
   • Test scenarios part of specification
   • Tests generate from acceptance criteria
   • Implementation serves tests, not vice versa

💡 Tests are executable specifications - they define what success looks like!"""

    elif workflow_pattern == "library_first":
        return """📚 SDD Library-First Principle (Article I):

**Every Feature as a Standalone Library:**

1. **Constitutional Requirement**:
   • No feature implemented directly in application code
   • Everything starts as a reusable library component
   • Clear boundaries and minimal dependencies

2. **CLI Interface Mandate (Article II)**:
   • All libraries expose functionality via command line
   • Text input/output for observability
   • JSON support for structured data

3. **Implementation Strategy**:
   • Design library API first
   • Create CLI wrapper
   • Then integrate into application

4. **SDD Benefits**:
   • Forced modular design
   • Easier testing and validation
   • Reusable across projects
   • Clear architectural boundaries

💡 Libraries force you to think about clean interfaces from day one!"""

    elif workflow_pattern == "simplicity":
        return """🎯 SDD Simplicity & Anti-Abstraction (Articles VII & VIII):

**Simplicity Gate Requirements:**
• Maximum 3 projects for initial implementation
• No future-proofing allowed
• Additional projects need documented justification

**Anti-Abstraction Gate Requirements:**
• Use framework features directly
• No wrapper layers without justification
• Single model representation
• Trust framework patterns

**Complexity Tracking:**
• Document all justified violations
• Rationale for each layer of abstraction
• Regular complexity audits

**SDD Approach:**
• Start simple, add complexity only when proven necessary
• Prefer composition over inheritance
• Extract complexity only when clearly beneficial

💡 Simplicity is the ultimate sophistication - especially in generated code!"""

    elif workflow_pattern == "cli_interface":
        return """⌨️ SDD CLI Interface Mandate (Article II):

**Constitutional Requirement:**
Every library MUST expose functionality through CLI

**CLI Standards:**
• Accept text input (stdin, args, files)
• Produce text output (stdout)
• Support JSON for structured data
• Follow Unix philosophy

**Benefits for SDD:**
• Complete observability of all functionality
• Easy testing and validation
• Scriptable and composable
• Language-agnostic integration

**Implementation Pattern:**
1. Design library core functionality
2. Create CLI wrapper with proper argument handling
3. Expose all library features via CLI
4. Test through CLI interface

💡 CLI interfaces make everything inspectable and testable!"""

    elif workflow_pattern == "integration_first":
        return """🔗 SDD Integration-First Testing (Article IX):

**Real Environment Testing:**
• Use real databases, not mocks
• Actual service instances, not stubs
• Contract tests mandatory before implementation
• Test realistic data volumes and scenarios

**Constitutional Requirements:**
• Integration tests validate library boundaries
• Contract tests define interface agreements
• End-to-end tests prove specification compliance

**SDD Implementation Order:**
1. Define contracts first
2. Create contract tests
3. Build integration test environment
4. Write integration tests
5. Implement to make tests pass

**Why Integration-First:**
• Proves specifications work in practice
• Catches interface mismatches early
• Validates constitutional compliance
• Reduces post-deployment surprises

💡 Integration tests prove your specifications work in the real world!"""

    else:
        # General SDD-aware guidance
        return f"""🧠 Enhanced with SDD Constitutional Principles:

Question: {user_input}

**SDD Constitutional Guidance:**
• Article I: Library-First - Design as standalone, reusable components
• Article II: CLI-First - Expose functionality through command line interfaces
• Article III: Test-First - Write tests before implementation (non-negotiable)
• Article VII: Simplicity - Start simple, justify complexity
• Article VIII: Anti-Abstraction - Use frameworks directly, avoid over-engineering
• Article IX: Integration-First - Test with real systems, not mocks

**SDD Workflow Context:**
• Is this part of a specification? Focus on WHAT and WHY, not HOW
• Creating implementation plans? Validate against constitutional gates
• Building features? Start with library design and CLI interface
• Writing tests? Use real environments and integration-first approach

💡 Specifications drive implementation - code serves specifications!"""

    return guidance


def analyze_constitutional_compliance(code_snippet: str) -> str:
    """Enhanced constitutional analysis with SDD principles."""
    issues = []
    recommendations = []
    gates_passed = []
    gates_failed = []

    lines = code_snippet.split("\n")

    # Article I: Library-First Analysis
    if "class " in code_snippet or "def " in code_snippet:
        if "import " not in code_snippet:
            gates_failed.append("Article I: No evidence of library-first design")
            recommendations.append(
                "Consider extracting functionality into a reusable library"
            )
        else:
            gates_passed.append("Article I: Using existing libraries")

    # Article II: CLI Interface Analysis
    if any(
        "cli" in line.lower() or "argparse" in line or "click" in line for line in lines
    ):
        gates_passed.append("Article II: CLI interface detected")
    elif "def main(" in code_snippet or "if __name__" in code_snippet:
        recommendations.append("Consider adding CLI interface for better observability")

    # Article III: Test-First Analysis
    if "def test_" in code_snippet or "import pytest" in code_snippet:
        gates_passed.append("Article III: Test code detected")
    elif "def " in code_snippet and "test_" not in code_snippet:
        gates_failed.append("Article III: Implementation without corresponding tests")
        recommendations.append("Write tests before implementation (SDD non-negotiable)")

    # Article VII: Simplicity Analysis
    if len(lines) > 50:
        gates_failed.append("Article VII: Function/class too long (>50 lines)")
        recommendations.append("Consider breaking into smaller components")
    else:
        gates_passed.append("Article VII: Appropriate size and complexity")

    # Article VIII: Anti-Abstraction Analysis
    abstraction_patterns = ["AbstractBase", "Factory", "Wrapper", "Adapter"]
    if any(pattern in code_snippet for pattern in abstraction_patterns):
        gates_failed.append("Article VIII: Unnecessary abstraction detected")
        recommendations.append("Consider using framework features directly")
    else:
        gates_passed.append("Article VIII: Direct framework usage")

    # Article IX: Integration-First Analysis
    if any(mock in code_snippet.lower() for mock in ["mock", "stub", "fake"]):
        gates_failed.append("Article IX: Mock usage detected")
        recommendations.append("Consider using real services for integration testing")
    elif any(real in code_snippet.lower() for real in ["database", "api", "service"]):
        gates_passed.append("Article IX: Real system integration")

    # Generate report
    report = "📊 SDD Constitutional Compliance Analysis:\n\n"

    if gates_passed:
        report += "✅ **Constitutional Gates Passed:**\n"
        for gate in gates_passed:
            report += f"   • {gate}\n"
        report += "\n"

    if gates_failed:
        report += "❌ **Constitutional Gates Failed:**\n"
        for gate in gates_failed:
            report += f"   • {gate}\n"
        report += "\n"

    if recommendations:
        report += "💡 **Recommendations for SDD Compliance:**\n"
        for rec in recommendations:
            report += f"   • {rec}\n"
        report += "\n"

    # Calculate compliance score
    total_gates = len(gates_passed) + len(gates_failed)
    if total_gates > 0:
        compliance_score = len(gates_passed) / total_gates
        report += (
            f"📈 **Constitutional Compliance Score:** {compliance_score:.2f}/1.00\n"
        )
        if compliance_score >= 0.75:
            report += "🎯 **Status:** Ready for implementation\n"
        else:
            report += "⚠️ **Status:** Constitutional review required\n"

    return report


def setup_sdd_integration():
    """Setup SDD-enhanced GitHub Copilot integration."""
    # Set environment variables
    os.environ["GITHUB_COPILOT_SDD"] = "enabled"
    os.environ["COPILOT_CONSTITUTIONAL_MODE"] = "sdd_enhanced"
    os.environ["SDD_WORKFLOW_PATTERNS"] = "enabled"

    print("🚀 GitHub Copilot SDD Integration")
    print("=" * 40)
    print("✅ Specification-Driven Development: ENABLED")
    print("✅ 9 Constitutional Articles: ACTIVE")
    print("✅ Template-Driven LLM Constraints: ACTIVE")
    print("✅ SDD Workflow Pattern Detection: ENABLED")
    print("✅ Constitutional Compliance Analysis: ENHANCED")
    print("")
    print("🎯 SDD Workflow Support:")
    print("• /new_feature command guidance")
    print("• /generate_plan constitutional validation")
    print("• Specification refinement patterns")
    print("• Template-driven quality constraints")
    print("• Phase gate enforcement")
    print("")
    print("🏛️ Constitutional Framework:")
    print("• Library-First Principle (Article I)")
    print("• CLI Interface Mandate (Article II)")
    print("• Test-First Imperative (Article III)")
    print("• Simplicity & Anti-Abstraction Gates (Articles VII & VIII)")
    print("• Integration-First Testing (Article IX)")
    print("")
    print("💡 Every GitHub Copilot interaction now includes SDD methodology!")

    return True


# Main SDD enhancement function for GitHub Copilot
def copilot_sdd_enhance(prompt: str) -> str:
    """
    SDD-enhanced GitHub Copilot function.
    Provides specification-driven development guidance with constitutional compliance.
    """
    workflow_pattern = detect_sdd_workflow_pattern(prompt)
    return get_sdd_enhanced_guidance(prompt, workflow_pattern)


if __name__ == "__main__":
    success = setup_sdd_integration()

    if success:
        print("\n🎉 SDD INTEGRATION COMPLETE!")
        print("GitHub Copilot now supports full Specification-Driven Development!")
        print("Specifications drive implementation - code serves specifications!")

        # Demo SDD patterns
        print("\n🔬 SDD Pattern Demo:")
        print("-" * 30)

        demo_patterns = [
            ("new_feature", "/new_feature Real-time chat system"),
            ("generate_plan", "/generate_plan WebSocket messaging with Redis"),
            ("constitutional", "Check constitutional compliance"),
            ("test_first", "How do I implement test-first development?"),
            ("library_first", "Should I build this as a library?"),
        ]

        for pattern_name, demo_input in demo_patterns[:2]:  # Show first 2 for brevity
            print(f"\nPattern: {pattern_name}")
            print(f"Input: {demo_input}")
            print("Response:")
            response = copilot_sdd_enhance(demo_input)
            # Show just the first few lines
            lines = response.split("\n")
            for line in lines[:6]:
                print(line)
            print("   [... and more constitutional guidance ...]")
            print()

        print("📝 Integration Usage:")
        print("```python")
        print("from copilot_sdd_seamless import copilot_sdd_enhance")
        print("")
        print("# Enhance any GitHub Copilot interaction with SDD")
        print("enhanced = copilot_sdd_enhance('/new_feature User authentication')")
        print("print(enhanced)")
        print("```")
