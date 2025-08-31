#!/usr/bin/env python3
"""
Mangle Validation Example Script

This example demonstrates the usage of Mangle validation capabilities
for output validation, tool execution controls, and method selection.
"""

import asyncio
import json
import os
import sys
from pathlib import Path

# Add parent directory to import path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.abilities.mangle.mangle_ability import MangleAbility
from src.abilities.mangle.mangle_validator import MangleValidator


async def setup_example():
    """Set up example data and directories."""
    # Create rules directory if it doesn't exist
    data_dir = Path("./data/mangle")
    data_dir.mkdir(parents=True, exist_ok=True)

    # Copy example rules if not already present
    example_rules = Path("./examples/mangle_rules_example.json")
    rules_file = data_dir / "rules.json"

    if example_rules.exists() and not rules_file.exists():
        rules_file.write_text(example_rules.read_text())
        print(f"✅ Copied example rules to {rules_file}")

    # Set up a mock Mangle binary if MANGLE_BIN_PATH not set
    if "MANGLE_BIN_PATH" not in os.environ:
        mock_path = Path("./mock_mangle.py")
        if not mock_path.exists():
            mock_content = """#!/usr/bin/env python3
import json
import sys

# Simple mock implementation of Mangle
def main():
    if len(sys.argv) > 1 and sys.argv[1] == "query":
        # Return empty results for any query
        print("[]")
    else:
        # Version info
        print("Mangle Mock v0.1")

if __name__ == "__main__":
    main()
"""
            mock_path.write_text(mock_content)
            mock_path.chmod(0o755)

        os.environ["MANGLE_BIN_PATH"] = str(mock_path)
        print(f"✅ Created mock Mangle binary at {mock_path}")


async def example_output_validation():
    """Example of validating output against policy rules."""
    print("\n--- Output Validation Example ---")

    mangle = MangleAbility()
    validator = MangleValidator(mangle)

    # Example financial advice text
    financial_text = """
    I recommend investing in tech stocks as they tend to perform well.
    You should put at least 30% of your portfolio in these high-growth assets.
    """

    # Validate against policy rules
    result = await validator.validate_output(
        output_text=financial_text,
        domain="finance",
        meta={"samples_valid": 3, "response_length": len(financial_text)}
    )

    print(f"Valid: {result['valid']}")
    if not result["valid"]:
        print(f"Violations: {result['violations']}")
        print(f"Confidence penalty: {result['confidence_penalty']}")

    # Example with disclaimer
    financial_text_with_disclaimer = """
    Some tech stocks have shown good historical performance.
    This is not financial advice, please consult with a financial advisor.
    """

    result = await validator.validate_output(
        output_text=financial_text_with_disclaimer,
        domain="finance"
    )

    print(f"\nWith disclaimer - Valid: {result['valid']}")


async def example_tool_validation():
    """Example of validating tool execution against policy rules."""
    print("\n--- Tool Execution Validation Example ---")

    mangle = MangleAbility()
    validator = MangleValidator(mangle)

    # Example risky tool execution
    result = await validator.validate_tool_execution(
        tool_name="file_write",
        params={"path": "/etc/passwd", "content": "new text"},
        context={"domain": "finance", "user_role": "standard"}
    )

    print(f"Tool authorized: {result['authorized']}")
    if not result['authorized']:
        print(f"Reason: {result['reason']}")

    # Example allowed tool execution
    result = await validator.validate_tool_execution(
        tool_name="file_read",
        params={"path": "/tmp/example.txt"},
        context={"domain": "development", "user_role": "admin"}
    )

    print(f"\nSafe tool authorized: {result['authorized']}")


async def example_method_selection():
    """Example of selecting a consensus method based on rules."""
    print("\n--- Consensus Method Selection Example ---")

    mangle = MangleAbility()
    validator = MangleValidator(mangle)

    # Technical domain selection
    result = await validator.select_consensus_method(
        domain="programming",
        sample_count=4,
        meta={"temperature": 0.7}
    )

    print(f"Technical domain - Selected method: {result['method']}")
    print(f"Reason: {result['reason']}")

    # Subjective domain selection
    result = await validator.select_consensus_method(
        domain="creative",
        sample_count=3
    )

    print(f"\nSubjective domain - Selected method: {result['method']}")
    print(f"Reason: {result['reason']}")


async def example_claim_verification():
    """Example of verifying claims in LLM output."""
    print("\n--- Claim Verification Example ---")

    mangle = MangleAbility()
    validator = MangleValidator(mangle)

    # Example with potentially invalid claims
    python_text = """
    Python 3.9 introduced the match statement, which provides pattern matching
    capabilities similar to switch statements in other languages.
    """

    result = await validator.verify_llm_claims(
        output_text=python_text,
        claims_type="software"
    )

    print(f"Claims verified: {result['verified']}")
    if not result['verified']:
        print(f"Invalid claims: {result['invalid_claims']}")
        print(f"Confidence adjustment: {result['confidence_adjustment']}")


async def main():
    """Run all examples."""
    await setup_example()
    await example_output_validation()
    await example_tool_validation()
    await example_method_selection()
    await example_claim_verification()


if __name__ == "__main__":
    asyncio.run(main())
