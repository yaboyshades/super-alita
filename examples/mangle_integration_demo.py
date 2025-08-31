#!/usr/bin/env python3
"""
Example usage of Mangle integration with Super Alita.

This script demonstrates how to use the Mangle integration for logical reasoning,
dependency analysis, and knowledge graph traversal.
"""

import asyncio
import sys
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.absolute()))

from src.abilities.mangle.mangle_ability import MangleAbility


async def demonstrate_dependency_analysis():
    """Demonstrate vulnerability detection with Mangle."""
    print("\n===== Vulnerability Analysis Example =====")

    # Initialize the Mangle ability
    mangle = MangleAbility()

    # Define some project dependencies
    dependencies = [
        {"name": "log4j", "version": "2.14.0"},
        {"name": "spring-core", "version": "5.3.20"},
        {"name": "junit", "version": "4.13.1"}
    ]

    # Add vulnerability data
    await mangle.add_fact("known_vulnerability('log4j', '2.14.0')")
    await mangle.add_fact("known_vulnerability('junit', '4.13.1')")

    # Analyze dependencies
    print("Analyzing dependencies for vulnerabilities...")
    result = await mangle.analyze_dependencies(dependencies)

    # Display results
    print(f"Found {result.get('count', 0)} vulnerable dependencies")
    for item in result.get("results", []):
        print(f"- {item.get('Name')} {item.get('Version')} is vulnerable")

    return result


async def demonstrate_knowledge_graph():
    """Demonstrate knowledge graph reasoning with Mangle."""
    print("\n===== Knowledge Graph Example =====")

    # Initialize the Mangle ability
    mangle = MangleAbility()

    # Add facts about software components
    print("Building knowledge graph with software component relationships...")
    await mangle.add_fact("component('Frontend')")
    await mangle.add_fact("component('Backend')")
    await mangle.add_fact("component('Database')")
    await mangle.add_fact("component('Authentication')")

    await mangle.add_fact("depends_on('Frontend', 'Backend')")
    await mangle.add_fact("depends_on('Backend', 'Database')")
    await mangle.add_fact("depends_on('Frontend', 'Authentication')")
    await mangle.add_fact("depends_on('Backend', 'Authentication')")

    # Add rule for transitive dependencies
    await mangle.add_rule(
        "transitive_dependency",
        """
        transitive_depends_on(X, Y) :- depends_on(X, Y).
        transitive_depends_on(X, Z) :- depends_on(X, Y), transitive_depends_on(Y, Z).
        """
    )

    # Add security context
    context = [
        {"relation": "has_vulnerability", "subject": "Authentication", "object": "CVE-2023-1234"}
    ]

    # Query for components affected by the vulnerability
    print("Querying for components affected by authentication vulnerability...")
    query = "affected_by_auth_vuln(X) :- component(X), transitive_depends_on(X, 'Authentication')"
    result = await mangle.knowledge_graph_query(query, context)

    # Display results
    print("Components potentially affected by the authentication vulnerability:")
    for item in result.get("results", []):
        print(f"- {item.get('X')}")

    return result


async def demonstrate_security_rules():
    """Demonstrate security policy rules with Mangle."""
    print("\n===== Security Policy Example =====")

    # Initialize the Mangle ability
    mangle = MangleAbility()

    # Add facts about security policies and configurations
    print("Defining security policies and configurations...")

    # Security policies
    await mangle.add_fact("security_policy('require_tls_1_2_minimum')")
    await mangle.add_fact("security_policy('require_authentication')")
    await mangle.add_fact("security_policy('no_default_passwords')")

    # Component configurations
    await mangle.add_fact("component_config('api_server', 'tls_version', '1.1')")
    await mangle.add_fact("component_config('api_server', 'authentication', 'enabled')")
    await mangle.add_fact("component_config('database', 'tls_version', '1.2')")
    await mangle.add_fact("component_config('database', 'default_password', 'none')")
    await mangle.add_fact("component_config('web_portal', 'tls_version', '1.2')")
    await mangle.add_fact("component_config('web_portal', 'authentication', 'enabled')")

    # Define security compliance rules
    await mangle.add_rule(
        "security_compliance",
        """
        tls_compliant(Component) :-
            component_config(Component, 'tls_version', Version),
            Version >= '1.2'.

        auth_compliant(Component) :-
            component_config(Component, 'authentication', 'enabled').

        password_compliant(Component) :-
            component_config(Component, 'default_password', 'none').

        fully_compliant(Component) :-
            tls_compliant(Component),
            auth_compliant(Component),
            password_compliant(Component).

        non_compliant(Component, 'tls_version') :-
            component_config(Component, 'tls_version', Version),
            Version < '1.2'.

        non_compliant(Component, 'authentication') :-
            component_config(Component, 'authentication', Status),
            Status != 'enabled'.

        non_compliant(Component, 'default_password') :-
            component_config(Component, 'default_password', Value),
            Value != 'none'.
        """
    )

    # Query for non-compliant components
    print("Finding non-compliant components...")
    result = await mangle.query("non_compliant(Component, Policy)")

    # Display results with explanation
    print("Non-compliant components:")
    for item in result.get("results", []):
        component = item.get("Component")
        policy = item.get("Policy")
        print(f"- {component} does not comply with {policy} policy")

    # Get explanation for a specific component
    print("\nGetting detailed compliance explanation for api_server...")
    explanation = await mangle.explain_query_results("non_compliant('api_server', Policy)")

    # Show explanation
    print("Explanation for api_server non-compliance:")
    explanation_text = explanation.get("explanation", "No explanation available")
    if isinstance(explanation_text, str):
        # Print first few lines of explanation
        lines = explanation_text.split('\n')
        for line in lines[:10]:
            print(f"  {line}")

    return result


async def main():
    """Run the Mangle integration examples."""
    print("=== Mangle Integration Demo ===")
    print("Demonstrating logical reasoning capabilities with Google's Mangle")

    # Run the examples
    await demonstrate_dependency_analysis()
    await demonstrate_knowledge_graph()
    await demonstrate_security_rules()

    print("\n=== Demo Complete ===")
    print("Mangle integration successfully demonstrated")


if __name__ == "__main__":
    asyncio.run(main())
