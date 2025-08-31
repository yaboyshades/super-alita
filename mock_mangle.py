#!/usr/bin/env python3
"""
Mock implementation of the Mangle binary for testing purposes.

This script simulates the behavior of Google's Mangle deductive database tool
by processing input files and returning expected JSON responses based on the
queries it receives. It's used during development to test the MangleAbility
integration without requiring the actual Mangle binary.

Usage:
    python mock_mangle.py query --format json input_file.mgl
"""

import argparse
import json
import re
import sys
from typing import Any


def parse_mangle_file(file_path: str) -> dict[str, Any]:
    """Parse a Mangle file and extract facts, rules, and queries.

    Args:
        file_path: Path to Mangle file

    Returns:
        Dictionary with parsed facts, rules, and queries
    """
    with open(file_path) as f:
        content = f.read()

    facts = []
    rules = []
    queries = []

    # Extract facts (simple statements ending with a period)
    fact_pattern = r"([a-zA-Z_][a-zA-Z0-9_]*\([^)]*\))\."
    facts = re.findall(fact_pattern, content)

    # Extract rules (statements with :-)
    rule_pattern = r"([a-zA-Z_][a-zA-Z0-9_]*\([^)]*\)\s*:-[^.]*)\."
    rules = re.findall(rule_pattern, content)

    # Extract queries (statements ending with a question mark)
    query_pattern = r"([a-zA-Z_][a-zA-Z0-9_]*\([^)]*\))\?"
    queries = re.findall(query_pattern, content)

    return {
        "facts": facts,
        "rules": rules,
        "queries": queries
    }


def process_query(
    query: str, parsed_data: dict[str, Any]
) -> list[dict[str, Any]]:
    """Process a Mangle query and return mock results.

    Args:
        query: Query string
        parsed_data: Parsed Mangle data

    Returns:
        List of result dictionaries
    """
    # Extract the query predicate name
    predicate_match = re.match(r"([a-zA-Z_][a-zA-Z0-9_]*)\(", query)
    if not predicate_match:
        return []

    predicate = predicate_match.group(1)

    # Extract variables in the query
    vars_match = re.search(r"\((.*)\)", query)
    if not vars_match:
        return []

    variables = [v.strip() for v in vars_match.group(1).split(",")]

    # Handle specific known queries with mock responses
    if predicate == "vulnerable" or predicate == "known_vulnerability":
        return [
            {"Name": "log4j", "Version": "2.14.0"},
            {"Name": "junit", "Version": "4.13.1"}
        ]
    elif predicate == "affected_by_auth_vuln":
        return [
            {"X": "Frontend"},
            {"X": "Backend"}
        ]
    elif predicate == "non_compliant":
        return [
            {"Component": "api_server", "Policy": "tls_version"},
            {"Component": "database", "Policy": "authentication"}
        ]
    elif "transitive_depends_on" in query:
        return [
            {"X": "Frontend", "Y": "Authentication"},
            {"X": "Backend", "Y": "Authentication"}
        ]

    # Default response based on the query variables
    # For example, if the query is component(X)?, return some component names
    if predicate == "component":
        return [
            {"X": "Frontend"},
            {"X": "Backend"},
            {"X": "Database"},
            {"X": "Authentication"}
        ]

    # Generic fallback response
    return [
        {variables[0]: "Result1"},
        {variables[0]: "Result2"}
    ]


def handle_explain(query: str) -> dict[str, Any]:
    """Generate a mock explanation for a query.

    Args:
        query: Query to explain

    Returns:
        Dictionary with explanation
    """
    return {
        "explanation": f"""
Explanation for query: {query}

1. The query matches the following facts:
   - component('api_server')
   - component_config('api_server', 'tls_version', '1.1')

2. This triggers the rule:
   non_compliant(Component, 'tls_version') :-
       component_config(Component, 'tls_version', Version),
       Version < '1.2'.

3. Since '1.1' < '1.2', the rule concludes:
   - non_compliant('api_server', 'tls_version')

This indicates that the api_server component is using TLS version 1.1,
which does not meet the minimum requirement of TLS 1.2 specified in the security policy.
"""
    }


def main():
    """Main function to process arguments and generate mock responses."""
    parser = argparse.ArgumentParser(description="Mock Mangle binary")
    parser.add_argument("command", help="Command to execute (e.g., query, explain)")
    parser.add_argument("--format", help="Output format", default="json")
    parser.add_argument("file", help="Input Mangle file")
    args = parser.parse_args()

    if args.command == "query":
        parsed_data = parse_mangle_file(args.file)
        if parsed_data["queries"]:
            query = parsed_data["queries"][0]
            results = process_query(query, parsed_data)
            print(json.dumps(results, indent=2))
        else:
            print(json.dumps([], indent=2))
    elif args.command == "explain":
        parsed_data = parse_mangle_file(args.file)
        if parsed_data["queries"]:
            query = parsed_data["queries"][0]
            explanation = handle_explain(query)
            print(json.dumps(explanation, indent=2))
        else:
            print(json.dumps({"explanation": "No query found"}, indent=2))
    else:
        sys.stderr.write(f"Unknown command: {args.command}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
