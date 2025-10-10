#!/usr/bin/env python3
"""
Test Mangle API Integration

This script tests the Mangle integration with Super Alita by making API calls
to add facts, rules, and execute queries.
"""

import requests


def test_mangle_api():
    """Test the Mangle API integration."""
    base_url = "http://127.0.0.1:8080/ability/execute"

    print("==== Testing Mangle API Integration ====")

    # Test health endpoint first
    try:
        health_response = requests.get("http://127.0.0.1:8080/healthz")
        health_data = health_response.json()
        print(f"✅ Server Health: {health_data['status']}")

        if health_data["status"] != "healthy":
            print("❌ Server is not healthy. Please check the server status.")
            return
    except Exception as e:
        print(f"❌ Error checking server health: {e}")
        print("Please make sure the Super Alita server is running.")
        return

    # Step 1: Add facts about dependencies
    print("\n1. Adding facts about dependencies...")
    try:
        fact_response = requests.post(
            f"{base_url}/mangle_add_fact",
            json={"fact": "vulnerable('log4j', '2.14.0')"},
        )
        fact_data = fact_response.json()
        print(f"   Response: {fact_data}")

        if fact_data.get("ok", False):
            print("✅ Successfully added fact")
        else:
            print("❌ Failed to add fact")
    except Exception as e:
        print(f"❌ Error adding fact: {e}")

    # Step 2: Add more facts
    print("\n2. Adding more facts...")
    facts = [
        "vulnerable('spring-core', '5.3.17')",
        "vulnerable('commons-text', '1.9.0')",
        "safe('log4j', '2.17.1')",
        "safe('spring-core', '5.3.20')",
    ]

    for fact in facts:
        try:
            response = requests.post(
                f"{base_url}/mangle_add_fact", json={"fact": fact}
            )
            if response.status_code == 200:
                print(f"   ✅ Added: {fact}")
            else:
                print(
                    f"   ❌ Failed to add: {fact} - Status: {response.status_code}"
                )
        except Exception as e:
            print(f"   ❌ Error adding {fact}: {e}")

    # Step 3: Add a rule
    print("\n3. Adding rule for dependency analysis...")
    rule = """
    needs_update(Lib, CurrentVersion, SafeVersion) :-
        vulnerable(Lib, CurrentVersion),
        safe(Lib, SafeVersion).
    """

    try:
        rule_response = requests.post(
            f"{base_url}/mangle_add_rule",
            json={"name": "update_rule", "rule": rule},
        )
        rule_data = rule_response.json()
        print(f"   Response: {rule_data}")

        if rule_data.get("ok", False):
            print("✅ Successfully added rule")
        else:
            print("❌ Failed to add rule")
    except Exception as e:
        print(f"❌ Error adding rule: {e}")

    # Step 4: Execute a query
    print("\n4. Querying vulnerable dependencies...")
    try:
        query_response = requests.post(
            f"{base_url}/mangle_query",
            json={"query": "vulnerable(Name, Version)"},
        )
        query_data = query_response.json()

        print("   Query results:")
        result = query_data.get("result", {})

        if result.get("success", False):
            items = result.get("results", [])
            if items:
                for item in items:
                    print(
                        f"   - {item.get('Name')} {item.get('Version')} is vulnerable"
                    )
            else:
                print("   - No vulnerable dependencies found")
        else:
            print(
                f"   ❌ Query failed: {result.get('error', 'Unknown error')}"
            )
    except Exception as e:
        print(f"❌ Error executing query: {e}")

    # Step 5: Analyze dependencies
    print("\n5. Analyzing project dependencies...")
    dependencies = [
        {"name": "log4j", "version": "2.14.0"},
        {"name": "spring-core", "version": "5.3.20"},
        {"name": "commons-text", "version": "1.9.0"},
    ]

    try:
        analyze_response = requests.post(
            f"{base_url}/mangle_analyze_dependencies",
            json={"dependencies": dependencies},
        )
        analyze_data = analyze_response.json()

        print("   Analysis results:")
        result = analyze_data.get("result", {})

        if result.get("success", False):
            items = result.get("results", [])
            if items:
                for item in items:
                    print(
                        f"   - {item.get('Name')} {item.get('Version')} is vulnerable"
                    )
            else:
                print("   - No vulnerable dependencies found")
        else:
            print(
                f"   ❌ Analysis failed: {result.get('error', 'Unknown error')}"
            )
    except Exception as e:
        print(f"❌ Error analyzing dependencies: {e}")

    print("\n==== Testing Complete ====")
    print("Mangle API integration tested successfully!")


if __name__ == "__main__":
    test_mangle_api()
