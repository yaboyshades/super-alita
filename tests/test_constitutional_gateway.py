#!/usr/bin/env python3
"""Test script for Constitutional Gateway API."""

import json

import requests

BASE_URL = "http://127.0.0.1:8081"


def test_constitutional_gateway():
    """Test the Constitutional Gateway endpoints."""
    print("🔬 Testing Constitutional Gateway API...")

    # Test health endpoint
    try:
        response = requests.get(
            f"{BASE_URL}/constitutional/health", timeout=10
        )
        print(f"✅ Health endpoint: {response.status_code}")
        print(f"   Response: {response.json()}")
    except Exception as e:
        print(f"❌ Health endpoint failed: {e}")

    # Test capabilities endpoint
    try:
        response = requests.get(
            f"{BASE_URL}/constitutional/capabilities", timeout=10
        )
        print(f"✅ Capabilities endpoint: {response.status_code}")
        print(f"   Response: {json.dumps(response.json(), indent=2)}")
    except Exception as e:
        print(f"❌ Capabilities endpoint failed: {e}")

    # Test workspace context endpoint
    try:
        payload = {
            "workspace_path": "d:\\Coding_Projects\\super-alita-clean",
            "file_pattern": "**/*.py",
            "include_content": False,
            "max_files": 10,
        }
        response = requests.post(
            f"{BASE_URL}/constitutional/context/workspace",
            json=payload,
            timeout=30,
        )
        print(f"✅ Workspace context endpoint: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"   Files found: {len(data.get('files', []))}")
            print(f"   Summary: {data.get('summary', {})}")
        else:
            print(f"   Error: {response.text}")
    except Exception as e:
        print(f"❌ Workspace context endpoint failed: {e}")

    # Test constitutional validation endpoint
    try:
        test_code = """
def hello_world():
    # TODO: implement this
    pass
"""
        payload = {"content": test_code}
        response = requests.post(
            f"{BASE_URL}/constitutional/enforce/validate",
            json=payload,
            timeout=30,
        )
        print(f"✅ Constitutional validation endpoint: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"   Compliance status: {data.get('compliance_status')}")
            print(f"   Score: {data.get('score')}/100")
            print(f"   Violations: {len(data.get('violations', []))}")
        else:
            print(f"   Error: {response.text}")
    except Exception as e:
        print(f"❌ Constitutional validation endpoint failed: {e}")

    print("\n🎉 Constitutional Gateway API testing completed!")


if __name__ == "__main__":
    test_constitutional_gateway()
