#!/usr/bin/env python3
"""Test autogen FastAPI integration."""

import requests


def test_autogen_routes():
    """Test autogen endpoints via HTTP requests."""
    base_url = "http://localhost:8000"
    
    # Test 1: List capabilities
    print("Testing /autogen/capabilities...")
    try:
        response = requests.get(f"{base_url}/autogen/capabilities", timeout=10)
        if response.status_code == 200:
            data = response.json()
            kinds_count = len(data.get('capability_kinds', []))
            print(f"✓ Capabilities: {kinds_count} kinds detected")
            print(f"  Available: {list(data.get('capability_kinds', []))}")
        else:
            print(f"✗ Capabilities failed: {response.status_code}")
    except Exception as e:
        print(f"✗ Capabilities error: {e}")
    
    # Test 2: Detect needs
    print("\nTesting /autogen/detect...")
    test_description = "I need a function to parse CSV files"
    try:
        response = requests.post(
            f"{base_url}/autogen/detect",
            json={"description": test_description},
            timeout=10
        )
        if response.status_code == 200:
            data = response.json()
            needs = data.get("detected_needs", [])
            print(f"✓ Detected {len(needs)} needs: {needs}")
        else:
            print(f"✗ Detect failed: {response.status_code}")
    except Exception as e:
        print(f"✗ Detect error: {e}")
    
    # Test 3: Trigger autogen (without actually running full pipeline)
    print("\nTesting /autogen/trigger...")
    try:
        response = requests.post(
            f"{base_url}/autogen/trigger",
            json={"description": "test capability for demo"},
            timeout=30
        )
        if response.status_code == 200:
            data = response.json()
            if data.get("status") == "success":
                print("✓ Trigger succeeded")
                print(f"  Result: {data.get('result', {})}")
            else:
                print(f"✗ Trigger failed: {data.get('error')}")
        else:
            print(f"✗ Trigger failed: {response.status_code}")
    except Exception as e:
        print(f"✗ Trigger error: {e}")

def main():
    """Main test runner."""
    print("Super Alita Autogen Integration Test")
    print("=" * 40)
    
    # Check if server is running
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            print("✓ Server is running")
        else:
            print("✗ Server health check failed")
            return
    except Exception:
        print("✗ Server not accessible - start server first")
        print("  Run: python src/main.py")
        return
    
    test_autogen_routes()
    print("\nTest completed!")

if __name__ == "__main__":
    main()