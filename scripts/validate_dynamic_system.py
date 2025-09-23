#!/usr/bin/env python3
"""
Super Alita Dynamic System Validation Script
Tests all dynamic components and provides comprehensive status
"""

import sys
from typing import Any

import requests


def test_health() -> bool:
    """Test basic health endpoint"""
    try:
        response = requests.get("http://127.0.0.1:8080/healthz", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health Check: {data['status']}")
            return True
        else:
            print(f"❌ Health Check Failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health Check Error: {e}")
        return False


def test_tools_catalog() -> dict[str, Any]:
    """Test dynamic tools catalog"""
    try:
        response = requests.get("http://127.0.0.1:8080/tools/catalog", timeout=5)
        if response.status_code == 200:
            tools = response.json()
            tool_names = [tool.get("name", "unnamed") for tool in tools]
            print(f"✅ Tools Catalog: {len(tools)} tools available")
            print(
                f"   📋 Tool Names: {', '.join(tool_names[:5])}{'...' if len(tool_names) > 5 else ''}"
            )

            # Check for key dynamic tools
            dynamic_tools = [
                "deepconf_consensus",
                "reug_start_turn",
                "reug_stream_next",
            ]
            found_dynamic = [tool for tool in dynamic_tools if tool in tool_names]
            print(f"   🎯 Dynamic Tools Found: {', '.join(found_dynamic)}")

            return {"success": True, "tools": tools, "count": len(tools)}
        else:
            print(f"❌ Tools Catalog Failed: {response.status_code}")
            return {"success": False}
    except Exception as e:
        print(f"❌ Tools Catalog Error: {e}")
        return {"success": False}


def test_consensus_ability() -> bool:
    """Test dynamic consensus ability"""
    try:
        payload = {
            "prompt": "What is machine learning?",
            "num_samples": 2,
            "temperature": 0.7,
        }
        response = requests.post(
            "http://127.0.0.1:8080/ability/execute/deepconf_consensus",
            json=payload,
            timeout=120,
        )
        if response.status_code == 200:
            result = response.json()
            consensus_text = result.get("result", {}).get("consensus_text", "")
            confidence = result.get("result", {}).get("consensus_confidence", 0)

            print("✅ Consensus Ability Working!")
            print(f"   🎯 Confidence: {confidence:.2f}")
            print(
                f"   📝 Response: {consensus_text[:100]}{'...' if len(consensus_text) > 100 else ''}"
            )
            return True
        else:
            print(f"❌ Consensus Test Failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Consensus Test Error: {e}")
        return False


def test_streaming_chat() -> bool:
    """Test streaming chat endpoint (POST with GET fallback)."""
    try:
        payload = {
            "message": "Hello! Can you help me test the system?",
            "session_id": "validation_test",
        }
        try:
            response = requests.post(
                "http://127.0.0.1:8080/v1/chat/stream",
                json=payload,
                timeout=10,
                stream=True,
            )
        except Exception:
            # Fallback to GET variant used by EventSource
            response = requests.get(
                "http://127.0.0.1:8080/v1/chat/stream",
                params={"q": payload["message"], "session_id": payload["session_id"]},
                timeout=10,
                stream=True,
            )
        if response.status_code == 200:
            first_chunk = next(response.iter_lines(decode_unicode=True), None)
            if first_chunk:
                print("✅ Streaming Chat: Working")
                print(
                    f"   📡 First Event: {first_chunk[:80]}"
                    f"{'...' if len(first_chunk) > 80 else ''}"
                )
                return True
            print("❌ Streaming Chat: No content received")
            return False
        print(f"❌ Streaming Chat Failed: {response.status_code}")
        return False
    except Exception as e:
        print(f"❌ Streaming Chat Error: {e}")
        return False


def main():
    """Run all validation tests"""
    print("🚀 Super Alita Dynamic System Validation")
    print("=" * 50)

    tests = [
        ("Health Check", test_health),
        ("Tools Catalog", lambda: test_tools_catalog()["success"]),
        ("Consensus Ability", test_consensus_ability),
        ("Streaming Chat", test_streaming_chat),
    ]

    results = []
    for name, test_func in tests:
        print(f"\n🧪 Testing {name}...")
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print(f"❌ {name} Exception: {e}")
            results.append((name, False))

    print("\n" + "=" * 50)
    print("📊 VALIDATION SUMMARY")
    print("=" * 50)

    passed = sum(1 for _, success in results if success)
    total = len(results)

    for name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status}: {name}")

    print(f"\n🎯 Overall: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 ALL SYSTEMS OPERATIONAL - Dynamic Discovery Working!")
        return 0
    else:
        print("⚠️  Some systems need attention")
        return 1


if __name__ == "__main__":
    sys.exit(main())
