#!/usr/bin/env python3
"""
Debug the 500 error in consensus tool execution.
Following Super Alita debugging patterns.
"""

import asyncio
import httpx
import json
import traceback
from pathlib import Path


async def debug_consensus_execution():
    """Debug the specific 500 error in consensus tool execution."""
    print("🔍 Debugging Consensus 500 Error...")
    print("Following Super Alita troubleshooting patterns")

    # Test direct ability registry access
    await test_ability_registry_state()

    # Test simplified consensus call
    await test_simplified_consensus()

    # Check server logs indirectly
    await test_server_error_patterns()


async def test_ability_registry_state():
    """Test if the ability registry has the consensus tool properly loaded."""
    print("\n🔧 Testing Ability Registry State...")

    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            # Check health to see if registry is working
            response = await client.get("http://127.0.0.1:8080/healthz")
            if response.status_code == 200:
                health = response.json()
                registry_status = health.get("components", {}).get("ability_registry")

                if isinstance(registry_status, dict):
                    status = registry_status.get("status", "unknown")
                else:
                    status = str(registry_status)

                print(f"✅ Ability Registry Status: {status}")

                if status != "ok":
                    print(f"❌ Registry issue detected: {registry_status}")
                    return False
                else:
                    print(f"✅ Registry is healthy")
                    return True
            else:
                print(f"❌ Health check failed: {response.status_code}")
                return False

        except Exception as e:
            print(f"❌ Registry test error: {e}")
            return False


async def test_simplified_consensus():
    """Test consensus with minimal parameters to isolate the issue."""
    print("\n🧪 Testing Simplified Consensus Call...")

    # Try the simplest possible consensus request
    simple_message = "Use deepconf_consensus tool with prompt='Hello' and num_samples=1"

    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            # Start simple REUG turn
            response = await client.post(
                "http://127.0.0.1:8080/v1/chat/stream",
                json={"message": simple_message, "session_id": "debug_simple"},
            )

            if response.status_code == 200:
                result = response.json()
                run_id = result.get("run_id")
                print(f"✅ Simple REUG started: {run_id}")

                # Try one stream iteration
                stream_response = await client.post(
                    "http://127.0.0.1:8080/tools/reug_stream_next",
                    json={"run_id": run_id},
                )

                if stream_response.status_code == 200:
                    stream_result = stream_response.json()
                    chunks = stream_result.get("chunks", [])
                    print(f"✅ Stream response: {len(chunks)} chunks")

                    for i, chunk in enumerate(chunks):
                        print(f"   Chunk {i}: {str(chunk)[:100]}...")

                    return True
                elif stream_response.status_code == 500:
                    print(f"❌ 500 error confirmed in simple test")
                    print(f"   This suggests the issue is in tool execution itself")
                    return False
                else:
                    print(f"❌ Stream error: {stream_response.status_code}")
                    return False
            else:
                print(f"❌ Simple REUG failed: {response.status_code}")
                return False

        except Exception as e:
            print(f"❌ Simple test error: {e}")
            return False


async def test_server_error_patterns():
    """Test patterns that might reveal server-side issues."""
    print("\n🔍 Testing Server Error Patterns...")

    async with httpx.AsyncClient(timeout=30.0) as client:
        # Test 1: Non-consensus tool to see if error is consensus-specific
        print("🔧 Testing non-consensus tool...")

        try:
            response = await client.post(
                "http://127.0.0.1:8080/v1/chat/stream",
                json={
                    "message": "List the files in current directory using fs_read tool",
                    "session_id": "debug_fs",
                },
            )

            if response.status_code == 200:
                result = response.json()
                run_id = result.get("run_id")

                # Try streaming
                stream_response = await client.post(
                    "http://127.0.0.1:8080/tools/reug_stream_next",
                    json={"run_id": run_id},
                )

                if stream_response.status_code == 200:
                    print("✅ Non-consensus tools work - issue is consensus-specific")
                elif stream_response.status_code == 500:
                    print("❌ 500 error affects all tools - broader system issue")
                else:
                    print(
                        f"⚠️  Non-consensus tool status: {stream_response.status_code}"
                    )

        except Exception as e:
            print(f"⚠️  Non-consensus test error: {e}")

        # Test 2: Check if LLM component issues affect tool execution
        print("\n🔧 Testing LLM component interaction...")

        try:
            # Send a simple non-tool message
            response = await client.post(
                "http://127.0.0.1:8080/v1/chat/stream",
                json={
                    "message": "Hello, how are you?",
                    "session_id": "debug_simple_chat",
                },
            )

            if response.status_code == 200:
                result = response.json()
                run_id = result.get("run_id")

                stream_response = await client.post(
                    "http://127.0.0.1:8080/tools/reug_stream_next",
                    json={"run_id": run_id},
                )

                if stream_response.status_code == 200:
                    print("✅ Simple chat works - issue is tool-execution specific")
                else:
                    print(f"❌ Simple chat fails: {stream_response.status_code}")

        except Exception as e:
            print(f"⚠️  Simple chat test error: {e}")


async def main():
    """Main debugging runner."""
    print("🚀 Consensus 500 Error Debug Session")
    print("=" * 50)

    await debug_consensus_execution()

    print("\n" + "=" * 50)
    print("🎯 Debug Summary:")
    print("Check the output above for specific error patterns.")
    print("\nNext steps based on findings:")
    print("1. If registry is unhealthy → Check ability registration")
    print("2. If 500 is consensus-specific → Check consensus tool implementation")
    print("3. If 500 affects all tools → Check REUG router or LLM integration")
    print("4. If simple chat fails → Check core streaming infrastructure")


if __name__ == "__main__":
    asyncio.run(main())
