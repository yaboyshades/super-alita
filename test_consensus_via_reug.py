#!/usr/bin/env python3
"""Test enhanced consensus via REUG streaming system."""

import asyncio
import httpx
import json


async def test_consensus_via_reug():
    """Test consensus through REUG streaming system."""
    print("🧪 Testing Enhanced Consensus via REUG...")

    base_url = "http://127.0.0.1:8080"

    # Start a REUG turn with consensus tool request
    async with httpx.AsyncClient(timeout=60.0) as client:
        print("🔄 Starting REUG turn with consensus request...")

        # Start turn
        start_response = await client.post(
            f"{base_url}/v1/chat/stream",
            json={
                "message": "Use the deepconf_consensus tool to find consensus on this question: What is the capital of Japan? Use weighted_vote method with 3 samples.",
                "session_id": "test_enhanced_consensus",
            },
        )

        if start_response.status_code != 200:
            print(f"❌ Failed to start turn: {start_response.status_code}")
            print(f"Error: {start_response.text}")
            return

        print("✅ REUG turn started, streaming response...")

        # Stream the response
        full_response = ""
        async for line in start_response.aiter_lines():
            if line.startswith("data: "):
                try:
                    data = json.loads(line[6:])
                    event_type = data.get("type", "")

                    if event_type == "LLMChunk":
                        content = data.get("data", {}).get("content", "")
                        full_response += content
                        print(content, end="", flush=True)
                    elif event_type == "TaskFinished":
                        print(f"\n\n✅ Task finished: {data.get('data', {})}")
                        break
                    elif event_type in ["ToolCall", "ToolResult"]:
                        print(f"\n🔧 {event_type}: {data.get('data', {})}")

                except json.JSONDecodeError:
                    continue

        print(f"\n\n📝 Full Response Length: {len(full_response)} characters")


async def test_direct_tool_call():
    """Test calling consensus tool directly if possible."""
    print("\n🎯 Testing Direct Tool Call...")

    # Check if there's a direct tool endpoint
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            # Try to use the tool through some other endpoint
            response = await client.post(
                "http://127.0.0.1:8080/api/tools/execute",  # Guessing endpoint
                json={
                    "tool": "deepconf_consensus",
                    "params": {
                        "prompt": "What is the capital of France?",
                        "num_samples": 3,
                        "method": "weighted_vote",
                    },
                },
            )

            if response.status_code == 200:
                result = response.json()
                print(f"✅ Direct call successful: {result}")
            else:
                print(f"❌ Direct call failed: {response.status_code}")

        except Exception as e:
            print(f"❌ Direct call error: {e}")


async def main():
    """Run consensus tests."""
    await test_consensus_via_reug()
    await test_direct_tool_call()


if __name__ == "__main__":
    asyncio.run(main())
