#!/usr/bin/env python3
"""Simple GPT-OSS test with non-streaming call."""

import asyncio
import os
import sys
from pathlib import Path

# Add src to path
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import httpx


async def test_simple_ollama():
    """Test simple non-streaming Ollama call."""
    print("🧪 Testing simple Ollama GPT-OSS call...")

    host = "http://127.0.0.1:11434"
    model = "gpt-oss:20b"

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": "Hello! Please respond with just 'Hi' and nothing else.",
            }
        ],
        "stream": False,
    }

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            print(f"📡 Calling {host}/api/chat with model {model}")

            response = await client.post(
                f"{host}/api/chat",
                headers={"Content-Type": "application/json"},
                json=payload,
            )

            if response.status_code == 200:
                data = response.json()
                message = data.get("message", {}).get("content", "")
                print(f"✅ Response: {message!r}")
                print(f"✅ Model: {data.get('model')}")
                print(f"✅ Duration: {data.get('total_duration', 0) / 1_000_000:.1f}ms")
                print("🎉 Simple Ollama test PASSED!")
                return True
            else:
                print(f"❌ HTTP {response.status_code}: {response.text}")
                return False

    except Exception as e:
        print(f"❌ Error: {e}")
        return False


async def test_ollama_streaming():
    """Test streaming Ollama call."""
    print("\n🧪 Testing streaming Ollama GPT-OSS call...")

    host = "http://127.0.0.1:11434"
    model = "gpt-oss:20b"

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Count to 3 slowly: one..."}],
        "stream": True,
    }

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            print(f"📡 Streaming {host}/api/chat with model {model}")

            async with client.stream(
                "POST", f"{host}/api/chat", json=payload
            ) as response:
                if response.status_code >= 400:
                    data = await response.aread()
                    print(f"❌ HTTP {response.status_code}: {data}")
                    return False

                chunks = []
                async for line in response.aiter_lines():
                    if not line:
                        continue

                    try:
                        import json

                        obj = json.loads(line)
                        msg = obj.get("message", {})
                        content = msg.get("content")
                        if content:
                            print(f"📝 Chunk: {content!r}")
                            chunks.append(content)

                        if obj.get("done"):
                            print("✅ Stream complete")
                            break

                    except json.JSONDecodeError:
                        continue

                full_response = "".join(chunks)
                print(f"✅ Complete response: {full_response!r}")
                print(f"✅ Total chunks: {len(chunks)}")
                print("🎉 Streaming Ollama test PASSED!")
                return True

    except Exception as e:
        print(f"❌ Streaming error: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_our_client():
    """Test our OllamaClient implementation."""
    print("\n🧪 Testing our OllamaClient implementation...")

    # Set environment for GPT-OSS
    os.environ["LLM_MODEL"] = "ollama:gpt-oss:20b"
    os.environ["OLLAMA_HOST"] = "http://127.0.0.1:11434"

    try:
        from reug_runtime.llm_client import get_llm_client

        client = get_llm_client(os.environ["LLM_MODEL"])
        print(f"✅ Client created: {type(client).__name__}")
        print(f"✅ Model name: {client.model_name}")

        # Test with longer timeout
        messages = [{"role": "user", "content": "Say 'Working!' briefly."}]
        print("🚀 Testing with 30 second timeout...")

        response_parts = []
        async for chunk in client.stream_chat(messages, timeout=30):
            content = chunk.get("content", "")
            if content:
                print(f"📝 Chunk: {content!r}")
                response_parts.append(content)

        full_response = "".join(response_parts)
        print(f"✅ Complete response: {full_response!r}")
        print("🎉 OllamaClient test PASSED!")
        return True

    except Exception as e:
        print(f"❌ OllamaClient test FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("🚀 GPT-OSS Ollama Integration Tests")
    print("=" * 60)

    async def run_all_tests():
        results = []

        # Test 1: Simple non-streaming
        results.append(await test_simple_ollama())

        # Test 2: Direct streaming
        results.append(await test_ollama_streaming())

        # Test 3: Our client implementation
        results.append(await test_our_client())

        return results

    results = asyncio.run(run_all_tests())

    print("\n" + "=" * 60)
    print("📊 SUMMARY:")
    print(f"  ✅ Simple call:      {'PASS' if results[0] else 'FAIL'}")
    print(f"  ✅ Direct streaming: {'PASS' if results[1] else 'FAIL'}")
    print(f"  ✅ OllamaClient:     {'PASS' if results[2] else 'FAIL'}")

    if all(results):
        print("\n🎉 All tests PASSED! GPT-OSS integration is working!")
    else:
        print("\n⚠️ Some tests failed. Check output above for details.")

    print("=" * 60)
