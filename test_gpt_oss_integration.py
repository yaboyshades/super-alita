#!/usr/bin/env python3
"""Test script to verify GPT-OSS via Ollama integration."""

import asyncio
import os
import sys
from pathlib import Path

# Add src to path
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from reug_runtime.llm_client import get_llm_client


async def test_gpt_oss_ollama():
    """Test GPT-OSS integration via Ollama."""
    # Set environment for GPT-OSS
    os.environ["LLM_MODEL"] = "ollama:gpt-oss:20b"
    os.environ["OLLAMA_HOST"] = "http://127.0.0.1:11434"

    print("🧪 Testing GPT-OSS via Ollama integration...")
    print(f"LLM_MODEL: {os.environ['LLM_MODEL']}")
    print(f"OLLAMA_HOST: {os.environ['OLLAMA_HOST']}")

    try:
        client = get_llm_client(os.environ["LLM_MODEL"])
        print(f"✅ Client created: {type(client).__name__}")
        print(f"✅ Model name: {client.model_name}")

        # Test streaming
        messages = [{"role": "user", "content": "Hello! Please respond briefly."}]
        print("\n🚀 Testing streaming response...")

        response_parts = []
        async for chunk in client.stream_chat(messages, timeout=10):
            content = chunk.get("content", "")
            if content:
                print(f"📝 Chunk: {content!r}")
                response_parts.append(content)

        full_response = "".join(response_parts)
        print(f"\n✅ Complete response: {full_response!r}")
        print(f"✅ Total chunks: {len(response_parts)}")
        print("🎉 GPT-OSS integration test PASSED!")

    except Exception as e:
        print(f"❌ Test FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    success = asyncio.run(test_gpt_oss_ollama())
    sys.exit(0 if success else 1)
