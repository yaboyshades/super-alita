#!/usr/bin/env python3
"""
Direct Ollama integration test - bypassing Super Alita's LLM client
"""

import httpx
import json
import asyncio


async def chat_with_ollama(message: str):
    """Direct Ollama integration bypassing Super Alita's LLM client"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "http://127.0.0.1:11434/api/chat",
                json={
                    "model": "gpt-oss:20b",
                    "messages": [{"role": "user", "content": message}],
                    "stream": False
                },
                timeout=30.0
            )
            
            if response.status_code == 200:
                return response.json()["message"]["content"]
            else:
                return f"Error: Status {response.status_code} - {response.text}"
                
    except Exception as e:
        return f"Ollama connection error: {str(e)}"


async def test_streaming_chat(message: str):
    """Test streaming chat with Ollama"""
    print(f"🔄 Streaming test: '{message}'")
    try:
        async with httpx.AsyncClient() as client:
            async with client.stream(
                "POST",
                "http://127.0.0.1:11434/api/chat",
                json={
                    "model": "gpt-oss:20b",
                    "messages": [{"role": "user", "content": message}],
                    "stream": True
                },
                timeout=30.0
            ) as response:
                
                if response.status_code == 200:
                    print("📡 Response: ", end="", flush=True)
                    async for line in response.aiter_lines():
                        if line.strip():
                            try:
                                data = json.loads(line)
                                content = data.get("message", {}).get("content", "")
                                if content:
                                    print(content, end="", flush=True)
                                if data.get("done"):
                                    print("\n✅ Stream complete")
                                    break
                            except json.JSONDecodeError:
                                continue
                else:
                    print(f"❌ Error: Status {response.status_code}")
                    
    except Exception as e:
        print(f"❌ Streaming error: {e}")


async def main():
    """Main test function"""
    print("🚀 Testing Direct Ollama Integration")
    print("=" * 50)
    
    # Test 1: Simple chat
    print("\n📝 Test 1: Simple Chat")
    result = await chat_with_ollama("Hello! What model are you?")
    print(f"Response: {result}")
    
    # Test 2: Streaming chat
    print("\n📝 Test 2: Streaming Chat")
    await test_streaming_chat("Explain what you are in one sentence.")
    
    # Test 3: Technical question
    print("\n📝 Test 3: Technical Question")
    await test_streaming_chat("What's the difference between Python async and threading?")
    
    print("\n🎉 All tests completed!")


if __name__ == "__main__":
    asyncio.run(main())