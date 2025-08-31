#!/usr/bin/env python3
"""
Test Ollama client integration with Super Alita
"""

import asyncio
import os
import sys

# Add the source directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

async def test_ollama_client():
    """Test the OllamaClient directly"""
    try:
        from reug_runtime.llm_client import OllamaClient
        
        print("🧪 Testing OllamaClient...")
        
        # Create client
        client = OllamaClient("gpt-oss:20b", "http://127.0.0.1:11434")
        
        # Test streaming
        messages = [{"role": "user", "content": "Hello, what model are you?"}]
        
        print("📡 Sending test message...")
        response_parts = []
        
        async for chunk in client.stream_chat(messages, timeout=30):
            content = chunk.get("content", "")
            if content:
                response_parts.append(content)
                print(f"📝 Chunk: {content}")
        
        full_response = "".join(response_parts)
        print(f"✅ Full response: {full_response}")
        
        return True
        
    except Exception as e:
        print(f"❌ OllamaClient test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_get_llm_client():
    """Test the get_llm_client function"""
    try:
        from reug_runtime.llm_client import get_llm_client
        
        print("\n🔧 Testing get_llm_client...")
        
        # Set environment
        os.environ["LLM_MODEL"] = "ollama:gpt-oss:20b"
        os.environ["OLLAMA_HOST"] = "http://127.0.0.1:11434"
        
        client = get_llm_client("ollama:gpt-oss:20b")
        print(f"✅ Got client: {type(client)}")
        
        # Test streaming
        messages = [{"role": "user", "content": "Hi"}]
        response_parts = []
        
        async for chunk in client.stream_chat(messages, timeout=30):
            content = chunk.get("content", "")
            if content:
                response_parts.append(content)
        
        full_response = "".join(response_parts)
        print(f"✅ Response: {full_response[:100]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ get_llm_client test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    print("🚀 Testing Ollama Integration\n")
    
    # Test direct client
    direct_ok = await test_ollama_client()
    
    # Test factory function  
    factory_ok = await test_get_llm_client()
    
    print("\n📋 Results:")
    print(f"  • Direct OllamaClient: {'✅' if direct_ok else '❌'}")
    print(f"  • get_llm_client factory: {'✅' if factory_ok else '❌'}")
    
    if direct_ok and factory_ok:
        print("\n🎉 Ollama integration is working! The issue may be elsewhere.")
    else:
        print("\n⚠️ Ollama integration has issues.")

if __name__ == "__main__":
    asyncio.run(main())