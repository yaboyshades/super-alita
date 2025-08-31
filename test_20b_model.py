#!/usr/bin/env python3
"""
Test script to verify the 20B model is working with Super Alita
"""


import requests


def test_ollama_direct():
    """Test Ollama directly"""
    print("🧪 Testing Ollama directly...")
    try:
        # Test ollama endpoint
        response = requests.post(
            "http://127.0.0.1:11434/api/generate",
            json={
                "model": "gpt-oss:20b",
                "prompt": "Hello, what model are you?",
                "stream": False
            },
            timeout=30
        )
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Direct Ollama Response: {result.get('response', 'No response')}")
            return True
        else:
            print(f"❌ Ollama failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Ollama error: {e}")
        return False

def test_super_alita_config():
    """Check Super Alita's LLM configuration"""
    print("\n🔧 Testing Super Alita LLM configuration...")
    try:
        # Check if the LLM model endpoint exists
        response = requests.get("http://127.0.0.1:8080/healthz", timeout=10)
        if response.status_code == 200:
            health = response.json()
            print(f"✅ Super Alita Health: {health}")
            llm_status = health.get("components", {}).get("llm", {}).get("status")
            print(f"📊 LLM Status: {llm_status}")
            return llm_status == "ok"
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

def test_memory_usage():
    """Check current memory usage"""
    print("\n📊 Checking memory usage...")
    try:
        # Run ollama ps to check loaded models
        import subprocess
        result = subprocess.run(["ollama", "ps"], capture_output=True, text=True)
        if result.returncode == 0:
            print("📋 Loaded models:")
            print(result.stdout)
            return True
        else:
            print(f"❌ Failed to check models: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Memory check error: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Testing GPT-OSS 20B Model Integration\n")
    
    # Run all tests
    ollama_ok = test_ollama_direct()
    config_ok = test_super_alita_config()
    memory_ok = test_memory_usage()
    
    print("\n📋 Test Results:")
    print(f"  • Ollama Direct: {'✅' if ollama_ok else '❌'}")
    print(f"  • Super Alita Config: {'✅' if config_ok else '❌'}")
    print(f"  • Memory Usage: {'✅' if memory_ok else '❌'}")
    
    if all([ollama_ok, config_ok, memory_ok]):
        print("\n🎉 All tests passed! The 20B model should be working.")
    else:
        print("\n⚠️ Some tests failed. Check the configuration.")