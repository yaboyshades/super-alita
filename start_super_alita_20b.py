#!/usr/bin/env python3
"""
Enhanced Super Alita startup with proper 20B model configuration
"""

import os
import subprocess
import sys
import time


def setup_environment():
    """Setup environment variables for 20B model"""
    print("🔧 Setting up environment for GPT-OSS 20B...")
    
    # Essential environment variables
    env_vars = {
        "LLM_MODEL": "ollama:gpt-oss:20b",
        "OLLAMA_HOST": "http://127.0.0.1:11434",
        "PYTHONPATH": "./src",
    }
    
    for key, value in env_vars.items():
        os.environ[key] = value
        print(f"  • {key} = {value}")

def verify_ollama():
    """Verify Ollama is running and model is available"""
    print("\n🧪 Verifying Ollama setup...")
    
    try:
        # Check if ollama is running
        result = subprocess.run(["ollama", "ps"], capture_output=True, text=True, timeout=10)
        if result.returncode != 0:
            print("❌ Ollama not running")
            return False
            
        # Check if our model is loaded
        if "gpt-oss:20b" in result.stdout:
            print("✅ GPT-OSS 20B model is loaded")
            return True
        else:
            print("⚠️ Loading GPT-OSS 20B model...")
            # Try to load the model
            load_result = subprocess.run(
                ["ollama", "run", "gpt-oss:20b", "test"], 
                capture_output=True, text=True, timeout=60
            )
            if load_result.returncode == 0:
                print("✅ Model loaded successfully")
                return True
            else:
                print(f"❌ Failed to load model: {load_result.stderr}")
                return False
                
    except subprocess.TimeoutExpired:
        print("❌ Ollama command timed out")
        return False
    except Exception as e:
        print(f"❌ Error checking Ollama: {e}")
        return False

def test_llm_client():
    """Test the LLM client before starting the server"""
    print("\n🔬 Testing LLM client integration...")
    
    # Add src to path
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
    
    try:
        from reug_runtime.llm_client import get_llm_client
        
        client = get_llm_client(os.environ["LLM_MODEL"])
        print(f"✅ LLM client created: {type(client).__name__}")
        
        # Quick test
        import asyncio
        async def quick_test():
            messages = [{"role": "user", "content": "hi"}]
            response = []
            async for chunk in client.stream_chat(messages, timeout=10):
                content = chunk.get("content", "")
                if content:
                    response.append(content)
                    if len(response) > 10:  # Stop early for test
                        break
            return "".join(response)
        
        response = asyncio.run(quick_test())
        print(f"✅ Test response: {response[:50]}...")
        return True
        
    except Exception as e:
        print(f"❌ LLM client test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def start_super_alita():
    """Start the Super Alita application"""
    print("\n🚀 Starting Super Alita...")
    
    try:
        # Import and run the startup script
        subprocess.run([sys.executable, "start_super_alita.py"], check=True)
    except KeyboardInterrupt:
        print("\n🛑 Shutdown requested")
    except subprocess.CalledProcessError as e:
        print(f"❌ Super Alita failed to start: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False
    
    return True

def main():
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║               🤖 Super Alita with GPT-OSS 20B               ║")
    print("║                  Hybrid RAM+VRAM Configuration              ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()
    
    # Step 1: Setup environment
    setup_environment()
    
    # Step 2: Verify Ollama
    if not verify_ollama():
        print("\n❌ Ollama verification failed. Please check your setup.")
        return 1
    
    # Step 3: Test LLM client
    if not test_llm_client():
        print("\n❌ LLM client test failed. Please check your configuration.")
        return 1
    
    print("\n✅ All checks passed! Starting Super Alita...")
    time.sleep(2)
    
    # Step 4: Start application
    if not start_super_alita():
        return 1
    
    print("\n👋 Super Alita with 20B model has stopped.")
    return 0

if __name__ == "__main__":
    sys.exit(main())