#!/usr/bin/env python3
"""
Start Super Alita with proper 20B model configuration
"""

import os
import subprocess
import sys


def main():
    print("🔧 Configuring Super Alita for GPT-OSS 20B...")
    
    # Set environment variables for the 20B model
    os.environ["LLM_MODEL"] = "ollama:gpt-oss:20b"
    os.environ["OLLAMA_HOST"] = "http://127.0.0.1:11434"
    
    # Verify model is available
    print("🧪 Verifying model availability...")
    try:
        result = subprocess.run(["ollama", "ps"], capture_output=True, text=True)
        if "gpt-oss:20b" in result.stdout:
            print("✅ GPT-OSS 20B model is loaded and ready!")
        else:
            print("⚠️ Loading GPT-OSS 20B model...")
            subprocess.run(["ollama", "run", "gpt-oss:20b", "hello"], capture_output=True)
            print("✅ Model loaded!")
    except Exception as e:
        print(f"❌ Error checking model: {e}")
        return 1
    
    print("🚀 Starting Super Alita with 20B model...")
    
    # Start Super Alita with the environment
    try:
        subprocess.run([sys.executable, "start_super_alita.py"])
    except KeyboardInterrupt:
        print("\n🛑 Shutting down Super Alita...")
    except Exception as e:
        print(f"❌ Error starting Super Alita: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())