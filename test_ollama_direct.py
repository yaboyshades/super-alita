#!/usr/bin/env python3
"""Test consensus tool directly via HTTP."""

import requests
import json


def test_consensus_direct():
    """Test calling consensus tool directly."""
    print("🧪 Testing consensus tool directly...")

    # First check if Ollama is working
    try:
        print("🔍 Testing Ollama directly...")
        ollama_response = requests.post(
            "http://localhost:11434/v1/chat/completions",
            json={
                "model": "gpt-oss:20b",
                "messages": [{"role": "user", "content": "What is 2+2?"}],
                "max_tokens": 50,
                "temperature": 0.3,
            },
            timeout=30,
        )

        if ollama_response.status_code == 200:
            ollama_data = ollama_response.json()
            if "choices" in ollama_data:
                content = ollama_data["choices"][0]["message"]["content"]
                print(f"✅ Ollama working: {content}")
            else:
                print(f"❌ Ollama response format issue: {ollama_data}")
        else:
            print(
                f"❌ Ollama error: {ollama_response.status_code} - {ollama_response.text}"
            )
            return

    except Exception as e:
        print(f"❌ Ollama test failed: {e}")
        return

    # Now test the Super Alita health to see the specific LLM error
    try:
        print("\n🏥 Checking Super Alita health...")
        health_response = requests.get("http://127.0.0.1:8080/healthz", timeout=10)
        if health_response.status_code in [200, 503]:
            health_data = health_response.json()
            print(f"📊 Health status: {health_data.get('status', 'unknown')}")

            # Check LLM component specifically
            components = health_data.get("components", {})
            llm_status = components.get("llm", {})
            if llm_status.get("status") == "unhealthy":
                error = llm_status.get("error", "Unknown error")
                print(f"❌ LLM component error: {error}")
            else:
                print(f"✅ LLM component: {llm_status.get('status', 'unknown')}")

        else:
            print(f"❌ Health check failed: {health_response.status_code}")

    except Exception as e:
        print(f"❌ Health check error: {e}")


if __name__ == "__main__":
    test_consensus_direct()
