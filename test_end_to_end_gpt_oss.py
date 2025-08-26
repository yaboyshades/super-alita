#!/usr/bin/env python3
"""Test the VS Code extension's direct Ollama integration."""

import requests


def test_extension_ollama_direct():
    """Test direct Ollama call like the extension does."""
    print("🧪 Testing VS Code extension's direct Ollama integration...")

    host = "http://127.0.0.1:11434"
    model = "gpt-oss:20b"
    prompt = "Hello from VS Code extension test! Please respond briefly."

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
    }

    try:
        print(f"📡 Calling {host}/api/chat with model {model}")
        response = requests.post(
            f"{host}/api/chat",
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=30,
        )

        if response.ok:
            data = response.json()
            message = data.get("message", {}).get("content", "")
            print(f"✅ Response: {message}")
            print("🎉 Extension's direct Ollama integration WORKS!")
            return True
        else:
            print(f"❌ HTTP {response.status_code}: {response.text}")
            return False

    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_runtime_endpoint():
    """Test the runtime chat endpoint."""
    print("\n🧪 Testing runtime chat endpoint...")

    url = "http://127.0.0.1:8080/v1/chat/stream"
    payload = {
        "session_id": "test",
        "message": "Hello from runtime test! Brief response please.",
    }

    try:
        print(f"📡 Calling {url}")
        response = requests.post(
            url,
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=30,
            stream=True,
        )

        if response.ok:
            print("✅ Runtime endpoint responded!")
            # Read first few chunks
            chunks = []
            for i, line in enumerate(response.iter_lines(decode_unicode=True)):
                if line and i < 10:  # Just first 10 chunks
                    chunks.append(line)
                    print(f"📝 Chunk {i}: {line[:50]}...")
            print(f"✅ Received {len(chunks)} chunks")
            print("🎉 Runtime chat endpoint WORKS!")
            return True
        else:
            print(f"❌ HTTP {response.status_code}: {response.text}")
            return False

    except requests.exceptions.ConnectionError:
        print("⚠️ Runtime server not available (expected if not running)")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("🚀 GPT-OSS End-to-End Integration Test")
    print("=" * 60)

    # Test 1: Direct Ollama (like VS Code extension)
    ollama_works = test_extension_ollama_direct()

    # Test 2: Runtime endpoint (like extension's chat command)
    runtime_works = test_runtime_endpoint()

    print("\n" + "=" * 60)
    print("📊 SUMMARY:")
    print(f"  ✅ Direct Ollama (Extension): {'PASS' if ollama_works else 'FAIL'}")
    print(f"  ✅ Runtime Endpoint:         {'PASS' if runtime_works else 'FAIL'}")

    if ollama_works:
        print("\n🎯 VS Code Insiders Integration Ready!")
        print("   • Use 'Alita: Invoke Agent (Ollama)' for direct calls")
        if runtime_works:
            print("   • Use 'Alita: Chat via Runtime (Stream)' for runtime calls")
        print("   • Model default: gpt-oss:20b")
        print("   • Ollama host: http://127.0.0.1:11434")
        print("   • Runtime host: http://127.0.0.1:8080")

    print("=" * 60)
