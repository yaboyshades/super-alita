#!/usr/bin/env python3
"""Simulate VS Code extension commands for GPT-OSS."""

import requests


def test_extension_invoke_agent():
    """Simulate the 'Alita: Invoke Agent (Ollama)' command."""
    print("🎯 Simulating 'Alita: Invoke Agent (Ollama)' command...")

    # This is exactly what the extension does in invokeOllama function
    host = "http://127.0.0.1:11434"
    model = "gpt-oss:20b"  # Default from extension configuration
    prompt = "Help me understand how GPT-OSS works. Brief response please."

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
    }

    try:
        print(f"📡 Extension calling {host}/api/chat")
        print(f"📝 Model: {model}")
        print(f"📝 Prompt: {prompt[:50]}...")

        response = requests.post(
            f"{host}/api/chat",
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=30,
        )

        if response.ok:
            data = response.json()
            message = data.get("message", {}).get("content", "")

            print(f"✅ Status: {response.status_code}")
            print(f"✅ Response length: {len(message)} chars")
            print(f"✅ Response preview: {message[:100]}...")
            print("🎉 Extension simulation SUCCESSFUL!")

            # This would be opened as a new document in VS Code
            print("\n📄 VS Code would open this as a markdown document:")
            print("-" * 50)
            print(message)
            print("-" * 50)

            return True
        else:
            print(f"❌ HTTP {response.status_code}: {response.text}")
            return False

    except Exception as e:
        print(f"❌ Extension simulation failed: {e}")
        return False


def test_extension_chat_runtime():
    """Simulate the 'Alita: Chat via Runtime (Stream)' command."""
    print("\n🎯 Simulating 'Alita: Chat via Runtime (Stream)' command...")

    # This is what the extension does in chatRuntime command
    base = "http://127.0.0.1:8080"  # Default from extension configuration
    url = f"{base}/v1/chat/stream"
    payload = {
        "session_id": "vscode-test",
        "message": "Hello from VS Code extension! Test response please.",
    }

    try:
        print(f"📡 Extension calling {url}")
        print(f"📝 Session: {payload['session_id']}")
        print(f"📝 Message: {payload['message'][:50]}...")

        response = requests.post(
            url,
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=30,
            stream=True,
        )

        if response.ok:
            print(f"✅ Status: {response.status_code}")
            print("📺 VS Code Output channel would show:")
            print("-" * 50)

            chunks = []
            for i, line in enumerate(response.iter_lines(decode_unicode=True)):
                if line and i < 20:  # Limit output
                    chunks.append(line)
                    print(line)
                if i >= 20:
                    print("... (truncated)")
                    break

            print("-" * 50)
            print("🎉 Runtime simulation SUCCESSFUL!")
            return True
        else:
            print(f"❌ HTTP {response.status_code}: {response.text}")
            return False

    except requests.exceptions.ConnectionError:
        print("⚠️ Runtime server not running (expected)")
        print("💡 Start runtime with: make run-ollama")
        return False
    except Exception as e:
        print(f"❌ Runtime simulation failed: {e}")
        return False


if __name__ == "__main__":
    print("=" * 70)
    print("🚀 VS Code Extension GPT-OSS Integration Test")
    print("=" * 70)

    # Test both extension commands
    ollama_works = test_extension_invoke_agent()
    runtime_works = test_extension_chat_runtime()

    print("\n" + "=" * 70)
    print("📊 VS CODE EXTENSION SIMULATION RESULTS:")
    print(
        f"  🎯 'Invoke Agent (Ollama)':    {'✅ WORKING' if ollama_works else '❌ FAILED'}"
    )
    print(
        f"  🎯 'Chat via Runtime (Stream)': {'✅ WORKING' if runtime_works else '⚠️ NEEDS RUNTIME'}"
    )

    if ollama_works:
        print("\n🎉 VS Code Insiders is ready to use with GPT-OSS!")
        print("\nNext steps:")
        print("1. Open VS Code Insiders")
        print("2. Install/enable the 'alita-language-tools' extension")
        print("3. Use Command Palette > 'Alita: Invoke Agent (Ollama)'")
        print("4. Enter your prompt and see GPT-OSS respond!")

        if not runtime_works:
            print("\nOptional: For runtime streaming:")
            print(
                "5. Start runtime: make run-ollama (or set env vars + python -m src.main)"
            )
            print("6. Use Command Palette > 'Alita: Chat via Runtime (Stream)'")

    print("=" * 70)
