#!/usr/bin/env python3
"""
Test Ollama models with proper parameters to avoid verbose output
"""

import requests


def test_model_simple(model_name, prompt="Hello! How are you?"):
    """Test a model with simple, constrained parameters"""

    url = "http://localhost:11434/v1/chat/completions"

    # Use very constrained parameters to avoid verbose output
    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant. Give brief, direct answers. Do not elaborate unless asked.",
            },
            {"role": "user", "content": prompt},
        ],
        "max_tokens": 20,  # Very short responses
        "temperature": 0.1,  # Low temperature for focused responses
        "top_p": 0.9,  # Focused sampling
        "frequency_penalty": 0.5,  # Reduce repetition
        "presence_penalty": 0.3,  # Encourage brevity
        "stop": ["\n\n", ".", "!", "?"],  # Stop at punctuation
    }

    print(f"\n🧪 Testing {model_name}")
    print(f"📋 Prompt: {prompt}")
    print(
        f"📋 Parameters: max_tokens={payload['max_tokens']}, temp={payload['temperature']}"
    )

    try:
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()

        data = response.json()
        if "choices" in data and len(data["choices"]) > 0:
            content = data["choices"][0]["message"]["content"]
            print(f"✅ Response: {content}")
            print(f"📊 Length: {len(content)} characters")

            # Check if response is reasonable
            if len(content) > 100:
                print("⚠️  Response seems too verbose")
            elif len(content) < 5:
                print("⚠️  Response seems too short")
            else:
                print("✅ Response length looks good")

        else:
            print("❌ No response content found")

    except Exception as e:
        print(f"❌ Error: {e}")


def test_models():
    """Test available models"""

    models_to_test = ["llama3.2:1b", "gpt-oss-20b-split"]

    print("🚀 Testing Ollama Models with Constrained Parameters")
    print("=" * 60)

    for model in models_to_test:
        test_model_simple(model, "Hi!")
        test_model_simple(model, "What is 2+2?")
        print("-" * 40)


if __name__ == "__main__":
    test_models()
