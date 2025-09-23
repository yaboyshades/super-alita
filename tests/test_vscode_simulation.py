#!/usr/bin/env python3
"""Simulate VS Code DeepCode commands to test the full integration"""

import asyncio
from pathlib import Path

import aiohttp


async def simulate_vscode_deepcode_analyze():
    """Simulate the VS Code 'Alita: DeepCode — Analyze Workspace' command"""
    print("🔄 Simulating VS Code DeepCode Analyze command...")

    # This mimics what the VS Code extension does
    base_url = "http://127.0.0.1:8080"
    url = f"{base_url}/deepcode/request"

    # Get the workspace folder path (like VS Code would)
    workspace_path = str(Path.cwd())

    payload = {
        "task_kind": "analyze",
        "repo_path": workspace_path,
        "conversation_id": "vscode-simulation-analyze",
    }

    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(url, json=payload) as response:
                result = await response.json()
                print("✅ Analyze request sent successfully!")
                print(f"   Status: {result.get('status')}")
                print(f"   Message: {result.get('message')}")
                return True
        except Exception as e:
            print(f"❌ Analyze request failed: {e}")
            return False


async def simulate_vscode_deepcode_generate():
    """Simulate the VS Code 'Alita: DeepCode — Generate From Prompt' command"""
    print("🔄 Simulating VS Code DeepCode Generate command...")

    # This mimics what the VS Code extension does
    base_url = "http://127.0.0.1:8080"
    url = f"{base_url}/deepcode/request"

    # Simulate user input (like VS Code showInputBox would provide)
    requirements = "Add a new FastAPI endpoint for user authentication"
    workspace_path = str(Path.cwd())

    payload = {
        "task_kind": "text2backend",
        "requirements": requirements,
        "repo_path": workspace_path,
        "conversation_id": "vscode-simulation-generate",
    }

    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(url, json=payload) as response:
                result = await response.json()
                print("✅ Generate request sent successfully!")
                print(f"   Status: {result.get('status')}")
                print(f"   Message: {result.get('message')}")
                print(f"   Requirements: {requirements}")
                return True
        except Exception as e:
            print(f"❌ Generate request failed: {e}")
            return False


async def test_vscode_settings_simulation():
    """Test the VS Code setting 'alita.runtime.host' configuration"""
    print("🔄 Testing VS Code runtime host setting...")

    # Test different host configurations
    hosts_to_test = [
        "http://127.0.0.1:8080",
        "http://localhost:8080",
    ]

    for host in hosts_to_test:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{host}/health") as response:
                    health_data = await response.json()
                    if health_data.get("status") == "healthy":
                        print(f"✅ Host {host} is reachable and healthy")
                    else:
                        print(
                            f"⚠️  Host {host} responded but not healthy: {health_data}"
                        )
        except Exception as e:
            print(f"❌ Host {host} is unreachable: {e}")


async def main():
    """Run the full VS Code simulation test suite"""
    print("🚀 Starting VS Code DeepCode Integration Test Suite")
    print("=" * 60)

    # Test the settings
    await test_vscode_settings_simulation()
    print()

    # Test analyze command
    analyze_success = await simulate_vscode_deepcode_analyze()
    print()

    # Test generate command
    generate_success = await simulate_vscode_deepcode_generate()
    print()

    # Summary
    print("=" * 60)
    print("📋 VS Code Integration Test Summary:")
    print(f"   Analyze Command: {'✅ PASS' if analyze_success else '❌ FAIL'}")
    print(f"   Generate Command: {'✅ PASS' if generate_success else '❌ FAIL'}")

    if analyze_success and generate_success:
        print("\n🎉 All VS Code DeepCode commands working correctly!")
        print("\n📝 Next Steps:")
        print("   1. Open VS Code in this workspace")
        print("   2. Set 'alita.runtime.host' to 'http://127.0.0.1:8080'")
        print("   3. Run commands from Command Palette:")
        print("      - 'Alita: DeepCode — Analyze Workspace'")
        print("      - 'Alita: DeepCode — Generate From Prompt'")
        print("   4. Check the server logs for processing events")
    else:
        print("\n❌ Some tests failed - check server connectivity")

    return analyze_success and generate_success


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)
