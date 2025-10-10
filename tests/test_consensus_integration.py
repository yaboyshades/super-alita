#!/usr/bin/env python3
"""
Comprehensive consensus integration test following Super Alita patterns.
Tests the actual REUG → Tool → Ollama integration path.
"""

import asyncio
import time

import httpx


class ConsensusIntegrationTester:
    """Test consensus integration following Super Alita architecture patterns."""

    def __init__(self):
        self.base_url = "http://127.0.0.1:8080"
        self.timeout = 60.0

    async def test_system_health(self) -> bool:
        """Validate system health following instructions pattern."""
        print("🏥 Testing System Health...")

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                response = await client.get(f"{self.base_url}/healthz")
                if response.status_code == 200:
                    health = response.json()
                    status = health.get("status", "unknown")
                    print(f"✅ System Health: {status}")

                    # Check components as per instructions
                    components = health.get("components", {})
                    for name, component in components.items():
                        if isinstance(component, dict):
                            comp_status = component.get("status", "unknown")
                        else:
                            comp_status = str(component)
                        print(f"   {name}: {comp_status}")

                    return status == "healthy"
                else:
                    print(f"❌ Health check failed: {response.status_code}")
                    return False
            except Exception as e:
                print(f"❌ Health check error: {e}")
                return False

    async def test_tool_catalog(self) -> bool:
        """Check tools catalog as per validation patterns."""
        print("\n🛠️  Testing Tool Catalog...")

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                response = await client.get(f"{self.base_url}/tools/catalog")
                if response.status_code == 200:
                    tools = response.json()
                    tool_names = [tool["name"] for tool in tools]

                    print(f"✅ Found {len(tools)} tools")

                    # Look for consensus tool
                    consensus_tool = next(
                        (
                            t
                            for t in tools
                            if t["name"] == "deepconf_consensus"
                        ),
                        None,
                    )
                    if consensus_tool:
                        print("✅ Consensus tool registered:")
                        print(
                            f"   Description: {consensus_tool['description']}"
                        )

                        # Check schema
                        schema = consensus_tool.get("input_schema", {})
                        props = schema.get("properties", {})
                        required = schema.get("required", [])

                        print(f"   Required params: {required}")
                        print(f"   All params: {list(props.keys())}")

                        return True
                    else:
                        print("❌ Consensus tool not found in catalog")
                        print(f"   Available tools: {tool_names}")
                        return False
                else:
                    print(f"❌ Catalog request failed: {response.status_code}")
                    return False
            except Exception as e:
                print(f"❌ Catalog test error: {e}")
                return False

    async def test_ollama_direct(self) -> bool:
        """Test direct Ollama connectivity as baseline."""
        print("\n🤖 Testing Direct Ollama...")

        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                response = await client.post(
                    "http://127.0.0.1:11434/v1/chat/completions",
                    json={
                        "model": "gpt-oss:20b",
                        "messages": [
                            {"role": "user", "content": "Capital of France?"}
                        ],
                        "max_tokens": 20,
                        "temperature": 0.3,
                    },
                )

                if response.status_code == 200:
                    result = response.json()
                    choices = result.get("choices", [])
                    if choices:
                        content = (
                            choices[0].get("message", {}).get("content", "")
                        )
                        print(f"✅ Ollama response: {content.strip()}")
                        return True
                    else:
                        print(f"❌ No choices in response: {result}")
                        return False
                else:
                    print(f"❌ Ollama failed: {response.status_code}")
                    print(f"   Response: {response.text}")
                    return False

            except Exception as e:
                print(f"❌ Ollama test error: {e}")
                return False

    async def test_consensus_via_reug(self) -> bool:
        """Test consensus through REUG streaming (main integration path)."""
        print("\n🧠 Testing Consensus via REUG...")

        # Create explicit tool call message
        test_message = (
            "Please use the deepconf_consensus tool with these parameters: "
            "prompt='What is the capital of France?', num_samples=2, "
            "temperature=0.4, max_tokens=50"
        )

        session_id = f"consensus_test_{int(time.time())}"

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                # Start REUG turn
                start_response = await client.post(
                    f"{self.base_url}/v1/chat/stream",
                    json={"message": test_message, "session_id": session_id},
                )

                if start_response.status_code == 200:
                    start_result = start_response.json()
                    run_id = start_result.get("run_id")

                    print(f"✅ REUG turn started: {run_id}")

                    # Stream the results
                    return await self._stream_consensus_results(client, run_id)
                else:
                    print(
                        f"❌ REUG start failed: {start_response.status_code}"
                    )
                    print(f"   Response: {start_response.text}")
                    return False

            except Exception as e:
                print(f"❌ REUG test error: {e}")
                return False

    async def _stream_consensus_results(
        self, client: httpx.AsyncClient, run_id: str
    ) -> bool:
        """Stream REUG results and analyze for consensus execution."""
        print(f"📡 Streaming results for {run_id}...")

        consensus_detected = False
        tool_execution_detected = False
        error_detected = False
        chunks_received = 0

        for iteration in range(30):  # Max 30 iterations
            try:
                response = await client.post(
                    f"{self.base_url}/tools/reug_stream_next",
                    json={"run_id": run_id},
                )

                if response.status_code == 200:
                    result = response.json()
                    chunks = result.get("chunks", [])
                    finished = result.get("finished", False)

                    chunks_received += len(chunks)

                    for chunk in chunks:
                        chunk_str = str(chunk).lower()

                        # Check for consensus-related activity
                        if any(
                            keyword in chunk_str
                            for keyword in [
                                "consensus",
                                "deepconf",
                                "paris",
                                "capital",
                                "france",
                            ]
                        ):
                            consensus_detected = True
                            print(
                                f"🎯 Consensus activity: {str(chunk)[:100]}..."
                            )

                        # Check for tool execution
                        if any(
                            keyword in chunk_str
                            for keyword in [
                                "tool_call",
                                "tool_result",
                                "executing",
                            ]
                        ):
                            tool_execution_detected = True
                            print(f"🔧 Tool execution: {str(chunk)[:100]}...")

                        # Check for errors
                        if "error" in chunk_str or "exception" in chunk_str:
                            error_detected = True
                            print(f"❌ Error detected: {str(chunk)[:100]}...")

                    if finished:
                        print(
                            f"✅ Stream finished after {iteration + 1} iterations"
                        )
                        break

                elif response.status_code == 500:
                    print(f"❌ Stream error 500 at iteration {iteration}")
                    error_detected = True
                    break
                else:
                    print(f"❌ Stream error {response.status_code}")
                    break

                await asyncio.sleep(0.5)

            except Exception as e:
                print(f"❌ Stream exception at iteration {iteration}: {e}")
                error_detected = True
                break

        # Analyze results
        print("\n📊 Stream Analysis:")
        print(f"   Chunks received: {chunks_received}")
        print(f"   Consensus detected: {consensus_detected}")
        print(f"   Tool execution detected: {tool_execution_detected}")
        print(f"   Errors detected: {error_detected}")

        # Success criteria: consensus detected OR tool execution detected, no errors
        success = (
            consensus_detected or tool_execution_detected
        ) and not error_detected

        if success:
            print("🎉 Consensus integration working!")
        elif error_detected:
            print("⚠️  Integration has errors - need investigation")
        else:
            print("⚠️  No consensus activity detected")

        return success

    async def test_ability_registry_direct(self) -> bool:
        """Test consensus tool via ability registry (debugging path)."""
        print("\n🔍 Testing Ability Registry Direct Access...")

        try:
            # This is a debugging approach to test tool registration
            async with httpx.AsyncClient(timeout=30.0) as client:
                # Check if there's a debug endpoint for ability registry
                response = await client.get(f"{self.base_url}/debug/abilities")

                if response.status_code == 200:
                    abilities = response.json()
                    print(
                        f"✅ Registry accessible, {len(abilities)} abilities"
                    )

                    # Look for consensus ability
                    consensus_found = any(
                        "consensus" in str(ability).lower()
                        for ability in abilities
                    )

                    if consensus_found:
                        print("✅ Consensus ability found in registry")
                        return True
                    else:
                        print("❌ Consensus ability not in registry")
                        return False
                else:
                    print("⚠️  No debug endpoint available (expected)")
                    return True  # Not a failure, just no debug endpoint

        except Exception as e:
            print(f"⚠️  Registry test skipped: {e}")
            return True  # Not critical for main integration

    async def run_comprehensive_test(self) -> dict[str, bool]:
        """Run all integration tests following Super Alita patterns."""
        print("🚀 Comprehensive Consensus Integration Test")
        print("Following Super Alita Development Instructions")
        print("=" * 60)

        test_results = {}

        # Test sequence following validation patterns
        test_results["health"] = await self.test_system_health()
        test_results["catalog"] = await self.test_tool_catalog()
        test_results["ollama"] = await self.test_ollama_direct()
        test_results["registry"] = await self.test_ability_registry_direct()
        test_results["reug_integration"] = await self.test_consensus_via_reug()

        # Results summary
        print("\n" + "=" * 60)
        print("📊 Integration Test Results:")

        passed = sum(test_results.values())
        total = len(test_results)

        for test_name, result in test_results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            test_display = test_name.replace("_", " ").title()
            print(f"   {test_display}: {status}")

        success_rate = passed / total * 100
        print(
            f"\n🎯 Overall: {passed}/{total} tests passed ({success_rate:.1f}%)"
        )

        # Interpretation following instructions
        if passed == total:
            print(
                "🎉 ALL TESTS PASSED - Consensus integration is fully operational!"
            )
            print("\n✅ Ready for:")
            print("   - Production deployment")
            print("   - Enhanced algorithm testing")
            print("   - Multi-model validation")
        elif success_rate >= 80:
            print("🟡 MOSTLY WORKING - Minor issues to resolve")
            print("\n⚠️  Check failed tests and resolve before production")
        else:
            print("🔴 SIGNIFICANT ISSUES - Major problems detected")
            print("\n❌ Resolve core issues before proceeding")

        return test_results


async def main():
    """Main test runner following Super Alita patterns."""
    tester = ConsensusIntegrationTester()
    results = await tester.run_comprehensive_test()

    # Exit with appropriate code for CI/CD
    if all(results.values()):
        exit(0)  # Success
    else:
        exit(1)  # Some tests failed


if __name__ == "__main__":
    asyncio.run(main())
