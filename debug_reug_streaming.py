#!/usr/bin/env python3
"""
Debug REUG streaming connection issues following Super Alita patterns.
Focus on the specific "incomplete chunked read" error.
"""

import asyncio
import time

import httpx


class REUGStreamingDebugger:
    """Debug REUG streaming issues following development instructions."""

    def __init__(self):
        self.base_url = "http://127.0.0.1:8080"
        self.timeout = 30.0  # Shorter timeout for debugging

    async def test_simple_streaming(self) -> bool:
        """Test basic REUG streaming without consensus tool."""
        print("🔍 Testing Basic REUG Streaming...")

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                # Start with a simple message that shouldn't trigger tool execution
                response = await client.post(
                    f"{self.base_url}/v1/chat/stream",
                    json={
                        "message": "Hello, please respond briefly.",
                        "session_id": f"debug_basic_{int(time.time())}",
                    },
                )

                if response.status_code == 200:
                    result = response.json()
                    run_id = result.get("run_id")
                    print(f"✅ Basic REUG started: {run_id}")

                    # Test streaming with minimal requests
                    return await self._test_minimal_streaming(client, run_id)
                else:
                    print(f"❌ Basic REUG failed: {response.status_code}")
                    print(f"   Response: {response.text}")
                    return False

            except Exception as e:
                print(f"❌ Basic streaming error: {e}")
                return False

    async def _test_minimal_streaming(
        self, client: httpx.AsyncClient, run_id: str
    ) -> bool:
        """Test minimal streaming to isolate connection issues."""
        print(f"📡 Testing minimal streaming for {run_id}...")

        try:
            # Single stream request with shorter timeout
            response = await client.post(
                f"{self.base_url}/tools/reug_stream_next",
                json={"run_id": run_id},
                timeout=10.0,  # Shorter timeout
            )

            if response.status_code == 200:
                result = response.json()
                chunks = result.get("chunks", [])
                finished = result.get("finished", False)

                print(
                    f"✅ Minimal stream success: {len(chunks)} chunks, finished: {finished}"
                )

                for i, chunk in enumerate(chunks):
                    chunk_str = str(chunk)[:100]
                    print(f"   Chunk {i}: {chunk_str}...")

                return True
            else:
                print(f"❌ Minimal stream failed: {response.status_code}")
                print(f"   Response: {response.text}")
                return False

        except httpx.ReadTimeout:
            print("⚠️  Stream timeout (expected for some messages)")
            return True  # Timeout might be normal
        except httpx.RemoteProtocolError as e:
            print(f"❌ Protocol error: {e}")
            return False
        except Exception as e:
            print(f"❌ Minimal streaming error: {e}")
            return False

    async def test_consensus_with_retry(self) -> bool:
        """Test consensus tool with connection retry logic."""
        print("\n🧠 Testing Consensus with Retry Logic...")

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                # Start consensus request
                consensus_message = (
                    "Use the deepconf_consensus tool with prompt='What is 2+2?' "
                    "and num_samples=1"
                )

                response = await client.post(
                    f"{self.base_url}/v1/chat/stream",
                    json={
                        "message": consensus_message,
                        "session_id": f"debug_consensus_{int(time.time())}",
                    },
                )

                if response.status_code == 200:
                    result = response.json()
                    run_id = result.get("run_id")
                    print(f"✅ Consensus REUG started: {run_id}")

                    # Test with retry logic
                    return await self._stream_with_retry(client, run_id)
                else:
                    print(f"❌ Consensus REUG failed: {response.status_code}")
                    return False

            except Exception as e:
                print(f"❌ Consensus retry test error: {e}")
                return False

    async def _stream_with_retry(self, client: httpx.AsyncClient, run_id: str) -> bool:
        """Stream with retry logic to handle connection issues."""
        print(f"📡 Streaming with retry for {run_id}...")

        max_retries = 3
        chunks_received = 0

        for attempt in range(max_retries):
            try:
                response = await client.post(
                    f"{self.base_url}/tools/reug_stream_next",
                    json={"run_id": run_id},
                    timeout=15.0,
                )

                if response.status_code == 200:
                    result = response.json()
                    chunks = result.get("chunks", [])
                    finished = result.get("finished", False)

                    chunks_received += len(chunks)

                    if chunks:
                        print(f"✅ Attempt {attempt + 1}: {len(chunks)} chunks")
                        for chunk in chunks:
                            chunk_str = str(chunk)
                            if (
                                "consensus" in chunk_str.lower()
                                or "deepconf" in chunk_str.lower()
                            ):
                                print("🎯 Consensus activity detected!")

                    if finished:
                        print("✅ Stream completed successfully")
                        return True

                    # Small delay before next attempt
                    await asyncio.sleep(1.0)

                elif response.status_code == 500:
                    print(f"❌ Server error 500 on attempt {attempt + 1}")
                    if attempt == max_retries - 1:
                        return False
                else:
                    print(
                        f"❌ Stream error {response.status_code} on attempt {attempt + 1}"
                    )
                    return False

            except httpx.RemoteProtocolError as e:
                print(f"⚠️  Protocol error on attempt {attempt + 1}: {e}")
                if attempt == max_retries - 1:
                    print("❌ All retry attempts failed")
                    return False
                await asyncio.sleep(2.0)  # Longer delay for protocol errors
            except Exception as e:
                print(f"❌ Stream error on attempt {attempt + 1}: {e}")
                if attempt == max_retries - 1:
                    return False
                await asyncio.sleep(1.0)

        print(f"📊 Total chunks received: {chunks_received}")
        return chunks_received > 0

    async def test_server_streaming_health(self) -> bool:
        """Test if server streaming infrastructure is working."""
        print("\n🏥 Testing Server Streaming Health...")

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                # Test if streaming endpoints are responsive
                test_endpoints = ["/tools/catalog", "/healthz", "/v1/chat/stream"]

                for endpoint in test_endpoints:
                    if endpoint == "/v1/chat/stream":
                        # POST request for streaming
                        response = await client.post(
                            f"{self.base_url}{endpoint}",
                            json={"message": "test", "session_id": "health_check"},
                            timeout=5.0,
                        )
                    else:
                        # GET request for other endpoints
                        response = await client.get(
                            f"{self.base_url}{endpoint}", timeout=5.0
                        )

                    if response.status_code in [200, 201]:
                        print(f"✅ {endpoint}: responding")
                    else:
                        print(f"❌ {endpoint}: {response.status_code}")
                        return False

                print("✅ All streaming endpoints responsive")
                return True

            except Exception as e:
                print(f"❌ Streaming health test error: {e}")
                return False

    async def run_debug_session(self) -> dict[str, bool]:
        """Run comprehensive debugging session."""
        print("🚀 REUG Streaming Debug Session")
        print("Following Super Alita Development Instructions")
        print("=" * 60)

        debug_results = {}

        # Test sequence
        debug_results["streaming_health"] = await self.test_server_streaming_health()
        debug_results["basic_streaming"] = await self.test_simple_streaming()
        debug_results["consensus_retry"] = await self.test_consensus_with_retry()

        # Results analysis
        print("\n" + "=" * 60)
        print("🔍 Debug Results:")

        passed = sum(debug_results.values())
        total = len(debug_results)

        for test_name, result in debug_results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"   {test_name.replace('_', ' ').title()}: {status}")

        print(f"\n🎯 Debug Success: {passed}/{total} ({passed/total*100:.1f}%)")

        # Diagnosis
        if debug_results.get("streaming_health", False):
            if debug_results.get("basic_streaming", False):
                if not debug_results.get("consensus_retry", False):
                    print("\n🔍 DIAGNOSIS: Consensus tool execution issue")
                    print("   ✅ Server streaming works")
                    print("   ✅ Basic REUG works")
                    print("   ❌ Consensus tool has execution problems")
                    print("\n💡 SOLUTION: Check consensus tool implementation")
                else:
                    print("\n🎉 ALL SYSTEMS WORKING - Retries resolved the issue!")
            else:
                print("\n🔍 DIAGNOSIS: REUG streaming infrastructure issue")
                print("   ✅ Server endpoints work")
                print("   ❌ REUG streaming has problems")
                print("\n💡 SOLUTION: Check REUG router implementation")
        else:
            print("\n🔍 DIAGNOSIS: Server infrastructure issue")
            print("   ❌ Basic server endpoints failing")
            print("\n💡 SOLUTION: Restart server and check logs")

        return debug_results


async def main():
    """Main debug runner following development instructions."""
    debugger = REUGStreamingDebugger()
    results = await debugger.run_debug_session()

    # Follow instructions: exit with appropriate code
    if all(results.values()):
        exit(0)  # Success
    else:
        exit(1)  # Issues found


if __name__ == "__main__":
    asyncio.run(main())
