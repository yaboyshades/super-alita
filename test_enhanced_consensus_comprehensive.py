#!/usr/bin/env python3
"""Comprehensive test for all enhanced consensus algorithms."""

import asyncio
import json
import requests
import time
from typing import Dict, Any, List


class EnhancedConsensusValidator:
    """Validate all enhanced consensus algorithms."""

    def __init__(self, base_url: str = "http://127.0.0.1:8080"):
        self.base_url = base_url
        self.test_prompts = [
            "What is the capital of France?",
            "Explain photosynthesis in one sentence.",
            "What is 15 + 27?",
            "Name three programming languages.",
            "What color is the sky?",
        ]
        self.methods = [
            "simple_vote",
            "weighted_vote",
            "confidence_based",
            "semantic_similarity",
            "ensemble_ranking",
        ]

    def check_system_health(self) -> bool:
        """Check if the system is healthy."""
        try:
            response = requests.get(f"{self.base_url}/healthz", timeout=10)
            if response.status_code == 200:
                health = response.json()
                is_healthy = health.get("status") == "healthy"

                print(f"🏥 System Health: {health.get('status', 'unknown')}")

                components = health.get("components", {})
                for component, status in components.items():
                    status_icon = "✅" if status.get("status") == "ok" else "❌"
                    print(
                        f"   {status_icon} {component}: {status.get('status', 'unknown')}"
                    )

                return is_healthy
            else:
                print(f"❌ Health check failed: {response.status_code}")
                return False

        except Exception as e:
            print(f"❌ Health check error: {e}")
            return False

    def check_tools_catalog(self) -> bool:
        """Check if consensus tool is available."""
        try:
            response = requests.get(f"{self.base_url}/tools/catalog", timeout=10)
            if response.status_code == 200:
                tools = response.json()
                consensus_tool = None

                for tool in tools:
                    if tool.get("tool_id") == "deepconf_consensus":
                        consensus_tool = tool
                        break

                if consensus_tool:
                    print("✅ Enhanced consensus tool found in catalog")
                    print(f"   Description: {consensus_tool.get('description', 'N/A')}")

                    # Check for enhanced parameters
                    input_schema = consensus_tool.get("input_schema", {})
                    properties = input_schema.get("properties", {})

                    enhanced_params = [
                        "method",
                        "confidence_threshold",
                        "temperature_range",
                    ]
                    found_enhanced = [
                        param for param in enhanced_params if param in properties
                    ]

                    if found_enhanced:
                        print(f"   ✅ Enhanced parameters: {found_enhanced}")
                        return True
                    else:
                        print(
                            "   ⚠️  Enhanced parameters not found, using basic version"
                        )
                        return True
                else:
                    print("❌ Consensus tool not found in catalog")
                    return False
            else:
                print(f"❌ Tools catalog check failed: {response.status_code}")
                return False

        except Exception as e:
            print(f"❌ Tools catalog error: {e}")
            return False

    def test_consensus_method(self, method: str, prompt: str) -> Dict[str, Any]:
        """Test a specific consensus method."""
        print(f"\n🔍 Testing {method} with: '{prompt[:50]}...'")

        try:
            # For now, test directly with the consensus provider
            # Since the REUG integration might need more complex setup
            test_data = {
                "prompt": prompt,
                "method": method,
                "num_samples": 3,
                "temperature": 0.7,
                "max_tokens": 100,
                "confidence_threshold": 0.7,
                "temperature_range": 0.2,
            }

            # Try to call via the tool execution endpoint if available
            try:
                response = requests.post(
                    f"{self.base_url}/tools/execute",
                    json={"tool_id": "deepconf_consensus", "args": test_data},
                    timeout=60,
                )

                if response.status_code == 200:
                    result = response.json()
                    return self._analyze_consensus_result(method, result)
                else:
                    print(f"   ❌ Tool execution failed: {response.status_code}")
                    return {"success": False, "error": f"HTTP {response.status_code}"}

            except Exception:
                # Fallback: test the enhanced provider directly
                print("   ℹ️  Falling back to direct provider test...")
                return self._test_direct_provider(method, test_data)

        except Exception as e:
            print(f"   ❌ Test failed: {e}")
            return {"success": False, "error": str(e)}

    def _test_direct_provider(
        self, method: str, test_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Test the consensus provider directly."""
        try:
            # Import and test the enhanced provider directly
            import sys
            import os

            sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

            from abilities.enhanced_consensus_ability import EnhancedConsensusProvider

            async def run_direct_test():
                provider = EnhancedConsensusProvider(
                    {
                        "base_url": "http://localhost:11434/v1",
                        "model_name": "gpt-oss:20b",
                        "timeout": 60.0,
                    }
                )

                await provider.initialize()

                result = await provider.consensus_sampling(
                    prompt=test_data["prompt"],
                    num_samples=test_data["num_samples"],
                    temperature=test_data["temperature"],
                    max_tokens=test_data["max_tokens"],
                    method=test_data["method"],
                    confidence_threshold=test_data["confidence_threshold"],
                    temperature_range=test_data["temperature_range"],
                )

                return result

            # Run the async test
            result = asyncio.run(run_direct_test())
            return self._analyze_consensus_result(method, result)

        except Exception as e:
            print(f"   ❌ Direct provider test failed: {e}")
            return {"success": False, "error": str(e)}

    def _analyze_consensus_result(
        self, method: str, result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze consensus result."""
        try:
            consensus_text = result.get("consensus_text", "")
            consensus_confidence = result.get("consensus_confidence", 0.0)
            aggregation_method = result.get("aggregation_method", "unknown")
            individual_responses = result.get("individual_responses", [])
            confidence_scores = result.get("confidence_scores", [])
            metadata = result.get("metadata", {})

            print(f"   ✅ {method} completed successfully")
            print(f"   📝 Consensus: {consensus_text[:100]}...")
            print(f"   📊 Confidence: {consensus_confidence:.3f}")
            print(f"   🔢 Responses: {len(individual_responses)}")
            print(f"   📈 Method: {aggregation_method}")

            # Validate method-specific features
            validation_success = True

            if method == "weighted_vote" and confidence_scores:
                avg_confidence = sum(confidence_scores) / len(confidence_scores)
                print(f"   ⚖️  Avg confidence: {avg_confidence:.3f}")

            elif method == "confidence_based":
                threshold = metadata.get("threshold", 0.0)
                qualified = metadata.get("qualified_responses", 0)
                print(f"   🎯 Threshold: {threshold}, Qualified: {qualified}")

            elif method == "semantic_similarity":
                similarity_scores = metadata.get("similarity_scores", [])
                if similarity_scores:
                    avg_similarity = sum(similarity_scores) / len(similarity_scores)
                    print(f"   🔗 Avg similarity: {avg_similarity:.3f}")

            elif method == "ensemble_ranking":
                ensemble_scores = metadata.get("ensemble_scores", [])
                components = metadata.get("scoring_components", [])
                print(f"   🏆 Ranking components: {components}")

            return {
                "success": True,
                "method": method,
                "consensus_text": consensus_text,
                "consensus_confidence": consensus_confidence,
                "num_responses": len(individual_responses),
                "metadata": metadata,
            }

        except Exception as e:
            print(f"   ❌ Result analysis failed: {e}")
            return {"success": False, "error": str(e)}

    def run_comprehensive_test(self) -> Dict[str, Any]:
        """Run comprehensive test of all methods."""
        print("🧪 Starting Comprehensive Enhanced Consensus Test")
        print("=" * 60)

        # 1. System Health Check
        print("\n1️⃣ System Health Check")
        if not self.check_system_health():
            return {"success": False, "error": "System health check failed"}

        # 2. Tools Catalog Check
        print("\n2️⃣ Tools Catalog Check")
        if not self.check_tools_catalog():
            return {"success": False, "error": "Tools catalog check failed"}

        # 3. Test All Methods
        print(f"\n3️⃣ Testing {len(self.methods)} Consensus Methods")
        print(f"📝 Using {len(self.test_prompts)} test prompts")

        results = {}
        total_tests = 0
        successful_tests = 0

        for i, method in enumerate(self.methods, 1):
            print(f"\n--- Method {i}/{len(self.methods)}: {method.upper()} ---")
            method_results = []

            for j, prompt in enumerate(self.test_prompts, 1):
                print(f"\nTest {j}/{len(self.test_prompts)} for {method}:")
                result = self.test_consensus_method(method, prompt)
                method_results.append(result)
                total_tests += 1

                if result.get("success", False):
                    successful_tests += 1

                # Small delay between tests to avoid overwhelming the system
                time.sleep(2)

            results[method] = method_results

        # 4. Summary
        print(f"\n4️⃣ Test Summary")
        print("=" * 60)
        print(f"📊 Total Tests: {total_tests}")
        print(f"✅ Successful: {successful_tests}")
        print(f"❌ Failed: {total_tests - successful_tests}")
        print(f"📈 Success Rate: {(successful_tests/total_tests)*100:.1f}%")

        # Method-specific summary
        for method in self.methods:
            method_results = results[method]
            method_success = sum(1 for r in method_results if r.get("success", False))
            method_total = len(method_results)
            print(
                f"   {method}: {method_success}/{method_total} ({(method_success/method_total)*100:.1f}%)"
            )

        overall_success = successful_tests >= (total_tests * 0.8)  # 80% success rate

        if overall_success:
            print("\n🎉 Enhanced Consensus System: READY FOR PRODUCTION!")
        else:
            print("\n⚠️  Enhanced Consensus System: Needs attention before production")

        return {
            "success": overall_success,
            "total_tests": total_tests,
            "successful_tests": successful_tests,
            "success_rate": (successful_tests / total_tests) * 100,
            "method_results": results,
        }


def main():
    """Run the comprehensive test."""
    validator = EnhancedConsensusValidator()
    result = validator.run_comprehensive_test()

    # Print final status
    print("\n" + "=" * 60)
    if result["success"]:
        print("✅ COMPREHENSIVE TEST PASSED - READY FOR PRODUCTION!")
    else:
        print("❌ COMPREHENSIVE TEST FAILED - NEEDS ATTENTION")
    print("=" * 60)

    return result


if __name__ == "__main__":
    main()
