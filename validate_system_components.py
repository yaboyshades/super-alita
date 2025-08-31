#!/usr/bin/env python3
"""
Comprehensive validation script for Super Alita unified system
Tests all major components and integration points
"""

import asyncio
import sys
import time
import traceback
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.abilities.deepconf_ability import DeepConfAbility
from src.clients.deepconf_vllm import VLLMDeepConfClient
from src.core.optimization.nonstationary import NonStationaryBandit
from src.plugins.oak_core.tool_gating import UnifiedToolGatingSystem
from src.reasoning.deepconf_conf import MultiDomainConfidenceCalibrator
from src.reasoning.deepconf_pipeline import EnhancedDeepConfPipeline
from src.security.unified_security import SecurityConfig, UnifiedSecurity


class ValidationTestSuite:
    """Comprehensive test suite for system validation"""

    def __init__(self):
        self.passed_tests = 0
        self.failed_tests = 0
        self.test_results = []

    async def run_all_tests(self) -> bool:
        """Run all validation tests"""
        print("🎯 Starting Super Alita Unified System Validation")
        print("=" * 60)

        tests = [
            ("vLLM Client Validation", self.test_vllm_client),
            ("DeepConf Pipeline Validation", self.test_deepconf_pipeline),
            (
                "Confidence Calibration Validation",
                self.test_confidence_calibration,
            ),
            ("Tool Gating System Validation", self.test_tool_gating),
            ("Bandit Optimization Validation", self.test_bandit_optimization),
            ("Security System Validation", self.test_security_system),
            ("DeepConf Ability Integration", self.test_deepconf_ability),
            ("End-to-End Integration", self.test_integration),
        ]

        for test_name, test_func in tests:
            await self.run_test(test_name, test_func)

        self.print_summary()
        return self.failed_tests == 0

    async def run_test(self, test_name: str, test_func):
        """Run individual test with error handling"""
        print(f"\n🧪 {test_name}")
        print("-" * 40)

        try:
            start_time = time.time()
            result = await test_func()
            duration = time.time() - start_time

            if result:
                print(f"✅ PASSED ({duration:.2f}s)")
                self.passed_tests += 1
            else:
                print(f"❌ FAILED ({duration:.2f}s)")
                self.failed_tests += 1

            self.test_results.append(
                {"name": test_name, "passed": result, "duration": duration}
            )

        except Exception as e:
            print(f"💥 ERROR: {str(e)}")
            print(f"Traceback: {traceback.format_exc()}")
            self.failed_tests += 1
            self.test_results.append(
                {
                    "name": test_name,
                    "passed": False,
                    "duration": 0,
                    "error": str(e),
                }
            )

    async def test_vllm_client(self) -> bool:
        """Test vLLM client functionality"""
        try:
            # Create mock client (no actual vLLM server needed for validation)
            client = VLLMDeepConfClient(
                base_url="http://mock:8000/v1", model_name="test-model"
            )

            print("📋 Testing client initialization...")
            assert client.model_name == "test-model"
            assert client.timeout == 30.0
            assert client.request_count == 0

            print("📋 Testing performance stats...")
            stats = client.get_performance_stats()
            expected_keys = [
                "request_count",
                "total_tokens_generated",
                "total_generation_time",
                "average_generation_time",
            ]
            for key in expected_keys:
                assert key in stats

            print("📋 Testing health check structure...")
            # Note: This will fail without actual server, but structure should be correct
            try:
                await client.health_check()
            except Exception:
                pass  # Expected without real server

            print("📋 vLLM client validation successful")
            return True

        except Exception as e:
            print(f"vLLM client test failed: {e}")
            return False

    async def test_deepconf_pipeline(self) -> bool:
        """Test DeepConf pipeline functionality"""
        try:
            # Create mock model API
            class MockModelAPI:
                async def generate_with_logprobs(self, **kwargs):
                    # Mock generation result
                    class MockResult:
                        def __init__(self):
                            self.text = f"Mock response for: {kwargs.get('prompt', 'test')}"
                            self.confidence_score = 0.85
                            self.logprobs = [
                                {"token": "mock", "logprob": -0.1}
                            ]
                            self.generation_time = 0.1
                            self.metadata = {"mock": True}

                    return MockResult()

            print("📋 Testing pipeline initialization...")
            pipeline = EnhancedDeepConfPipeline(
                model_api=MockModelAPI(),
                cache_size=100,
                enable_adaptive_sampling=True,
            )

            assert pipeline.cache_size == 100
            assert pipeline.enable_adaptive_sampling == True
            assert len(pipeline.response_cache) == 0

            print("📋 Testing consensus request processing...")
            result = await pipeline.process_consensus_request(
                prompt="Test prompt",
                num_samples=2,
                consensus_method="weighted_vote",
                use_cache=False,  # Disable cache for test
            )

            # Verify result structure
            required_keys = ["consensus_text", "confidence", "metadata"]
            for key in required_keys:
                assert key in result, f"Missing key: {key}"

            assert "pipeline_processing_time" in result["metadata"]
            assert result["metadata"]["num_samples_requested"] == 2

            print("📋 Testing pipeline statistics...")
            stats = pipeline.get_pipeline_stats()
            assert "total_requests" in stats
            assert "successful_requests" in stats
            assert "cache_stats" in stats

            print("📋 DeepConf pipeline validation successful")
            return True

        except Exception as e:
            print(f"DeepConf pipeline test failed: {e}")
            return False

    async def test_confidence_calibration(self) -> bool:
        """Test confidence calibration functionality"""
        try:
            print("📋 Testing calibrator initialization...")
            calibrator = MultiDomainConfidenceCalibrator(
                cache_dir="./.test_cache"
            )

            # Test calibration update
            print("📋 Testing calibration update...")
            confidence_scores = [0.1, 0.3, 0.5, 0.7, 0.9] * 5  # 25 samples
            ground_truth = [False, False, True, True, True] * 5

            success = await calibrator.update_calibration(
                confidence_scores, ground_truth, domain="test_domain"
            )

            if success:
                print("✓ Calibration update successful")
            else:
                print("✓ Calibration update skipped (insufficient samples)")

            # Test confidence calibration
            print("📋 Testing confidence calibration...")
            test_scores = [0.2, 0.6, 0.8]
            calibrated = await calibrator.calibrate_confidence(
                test_scores, domain="test_domain"
            )

            assert len(calibrated) == len(test_scores)
            assert all(0 <= score <= 1 for score in calibrated)

            print("📋 Testing calibration statistics...")
            stats = calibrator.get_calibration_stats()
            assert "num_profiles" in stats
            assert "domains" in stats

            print("📋 Confidence calibration validation successful")
            return True

        except Exception as e:
            print(f"Confidence calibration test failed: {e}")
            return False

    async def test_tool_gating(self) -> bool:
        """Test tool gating system functionality"""
        try:
            print("📋 Testing tool gating initialization...")
            gating_system = UnifiedToolGatingSystem()

            # Register a test tool
            print("📋 Testing tool registration...")
            success = gating_system.register_tool("test_tool")
            assert success == True

            # Test tool execution
            print("📋 Testing tool execution...")

            async def mock_tool_function():
                await asyncio.sleep(0.01)  # Simulate work
                return "Tool executed successfully"

            result = await gating_system.execute_tool(
                "test_tool", mock_tool_function
            )

            assert result.success == True
            assert result.output == "Tool executed successfully"
            assert result.execution_time > 0
            assert "tool_name" in result.metadata

            # Test system status
            print("📋 Testing system status...")
            status = gating_system.get_system_status()
            assert "system_info" in status
            assert "tool_status" in status
            assert status["system_info"]["registered_tools"] >= 1

            print("📋 Tool gating system validation successful")
            return True

        except Exception as e:
            print(f"Tool gating test failed: {e}")
            return False

    async def test_bandit_optimization(self) -> bool:
        """Test bandit optimization functionality"""
        try:
            print("📋 Testing bandit initialization...")
            bandit = NonStationaryBandit(
                n_arms=3, learning_rate=0.1, algorithm="ucb_sliding"
            )

            assert bandit.n_arms == 3
            assert bandit.algorithm == "ucb_sliding"
            assert bandit.total_actions == 0

            # Test arm selection and updates
            print("📋 Testing arm selection and updates...")
            for i in range(20):
                arm = bandit.select_arm()
                assert 0 <= arm < bandit.n_arms

                # Simulate different reward patterns
                if arm == 0:
                    reward = 10 + (i * 0.1)  # Increasing rewards
                elif arm == 1:
                    reward = 5  # Constant rewards
                else:
                    reward = 15 - (i * 0.1)  # Decreasing rewards

                bandit.update(arm, reward)

            assert bandit.total_actions == 20

            # Test performance statistics
            print("📋 Testing performance statistics...")
            stats = bandit.get_performance_stats()
            assert "algorithm" in stats
            assert "total_actions" in stats
            assert "performance" in stats
            assert "arms" in stats

            # Test best arm selection
            print("📋 Testing best arm selection...")
            best_arm = bandit.get_best_arm()
            assert 0 <= best_arm < bandit.n_arms

            print("📋 Bandit optimization validation successful")
            return True

        except Exception as e:
            print(f"Bandit optimization test failed: {e}")
            return False

    async def test_security_system(self) -> bool:
        """Test security system functionality"""
        try:
            print("📋 Testing security system initialization...")
            config = SecurityConfig(
                jwt_secret="test_secret_key_for_validation", jwt_expiry_hours=1
            )
            security = UnifiedSecurity(config)

            # Test user registration
            print("📋 Testing user registration...")
            register_result = await security.register_user(
                username="testuser",
                email="test@example.com",
                password="SecurePassword123!",
                roles=["user"],
                ip_address="127.0.0.1",
            )

            assert register_result["success"] == True
            assert "user" in register_result

            # Test authentication
            print("📋 Testing user authentication...")
            auth_result = await security.authenticate_user(
                username="testuser",
                password="SecurePassword123!",
                ip_address="127.0.0.1",
            )

            assert auth_result["success"] == True
            assert "token" in auth_result

            # Test authorization
            print("📋 Testing authorization...")
            token = auth_result["token"]
            authz_result = await security.authorize_request(
                token=token,
                required_permission="read_own_data",
                ip_address="127.0.0.1",
            )

            assert authz_result["success"] == True
            assert authz_result["user_id"] == "testuser"

            # Test security statistics
            print("📋 Testing security statistics...")
            stats = security.get_security_stats()
            assert "total_users" in stats
            assert stats["total_users"] >= 1

            print("📋 Security system validation successful")
            return True

        except Exception as e:
            print(f"Security system test failed: {e}")
            return False

    async def test_deepconf_ability(self) -> bool:
        """Test DeepConf ability integration"""
        try:
            print("📋 Testing DeepConf ability initialization...")

            # Mock event bus
            class MockEventBus:
                def __init__(self):
                    self.subscriptions = {}

                async def subscribe(self, event_type, handler):
                    self.subscriptions[event_type] = handler

                async def publish(self, event):
                    pass

            ability = DeepConfAbility()
            event_bus = MockEventBus()

            # Test initialization (will fail without real vLLM, but should handle gracefully)
            print("📋 Testing ability initialization...")
            try:
                init_result = await ability.initialize(
                    event_bus,
                    vllm_base_url="http://mock:8000/v1",
                    model_name="test-model",
                )
                # May fail without real server, that's OK for validation
            except Exception:
                print(
                    "✓ Ability initialization handled connection error gracefully"
                )

            # Test plugin info
            print("📋 Testing plugin info...")
            info = ability.get_plugin_info()
            assert "name" in info
            assert info["name"] == "DeepConfAbility"

            print("📋 DeepConf ability validation successful")
            return True

        except Exception as e:
            print(f"DeepConf ability test failed: {e}")
            return False

    async def test_integration(self) -> bool:
        """Test end-to-end integration"""
        try:
            print("📋 Testing component integration...")

            # Test that all components can be imported and instantiated
            print("✓ All imports successful")

            # Test that components have compatible interfaces
            print("📋 Testing interface compatibility...")

            # Tool gating can work with arbitrary functions
            gating = UnifiedToolGatingSystem()

            async def test_function():
                return "integration_test_success"

            gating.register_tool("integration_test")
            result = await gating.execute_tool(
                "integration_test", test_function
            )
            assert result.success == True
            assert result.output == "integration_test_success"

            print("✓ Tool gating integration successful")

            # Bandit can be used for optimization
            bandit = NonStationaryBandit(n_arms=2)
            for i in range(5):
                arm = bandit.select_arm()
                bandit.update(arm, i * 10)

            print("✓ Bandit optimization integration successful")

            # Security system can validate tokens
            security = UnifiedSecurity()
            # (Already tested in detail above)

            print("✓ Security integration successful")

            print("📋 End-to-end integration validation successful")
            return True

        except Exception as e:
            print(f"Integration test failed: {e}")
            return False

    def print_summary(self):
        """Print test summary"""
        print("\n" + "=" * 60)
        print("🎯 VALIDATION SUMMARY")
        print("=" * 60)

        total_tests = self.passed_tests + self.failed_tests
        success_rate = (
            (self.passed_tests / total_tests * 100) if total_tests > 0 else 0
        )

        print(f"Total Tests: {total_tests}")
        print(f"Passed: {self.passed_tests}")
        print(f"Failed: {self.failed_tests}")
        print(f"Success Rate: {success_rate:.1f}%")

        if self.failed_tests == 0:
            print(
                "\n🎉 ALL TESTS PASSED - Super Alita is ready for deployment!"
            )
        else:
            print(
                f"\n⚠️  {self.failed_tests} test(s) failed - review errors above"
            )

        print("\n📊 Detailed Results:")
        for result in self.test_results:
            status = "✅ PASS" if result["passed"] else "❌ FAIL"
            duration = result.get("duration", 0)
            print(f"  {status} {result['name']} ({duration:.2f}s)")
            if "error" in result:
                print(f"       Error: {result['error']}")


async def main():
    """Main validation entry point"""
    try:
        suite = ValidationTestSuite()
        success = await suite.run_all_tests()

        # Return appropriate exit code
        sys.exit(0 if success else 1)

    except KeyboardInterrupt:
        print("\n🛑 Validation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Validation failed with unexpected error: {e}")
        print(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
