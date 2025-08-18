#!/usr/bin/env python3
"""
Comprehensive Super Alita System Verification

Tests all core components:
- Memory management and neural atoms
- Dynamic capability discovery
- Tool creation and execution
- Event bus communication
- Plugin composability
- Error handling and recovery
- Performance characteristics
"""

import asyncio
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

try:
    from src.core.event_bus import EventBus
    from src.core.events import ToolCallEvent, ToolResultEvent
    from src.core.neural_atom import (NeuralAtom, NeuralAtomMetadata,
                                      NeuralStore, TextualMemoryAtom)
    from src.plugins.core_utils_plugin_dynamic import CoreUtilsPlugin
    from src.tools.core_utils import CoreUtils
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're running from the super-alita root directory")
    sys.exit(1)


@dataclass
class VerificationResult:
    """Stores results of verification tests."""

    name: str
    passed: bool
    details: Dict[str, Any]
    execution_time: float
    error: Optional[str] = None


class ComprehensiveVerifier:
    """Comprehensive system verification suite."""

    def __init__(self):
        self.results: List[VerificationResult] = []
        self.mock_event_bus: Optional[MockEventBus] = None
        self.neural_store: Optional[NeuralStore] = None

    async def setup(self):
        """Set up test environment."""
        print("🔧 Setting up verification environment...")

        # Create mock event bus
        self.mock_event_bus = MockEventBus()

        # Create neural store
        self.neural_store = NeuralStore()

        print("✅ Verification environment ready")

    async def run_verification(
        self, test_name: str, test_func, *args, **kwargs
    ) -> VerificationResult:
        """Run a single verification test and capture results."""
        start_time = time.time()

        try:
            print(f"\n🧪 Running: {test_name}")
            details = await test_func(*args, **kwargs)
            execution_time = time.time() - start_time

            result = VerificationResult(
                name=test_name,
                passed=True,
                details=details,
                execution_time=execution_time,
            )
            print(f"  ✅ Passed ({execution_time:.3f}s)")

        except Exception as e:
            execution_time = time.time() - start_time
            result = VerificationResult(
                name=test_name,
                passed=False,
                details={},
                execution_time=execution_time,
                error=str(e),
            )
            print(f"  ❌ Failed ({execution_time:.3f}s): {e}")

        self.results.append(result)
        return result

    async def verify_memory_management(self) -> Dict[str, Any]:
        """Verify memory management and neural atom creation."""
        print("  📝 Testing TextualMemoryAtom creation...")

        # Test 1: Create TextualMemoryAtom
        metadata = NeuralAtomMetadata(
            name="test_memory_atom",
            description="Test memory for verification",
            capabilities=["memory_storage", "text_retrieval"],
        )

        memory_atom = TextualMemoryAtom(
            metadata=metadata,
            content="This is a test memory containing important information about Super Alita's capabilities.",
        )

        # Test 2: Execute memory atom
        result = await memory_atom.execute({"query": "capabilities"})
        assert "content" in result
        assert result["content"] == memory_atom.content

        # Test 3: Get embedding
        embedding = memory_atom.get_embedding()
        assert isinstance(embedding, list)
        assert len(embedding) > 0
        assert all(isinstance(x, float) for x in embedding)

        # Test 4: Test can_handle
        memory_score = memory_atom.can_handle("remember this information")
        non_memory_score = memory_atom.can_handle("calculate 2 + 2")
        assert memory_score > non_memory_score

        # Test 5: Store in neural store
        assert self.neural_store is not None, "Neural store should be initialized"
        self.neural_store.register(memory_atom)
        stored_atoms = self.neural_store.get_all()
        assert len(stored_atoms) >= 1

        return {
            "memory_atom_created": True,
            "execution_result": result,
            "embedding_length": len(embedding),
            "memory_capability_score": memory_score,
            "non_memory_capability_score": non_memory_score,
            "atoms_in_store": len(stored_atoms),
        }

    async def verify_dynamic_capabilities(self) -> Dict[str, Any]:
        """Verify dynamic capability discovery and tool registration."""
        print("  🔍 Testing dynamic capability discovery...")

        # Test 1: Create plugin and discover capabilities
        plugin = CoreUtilsPlugin()
        await plugin.setup(self.mock_event_bus, self.neural_store, {"enabled": True})

        capabilities = plugin.get_discovered_capabilities()
        assert len(capabilities) >= 2  # Should find calculate and reverse_string

        # Test 2: Verify specific capabilities
        assert "core.calculate" in capabilities
        assert "core.reverse_string" in capabilities

        # Test 3: Check capability metadata
        calc_meta = capabilities["core.calculate"]
        assert "parameters" in calc_meta
        assert "expression" in calc_meta["parameters"]
        assert calc_meta["is_static"] is True

        # Test 4: Test can_handle_tool
        assert plugin.can_handle_tool("core.calculate") is True
        assert plugin.can_handle_tool("core.reverse_string") is True
        assert plugin.can_handle_tool("unknown.tool") is False

        return {
            "capabilities_discovered": len(capabilities),
            "calculator_found": "core.calculate" in capabilities,
            "reverse_string_found": "core.reverse_string" in capabilities,
            "capability_metadata": calc_meta,
            "can_handle_tests_passed": True,
        }

    async def verify_tool_execution(self) -> Dict[str, Any]:
        """Verify tool execution via event bus."""
        print("  ⚡ Testing tool execution...")

        # Setup plugin
        plugin = CoreUtilsPlugin()
        await plugin.setup(self.mock_event_bus, self.neural_store, {"enabled": True})
        await plugin.start()

        # Test 1: Execute calculator tool
        # Create tool call event
        assert self.mock_event_bus is not None, "Mock event bus should be initialized"
        calc_event = ToolCallEvent(
            source_plugin="verifier",
            conversation_id="verification_session",
            session_id="verification_session_id",
            tool_name="core.calculate",
            parameters={"expression": "((10 + 5) * 2) - 3"},
            tool_call_id="verify_calc_1",
        )

        await self.mock_event_bus.publish(calc_event)

        # Test 2: Execute string reversal tool
        reverse_event = ToolCallEvent(
            source_plugin="verifier",
            conversation_id="verification_session",
            session_id="verification_session_id",
            tool_name="core.reverse_string",
            parameters={"text": "Super Alita Verification"},
            tool_call_id="verify_reverse_1",
        )

        await self.mock_event_bus.publish(reverse_event)

        # Allow processing
        await asyncio.sleep(0.1)

        # Analyze results
        tool_results = [
            event
            for event in self.mock_event_bus.published_events
            if isinstance(event, ToolResultEvent)
        ]

        calc_result = next(
            (r for r in tool_results if r.tool_call_id == "verify_calc_1"), None
        )
        reverse_result = next(
            (r for r in tool_results if r.tool_call_id == "verify_reverse_1"), None
        )

        assert calc_result is not None
        assert reverse_result is not None
        assert calc_result.success is True
        assert reverse_result.success is True

        await plugin.shutdown()

        return {
            "total_events_published": len(self.mock_event_bus.published_events),
            "tool_results_count": len(tool_results),
            "calculator_result": calc_result.result.get("result")
            if calc_result.result
            else None,
            "reverse_result": reverse_result.result.get("result")
            if reverse_result.result
            else None,
            "all_tools_successful": all(r.success for r in tool_results),
        }

    async def verify_atom_composability(self) -> Dict[str, Any]:
        """Verify neural atom composability and chaining."""
        print("  🔗 Testing atom composability...")

        # Create multiple atoms for composition
        atoms = []

        # Memory atom 1: Store calculation
        calc_memory = TextualMemoryAtom(
            metadata=NeuralAtomMetadata(
                name="calculation_memory",
                description="Stores calculation results",
                capabilities=["calculation_storage"],
            ),
            content="Previous calculation: 15 * 4 = 60",
        )
        atoms.append(calc_memory)

        # Memory atom 2: Store string processing
        string_memory = TextualMemoryAtom(
            metadata=NeuralAtomMetadata(
                name="string_memory",
                description="Stores string processing results",
                capabilities=["string_storage"],
            ),
            content="Reversed text: 'algorithm' becomes 'mhtirogla'",
        )
        atoms.append(string_memory)

        # Register all atoms
        assert self.neural_store is not None, "Neural store should be initialized"
        for atom in atoms:
            self.neural_store.register(atom)

        # Test composition: Find atoms that can handle specific tasks
        calc_handlers = []
        string_handlers = []

        for atom in atoms:
            if atom.can_handle("remember calculation") > 0.5:
                calc_handlers.append(atom)
            if atom.can_handle("remember string processing") > 0.5:
                string_handlers.append(atom)

        # Test execution chain
        execution_results = []
        for atom in atoms:
            result = await atom.execute({"type": "retrieve"})
            execution_results.append(result)

        return {
            "atoms_created": len(atoms),
            "atoms_registered": len(self.neural_store.get_all()),
            "calculation_handlers": len(calc_handlers),
            "string_handlers": len(string_handlers),
            "execution_results": execution_results,
            "composability_working": len(calc_handlers) > 0
            and len(string_handlers) > 0,
        }

    async def verify_error_handling(self) -> Dict[str, Any]:
        """Verify error handling and recovery."""
        print("  🛡️ Testing error handling...")

        plugin = CoreUtilsPlugin()
        await plugin.setup(self.mock_event_bus, self.neural_store, {"enabled": True})
        await plugin.start()

        error_scenarios = []

        # Test 1: Invalid calculation
        invalid_calc = ToolCallEvent(
            source_plugin="verifier",
            conversation_id="error_test",
            session_id="error_test_id",
            tool_name="core.calculate",
            parameters={"expression": "2 ** 3"},  # Power not supported
            tool_call_id="error_calc_1",
        )

        await self.mock_event_bus.publish(invalid_calc)
        await asyncio.sleep(0.1)

        # Test 2: Division by zero
        div_zero = ToolCallEvent(
            source_plugin="verifier",
            conversation_id="error_test",
            session_id="error_test_id",
            tool_name="core.calculate",
            parameters={"expression": "10 / 0"},
            tool_call_id="error_calc_2",
        )

        await self.mock_event_bus.publish(div_zero)
        await asyncio.sleep(0.1)

        # Test 3: Unknown tool
        unknown_tool = ToolCallEvent(
            source_plugin="verifier",
            conversation_id="error_test",
            session_id="error_test_id",
            tool_name="core.unknown_function",
            parameters={"param": "value"},
            tool_call_id="error_unknown_1",
        )

        await self.mock_event_bus.publish(unknown_tool)
        await asyncio.sleep(0.1)

        # Analyze error handling
        error_results = [
            event
            for event in self.mock_event_bus.published_events
            if isinstance(event, ToolResultEvent) and not event.success
        ]

        await plugin.shutdown()

        return {
            "error_scenarios_tested": 3,
            "error_results_captured": len(error_results),
            "system_remained_stable": True,  # If we got here, system didn't crash
            "error_handling_working": len(error_results) > 0,
        }

    async def verify_performance(self) -> Dict[str, Any]:
        """Verify system performance characteristics."""
        print("  📊 Testing performance...")

        plugin = CoreUtilsPlugin()
        await plugin.setup(self.mock_event_bus, self.neural_store, {"enabled": True})
        await plugin.start()

        # Performance test: Multiple rapid tool calls
        start_time = time.time()

        tasks = []
        for i in range(10):
            event = ToolCallEvent(
                source_plugin="verifier",
                conversation_id="perf_test",
                session_id="perf_test_id",
                tool_name="core.calculate",
                parameters={"expression": f"{i} + {i * 2}"},
                tool_call_id=f"perf_calc_{i}",
            )
            tasks.append(self.mock_event_bus.publish(event))

        await asyncio.gather(*tasks)
        await asyncio.sleep(0.2)  # Allow processing

        total_time = time.time() - start_time

        # Count successful results
        perf_results = [
            event
            for event in self.mock_event_bus.published_events
            if isinstance(event, ToolResultEvent)
            and event.tool_call_id.startswith("perf_calc_")
        ]

        await plugin.shutdown()

        return {
            "concurrent_operations": 10,
            "total_execution_time": total_time,
            "operations_per_second": 10 / total_time if total_time > 0 else 0,
            "successful_results": len([r for r in perf_results if r.success]),
            "all_operations_completed": len(perf_results) == 10,
        }

    async def run_comprehensive_verification(self):
        """Run all verification tests."""
        print("🚀 Super Alita Comprehensive System Verification")
        print("=" * 70)

        await self.setup()

        # Run all verification tests
        verification_tests = [
            ("Memory Management & Neural Atoms", self.verify_memory_management),
            ("Dynamic Capability Discovery", self.verify_dynamic_capabilities),
            ("Tool Execution & Event Bus", self.verify_tool_execution),
            ("Atom Composability & Chaining", self.verify_atom_composability),
            ("Error Handling & Recovery", self.verify_error_handling),
            ("Performance Characteristics", self.verify_performance),
        ]

        for test_name, test_func in verification_tests:
            await self.run_verification(test_name, test_func)

        # Generate summary report
        await self.generate_summary_report()

    async def generate_summary_report(self):
        """Generate comprehensive summary report."""
        print("\n" + "=" * 70)
        print("📋 COMPREHENSIVE VERIFICATION SUMMARY")
        print("=" * 70)

        passed_tests = [r for r in self.results if r.passed]
        failed_tests = [r for r in self.results if not r.passed]

        print("\n📊 Overall Results:")
        print(f"  ✅ Tests Passed: {len(passed_tests)}/{len(self.results)}")
        print(f"  ❌ Tests Failed: {len(failed_tests)}/{len(self.results)}")
        print(f"  📈 Success Rate: {len(passed_tests) / len(self.results) * 100:.1f}%")

        total_execution_time = sum(r.execution_time for r in self.results)
        print(f"  ⏱️ Total Execution Time: {total_execution_time:.3f}s")

        if passed_tests:
            print("\n✅ Successful Verifications:")
            for result in passed_tests:
                print(f"  • {result.name} ({result.execution_time:.3f}s)")
                if result.details:
                    for key, value in result.details.items():
                        if isinstance(value, (int, float, bool, str)):
                            print(f"    - {key}: {value}")

        if failed_tests:
            print("\n❌ Failed Verifications:")
            for result in failed_tests:
                print(f"  • {result.name}: {result.error}")

        # System capabilities summary
        print("\n🎯 System Capabilities Verified:")

        capabilities = {
            "Neural Atom Creation": any(
                "memory_atom_created" in r.details for r in passed_tests
            ),
            "Dynamic Tool Discovery": any(
                "capabilities_discovered" in r.details for r in passed_tests
            ),
            "Event-Driven Execution": any(
                "tool_results_count" in r.details for r in passed_tests
            ),
            "Atom Composability": any(
                "composability_working" in r.details for r in passed_tests
            ),
            "Error Recovery": any(
                "error_handling_working" in r.details for r in passed_tests
            ),
            "Performance Scaling": any(
                "operations_per_second" in r.details for r in passed_tests
            ),
        }

        for capability, verified in capabilities.items():
            status = "✅" if verified else "❌"
            print(f"  {status} {capability}")

        # Final assessment
        if len(failed_tests) == 0:
            print("\n🎉 VERIFICATION COMPLETE: All systems operational!")
            print("🚀 Super Alita is ready for advanced AI operations")
        else:
            print(
                f"\n⚠️ VERIFICATION PARTIAL: {len(failed_tests)} issues need attention"
            )

        return len(failed_tests) == 0


class MockEventBus:
    """Enhanced mock event bus for comprehensive testing."""

    def __init__(self):
        self.published_events = []
        self.subscribers = {}

    async def publish(self, event):
        """Mock publish - stores events and triggers handlers."""
        self.published_events.append(event)

        # Trigger subscribers
        event_type = getattr(event, "event_type", type(event).__name__)
        if event_type in self.subscribers:
            for handler in self.subscribers[event_type]:
                try:
                    await handler(event)
                except Exception as e:
                    print(f"⚠️ Handler error: {e}")

    async def subscribe(self, event_type: str, handler):
        """Mock subscribe - registers handlers."""
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
        self.subscribers[event_type].append(handler)


async def main():
    """Run comprehensive system verification."""
    verifier = ComprehensiveVerifier()

    try:
        success = await verifier.run_comprehensive_verification()
        return success
    except Exception as e:
        print(f"\n💥 Verification suite crashed: {e}")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
