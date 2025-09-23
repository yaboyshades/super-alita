#!/usr/bin/env python3
"""
EOS-Mangle Integration Demonstration

This script demonstrates the integration between the E-UPUSF Orchestration Schema (EOS)
and Google's Mangle deductive reasoning engine for enhanced context-adaptive problem solving.

Features demonstrated:
- EOS adaptive orchestration with Mangle reasoning
- Constitutional compliance validation via deductive rules
- Expert routing enhanced by logical inference
- Knowledge graph integration for decision-making
- Semantic lifting with logical justification
"""

import asyncio
import logging
import time
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger(__name__)


async def demo_eos_mangle_integration():
    """Demonstrate EOS-Mangle integration capabilities."""
    print("🎯 E-UPUSF Orchestration Schema + Mangle Integration Demo")
    print("=" * 65)
    print()
    
    # Step 1: Check system availability
    print("📋 Step 1: System Availability Check")
    print("-" * 40)
    
    # Check EOS availability
    try:
        # Import to check availability
        import src.eos.orchestrator  # noqa: F401
        import src.eos.schema  # noqa: F401
        eos_available = True
        print("✅ EOS System: Available")
    except ImportError as e:
        eos_available = False
        print(f"❌ EOS System: Not available ({e})")
    
    # Check Mangle availability
    try:
        # Import to check availability
        import src.abilities.mangle.mangle_ability  # noqa: F401
        import src.unified_intelligence.mangle_bridge  # noqa: F401
        mangle_available = True
        print("✅ Mangle System: Available")
    except ImportError as e:
        mangle_available = False
        print(f"❌ Mangle System: Not available ({e})")
    
    # Check EOS-Mangle integration
    try:
        from src.eos.mangle_integration import EOSMangleOrchestrator, EOSMangleConfig
        integration_available = True
        print("✅ EOS-Mangle Integration: Available")
    except ImportError as e:
        integration_available = False
        print(f"❌ EOS-Mangle Integration: Not available ({e})")
    
    # Check Unified Intelligence integration
    try:
        from src.unified_intelligence import (
            UnifiedIntelligenceEngine, EOS_AVAILABLE
        )
        unified_available = True
        eos_status = "Yes" if EOS_AVAILABLE else "No"
        print(f"✅ Unified Intelligence: Available (EOS: {eos_status})")
    except ImportError as e:
        unified_available = False
        print(f"❌ Unified Intelligence: Not available ({e})")
    
    print()
    
    # Step 2: Basic Integration Test
    print("🧪 Step 2: Basic Integration Test")
    print("-" * 40)
    
    if unified_available:
        try:
            engine = UnifiedIntelligenceEngine(enable_eos=True)
            status = {
                "unified_engine_initialized": True,
                "eos_enabled": engine.enable_eos,
                "eos_available": getattr(engine, 'EOS_AVAILABLE', False) if hasattr(engine, 'EOS_AVAILABLE') else False,
                "mangle_bridge_available": engine.mangle_bridge.is_available() if engine.mangle_bridge else False
            }
            
            print(f"✅ Unified Engine: {status['unified_engine_initialized']}")
            print(f"🎯 EOS Enabled: {status['eos_enabled']}")
            print(f"🧠 Mangle Available: {status['mangle_bridge_available']}")
            
        except Exception as e:
            print(f"❌ Integration test failed: {e}")
            status = {"error": str(e)}
    else:
        print("⚠️ Skipping - Unified Intelligence not available")
        status = {"skipped": True}
    
    print()
    
    # Step 3: Enhanced Interaction Demo
    print("🚀 Step 3: Enhanced Interaction Demo")
    print("-" * 40)
    
    if unified_available:
        try:
            engine = UnifiedIntelligenceEngine(enable_eos=True)
            
            # Test query that should trigger EOS-Mangle integration
            test_query = (
                "Create a user authentication system with secure password handling"
            )
            
            print(f"📝 Query: {test_query}")
            print()
            
            start_time = time.time()
            result = await engine.enhance_interaction(test_query, {
                "session_id": "demo_session",
                "request_type": "feature_creation",
                "complexity": "medium"
            })
            duration = (time.time() - start_time) * 1000
            
            print(f"⏱️ Processing Time: {duration:.0f}ms")
            print(f"🔍 Detected Pattern: {result.get('detected_pattern', 'None')}")
            print(f"⚖️ Constitutional Score: {result.get('constitutional_compliance', {}).get('score', 'N/A')}")
            print(f"🧠 Mangle Insights: {'Available' if result.get('mangle_insights', {}).get('available') else 'Not Available'}")
            print(f"🎯 EOS Insights: {'Available' if result.get('eos_insights', {}).get('available') else 'Not Available'}")
            print(f"📋 Recommendations: {len(result.get('recommendations', []))}")
            
            # Show some recommendations
            recommendations = result.get('recommendations', [])
            if recommendations:
                print("\n📋 Top Recommendations:")
                for i, rec in enumerate(recommendations[:3], 1):
                    print(f"   {i}. [{rec.get('type', 'general')}] {rec.get('message', 'No message')}")
            
        except Exception as e:
            print(f"❌ Enhanced interaction demo failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("⚠️ Skipping - Unified Intelligence not available")
    
    print()
    
    # Step 4: Direct EOS-Mangle Integration (if available)
    print("🔬 Step 4: Direct EOS-Mangle Integration")
    print("-" * 40)
    
    if integration_available and eos_available:
        try:
            # Check if reference spec exists
            spec_path = Path("examples/eos/reference-orchestration.yaml")
            if spec_path.exists():
                from src.eos.mangle_integration import EOSMangleOrchestrator, EOSMangleConfig
                
                # Load EOS specification
                eos_schema = EOSSchema()
                validation_result, eos_spec = eos_schema.load_and_validate(str(spec_path))
                
                if validation_result.valid:
                    print(f"✅ EOS Spec loaded: {spec_path}")
                    
                    # Create EOS-Mangle orchestrator
                    config = EOSMangleConfig(
                        enable_mangle_routing=mangle_available,
                        enable_deductive_validation=mangle_available,
                        enable_semantic_enhancement=True,
                        enable_knowledge_graph=mangle_available,
                        mangle_timeout=10.0
                    )
                    
                    orchestrator = EOSMangleOrchestrator(eos_spec, config)
                    integration_status = orchestrator.get_integration_status()
                    
                    print(f"🧠 Mangle Available: {integration_status['mangle_available']}")
                    print(f"🎯 Knowledge Base Init: {integration_status['knowledge_base_initialized']}")
                    print(f"⚖️ Constitutional Rules: {integration_status['constitutional_rules_loaded']}")
                    
                    # Test orchestration (with timeout)
                    print("\n🚀 Testing EOS-Mangle orchestration...")
                    start_time = time.time()
                    
                    try:
                        result = await asyncio.wait_for(
                            orchestrator.orchestrate_with_mangle(),
                            timeout=15.0
                        )
                        duration = (time.time() - start_time) * 1000
                        
                        print(f"✅ Orchestration Success: {result.success}")
                        print(f"⏱️ Duration: {duration:.0f}ms")
                        print(f"🏁 Final State: {result.final_state.value}")
                        print(f"📦 Artifacts: {len(result.artifacts)}")
                        
                        if 'mangle_constitutional_score' in result.governance_report:
                            score = result.governance_report['mangle_constitutional_score']
                            print(f"⚖️ Mangle Constitutional Score: {score:.2f}")
                        
                    except asyncio.TimeoutError:
                        print("⏰ Orchestration timed out (15s limit)")
                    except Exception as e:
                        print(f"❌ Orchestration failed: {e}")
                        
                else:
                    print(f"❌ Invalid EOS spec: {validation_result.errors}")
            else:
                print(f"⚠️ EOS reference spec not found: {spec_path}")
                
        except Exception as e:
            print(f"❌ Direct integration test failed: {e}")
    else:
        missing = []
        if not integration_available:
            missing.append("EOS-Mangle Integration")
        if not eos_available:
            missing.append("EOS System")
        print(f"⚠️ Skipping - Missing: {', '.join(missing)}")
    
    print()
    
    # Step 5: Summary and Next Steps
    print("📊 Step 5: Integration Summary")
    print("-" * 40)
    
    summary = {
        "eos_available": eos_available,
        "mangle_available": mangle_available,
        "integration_available": integration_available,
        "unified_available": unified_available,
        "full_integration": all([eos_available, mangle_available, integration_available, unified_available])
    }
    
    print(f"🎯 EOS System: {'✅' if summary['eos_available'] else '❌'}")
    print(f"🧠 Mangle System: {'✅' if summary['mangle_available'] else '❌'}")
    print(f"🔗 Integration Layer: {'✅' if summary['integration_available'] else '❌'}")
    print(f"🌟 Unified Intelligence: {'✅' if summary['unified_available'] else '❌'}")
    print()
    print(f"🏆 Full Integration Status: {'✅ OPERATIONAL' if summary['full_integration'] else '⚠️ PARTIAL'}")
    
    if summary['full_integration']:
        print("\n🎉 EOS-Mangle Integration is fully operational!")
        print("💡 You can now use:")
        print("   • Enhanced problem-solving with adaptive orchestration")
        print("   • Constitutional compliance via deductive reasoning")
        print("   • Expert routing with logical justification")
        print("   • Semantic lifting with knowledge graphs")
    else:
        print("\n📋 Next Steps:")
        if not summary['eos_available']:
            print("   • Install EOS system dependencies")
        if not summary['mangle_available']:
            print("   • Install Mangle deductive reasoning engine")
        if not summary['integration_available']:
            print("   • Check EOS-Mangle integration module")
        if not summary['unified_available']:
            print("   • Verify Unified Intelligence system")
    
    return summary


async def main():
    """Main demonstration entry point."""
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    try:
        summary = await demo_eos_mangle_integration()
        exit_code = 0 if summary.get('full_integration', False) else 1
        return exit_code
    except KeyboardInterrupt:
        print("\n🛑 Demo interrupted by user")
        return 1
    except Exception as e:
        print(f"\n💥 Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)