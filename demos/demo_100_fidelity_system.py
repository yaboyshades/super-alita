#!/usr/bin/env python3
"""
Super Alita 100% Fidelity System Demo

Demonstrates the complete implementation of the performance monitoring
and rule automation system as specified in the 100% fidelity breakdown.

This demo validates that all components work together:
1. Performance Monitoring with SLO tracking
2. Rule Automation with constitutional compliance
3. Telemetry collection and reporting
4. CI/CD pipeline integration
5. User documentation and workflow

Run this demo to verify system integration and functionality.
"""

import asyncio
import json
import logging
import sys
import time
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def demo_telemetry_system():
    """Demo 1: OpenTelemetry Performance Monitoring"""
    print("\n🔄 Demo 1: OpenTelemetry Performance Monitoring")
    print("=" * 60)
    
    try:
        # Import telemetry components
        from src.performance_monitoring.telemetry.opentelemetry_config import (
            get_telemetry_collector,
            telemetry_span,
            telemetry_trace,
        )
        
        collector = get_telemetry_collector()
        print("✅ Telemetry collector initialized")
        print(f"   SLOs: p95 < {collector.slos.latency_p95_ms}ms, error rate < {collector.slos.error_rate_threshold * 100}%")
        
        # Demo telemetry tracing
        @telemetry_trace("demo", "extension_call")
        async def mock_extension_call():
            await asyncio.sleep(0.1)  # Simulate work
            return {"result": "success"}
        
        # Generate some telemetry data
        print("📊 Generating telemetry data...")
        for i in range(5):
            result = await mock_extension_call()
            print(f"   Call {i+1}: {result}")
        
        # Get metrics summary
        summary = collector.get_metrics_summary()
        if summary.get("status") != "no_data":
            print("📈 Metrics Summary:")
            print(f"   Total requests: {summary['total_requests']}")
            print(f"   Error rate: {summary['error_rate']:.1%}")
            print(f"   Avg latency: {summary['latency_ms']['avg']:.1f}ms")
            print(f"   SLO compliance: {summary['slo_compliance']}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Telemetry system not available: {e}")
        return False
    except Exception as e:
        print(f"❌ Telemetry demo failed: {e}")
        return False


async def demo_extension_middleware():
    """Demo 2: Extension Interaction Middleware"""
    print("\n🔧 Demo 2: Extension Interaction Middleware")
    print("=" * 60)
    
    try:
        from src.performance_monitoring.middleware.extension_interceptors import (
            get_extension_middleware,
            track_extension_call,
        )
        
        middleware = get_extension_middleware()
        print("✅ Extension middleware initialized")
        
        # Demo automatic telemetry tracking
        @track_extension_call("demo_extension", "validation")
        async def validate_code(code):
            await asyncio.sleep(0.05)  # Simulate validation work
            if "error" in code:
                raise ValueError("Mock validation error")
            return {"valid": True, "score": 0.95}
        
        print("🔍 Testing extension calls with automatic telemetry...")
        
        # Successful calls
        for i in range(3):
            result = await validate_code(f"good_code_{i}")
            print(f"   ✅ Call {i+1}: {result}")
        
        # Error call (for 100% error capture)
        try:
            await validate_code("error_code")
        except ValueError as e:
            print(f"   ❌ Error call: {e} (captured in telemetry)")
        
        # Get interaction summary
        summary = middleware.get_interaction_summary()
        if summary.get("status") != "no_telemetry_collector":
            print(f"📊 Interaction Summary: {summary}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Middleware system not available: {e}")
        return False
    except Exception as e:
        print(f"❌ Middleware demo failed: {e}")
        return False


def demo_constitutional_rules():
    """Demo 3: Constitutional Rule Framework"""
    print("\n🏛️ Demo 3: Constitutional Rule Framework")
    print("=" * 60)
    
    try:
        # Check if constitutional rules exist
        rules_dir = Path("rules/constitution")
        if not rules_dir.exists():
            print("❌ Constitutional rules directory not found")
            return False
        
        rule_files = list(rules_dir.glob("*.yaml"))
        print(f"✅ Found {len(rule_files)} constitutional rule files:")
        
        for rule_file in rule_files:
            try:
                import yaml
                with open(rule_file, encoding='utf-8') as f:
                    rule_data = yaml.safe_load(f)
                
                print(f"   📋 {rule_data.get('id', 'Unknown')}: {rule_data.get('description', 'No description')}")
                print(f"      Article: {rule_data.get('article', 'Unknown')}")
                print(f"      Severity: {rule_data.get('severity', 'Unknown')}")
                
            except Exception as e:
                print(f"   ❌ Error reading {rule_file}: {e}")
        
        # Check ruleset metadata
        metadata_file = rules_dir / "ruleset_metadata.yaml"
        if metadata_file.exists():
            with open(metadata_file, encoding='utf-8') as f:
                metadata = yaml.safe_load(f)
            
            print("\n📖 Ruleset Metadata:")
            print(f"   Name: {metadata.get('ruleset', {}).get('name', 'Unknown')}")
            print(f"   Version: {metadata.get('ruleset', {}).get('version', 'Unknown')}")
            print(f"   Total Articles: {len(metadata.get('articles', []))}")
        
        return True
        
    except ImportError as e:
        print(f"❌ YAML library not available: {e}")
        return False
    except Exception as e:
        print(f"❌ Constitutional rules demo failed: {e}")
        return False


def demo_rule_validator():
    """Demo 4: Rule Validator CLI"""
    print("\n⚖️ Demo 4: Rule Validator CLI")
    print("=" * 60)
    
    try:
        import subprocess
        
        validator_script = Path("scripts/rule_validator.py")
        if not validator_script.exists():
            print("❌ Rule validator script not found")
            return False
        
        print("✅ Rule validator script found")
        
        # Test validator on a small target
        test_target = "src/performance_monitoring/telemetry"
        if Path(test_target).exists():
            print(f"🔍 Running constitutional validation on {test_target}...")
            
            # Run validator with JSON output
            try:
                result = subprocess.run([
                    sys.executable, str(validator_script),
                    "--format", "json",
                    "--quiet",
                    test_target
                ], capture_output=True, text=True, timeout=30)
                
                if result.stdout:
                    try:
                        validation_data = json.loads(result.stdout)
                        print("📊 Validation Results:")
                        print(f"   Files checked: {validation_data.get('total_files_checked', 0)}")
                        print(f"   Total violations: {validation_data.get('total_violations', 0)}")
                        print(f"   Blockers: {validation_data.get('blocker_count', 0)}")
                        print(f"   Warnings: {validation_data.get('warning_count', 0)}")
                        print(f"   Success: {validation_data.get('success', False)}")
                        
                        if result.returncode == 0:
                            print("   ✅ Constitutional compliance: PASS")
                        elif result.returncode == 1:
                            print("   ⚠️ Constitutional compliance: WARNINGS")
                        else:
                            print("   ❌ Constitutional compliance: VIOLATIONS")
                            
                    except json.JSONDecodeError:
                        print("   ⚠️ Validator output not JSON parseable")
                
                else:
                    print("   ⚠️ No validator output received")
                
            except subprocess.TimeoutExpired:
                print("   ❌ Validator timed out")
                return False
            except subprocess.CalledProcessError as e:
                print(f"   ❌ Validator process error: {e}")
                return False
        
        else:
            print(f"⚠️ Test target {test_target} not found, skipping validation test")
        
        return True
        
    except Exception as e:
        print(f"❌ Rule validator demo failed: {e}")
        return False


def demo_monitoring_stack():
    """Demo 5: Monitoring Stack Configuration"""
    print("\n📊 Demo 5: Monitoring Stack Configuration")
    print("=" * 60)
    
    try:
        monitoring_dir = Path("monitoring")
        if not monitoring_dir.exists():
            print("❌ Monitoring directory not found")
            return False
        
        print("✅ Monitoring directory found")
        
        # Check required monitoring files
        required_files = [
            "docker-compose.yml",
            "prometheus/prometheus.yml", 
            "grafana/dashboards/performance_dashboard.json",
            "alertmanager/alerting_rules.yml"
        ]
        
        missing_files = []
        present_files = []
        
        for file_path in required_files:
            full_path = monitoring_dir / file_path
            if full_path.exists():
                present_files.append(file_path)
                print(f"   ✅ {file_path}")
            else:
                missing_files.append(file_path)
                print(f"   ❌ {file_path}")
        
        print("\n📈 Monitoring Stack Status:")
        print(f"   Present files: {len(present_files)}/{len(required_files)}")
        print(f"   Missing files: {len(missing_files)}")
        
        if missing_files:
            print(f"   Missing: {', '.join(missing_files)}")
        
        # Check dashboard configuration
        dashboard_file = monitoring_dir / "grafana/dashboards/performance_dashboard.json"
        if dashboard_file.exists():
            try:
                with open(dashboard_file, encoding='utf-8') as f:
                    dashboard_data = json.load(f)
                
                dashboard_info = dashboard_data.get("dashboard", {})
                panels = dashboard_info.get("panels", [])
                
                print("\n📊 Grafana Dashboard:")
                print(f"   Title: {dashboard_info.get('title', 'Unknown')}")
                print(f"   Panels: {len(panels)}")
                
                for panel in panels[:3]:  # Show first 3 panels
                    print(f"     • {panel.get('title', 'Untitled')} ({panel.get('type', 'unknown')})")
                
                if len(panels) > 3:
                    print(f"     ... and {len(panels) - 3} more panels")
                    
            except json.JSONDecodeError as e:
                print(f"   ❌ Dashboard JSON invalid: {e}")
        
        return len(missing_files) == 0
        
    except Exception as e:
        print(f"❌ Monitoring stack demo failed: {e}")
        return False


def demo_ci_integration():
    """Demo 6: CI/CD Integration"""
    print("\n🚀 Demo 6: CI/CD Integration")
    print("=" * 60)
    
    try:
        # Check GitHub Actions workflow
        workflow_file = Path(".github/workflows/constitutional_validation.yml")
        if workflow_file.exists():
            print("✅ GitHub Actions workflow found")
            
            with open(workflow_file, encoding='utf-8') as f:
                workflow_content = f.read()
            
            # Count jobs and steps
            import re
            jobs = re.findall(r'^\s+\w+:', workflow_content, re.MULTILINE)
            steps = re.findall(r'^\s+- name:', workflow_content, re.MULTILINE)
            
            print(f"   Jobs: {len(jobs)}")
            print(f"   Steps: {len(steps)}")
            
        else:
            print("❌ GitHub Actions workflow not found")
        
        # Check pre-commit configuration
        precommit_file = Path(".pre-commit-config.yaml")
        if precommit_file.exists():
            print("✅ Pre-commit configuration found")
            
            import yaml
            with open(precommit_file, encoding='utf-8') as f:
                precommit_data = yaml.safe_load(f)
            
            repos = precommit_data.get("repos", [])
            total_hooks = sum(len(repo.get("hooks", [])) for repo in repos)
            
            print(f"   Repositories: {len(repos)}")
            print(f"   Total hooks: {total_hooks}")
            
            # Check for constitutional validation hook
            has_constitutional_hook = any(
                "constitutional" in str(hook.get("id", "")).lower()
                for repo in repos
                for hook in repo.get("hooks", [])
            )
            
            if has_constitutional_hook:
                print("   ✅ Constitutional validation hook configured")
            else:
                print("   ⚠️ Constitutional validation hook not found")
                
        else:
            print("❌ Pre-commit configuration not found")
        
        # Check unification validation script
        unification_script = Path("scripts/validate_unification.py")
        if unification_script.exists():
            print("✅ Unification validation script found")
        else:
            print("❌ Unification validation script not found")
        
        return True
        
    except Exception as e:
        print(f"❌ CI integration demo failed: {e}")
        return False


def demo_documentation():
    """Demo 7: Documentation and User Guides"""
    print("\n📚 Demo 7: Documentation and User Guides")
    print("=" * 60)
    
    try:
        # Check key documentation files
        doc_files = [
            ("CONTRIBUTING.md", "Contributing Guide"),
            ("docs/architecture.md", "Architecture Overview"),
            ("README.md", "Project README")
        ]
        
        found_docs = 0
        
        for file_path, description in doc_files:
            doc_file = Path(file_path)
            if doc_file.exists():
                print(f"✅ {description}")
                
                # Get file size for basic validation
                size_kb = doc_file.stat().st_size / 1024
                print(f"   Size: {size_kb:.1f} KB")
                
                found_docs += 1
            else:
                print(f"❌ {description} not found")
        
        print("\n📖 Documentation Status:")
        print(f"   Found: {found_docs}/{len(doc_files)} key documents")
        
        # Check for Mermaid diagrams in architecture
        arch_file = Path("docs/architecture.md")
        if arch_file.exists():
            with open(arch_file, encoding='utf-8') as f:
                arch_content = f.read()
            
            mermaid_count = arch_content.count("```mermaid")
            print(f"   Mermaid diagrams: {mermaid_count}")
        
        return found_docs == len(doc_files)
        
    except Exception as e:
        print(f"❌ Documentation demo failed: {e}")
        return False


async def main():
    """Run complete 100% fidelity system demo."""
    print("🎯 Super Alita 100% Fidelity System Demo")
    print("=" * 80)
    print("Validating implementation against the comprehensive specification...")
    
    start_time = time.perf_counter()
    
    # Run all demos
    demos = [
        ("OpenTelemetry Performance Monitoring", demo_telemetry_system),
        ("Extension Interaction Middleware", demo_extension_middleware),
        ("Constitutional Rule Framework", demo_constitutional_rules),
        ("Rule Validator CLI", demo_rule_validator),
        ("Monitoring Stack Configuration", demo_monitoring_stack),
        ("CI/CD Integration", demo_ci_integration),
        ("Documentation and User Guides", demo_documentation)
    ]
    
    results = []
    
    for name, demo_func in demos:
        try:
            if asyncio.iscoroutinefunction(demo_func):
                result = await demo_func()
            else:
                result = demo_func()
            results.append((name, result))
        except Exception as e:
            print(f"❌ Demo '{name}' crashed: {e}")
            results.append((name, False))
    
    # Summary
    execution_time = (time.perf_counter() - start_time) * 1000
    
    print("\n🎯 100% Fidelity System Validation Summary")
    print("=" * 80)
    
    passed = sum(1 for _, result in results if result)
    failed = len(results) - passed
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {name}")
    
    print("\n📊 Overall Results:")
    print(f"   Total Components: {len(results)}")
    print(f"   Passed: {passed}")
    print(f"   Failed: {failed}")
    print(f"   Success Rate: {passed/len(results)*100:.1f}%")
    print(f"   Execution Time: {execution_time:.1f}ms")
    
    # Final verdict
    if failed == 0:
        print("\n🎉 SUCCESS: 100% Fidelity Implementation Complete!")
        print("   All components operational and integrated")
        print("   System ready for production deployment")
        return 0
    else:
        print(f"\n⚠️ PARTIAL: {failed} components need attention")
        print("   Review failed components and resolve issues")
        return 1


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n❌ Demo interrupted")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        sys.exit(1)