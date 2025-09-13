#!/usr/bin/env python3
"""
Usage Examples - Security & Resilience Features

This file demonstrates how to use the new security and resilience features
in Super Alita applications.
"""

import asyncio
import os
from typing import Dict, Any


async def example_1_policy_enforcement():
    """Example 1: Policy enforcement for tool access control"""
    print("🔒 Example 1: Policy Enforcement")
    print("-" * 40)
    
    from src.security.policy_manager import create_policy_manager_from_env
    
    # Create policy manager
    policy_manager = create_policy_manager_from_env()
    
    # Test different scenarios
    scenarios = [
        {
            "actor": "authenticated_user",
            "resource": "github_search_repos", 
            "environment": {"authenticated": True, "env": "development"},
            "expected": "ALLOW"
        },
        {
            "actor": "anonymous_user",
            "resource": "github_search_repos",
            "environment": {"authenticated": False},
            "expected": "DENY"
        },
        {
            "actor": "any_user",
            "resource": "echo",
            "environment": {},
            "expected": "ALLOW (core tool)"
        }
    ]
    
    for scenario in scenarios:
        allowed = await policy_manager.can_call(
            actor=scenario["actor"],
            resource=scenario["resource"],
            environment=scenario["environment"]
        )
        
        status = "✅ ALLOWED" if allowed else "❌ DENIED"
        print(f"{status} {scenario['actor']} -> {scenario['resource']}")
        print(f"  Expected: {scenario['expected']}")
        print()


async def example_2_secret_management():
    """Example 2: Secure secret resolution from multiple backends"""
    print("🔑 Example 2: Secret Management") 
    print("-" * 40)
    
    from src.security.secret_manager import SecretManager, MemorySecretBackend
    
    # Create secret manager with test data
    memory_backend = MemorySecretBackend({
        "openai_api_key": "sk-test-1234567890",
        "github_token": "ghp_test_token_123",
        "database_password": "super_secure_password_456"
    })
    
    secret_manager = SecretManager([memory_backend])
    
    # Demonstrate secret retrieval
    secrets_to_test = ["openai_api_key", "github_token", "nonexistent_secret"]
    
    for secret_name in secrets_to_test:
        value = await secret_manager.get_secret(secret_name)
        if value:
            # Mask the secret for display
            masked_value = value[:8] + "..." if len(value) > 8 else "***"
            print(f"✅ {secret_name}: {masked_value}")
        else:
            print(f"❌ {secret_name}: Not found")
    
    # Show fallback behavior
    fallback_value = await secret_manager.get_secret_with_fallback(
        "missing_secret", "default_value"
    )
    print(f"🔄 Fallback example: {fallback_value}")
    print()


async def example_3_resilient_api_calls():
    """Example 3: Resilient API calls with circuit breakers and bulkheads"""
    print("🛡️ Example 3: Resilient API Calls")
    print("-" * 40)
    
    from src.security.resilience_manager import ResilienceManager, ServiceCategory
    import random
    import time
    
    resilience_manager = ResilienceManager()
    
    # Simulate unreliable external service
    async def unreliable_api_call(fail_rate: float = 0.3):
        """Simulates an API call that sometimes fails"""
        await asyncio.sleep(0.1)  # Simulate network delay
        if random.random() < fail_rate:
            raise Exception("External API temporarily unavailable")
        return {"status": "success", "data": f"response_at_{time.time():.2f}"}
    
    # Test multiple calls with circuit breaker protection
    print("Making 10 API calls with circuit breaker protection...")
    success_count = 0
    
    for i in range(10):
        try:
            result = await resilience_manager.execute_resilient_call(
                service_name="external_api",
                func=unreliable_api_call,
                category=ServiceCategory.EXTERNAL_API,
                use_hedging=False,  # Disable hedging for clearer demo
                fail_rate=0.4  # 40% failure rate
            )
            success_count += 1
            print(f"  Call {i+1}: ✅ Success")
        except Exception as e:
            print(f"  Call {i+1}: ❌ Failed - {str(e)[:50]}...")
    
    print(f"\nSuccess rate: {success_count}/10 ({success_count*10}%)")
    
    # Show system health
    health = resilience_manager.get_system_health()
    print(f"System health score: {health['health_score']:.2f}")
    print()


async def example_4_integrated_workflow():
    """Example 4: Complete workflow with all security features"""
    print("🔧 Example 4: Integrated Security Workflow")
    print("-" * 40)
    
    from src.security.integration import SecurityResilienceMiddleware
    
    # Create integrated middleware
    middleware = SecurityResilienceMiddleware()
    
    # Simulate a secure tool call workflow
    async def secure_github_search(query: str):
        """Simulates a GitHub search that requires authentication"""
        await asyncio.sleep(0.2)  # Simulate API call
        return {
            "query": query,
            "results": [
                {"name": "super-alita", "stars": 100},
                {"name": "example-repo", "stars": 50}
            ]
        }
    
    # Test with different actors and environments
    test_scenarios = [
        {
            "actor": "authenticated_dev",
            "environment": {"authenticated": True, "env": "development"},
            "should_succeed": True
        },
        {
            "actor": "anonymous_user", 
            "environment": {"authenticated": False},
            "should_succeed": False
        }
    ]
    
    for scenario in test_scenarios:
        try:
            result = await middleware.execute_tool_call(
                actor=scenario["actor"],
                tool_name="github_search_repos",
                tool_function=lambda: secure_github_search("super-alita"),
                args={"query": "super-alita"},
                environment=scenario["environment"]
            )
            
            if scenario["should_succeed"]:
                print(f"✅ {scenario['actor']}: Success (as expected)")
                print(f"   Found {len(result.get('results', []))} repositories")
            else:
                print(f"❌ {scenario['actor']}: Unexpected success")
                
        except PermissionError as e:
            if not scenario["should_succeed"]:
                print(f"✅ {scenario['actor']}: Properly denied access")
                print(f"   Reason: {e}")
            else:
                print(f"❌ {scenario['actor']}: Unexpected denial")
        except Exception as e:
            print(f"❌ {scenario['actor']}: Unexpected error - {e}")
    
    print()


async def example_5_monitoring_and_health():
    """Example 5: Monitoring and health check capabilities"""
    print("📊 Example 5: Monitoring and Health Checks")
    print("-" * 40)
    
    from src.security.integration import SecurityResilienceMiddleware
    
    middleware = SecurityResilienceMiddleware()
    
    # Get comprehensive health status
    health_status = await middleware.get_health_status()
    
    print("Component Health Status:")
    for component, details in health_status.items():
        status_emoji = "✅" if details["status"] == "healthy" else "⚠️"
        print(f"  {status_emoji} {component.replace('_', ' ').title()}: {details['status']}")
        
        # Show key metrics
        if "metrics" in details:
            metrics = details["metrics"]
            if component == "policy_manager":
                print(f"     Policy evaluations: {metrics.get('total_evaluations', 0)}")
                print(f"     Avg response time: {metrics.get('avg_evaluation_time_ms', 0):.2f}ms")
            elif component == "secret_manager":
                print(f"     Cache size: {metrics.get('cache_size', 0)}")
                print(f"     Backend count: {metrics.get('backend_count', 0)}")
            elif component == "resilience_manager":
                score = details.get('health_score', 0)
                print(f"     Health score: {score:.2f}")
    
    print("\nConfiguration Summary:")
    print("  🔒 Security: Deny-by-default policy with explicit allow rules")
    print("  🔑 Secrets: Multi-backend resolution with encrypted caching")
    print("  🛡️ Resilience: Circuit breakers and bulkhead isolation")
    print("  📊 Monitoring: Real-time health checks and metrics")


async def main():
    """Run all examples"""
    print("🚀 Super Alita Security & Resilience Examples")
    print("=" * 50)
    print()
    
    examples = [
        example_1_policy_enforcement,
        example_2_secret_management,
        example_3_resilient_api_calls,
        example_4_integrated_workflow,
        example_5_monitoring_and_health
    ]
    
    for example in examples:
        await example()
        print()
    
    print("✨ All examples completed successfully!")
    print("📚 See SECURITY_RESILIENCE_IMPLEMENTATION.md for detailed documentation.")


if __name__ == "__main__":
    # Set up minimal environment for examples
    os.environ.setdefault("SUPER_ALITA_POLICY_ENGINE", "yaml")
    os.environ.setdefault("SUPER_ALITA_POLICY_FILE", "config/security_policies.yaml")
    
    asyncio.run(main())