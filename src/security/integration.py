#!/usr/bin/env python3
"""
Security & Resilience Integration - Integration layer for Super Alita

This module integrates the new security and resilience components into the existing
Super Alita application architecture.

Features:
1. PolicyManager integration with ability registry
2. SecretManager integration with configuration system
3. ResilienceManager integration with external API calls
4. Middleware for automatic policy enforcement
5. Health checks and monitoring endpoints
"""

import asyncio
import logging
from typing import Any, Callable, Dict, Optional

# Use absolute imports since we're testing as __main__
try:
    from .policy_manager import PolicyManager, create_policy_manager_from_env
    from .secret_manager import SecretManager, get_global_secret_manager
    from .resilience_manager import ResilienceManager, get_global_resilience_manager, ServiceCategory
except ImportError:
    # Fallback for direct execution
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent))
    from policy_manager import PolicyManager, create_policy_manager_from_env
    from secret_manager import SecretManager, get_global_secret_manager
    from resilience_manager import ResilienceManager, get_global_resilience_manager, ServiceCategory

logger = logging.getLogger(__name__)


class SecurityResilienceMiddleware:
    """Middleware to automatically apply security and resilience patterns"""
    
    def __init__(
        self,
        policy_manager: Optional[PolicyManager] = None,
        secret_manager: Optional[SecretManager] = None,
        resilience_manager: Optional[ResilienceManager] = None
    ):
        self.policy_manager = policy_manager or create_policy_manager_from_env()
        self.secret_manager = secret_manager or get_global_secret_manager()
        self.resilience_manager = resilience_manager or get_global_resilience_manager()
        
        logger.info("Security & Resilience Middleware initialized")
    
    async def execute_tool_call(
        self,
        actor: str,
        tool_name: str,
        tool_function: Callable,
        args: Dict[str, Any],
        **kwargs
    ) -> Any:
        """Execute tool call with full security and resilience protection"""
        
        # 1. Policy Check - ensure actor can call this tool
        environment = kwargs.get('environment', {})
        can_call = await self.policy_manager.can_call(
            actor=actor,
            resource=tool_name,
            action="execute",
            environment=environment,
            session_id=kwargs.get('session_id')
        )
        
        if not can_call:
            logger.warning(f"Policy denied: {actor} -> {tool_name}")
            raise PermissionError(f"Access denied: {actor} not authorized to call {tool_name}")
        
        # 2. Secret Resolution - replace any secret references in args
        resolved_args = await self._resolve_secrets(args)
        
        # 3. Resilient Execution - execute with circuit breaker and bulkhead protection
        service_category = self._determine_service_category(tool_name)
        use_hedging = self._should_use_hedging(tool_name)
        
        try:
            result = await self.resilience_manager.execute_resilient_call(
                service_name=tool_name,
                func=lambda: tool_function(**resolved_args),
                category=service_category,
                use_hedging=use_hedging
            )
            
            logger.debug(f"Tool call successful: {actor} -> {tool_name}")
            return result
            
        except Exception as e:
            logger.error(f"Tool call failed: {actor} -> {tool_name}: {e}")
            raise
    
    async def _resolve_secrets(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve any secret references in arguments"""
        resolved_args = {}
        
        for key, value in args.items():
            if isinstance(value, str) and value.startswith("secret:"):
                # Extract secret name
                secret_name = value[7:]  # Remove "secret:" prefix
                secret_value = await self.secret_manager.get_secret(secret_name)
                
                if secret_value is None:
                    logger.warning(f"Secret '{secret_name}' not found, using fallback")
                    resolved_args[key] = f"MISSING_SECRET_{secret_name}"
                else:
                    resolved_args[key] = secret_value
            else:
                resolved_args[key] = value
        
        return resolved_args
    
    def _determine_service_category(self, tool_name: str) -> ServiceCategory:
        """Determine service category based on tool name"""
        tool_lower = tool_name.lower()
        
        if any(keyword in tool_lower for keyword in ["github", "api", "http", "rest"]):
            return ServiceCategory.EXTERNAL_API
        elif any(keyword in tool_lower for keyword in ["openai", "llm", "gpt", "anthropic"]):
            return ServiceCategory.LLM_API
        elif any(keyword in tool_lower for keyword in ["db", "database", "redis", "sql"]):
            return ServiceCategory.DATABASE
        elif any(keyword in tool_lower for keyword in ["file", "fs", "read", "write"]):
            return ServiceCategory.FILE_SYSTEM
        elif any(keyword in tool_lower for keyword in ["process", "cpu", "compute"]):
            return ServiceCategory.CPU_INTENSIVE
        else:
            return ServiceCategory.EXTERNAL_API  # Default fallback
    
    def _should_use_hedging(self, tool_name: str) -> bool:
        """Determine if hedging should be used for this tool"""
        # Use hedging for external APIs and LLM calls where latency matters
        tool_lower = tool_name.lower()
        return any(keyword in tool_lower for keyword in [
            "api", "http", "llm", "openai", "search", "external"
        ])
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Get health status of all security and resilience components"""
        policy_metrics = self.policy_manager.get_metrics()
        secret_metrics = self.secret_manager.get_metrics()
        secret_health = await self.secret_manager.health_check()
        resilience_health = self.resilience_manager.get_system_health()
        
        return {
            "policy_manager": {
                "status": "healthy",
                "metrics": policy_metrics
            },
            "secret_manager": {
                "status": "healthy" if any(secret_health.values()) else "degraded",
                "metrics": secret_metrics,
                "backend_health": secret_health
            },
            "resilience_manager": {
                "status": "healthy" if resilience_health["health_score"] > 0.5 else "degraded",
                "health_score": resilience_health["health_score"],
                "metrics": resilience_health
            }
        }


class SecureAbilityRegistry:
    """Enhanced ability registry with integrated security and resilience"""
    
    def __init__(self, base_registry, middleware: Optional[SecurityResilienceMiddleware] = None):
        self.base_registry = base_registry
        self.middleware = middleware or SecurityResilienceMiddleware()
        
        # Track protected executions
        self.execution_stats = {
            "total_calls": 0,
            "policy_denials": 0,
            "successful_calls": 0,
            "failed_calls": 0,
            "secret_resolutions": 0
        }
    
    async def execute(self, tool_name: str, args: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Execute tool with security and resilience protection"""
        self.execution_stats["total_calls"] += 1
        
        # Determine actor (could be from session, API key, etc.)
        actor = kwargs.pop('actor', 'unknown_actor')  # Remove from kwargs to avoid duplication
        
        # Get the original tool function
        if hasattr(self.base_registry, 'execute'):
            original_func = lambda: self.base_registry.execute(tool_name, args)
        else:
            # Fallback if base registry doesn't have execute method
            original_func = lambda: {"error": "Tool execution not available"}
        
        try:
            # Execute through security middleware
            result = await self.middleware.execute_tool_call(
                actor=actor,
                tool_name=tool_name,
                tool_function=original_func,
                args=args,
                **kwargs
            )
            
            self.execution_stats["successful_calls"] += 1
            return result
            
        except PermissionError as e:
            self.execution_stats["policy_denials"] += 1
            logger.warning(f"Tool execution denied: {e}")
            return {"error": "Access denied", "details": str(e)}
            
        except Exception as e:
            self.execution_stats["failed_calls"] += 1
            logger.error(f"Tool execution failed: {e}")
            return {"error": "Execution failed", "details": str(e)}
    
    def get_stats(self) -> Dict[str, Any]:
        """Get execution statistics"""
        return self.execution_stats.copy()
    
    # Delegate other methods to base registry
    def __getattr__(self, name):
        return getattr(self.base_registry, name)


def integrate_security_resilience(app):
    """
    Integrate security and resilience components into FastAPI app
    
    This function adds the new security and resilience features to an existing
    Super Alita application instance.
    """
    
    # Create middleware instance
    middleware = SecurityResilienceMiddleware()
    
    # Wrap existing ability registry if available
    if hasattr(app.state, 'ability_registry'):
        original_registry = app.state.ability_registry
        app.state.ability_registry = SecureAbilityRegistry(original_registry, middleware)
        logger.info("Ability registry wrapped with security & resilience protection")
    
    # Store middleware on app state for access by other components
    app.state.security_resilience_middleware = middleware
    
    # Add health check endpoints
    @app.get("/security/health")
    async def security_health():
        """Security and resilience health check endpoint"""
        return await middleware.get_health_status()
    
    @app.get("/security/policy/status")
    async def policy_status():
        """Policy manager status endpoint"""
        return middleware.policy_manager.get_metrics()
    
    @app.get("/security/secrets/status")
    async def secrets_status():
        """Secret manager status endpoint"""
        metrics = middleware.secret_manager.get_metrics()
        health = await middleware.secret_manager.health_check()
        return {"metrics": metrics, "backend_health": health}
    
    @app.get("/security/resilience/status")
    async def resilience_status():
        """Resilience manager status endpoint"""
        return middleware.resilience_manager.get_system_health()
    
    @app.post("/security/policy/reload")
    async def reload_policies():
        """Reload security policies"""
        success = await middleware.policy_manager.reload_policies()
        return {"success": success, "message": "Policies reloaded" if success else "Failed to reload policies"}
    
    @app.post("/security/secrets/clear-cache")
    async def clear_secret_cache():
        """Clear secret cache"""
        middleware.secret_manager.clear_cache()
        return {"success": True, "message": "Secret cache cleared"}
    
    logger.info("Security & Resilience integration completed")
    return middleware


# Example usage for testing
async def test_integration():
    """Test the security and resilience integration"""
    
    # Mock ability registry for testing
    class MockAbilityRegistry:
        async def execute(self, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
            # Simulate tool execution
            await asyncio.sleep(0.1)
            if "fail" in tool_name:
                raise Exception("Simulated tool failure")
            return {"success": True, "tool": tool_name, "args": args}
    
    # Create secure registry
    mock_registry = MockAbilityRegistry()
    middleware = SecurityResilienceMiddleware()
    secure_registry = SecureAbilityRegistry(mock_registry, middleware)
    
    print("Security & Resilience Integration Test")
    print("=" * 50)
    
    # Test cases
    test_cases = [
        ("system_actor", "echo", {"message": "hello"}),
        ("github_client", "github_search_repos", {"query": "test"}),
        ("unauthorized_actor", "dangerous_tool", {"action": "delete"}),
        ("llm_client", "openai_completion", {"prompt": "test", "api_key": "secret:openai_key"})
    ]
    
    for actor, tool, args in test_cases:
        try:
            result = await secure_registry.execute(
                tool, args, actor=actor, environment={"env": "development"}
            )
            status = "✅ Success" if "error" not in result else "❌ Error"
            print(f"{status} {actor} -> {tool}: {result}")
        except Exception as e:
            print(f"❌ Exception {actor} -> {tool}: {e}")
    
    # Show statistics
    print(f"\nExecution Statistics:")
    stats = secure_registry.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Show health status
    print(f"\nHealth Status:")
    health = await middleware.get_health_status()
    for component, status in health.items():
        print(f"  {component}: {status['status']}")


if __name__ == "__main__":
    asyncio.run(test_integration())