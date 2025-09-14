#!/usr/bin/env python3
"""
Policy Manager - Lightweight policy enforcement layer for Super Alita

This module provides a flexible policy enforcement system that can:
1. Control which tools/APIs atoms can access
2. Support both YAML-based rules and external policy engines (OPA/Cedar)
3. Integrate with the existing security framework
4. Provide audit logging for all policy decisions

Design Philosophy:
- Start simple with YAML rules, evolve to external engines
- Fail-safe: deny by default, explicit allow
- Audit everything for compliance and debugging
- Minimal performance impact on critical paths
"""

import json
import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


class PolicyDecision(Enum):
    """Policy decision outcomes"""
    ALLOW = "allow"
    DENY = "deny"
    UNKNOWN = "unknown"


class PolicyEngine(Enum):
    """Supported policy engines"""
    YAML = "yaml"
    OPA = "opa" 
    CEDAR = "cedar"


@dataclass
class PolicyContext:
    """Context information for policy evaluation"""
    actor: str  # Which atom/component is making the call
    action: str  # What action is being attempted
    resource: str  # What resource/tool is being accessed
    environment: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    session_id: str | None = None
    ip_address: str | None = None
    user_agent: str | None = None


@dataclass 
class PolicyResult:
    """Result of policy evaluation"""
    decision: PolicyDecision
    reason: str
    rule_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    evaluation_time_ms: float = 0.0


@dataclass
class PolicyRule:
    """Individual policy rule for YAML engine"""
    id: str
    description: str
    effect: PolicyDecision  # ALLOW or DENY
    conditions: dict[str, Any]
    priority: int = 100  # Lower number = higher priority


class PolicyEngineInterface(ABC):
    """Abstract interface for policy engines"""
    
    @abstractmethod
    async def evaluate(self, context: PolicyContext) -> PolicyResult:
        """Evaluate policy for given context"""
        pass
    
    @abstractmethod
    async def reload_policies(self) -> bool:
        """Reload policies from source"""
        pass


class YAMLPolicyEngine(PolicyEngineInterface):
    """YAML-based policy engine for simple rules"""
    
    def __init__(self, policy_file: str = "config/security_policies.yaml"):
        self.policy_file = Path(policy_file)
        self.rules: list[PolicyRule] = []
        self.default_decision = PolicyDecision.DENY
        self._last_loaded = 0.0
        
    async def evaluate(self, context: PolicyContext) -> PolicyResult:
        """Evaluate context against YAML rules"""
        start_time = time.time()
        
        # Auto-reload if file changed
        if self._should_reload():
            await self.reload_policies()
        
        # Evaluate rules in priority order
        for rule in sorted(self.rules, key=lambda r: r.priority):
            if self._rule_matches(rule, context):
                evaluation_time = (time.time() - start_time) * 1000
                return PolicyResult(
                    decision=rule.effect,
                    reason=rule.description,
                    rule_id=rule.id,
                    evaluation_time_ms=evaluation_time
                )
        
        # No rules matched, use default
        evaluation_time = (time.time() - start_time) * 1000
        return PolicyResult(
            decision=self.default_decision,
            reason="No matching rules found, using default policy",
            evaluation_time_ms=evaluation_time
        )
    
    async def reload_policies(self) -> bool:
        """Load policies from YAML file"""
        try:
            if not self.policy_file.exists():
                # Create default policy file
                self._create_default_policies()
            
            with open(self.policy_file) as f:
                data = yaml.safe_load(f) or {}
            
            self.rules = []
            self.default_decision = PolicyDecision(
                data.get('default_decision', 'deny')
            )
            
            for rule_data in data.get('rules', []):
                rule = PolicyRule(
                    id=rule_data['id'],
                    description=rule_data['description'],
                    effect=PolicyDecision(rule_data['effect']),
                    conditions=rule_data['conditions'],
                    priority=rule_data.get('priority', 100)
                )
                self.rules.append(rule)
            
            self._last_loaded = time.time()
            logger.info(f"Loaded {len(self.rules)} policy rules from {self.policy_file}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load policies from {self.policy_file}: {e}")
            return False
    
    def _should_reload(self) -> bool:
        """Check if policies should be reloaded"""
        if not self.policy_file.exists():
            return True
        
        try:
            file_mtime = self.policy_file.stat().st_mtime
            return file_mtime > self._last_loaded
        except OSError:
            return True
    
    def _rule_matches(self, rule: PolicyRule, context: PolicyContext) -> bool:
        """Check if rule conditions match context"""
        conditions = rule.conditions
        
        # Check actor patterns
        if 'actors' in conditions:
            actor_patterns = conditions['actors']
            if not self._matches_patterns(context.actor, actor_patterns):
                return False
        
        # Check action patterns  
        if 'actions' in conditions:
            action_patterns = conditions['actions']
            if not self._matches_patterns(context.action, action_patterns):
                return False
        
        # Check resource patterns
        if 'resources' in conditions:
            resource_patterns = conditions['resources']
            if not self._matches_patterns(context.resource, resource_patterns):
                return False
        
        # Check environment conditions
        if 'environment' in conditions:
            env_conditions = conditions['environment']
            for key, expected in env_conditions.items():
                actual = context.environment.get(key)
                if actual != expected:
                    return False
        
        return True
    
    def _matches_patterns(self, value: str, patterns: list[str]) -> bool:
        """Check if value matches any of the patterns"""
        import fnmatch
        
        for pattern in patterns:
            if fnmatch.fnmatch(value, pattern):
                return True
        return False
    
    def _create_default_policies(self):
        """Create default policy file"""
        self.policy_file.parent.mkdir(parents=True, exist_ok=True)
        
        default_policies = {
            'default_decision': 'deny',
            'description': 'Super Alita Security Policies - Deny by default, explicit allow',
            'rules': [
                {
                    'id': 'allow_core_tools',
                    'description': 'Allow core system tools for all actors',
                    'effect': 'allow',
                    'priority': 10,
                    'conditions': {
                        'resources': [
                            'echo',
                            'secure_scan_code',
                            'python_import_smoke',
                            'pytest_run'
                        ]
                    }
                },
                {
                    'id': 'allow_authenticated_github',
                    'description': 'Allow GitHub tools for authenticated requests',
                    'effect': 'allow',
                    'priority': 20,
                    'conditions': {
                        'resources': ['github_*'],
                        'environment': {
                            'authenticated': True
                        }
                    }
                },
                {
                    'id': 'allow_development_tools',
                    'description': 'Allow development tools in dev environment',
                    'effect': 'allow',
                    'priority': 30,
                    'conditions': {
                        'resources': [
                            'code_synthesize*',
                            'ladder_reug_generate',
                            'unified_execute'
                        ],
                        'environment': {
                            'env': 'development'
                        }
                    }
                },
                {
                    'id': 'deny_destructive_production',
                    'description': 'Deny destructive operations in production',
                    'effect': 'deny',
                    'priority': 5,
                    'conditions': {
                        'actions': ['delete', 'destroy', 'overwrite'],
                        'environment': {
                            'env': 'production'
                        }
                    }
                }
            ]
        }
        
        with open(self.policy_file, 'w') as f:
            yaml.dump(default_policies, f, default_flow_style=False, indent=2)
        
        logger.info(f"Created default policy file at {self.policy_file}")


class OPAPolicyEngine(PolicyEngineInterface):
    """Open Policy Agent (OPA) integration"""
    
    def __init__(self, opa_url: str = "http://localhost:8181"):
        self.opa_url = opa_url
        self.policy_path = "/v1/data/super_alita/allow"
        
    async def evaluate(self, context: PolicyContext) -> PolicyResult:
        """Evaluate policy via OPA REST API"""
        start_time = time.time()
        
        try:
            import httpx
            
            # Prepare input for OPA
            input_data = {
                "input": {
                    "actor": context.actor,
                    "action": context.action,
                    "resource": context.resource,
                    "environment": context.environment,
                    "timestamp": context.timestamp
                }
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.opa_url}{self.policy_path}",
                    json=input_data,
                    timeout=5.0
                )
                
                if response.status_code == 200:
                    result = response.json()
                    allowed = result.get("result", False)
                    
                    evaluation_time = (time.time() - start_time) * 1000
                    return PolicyResult(
                        decision=PolicyDecision.ALLOW if allowed else PolicyDecision.DENY,
                        reason=f"OPA decision: {allowed}",
                        metadata={"opa_result": result},
                        evaluation_time_ms=evaluation_time
                    )
                else:
                    logger.error(f"OPA returned status {response.status_code}")
                    
        except Exception as e:
            logger.error(f"OPA evaluation failed: {e}")
        
        # Fallback to deny on error
        evaluation_time = (time.time() - start_time) * 1000
        return PolicyResult(
            decision=PolicyDecision.DENY,
            reason="OPA evaluation failed, defaulting to deny",
            evaluation_time_ms=evaluation_time
        )
    
    async def reload_policies(self) -> bool:
        """OPA policies are managed externally"""
        return True


class PolicyManager:
    """Main policy management system"""
    
    def __init__(
        self,
        engine_type: PolicyEngine = PolicyEngine.YAML,
        engine_config: dict[str, Any] | None = None
    ):
        self.engine_type = engine_type
        self.engine_config = engine_config or {}
        self.engine = self._create_engine()
        
        # Audit logging
        self.audit_file = Path("logs/policy_audit.jsonl")
        self.audit_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Performance metrics
        self.total_evaluations = 0
        self.total_evaluation_time = 0.0
        self.cache: dict[str, PolicyResult] = {}
        self.cache_ttl = 300  # 5 minutes
        
    def _create_engine(self) -> PolicyEngineInterface:
        """Create policy engine based on configuration"""
        if self.engine_type == PolicyEngine.YAML:
            policy_file = self.engine_config.get(
                'policy_file', 'config/security_policies.yaml'
            )
            return YAMLPolicyEngine(policy_file)
        elif self.engine_type == PolicyEngine.OPA:
            opa_url = self.engine_config.get('opa_url', 'http://localhost:8181')
            return OPAPolicyEngine(opa_url)
        else:
            raise ValueError(f"Unsupported policy engine: {self.engine_type}")
    
    async def can_call(
        self,
        actor: str,
        resource: str,
        action: str = "execute",
        environment: dict[str, Any] | None = None,
        session_id: str | None = None
    ) -> bool:
        """Main policy check method - returns True if call is allowed"""
        context = PolicyContext(
            actor=actor,
            action=action,
            resource=resource,
            environment=environment or {},
            session_id=session_id
        )
        
        result = await self.evaluate(context)
        return result.decision == PolicyDecision.ALLOW
    
    async def evaluate(self, context: PolicyContext) -> PolicyResult:
        """Evaluate policy and log the decision"""
        # Check cache first
        cache_key = self._cache_key(context)
        cached_result = self._get_cached_result(cache_key)
        if cached_result:
            return cached_result
        
        # Evaluate via engine
        start_time = time.time()
        result = await self.engine.evaluate(context)
        evaluation_time = time.time() - start_time
        
        # Update metrics
        self.total_evaluations += 1
        self.total_evaluation_time += evaluation_time
        
        # Cache result
        self._cache_result(cache_key, result)
        
        # Audit log
        await self._audit_log(context, result)
        
        return result
    
    async def reload_policies(self) -> bool:
        """Reload policies and clear cache"""
        success = await self.engine.reload_policies()
        if success:
            self.cache.clear()
            logger.info("Policies reloaded and cache cleared")
        return success
    
    def _cache_key(self, context: PolicyContext) -> str:
        """Generate cache key for context"""
        # Simple hash of key fields
        import hashlib
        key_data = f"{context.actor}:{context.action}:{context.resource}:{json.dumps(context.environment, sort_keys=True)}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _get_cached_result(self, cache_key: str) -> PolicyResult | None:
        """Get cached result if still valid"""
        if cache_key not in self.cache:
            return None
        
        result, cached_at = self.cache[cache_key]
        if time.time() - cached_at > self.cache_ttl:
            del self.cache[cache_key]
            return None
        
        return result
    
    def _cache_result(self, cache_key: str, result: PolicyResult):
        """Cache policy result"""
        self.cache[cache_key] = (result, time.time())
        
        # Limit cache size
        if len(self.cache) > 1000:
            # Remove oldest 20% of entries
            sorted_items = sorted(
                self.cache.items(), 
                key=lambda x: x[1][1]  # Sort by timestamp
            )
            keep_count = int(len(sorted_items) * 0.8)
            self.cache = dict(sorted_items[-keep_count:])
    
    async def _audit_log(self, context: PolicyContext, result: PolicyResult):
        """Log policy decision for audit purposes"""
        try:
            audit_entry = {
                "timestamp": datetime.now(UTC).isoformat(),
                "actor": context.actor,
                "action": context.action,
                "resource": context.resource,
                "environment": context.environment,
                "decision": result.decision.value,
                "reason": result.reason,
                "rule_id": result.rule_id,
                "evaluation_time_ms": result.evaluation_time_ms,
                "session_id": context.session_id,
                "ip_address": context.ip_address
            }
            
            with open(self.audit_file, 'a') as f:
                f.write(json.dumps(audit_entry) + '\n')
                
        except Exception as e:
            logger.error(f"Failed to write audit log: {e}")
    
    def get_metrics(self) -> dict[str, Any]:
        """Get performance and usage metrics"""
        avg_evaluation_time = (
            self.total_evaluation_time / max(1, self.total_evaluations) * 1000
        )
        
        return {
            "total_evaluations": self.total_evaluations,
            "avg_evaluation_time_ms": avg_evaluation_time,
            "cache_size": len(self.cache),
            "cache_ttl_seconds": self.cache_ttl,
            "engine_type": self.engine_type.value
        }


# Integration helper for existing systems
def create_policy_manager_from_env() -> PolicyManager:
    """Create policy manager from environment variables"""
    engine_type_str = os.getenv("SUPER_ALITA_POLICY_ENGINE", "yaml").lower()
    
    try:
        engine_type = PolicyEngine(engine_type_str)
    except ValueError:
        logger.warning(f"Unknown policy engine '{engine_type_str}', using YAML")
        engine_type = PolicyEngine.YAML
    
    config = {}
    if engine_type == PolicyEngine.YAML:
        config['policy_file'] = os.getenv(
            "SUPER_ALITA_POLICY_FILE", 
            "config/security_policies.yaml"
        )
    elif engine_type == PolicyEngine.OPA:
        config['opa_url'] = os.getenv(
            "SUPER_ALITA_OPA_URL", 
            "http://localhost:8181"
        )
    
    return PolicyManager(engine_type, config)


# Example usage and testing
if __name__ == "__main__":
    import asyncio
    
    async def test_policy_manager():
        """Test policy manager functionality"""
        pm = create_policy_manager_from_env()
        
        # Ensure policies are loaded
        await pm.reload_policies()
        
        # Test cases
        test_cases = [
            ("atom_github_client", "github_search_repos", "execute", {"authenticated": True}),
            ("atom_github_client", "github_search_repos", "execute", {"authenticated": False}),
            ("atom_code_gen", "code_synthesize", "execute", {"env": "development"}),
            ("atom_code_gen", "code_synthesize", "execute", {"env": "production"}),
            ("atom_destructive", "delete_files", "delete", {"env": "production"}),
            ("atom_core", "echo", "execute", {}),
        ]
        
        print("Policy Manager Test Results:")
        print("=" * 50)
        
        for actor, resource, action, env in test_cases:
            allowed = await pm.can_call(actor, resource, action, env)
            status = "✅ ALLOW" if allowed else "❌ DENY"
            print(f"{status} {actor} -> {resource} ({action}) [env: {env}]")
        
        # Show metrics
        print("\nMetrics:")
        metrics = pm.get_metrics()
        for key, value in metrics.items():
            print(f"  {key}: {value}")
    
    asyncio.run(test_policy_manager())