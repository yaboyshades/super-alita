"""Ability registry service with tool management."""

from __future__ import annotations

from typing import Any, Dict, List, Callable, Optional
import json

from .base import BaseService
from ..governance import ConstitutionalViolationError

class AbilityRegistryService(BaseService):
    """Clean ability registry with constitutional integration."""
    
    def __init__(self, config, registry):
        super().__init__(config, registry)
        self.tools: Dict[str, Dict[str, Any]] = {}
        self.executors: Dict[str, Callable] = {}
        self.execution_stats: Dict[str, int] = {}
    
    async def initialize(self) -> None:
        """Initialize ability registry with core tools."""
        # Register core tools
        await self._register_core_tools()
        
        # Register enhanced tools if features are enabled
        if self.config.features.enhanced_consensus:
            await self._register_consensus_tools()
        
        if self.config.features.z3_verifier:
            await self._register_z3_tools()
        
        self._initialized = True
        self.logger.info(f"Ability registry initialized with {len(self.tools)} tools")
    
    async def _register_core_tools(self) -> None:
        """Register core built-in tools."""
        # Echo tool
        await self.register_tool(
            contract={
                "tool_id": "echo",
                "description": "Echo back the provided payload",
                "input_schema": {
                    "type": "object",
                    "properties": {"payload": {"type": "string"}},
                    "required": ["payload"]
                },
                "output_schema": {"type": "object"}
            },
            executor=self._echo_executor
        )
        
        # GitHub fetch tool
        await self.register_tool(
            contract={
                "tool_id": "fetch_github_raw",
                "description": "Fetch raw file content from GitHub",
                "input_schema": {
                    "type": "object",
                    "required": ["owner", "repo", "path"],
                    "properties": {
                        "owner": {"type": "string"},
                        "repo": {"type": "string"},
                        "path": {"type": "string"},
                        "ref": {"type": "string", "default": "main"}
                    }
                },
                "output_schema": {"type": "object"}
            },
            executor=self._github_fetch_executor
        )
        
        # Code security scanner
        await self.register_tool(
            contract={
                "tool_id": "secure_scan_code",
                "description": "Scan code for security vulnerabilities",
                "input_schema": {
                    "type": "object",
                    "required": ["code"],
                    "properties": {"code": {"type": "string"}}
                },
                "output_schema": {"type": "object"}
            },
            executor=self._security_scan_executor
        )
    
    async def _register_consensus_tools(self) -> None:
        """Register enhanced consensus tools."""
        try:
            from ..abilities.enhanced_consensus_ability import EnhancedConsensusProvider
            
            consensus_provider = EnhancedConsensusProvider({
                "base_url": f"{self.config.llm.ollama_host}/v1",
                "model_name": self.config.llm.model.replace("ollama:", ""),
                "timeout": float(self.config.llm.timeout)
            })
            
            await consensus_provider.initialize()
            
            await self.register_tool(
                contract={
                    "tool_id": "deepconf_consensus",
                    "description": "Enhanced consensus sampling with multiple aggregation methods",
                    "input_schema": {
                        "type": "object",
                        "required": ["prompt"],
                        "properties": {
                            "prompt": {"type": "string"},
                            "num_samples": {"type": "integer", "default": 3},
                            "method": {"type": "string", "default": "weighted_vote"}
                        }
                    },
                    "output_schema": {"type": "object"}
                },
                executor=lambda args: consensus_provider.consensus_sampling(
                    prompt=args["prompt"],
                    num_samples=args.get("num_samples", 3),
                    method=args.get("method", "weighted_vote")
                )
            )
            
            self.logger.info("✅ Consensus tools registered")
            
        except Exception as e:
            self.logger.error(f"Failed to register consensus tools: {e}")
    
    async def _register_z3_tools(self) -> None:
        """Register Z3 verification tools."""
        try:
            from ..cognitive.z3_verifier import ScalableZ3Verifier
            
            z3_verifier = ScalableZ3Verifier()
            
            await self.register_tool(
                contract={
                    "tool_id": "z3_verify",
                    "description": "Verify constraints using Z3 solver",
                    "input_schema": {
                        "type": "object",
                        "required": ["constraints"],
                        "properties": {
                            "constraints": {"type": "array"},
                            "timeout_s": {"type": "integer", "default": 10}
                        }
                    },
                    "output_schema": {"type": "object"}
                },
                executor=lambda args: z3_verifier.verify(
                    args["constraints"],
                    timeout_s=args.get("timeout_s", 10)
                )
            )
            
            self.logger.info("✅ Z3 verification tools registered")
            
        except Exception as e:
            self.logger.error(f"Failed to register Z3 tools: {e}")
    
    async def register_tool(self, contract: Dict[str, Any], executor: Callable) -> None:
        """Register a new tool with contract and executor."""
        tool_id = contract.get("tool_id")
        if not tool_id:
            raise ValueError("Tool contract must include 'tool_id'")
        
        self.tools[tool_id] = contract
        self.executors[tool_id] = executor
        self.execution_stats[tool_id] = 0
        
        self.logger.info(f"Registered tool: {tool_id}")
    
    async def execute(self, tool_id: str, args: Dict[str, Any], context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute a tool with constitutional checking."""
        if tool_id not in self.tools:
            return {"error": f"Unknown tool: {tool_id}"}
        
        # Validate arguments
        validation_error = self._validate_args(tool_id, args)
        if validation_error:
            return {"error": validation_error}
        
        # Constitutional check
        constitutional_service = self.get_service("constitutional")
        if constitutional_service:
            try:
                action = {"tool_id": tool_id, "args": args}
                evaluation_context = context or {}
                
                approved, reasoning = await constitutional_service.evaluate_action(action, evaluation_context)
                if not approved:
                    return {
                        "error": "constitutional_violation",
                        "reasoning": reasoning
                    }
            except Exception as e:
                self.logger.error(f"Constitutional evaluation failed: {e}")
                # Continue with execution but log the issue
        
        # Execute tool
        try:
            executor = self.executors[tool_id]
            result = executor(args)
            
            # Handle async executors
            if hasattr(result, '__await__'):
                result = await result
            
            # Update stats
            self.execution_stats[tool_id] += 1
            
            # Emit execution event
            event_bus = self.get_service("event_bus")
            if event_bus:
                await event_bus.emit("tool_executed", {
                    "tool_id": tool_id,
                    "success": True,
                    "execution_count": self.execution_stats[tool_id]
                })
            
            return result
            
        except Exception as e:
            self.logger.error(f"Tool execution failed ({tool_id}): {e}")
            
            # Emit failure event
            event_bus = self.get_service("event_bus")
            if event_bus:
                await event_bus.emit("tool_execution_failed", {
                    "tool_id": tool_id,
                    "error": str(e)
                })
            
            return {"error": f"Execution failed: {str(e)}"}
    
    def _validate_args(self, tool_id: str, args: Dict[str, Any]) -> Optional[str]:
        """Validate tool arguments against schema."""
        contract = self.tools[tool_id]
        schema = contract.get("input_schema", {})
        required = schema.get("required", [])
        
        # Check required fields
        for field in required:
            if field not in args:
                return f"Missing required field: {field}"
        
        # TODO: Add full JSON schema validation
        return None
    
    def get_available_tools(self) -> List[Dict[str, Any]]:
        """Get list of available tools."""
        return list(self.tools.values())
    
    def get_execution_stats(self) -> Dict[str, int]:
        """Get tool execution statistics."""
        return self.execution_stats.copy()
    
    # Built-in tool executors
    async def _echo_executor(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Echo tool executor."""
        return {"echo": args.get("payload", "")}
    
    async def _github_fetch_executor(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """GitHub fetch tool executor."""
        import urllib.request
        
        owner = args["owner"]
        repo = args["repo"]
        path = args["path"]
        ref = args.get("ref", "main")
        
        url = f"https://raw.githubusercontent.com/{owner}/{repo}/{ref}/{path}"
        
        try:
            with urllib.request.urlopen(url, timeout=10) as response:
                content = response.read().decode('utf-8')
            
            return {
                "content": content,
                "url": url,
                "success": True
            }
            
        except Exception as e:
            return {
                "error": str(e),
                "url": url,
                "success": False
            }
    
    async def _security_scan_executor(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Security scan tool executor."""
        import re
        
        code = args["code"]
        issues = []
        
        # Security patterns
        patterns = [
            (r"exec\(", "HIGH", "Dangerous exec() usage"),
            (r"eval\(", "HIGH", "Dangerous eval() usage"),
            (r"os\.system\(", "MEDIUM", "System command execution"),
            (r"subprocess", "MEDIUM", "Subprocess usage"),
            (r"__import__\(", "LOW", "Dynamic import")
        ]
        
        for pattern, severity, description in patterns:
            matches = list(re.finditer(pattern, code))
            for match in matches:
                line = code[:match.start()].count('\n') + 1
                issues.append({
                    "severity": severity,
                    "description": description,
                    "line": line,
                    "pattern": pattern
                })
        
        return {
            "issues": issues,
            "issue_count": len(issues),
            "scan_result": "clean" if len(issues) == 0 else "issues_found"
        }