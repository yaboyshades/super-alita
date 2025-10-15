"""Core agent abilities implementation."""

import logging
from typing import Dict, Any, List
from dataclasses import dataclass
from .adapters.github_adapter import GitHubAdapter
from .adapters.code_adapter import CodeAdapter

@dataclass
class AbilityMetadata:
    name: str
    description: str
    risk_level: str  # low, medium, high
    requires_auth: bool
    cost_estimate: float
    timeout_seconds: int

class CoreAbility:
    """Base class for core agent abilities."""
    
    def __init__(self, metadata: AbilityMetadata, adapter=None):
        self.metadata = metadata
        self.adapter = adapter
        self.logger = logging.getLogger(__name__)
    
    async def execute(self, parameters: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the ability with given parameters."""
        if self.adapter:
            return await self.adapter.execute(self.metadata.name, parameters, context)
        else:
            return await self.default_execution(parameters, context)
    
    async def default_execution(self, parameters: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Default execution when no adapter is available."""
        return {
            "status": "executed",
            "ability": self.metadata.name,
            "parameters": parameters,
            "message": f"Executed {self.metadata.name} successfully"
        }
    
    async def dry_run(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Perform a dry run without executing."""
        return {
            "dry_run": True,
            "ability": self.metadata.name,
            "parameters": parameters,
            "estimated_cost": self.metadata.cost_estimate,
            "risk_level": self.metadata.risk_level
        }
    
    async def validate_parameters(self, parameters: Dict[str, Any]) -> List[str]:
        """Validate parameters and return list of errors."""
        errors = []
        
        # Basic validation - override in subclasses
        if not isinstance(parameters, dict):
            errors.append("Parameters must be a dictionary")
        
        return errors

class GitHubAbility(CoreAbility):
    """GitHub integration ability."""
    
    def __init__(self):
        metadata = AbilityMetadata(
            name="github_integration",
            description="GitHub repository operations and integration",
            risk_level="medium",
            requires_auth=True,
            cost_estimate=0.01,
            timeout_seconds=30
        )
        adapter = GitHubAdapter()
        super().__init__(metadata, adapter)

class CodeGenerationAbility(CoreAbility):
    """Code generation ability."""
    
    def __init__(self):
        metadata = AbilityMetadata(
            name="code_generation",
            description="Generate and analyze code based on specifications",
            risk_level="medium",
            requires_auth=False,
            cost_estimate=0.05,
            timeout_seconds=60
        )
        adapter = CodeAdapter()
        super().__init__(metadata, adapter)

class DataAnalysisAbility(CoreAbility):
    """Data analysis ability."""
    
    def __init__(self):
        metadata = AbilityMetadata(
            name="data_analysis",
            description="Analyze data and generate insights",
            risk_level="low",
            requires_auth=False,
            cost_estimate=0.02,
            timeout_seconds=45
        )
        super().__init__(metadata)
    
    async def default_execution(self, parameters: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Basic data analysis execution."""
        data = parameters.get("data", [])
        analysis_type = parameters.get("analysis_type", "summary")
        
        if analysis_type == "summary" and isinstance(data, list):
            return {
                "analysis_type": "summary",
                "data_points": len(data),
                "summary": f"Analyzed {len(data)} data points",
                "status": "completed"
            }
        
        return {
            "analysis_type": analysis_type,
            "status": "completed",
            "message": "Basic data analysis performed"
        }

# Factory functions
def create_core_abilities() -> List[CoreAbility]:
    """Create all core abilities."""
    return [
        GitHubAbility(),
        CodeGenerationAbility(),
        DataAnalysisAbility()
    ]