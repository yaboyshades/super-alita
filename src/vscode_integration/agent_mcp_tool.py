#!/usr/bin/env python3
"""MCP Tool for Enhanced Super Alita Agent Integration

This MCP tool allows VS Code to interact with the enhanced agent cycle,
providing commands to start/stop the continuous cycle, get real-time status,
and trigger specific agent operations.
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from vscode_integration.enhanced_agent_cycle import EnhancedAgentCycle
    from vscode_integration.agent_integration import SuperAlitaAgent
except ImportError as e:
    print(f"Warning: {e}")
    EnhancedAgentCycle = None
    SuperAlitaAgent = None


class AgentCycleMCPTool:
    """MCP tool for enhanced agent cycle integration."""
    
    def __init__(self):
        self.enhanced_agent: Optional[EnhancedAgentCycle] = None
        self.base_agent: Optional[SuperAlitaAgent] = None
        self.cycle_task: Optional[asyncio.Task] = None
        self.workspace_folder = Path.cwd()
        
    async def initialize_agent(self) -> Dict[str, Any]:
        """Initialize the enhanced agent cycle."""
        try:
            if EnhancedAgentCycle is None or SuperAlitaAgent is None:
                return {
                    "success": False,
                    "error": "Agent classes not available - import failed",
                    "result": ""
                }
                
            print("🚀 Initializing Enhanced Agent Cycle via MCP...")
            
            # Initialize enhanced agent
            self.enhanced_agent = EnhancedAgentCycle(self.workspace_folder)
            success = await self.enhanced_agent.initialize()
            
            if not success:
                return {
                    "success": False,
                    "error": "Failed to initialize enhanced agent",
                    "result": ""
                }
                
            # Initialize base agent for commands
            self.base_agent = SuperAlitaAgent(self.workspace_folder)
            await self.base_agent.initialize()
            
            return {
                "success": True,
                "error": "",
                "result": "✅ Enhanced Agent Cycle initialized successfully via MCP"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "result": ""
            }
    
    async def start_continuous_cycle(self, interval_seconds: int = 30) -> Dict[str, Any]:
        """Start the continuous agent cycle."""
        try:
            if not self.enhanced_agent:
                init_result = await self.initialize_agent()
                if not init_result["success"]:
                    return init_result
                    
            if self.cycle_task and not self.cycle_task.done():
                return {
                    "success": False,
                    "error": "Continuous cycle is already running",
                    "result": ""
                }
                
            # Set cycle interval
            self.enhanced_agent.cycle_interval = interval_seconds
            
            # Start continuous cycle as background task
            self.cycle_task = asyncio.create_task(
                self.enhanced_agent.start_continuous_cycle()
            )
            
            return {
                "success": True,
                "error": "",
                "result": f"🔄 Started continuous agent cycle (interval: {interval_seconds}s)"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "result": ""
            }
    
    async def stop_continuous_cycle(self) -> Dict[str, Any]:
        """Stop the continuous agent cycle."""
        try:
            if not self.enhanced_agent:
                return {
                    "success": False,
                    "error": "Enhanced agent not initialized",
                    "result": ""
                }
                
            if self.cycle_task and not self.cycle_task.done():
                self.enhanced_agent.cycle_running = False
                self.cycle_task.cancel()
                
                try:
                    await self.cycle_task
                except asyncio.CancelledError:
                    pass
                    
                self.cycle_task = None
                
                return {
                    "success": True,
                    "error": "",
                    "result": "🛑 Stopped continuous agent cycle"
                }
            else:
                return {
                    "success": False,
                    "error": "No continuous cycle is running",
                    "result": ""
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "result": ""
            }
    
    async def execute_single_cycle(self) -> Dict[str, Any]:
        """Execute a single agent cycle."""
        try:
            if not self.enhanced_agent:
                init_result = await self.initialize_agent()
                if not init_result["success"]:
                    return init_result
                    
            print("🔄 Executing single agent cycle...")
            await self.enhanced_agent._execute_cycle()
            
            # Get cycle metrics
            metrics = {
                "cycle_count": self.enhanced_agent.cycle_count,
                "tasks_completed": self.enhanced_agent.tasks_completed_this_session,
                "recommendations_generated": self.enhanced_agent.recommendations_generated,
                "code_improvements": self.enhanced_agent.code_improvements_made,
                "last_cycle_time": self.enhanced_agent.last_cycle_time
            }
            
            return {
                "success": True,
                "error": "",
                "result": f"✅ Single cycle completed in {metrics['last_cycle_time']:.2f}s",
                "metrics": metrics
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "result": ""
            }
    
    async def get_agent_status(self) -> Dict[str, Any]:
        """Get comprehensive agent status."""
        try:
            if not self.base_agent:
                return {
                    "success": False,
                    "error": "Base agent not initialized",
                    "result": ""
                }
                
            # Get development status
            dev_status = await self.base_agent.get_development_status()
            
            # Get agent metrics if enhanced agent is available
            agent_metrics = {}
            if self.enhanced_agent:
                agent_metrics = {
                    "cycle_count": self.enhanced_agent.cycle_count,
                    "cycle_running": self.enhanced_agent.cycle_running,
                    "tasks_completed_session": self.enhanced_agent.tasks_completed_this_session,
                    "recommendations_generated": self.enhanced_agent.recommendations_generated,
                    "code_improvements_made": self.enhanced_agent.code_improvements_made,
                    "last_cycle_time": self.enhanced_agent.last_cycle_time,
                    "average_cycle_time": self.enhanced_agent.perf_monitor.average_cycle_time
                }
            
            status_report = {
                "development_status": dev_status,
                "agent_metrics": agent_metrics,
                "continuous_cycle_active": self.cycle_task is not None and not self.cycle_task.done()
            }
            
            return {
                "success": True,
                "error": "",
                "result": "📊 Agent status retrieved",
                "status": status_report
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "result": ""
            }
    
    async def create_development_task(
        self, 
        title: str, 
        description: str, 
        priority: str = "medium"
    ) -> Dict[str, Any]:
        """Create a new development task via the agent."""
        try:
            if not self.base_agent:
                init_result = await self.initialize_agent()
                if not init_result["success"]:
                    return init_result
                    
            task = await self.base_agent.create_development_task(
                title=title,
                description=description,
                priority=priority
            )
            
            return {
                "success": True,
                "error": "",
                "result": f"✅ Created task: {title}",
                "task": task
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "result": ""
            }
    
    async def complete_development_task(
        self, 
        task_id: str, 
        notes: str = ""
    ) -> Dict[str, Any]:
        """Complete a development task via the agent."""
        try:
            if not self.base_agent:
                return {
                    "success": False,
                    "error": "Base agent not initialized",
                    "result": ""
                }
                
            completed_task = await self.base_agent.complete_development_task(
                task_id=task_id,
                notes=notes
            )
            
            if completed_task:
                return {
                    "success": True,
                    "error": "",
                    "result": f"✅ Completed task: {completed_task['title']}",
                    "completed_task": completed_task
                }
            else:
                return {
                    "success": False,
                    "error": f"Task {task_id} not found",
                    "result": ""
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "result": ""
            }
    
    async def get_agent_recommendations(self) -> Dict[str, Any]:
        """Get intelligent recommendations from the agent."""
        try:
            if not self.base_agent:
                return {
                    "success": False,
                    "error": "Base agent not initialized",
                    "result": ""
                }
                
            recommendations = await self.base_agent.get_agent_recommendations()
            
            return {
                "success": True,
                "error": "",
                "result": f"💡 Generated {len(recommendations)} recommendations",
                "recommendations": recommendations
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "result": ""
            }
    
    async def trigger_code_analysis(self) -> Dict[str, Any]:
        """Trigger code analysis via the enhanced agent."""
        try:
            if not self.enhanced_agent:
                return {
                    "success": False,
                    "error": "Enhanced agent not initialized",
                    "result": ""
                }
                
            # Trigger code analysis
            await self.enhanced_agent._perform_code_analysis()
            
            return {
                "success": True,
                "error": "",
                "result": "🔍 Code analysis completed",
                "improvements_made": self.enhanced_agent.code_improvements_made
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "result": ""
            }
    
    async def generate_documentation(self) -> Dict[str, Any]:
        """Trigger documentation generation via the enhanced agent."""
        try:
            if not self.enhanced_agent:
                return {
                    "success": False,
                    "error": "Enhanced agent not initialized",
                    "result": ""
                }
                
            # Trigger documentation generation
            await self.enhanced_agent._generate_documentation()
            
            return {
                "success": True,
                "error": "",
                "result": "📚 Documentation generation completed"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "result": ""
            }
    
    async def shutdown_agent(self) -> Dict[str, Any]:
        """Shutdown the agent cycle."""
        try:
            # Stop continuous cycle if running
            if self.cycle_task and not self.cycle_task.done():
                await self.stop_continuous_cycle()
                
            # Shutdown agents
            if self.enhanced_agent:
                await self.enhanced_agent.shutdown()
                
            if self.base_agent:
                await self.base_agent.shutdown()
                
            return {
                "success": True,
                "error": "",
                "result": "✅ Agent cycle shutdown complete"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "result": ""
            }


# Global tool instance
agent_cycle_tool = AgentCycleMCPTool()


async def mcp_agent_initialize() -> Dict[str, Any]:
    """MCP command: Initialize the enhanced agent cycle."""
    return await agent_cycle_tool.initialize_agent()


async def mcp_agent_start_cycle(interval_seconds: int = 30) -> Dict[str, Any]:
    """MCP command: Start continuous agent cycle."""
    return await agent_cycle_tool.start_continuous_cycle(interval_seconds)


async def mcp_agent_stop_cycle() -> Dict[str, Any]:
    """MCP command: Stop continuous agent cycle."""
    return await agent_cycle_tool.stop_continuous_cycle()


async def mcp_agent_execute_cycle() -> Dict[str, Any]:
    """MCP command: Execute single agent cycle."""
    return await agent_cycle_tool.execute_single_cycle()


async def mcp_agent_status() -> Dict[str, Any]:
    """MCP command: Get agent status."""
    return await agent_cycle_tool.get_agent_status()


async def mcp_agent_create_task(
    title: str, 
    description: str, 
    priority: str = "medium"
) -> Dict[str, Any]:
    """MCP command: Create development task."""
    return await agent_cycle_tool.create_development_task(title, description, priority)


async def mcp_agent_complete_task(task_id: str, notes: str = "") -> Dict[str, Any]:
    """MCP command: Complete development task."""
    return await agent_cycle_tool.complete_development_task(task_id, notes)


async def mcp_agent_recommendations() -> Dict[str, Any]:
    """MCP command: Get agent recommendations."""
    return await agent_cycle_tool.get_agent_recommendations()


async def mcp_agent_analyze_code() -> Dict[str, Any]:
    """MCP command: Trigger code analysis."""
    return await agent_cycle_tool.trigger_code_analysis()


async def mcp_agent_generate_docs() -> Dict[str, Any]:
    """MCP command: Generate documentation."""
    return await agent_cycle_tool.generate_documentation()


async def mcp_agent_shutdown() -> Dict[str, Any]:
    """MCP command: Shutdown agent."""
    return await agent_cycle_tool.shutdown_agent()


async def demo_mcp_integration():
    """Demonstrate the MCP integration with enhanced agent cycle."""
    print("🔧 Enhanced Agent MCP Integration Demo")
    print("=" * 50)
    
    # Initialize agent
    print("\n1. Initializing agent...")
    result = await mcp_agent_initialize()
    print(f"   Result: {result['result']}")
    
    # Get status
    print("\n2. Getting agent status...")
    result = await mcp_agent_status()
    if result["success"]:
        status = result["status"]
        dev_status = status["development_status"]
        print(f"   📁 Workspace: {dev_status['workspace']}")
        print(f"   📋 Tasks: {dev_status['task_summary']['total']} total")
        print(f"   🎯 Completion: {dev_status['completion_rate']:.1%}")
    
    # Execute single cycle
    print("\n3. Executing single cycle...")
    result = await mcp_agent_execute_cycle()
    print(f"   Result: {result['result']}")
    if "metrics" in result:
        metrics = result["metrics"]
        print(f"   📊 Metrics: {metrics['cycle_count']} cycles, {metrics['tasks_completed']} tasks completed")
    
    # Get recommendations
    print("\n4. Getting recommendations...")
    result = await mcp_agent_recommendations()
    if result["success"]:
        recommendations = result["recommendations"]
        print(f"   💡 Found {len(recommendations)} recommendations:")
        for i, rec in enumerate(recommendations[:3], 1):
            print(f"      {i}. {rec}")
    
    # Create a test task
    print("\n5. Creating test task...")
    result = await mcp_agent_create_task(
        title="MCP Integration Test",
        description="Test task created via MCP integration",
        priority="medium"
    )
    print(f"   Result: {result['result']}")
    
    # Trigger code analysis
    print("\n6. Triggering code analysis...")
    result = await mcp_agent_analyze_code()
    print(f"   Result: {result['result']}")
    
    # Final status
    print("\n7. Final status check...")
    result = await mcp_agent_status()
    if result["success"]:
        status = result["status"]
        dev_status = status["development_status"]
        print(f"   📋 Final task count: {dev_status['task_summary']['total']}")
    
    # Shutdown
    print("\n8. Shutting down...")
    result = await mcp_agent_shutdown()
    print(f"   Result: {result['result']}")
    
    print("\n🎉 MCP Integration Demo Complete!")
    return True


if __name__ == "__main__":
    # Run demo
    asyncio.run(demo_mcp_integration())