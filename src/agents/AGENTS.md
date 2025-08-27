# Agent Implementations - Agent Instructions

## Overview
The `src/agents/` directory contains specific agent implementations and behaviors:
- **Agent Classes** - Concrete agent implementations with specific capabilities
- **Agent Behaviors** - Behavioral patterns and decision-making logic
- **Agent Communication** - Inter-agent communication protocols
- **Agent Lifecycle** - Agent creation, management, and termination

## Key Files & Responsibilities

### Agent Components
- Agent implementation classes
- Behavior definition modules
- Communication protocol handlers
- Lifecycle management utilities

## Development Guidelines

### Agent Base Classes
```python
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import uuid
from datetime import datetime, timezone

class AgentState(Enum):
    CREATED = "created"
    INITIALIZING = "initializing"
    ACTIVE = "active"
    PAUSED = "paused"
    TERMINATING = "terminating"
    TERMINATED = "terminated"
    ERROR = "error"

class AgentPriority(Enum):
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4

@dataclass
class AgentCapability:
    """Represents an agent capability"""
    name: str
    description: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    version: str = "1.0.0"

@dataclass
class AgentMessage:
    """Message structure for agent communication"""
    id: str
    sender_id: str
    recipient_id: str
    message_type: str
    content: Dict[str, Any]
    timestamp: datetime
    priority: AgentPriority = AgentPriority.NORMAL
    reply_to: Optional[str] = None
    
    def __post_init__(self):
        if isinstance(self.timestamp, str):
            self.timestamp = datetime.fromisoformat(self.timestamp)

@dataclass
class AgentProfile:
    """Agent profile and configuration"""
    agent_id: str
    agent_type: str
    name: str
    description: str
    capabilities: List[AgentCapability] = field(default_factory=list)
    configuration: Dict[str, Any] = field(default_factory=dict)
    tags: Set[str] = field(default_factory=set)
    
class BaseAgent(ABC):
    """Base class for all agents"""
    
    def __init__(self, profile: AgentProfile, event_bus=None):
        self.profile = profile
        self.event_bus = event_bus
        self.state = AgentState.CREATED
        self.message_queue: List[AgentMessage] = []
        self.subscriptions: Set[str] = set()
        self.last_activity = datetime.now(timezone.utc)
        self.metrics: Dict[str, Any] = {
            "messages_sent": 0,
            "messages_received": 0,
            "tasks_completed": 0,
            "errors": 0,
            "uptime_start": datetime.now(timezone.utc)
        }
        
    @property
    def agent_id(self) -> str:
        return self.profile.agent_id
        
    @property
    def agent_type(self) -> str:
        return self.profile.agent_type
        
    async def initialize(self) -> bool:
        """Initialize the agent"""
        try:
            self.state = AgentState.INITIALIZING
            
            # Initialize capabilities
            for capability in self.profile.capabilities:
                await self._initialize_capability(capability)
                
            # Subscribe to relevant events
            await self._setup_subscriptions()
            
            # Run agent-specific initialization
            await self._agent_initialize()
            
            self.state = AgentState.ACTIVE
            self.last_activity = datetime.now(timezone.utc)
            
            return True
            
        except Exception as e:
            self.state = AgentState.ERROR
            self.metrics["errors"] += 1
            logger.error(f"Agent {self.agent_id} initialization failed: {e}")
            return False
            
    async def shutdown(self):
        """Shutdown the agent"""
        try:
            self.state = AgentState.TERMINATING
            
            # Process remaining messages
            await self._process_remaining_messages()
            
            # Run agent-specific cleanup
            await self._agent_shutdown()
            
            # Unsubscribe from events
            await self._cleanup_subscriptions()
            
            self.state = AgentState.TERMINATED
            
        except Exception as e:
            self.state = AgentState.ERROR
            logger.error(f"Agent {self.agent_id} shutdown failed: {e}")
            
    async def send_message(self, recipient_id: str, message_type: str, content: Dict[str, Any]) -> bool:
        """Send message to another agent"""
        try:
            message = AgentMessage(
                id=str(uuid.uuid4()),
                sender_id=self.agent_id,
                recipient_id=recipient_id,
                message_type=message_type,
                content=content,
                timestamp=datetime.now(timezone.utc)
            )
            
            # Send via event bus
            if self.event_bus:
                event = create_event(
                    "agent_message",
                    message=message.__dict__,
                    source=self.agent_id
                )
                await self.event_bus.publish(event)
                
            self.metrics["messages_sent"] += 1
            self.last_activity = datetime.now(timezone.utc)
            
            return True
            
        except Exception as e:
            self.metrics["errors"] += 1
            logger.error(f"Agent {self.agent_id} message send failed: {e}")
            return False
            
    async def receive_message(self, message: AgentMessage):
        """Receive message from another agent"""
        try:
            self.message_queue.append(message)
            self.metrics["messages_received"] += 1
            self.last_activity = datetime.now(timezone.utc)
            
            # Process message immediately if active
            if self.state == AgentState.ACTIVE:
                await self._process_message(message)
                
        except Exception as e:
            self.metrics["errors"] += 1
            logger.error(f"Agent {self.agent_id} message receive failed: {e}")
            
    @abstractmethod
    async def _agent_initialize(self):
        """Agent-specific initialization logic"""
        pass
        
    @abstractmethod
    async def _agent_shutdown(self):
        """Agent-specific shutdown logic"""
        pass
        
    @abstractmethod
    async def _process_message(self, message: AgentMessage):
        """Process received message"""
        pass
        
    async def _initialize_capability(self, capability: AgentCapability):
        """Initialize a specific capability"""
        # Default implementation - override in subclasses
        pass
        
    async def _setup_subscriptions(self):
        """Setup event subscriptions"""
        if self.event_bus:
            await self.event_bus.subscribe("agent_message", self._handle_agent_message)
            
    async def _cleanup_subscriptions(self):
        """Cleanup event subscriptions"""
        if self.event_bus:
            await self.event_bus.unsubscribe("agent_message", self._handle_agent_message)
            
    async def _handle_agent_message(self, event: Dict[str, Any]):
        """Handle agent message event"""
        message_data = event.get("data", {}).get("message", {})
        
        # Check if message is for this agent
        if message_data.get("recipient_id") == self.agent_id:
            message = AgentMessage(**message_data)
            await self.receive_message(message)
            
    async def _process_remaining_messages(self):
        """Process any remaining messages in queue"""
        while self.message_queue:
            message = self.message_queue.pop(0)
            await self._process_message(message)
            
    def get_metrics(self) -> Dict[str, Any]:
        """Get agent metrics"""
        current_time = datetime.now(timezone.utc)
        uptime = current_time - self.metrics["uptime_start"]
        
        return {
            **self.metrics,
            "uptime_seconds": uptime.total_seconds(),
            "state": self.state.value,
            "queue_size": len(self.message_queue),
            "last_activity": self.last_activity.isoformat()
        }
```

### Specialized Agent Types
```python
class TaskAgent(BaseAgent):
    """Agent specialized for task execution"""
    
    def __init__(self, profile: AgentProfile, event_bus=None):
        super().__init__(profile, event_bus)
        self.active_tasks: Dict[str, Dict[str, Any]] = {}
        self.task_history: List[Dict[str, Any]] = []
        
    async def _agent_initialize(self):
        """Initialize task agent"""
        # Setup task-specific capabilities
        await self._setup_task_capabilities()
        
    async def _agent_shutdown(self):
        """Shutdown task agent"""
        # Complete or cancel active tasks
        for task_id in list(self.active_tasks.keys()):
            await self._cancel_task(task_id)
            
    async def _process_message(self, message: AgentMessage):
        """Process task-related messages"""
        if message.message_type == "task_request":
            await self._handle_task_request(message)
        elif message.message_type == "task_cancel":
            await self._handle_task_cancel(message)
        elif message.message_type == "task_status":
            await self._handle_task_status(message)
        else:
            logger.warning(f"Unknown message type: {message.message_type}")
            
    async def _handle_task_request(self, message: AgentMessage):
        """Handle task execution request"""
        task_data = message.content
        task_id = task_data.get("task_id", str(uuid.uuid4()))
        
        # Validate task
        if not self._can_execute_task(task_data):
            await self._send_task_response(
                message.sender_id, task_id, "rejected", 
                "Task not compatible with agent capabilities"
            )
            return
            
        # Start task execution
        self.active_tasks[task_id] = {
            "task_data": task_data,
            "requester": message.sender_id,
            "start_time": datetime.now(timezone.utc),
            "status": "running"
        }
        
        try:
            result = await self._execute_task(task_data)
            
            # Task completed successfully
            self.active_tasks[task_id]["status"] = "completed"
            self.active_tasks[task_id]["result"] = result
            
            await self._send_task_response(
                message.sender_id, task_id, "completed", result
            )
            
            self.metrics["tasks_completed"] += 1
            
        except Exception as e:
            # Task failed
            self.active_tasks[task_id]["status"] = "failed"
            self.active_tasks[task_id]["error"] = str(e)
            
            await self._send_task_response(
                message.sender_id, task_id, "failed", str(e)
            )
            
            self.metrics["errors"] += 1
            
        finally:
            # Move to history
            self.task_history.append(self.active_tasks.pop(task_id))
            
    async def _execute_task(self, task_data: Dict[str, Any]) -> Any:
        """Execute the actual task - to be implemented by subclasses"""
        task_type = task_data.get("type")
        
        if task_type == "computation":
            return await self._execute_computation_task(task_data)
        elif task_type == "data_processing":
            return await self._execute_data_processing_task(task_data)
        else:
            raise ValueError(f"Unknown task type: {task_type}")
            
    async def _send_task_response(self, requester_id: str, task_id: str, status: str, result: Any):
        """Send task completion response"""
        await self.send_message(
            requester_id,
            "task_response",
            {
                "task_id": task_id,
                "status": status,
                "result": result,
                "agent_id": self.agent_id
            }
        )

class CoordinatorAgent(BaseAgent):
    """Agent specialized for coordinating other agents"""
    
    def __init__(self, profile: AgentProfile, event_bus=None):
        super().__init__(profile, event_bus)
        self.managed_agents: Dict[str, Dict[str, Any]] = {}
        self.coordination_tasks: Dict[str, Dict[str, Any]] = {}
        
    async def _agent_initialize(self):
        """Initialize coordinator agent"""
        await self._discover_agents()
        
    async def _agent_shutdown(self):
        """Shutdown coordinator agent"""
        # Stop coordination tasks
        for task_id in list(self.coordination_tasks.keys()):
            await self._stop_coordination_task(task_id)
            
    async def _process_message(self, message: AgentMessage):
        """Process coordination messages"""
        if message.message_type == "agent_registration":
            await self._handle_agent_registration(message)
        elif message.message_type == "coordination_request":
            await self._handle_coordination_request(message)
        elif message.message_type == "agent_status":
            await self._handle_agent_status(message)
        else:
            logger.warning(f"Unknown message type: {message.message_type}")
            
    async def _discover_agents(self):
        """Discover available agents"""
        if self.event_bus:
            # Request agent discovery
            event = create_event(
                "agent_discovery_request",
                coordinator_id=self.agent_id,
                source=self.agent_id
            )
            await self.event_bus.publish(event)
            
    async def _coordinate_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Coordinate multi-agent task"""
        task_id = str(uuid.uuid4())
        
        # Break down task into subtasks
        subtasks = await self._decompose_task(task_data)
        
        # Assign subtasks to appropriate agents
        assignments = {}
        for subtask in subtasks:
            agent_id = await self._select_agent_for_task(subtask)
            if agent_id:
                assignments[subtask["id"]] = agent_id
                
        # Execute coordination
        results = {}
        for subtask_id, agent_id in assignments.items():
            subtask = next(st for st in subtasks if st["id"] == subtask_id)
            
            # Send task to agent
            await self.send_message(
                agent_id,
                "task_request",
                subtask
            )
            
        return {"task_id": task_id, "assignments": assignments}

class LearningAgent(BaseAgent):
    """Agent with learning capabilities"""
    
    def __init__(self, profile: AgentProfile, event_bus=None):
        super().__init__(profile, event_bus)
        self.knowledge_base: Dict[str, Any] = {}
        self.learning_data: List[Dict[str, Any]] = []
        self.model_version = "1.0.0"
        
    async def _agent_initialize(self):
        """Initialize learning agent"""
        await self._load_knowledge_base()
        await self._initialize_learning_model()
        
    async def _agent_shutdown(self):
        """Shutdown learning agent"""
        await self._save_knowledge_base()
        await self._save_learning_data()
        
    async def _process_message(self, message: AgentMessage):
        """Process learning-related messages"""
        if message.message_type == "learning_data":
            await self._handle_learning_data(message)
        elif message.message_type == "knowledge_query":
            await self._handle_knowledge_query(message)
        elif message.message_type == "model_update":
            await self._handle_model_update(message)
        else:
            logger.warning(f"Unknown message type: {message.message_type}")
            
    async def _handle_learning_data(self, message: AgentMessage):
        """Handle new learning data"""
        learning_data = message.content
        
        # Validate and store learning data
        if self._validate_learning_data(learning_data):
            self.learning_data.append({
                "data": learning_data,
                "timestamp": datetime.now(timezone.utc),
                "source": message.sender_id
            })
            
            # Trigger learning if enough data accumulated
            if len(self.learning_data) >= 100:  # Configurable threshold
                await self._trigger_learning()
                
    async def _trigger_learning(self):
        """Trigger learning process"""
        try:
            # Extract patterns from learning data
            patterns = await self._extract_patterns(self.learning_data)
            
            # Update knowledge base
            await self._update_knowledge_base(patterns)
            
            # Clear processed learning data
            self.learning_data.clear()
            
            logger.info(f"Learning agent {self.agent_id} completed learning cycle")
            
        except Exception as e:
            self.metrics["errors"] += 1
            logger.error(f"Learning process failed: {e}")
```

### Agent Communication Protocols
```python
class AgentCommunicationProtocol:
    """Protocol for agent-to-agent communication"""
    
    def __init__(self, event_bus):
        self.event_bus = event_bus
        self.message_handlers: Dict[str, List[Callable]] = {}
        
    def register_handler(self, message_type: str, handler: Callable):
        """Register message handler"""
        if message_type not in self.message_handlers:
            self.message_handlers[message_type] = []
        self.message_handlers[message_type].append(handler)
        
    async def broadcast_message(self, message_type: str, content: Dict[str, Any], sender_id: str):
        """Broadcast message to all agents"""
        event = create_event(
            "agent_broadcast",
            message_type=message_type,
            content=content,
            sender_id=sender_id,
            source=sender_id
        )
        await self.event_bus.publish(event)
        
    async def send_direct_message(
        self, 
        sender_id: str, 
        recipient_id: str, 
        message_type: str, 
        content: Dict[str, Any]
    ):
        """Send direct message between agents"""
        message = AgentMessage(
            id=str(uuid.uuid4()),
            sender_id=sender_id,
            recipient_id=recipient_id,
            message_type=message_type,
            content=content,
            timestamp=datetime.now(timezone.utc)
        )
        
        event = create_event(
            "agent_direct_message",
            message=message.__dict__,
            source=sender_id
        )
        await self.event_bus.publish(event)
        
    async def handle_message(self, message: AgentMessage):
        """Handle incoming message"""
        handlers = self.message_handlers.get(message.message_type, [])
        
        for handler in handlers:
            try:
                await handler(message)
            except Exception as e:
                logger.error(f"Message handler failed: {e}")

class AgentDiscoveryService:
    """Service for agent discovery and registration"""
    
    def __init__(self, event_bus):
        self.event_bus = event_bus
        self.agent_registry: Dict[str, AgentProfile] = {}
        
    async def register_agent(self, agent: BaseAgent):
        """Register agent in discovery service"""
        self.agent_registry[agent.agent_id] = agent.profile
        
        # Announce agent registration
        event = create_event(
            "agent_registered",
            agent_profile=agent.profile.__dict__,
            source="discovery_service"
        )
        await self.event_bus.publish(event)
        
    async def unregister_agent(self, agent_id: str):
        """Unregister agent from discovery service"""
        if agent_id in self.agent_registry:
            profile = self.agent_registry.pop(agent_id)
            
            # Announce agent unregistration
            event = create_event(
                "agent_unregistered",
                agent_id=agent_id,
                agent_profile=profile.__dict__,
                source="discovery_service"
            )
            await self.event_bus.publish(event)
            
    def find_agents_by_capability(self, capability_name: str) -> List[AgentProfile]:
        """Find agents with specific capability"""
        matching_agents = []
        
        for profile in self.agent_registry.values():
            for capability in profile.capabilities:
                if capability.name == capability_name and capability.enabled:
                    matching_agents.append(profile)
                    break
                    
        return matching_agents
        
    def find_agents_by_type(self, agent_type: str) -> List[AgentProfile]:
        """Find agents of specific type"""
        return [
            profile for profile in self.agent_registry.values()
            if profile.agent_type == agent_type
        ]
        
    def get_all_agents(self) -> List[AgentProfile]:
        """Get all registered agents"""
        return list(self.agent_registry.values())
```

### Agent Lifecycle Management
```python
class AgentManager:
    """Manages agent lifecycle and coordination"""
    
    def __init__(self, event_bus):
        self.event_bus = event_bus
        self.agents: Dict[str, BaseAgent] = {}
        self.discovery_service = AgentDiscoveryService(event_bus)
        self.communication_protocol = AgentCommunicationProtocol(event_bus)
        
    async def create_agent(self, agent_class: type, profile: AgentProfile) -> BaseAgent:
        """Create and initialize new agent"""
        try:
            # Create agent instance
            agent = agent_class(profile, self.event_bus)
            
            # Initialize agent
            success = await agent.initialize()
            
            if success:
                # Register agent
                self.agents[agent.agent_id] = agent
                await self.discovery_service.register_agent(agent)
                
                logger.info(f"Agent {agent.agent_id} created and initialized")
                return agent
            else:
                logger.error(f"Agent {profile.agent_id} initialization failed")
                return None
                
        except Exception as e:
            logger.error(f"Agent creation failed: {e}")
            return None
            
    async def shutdown_agent(self, agent_id: str):
        """Shutdown and remove agent"""
        if agent_id in self.agents:
            agent = self.agents[agent_id]
            
            try:
                # Shutdown agent
                await agent.shutdown()
                
                # Unregister agent
                await self.discovery_service.unregister_agent(agent_id)
                
                # Remove from manager
                del self.agents[agent_id]
                
                logger.info(f"Agent {agent_id} shutdown completed")
                
            except Exception as e:
                logger.error(f"Agent {agent_id} shutdown failed: {e}")
                
    async def shutdown_all_agents(self):
        """Shutdown all managed agents"""
        shutdown_tasks = []
        
        for agent_id in list(self.agents.keys()):
            shutdown_tasks.append(self.shutdown_agent(agent_id))
            
        await asyncio.gather(*shutdown_tasks, return_exceptions=True)
        
    def get_agent(self, agent_id: str) -> Optional[BaseAgent]:
        """Get agent by ID"""
        return self.agents.get(agent_id)
        
    def get_agents_by_type(self, agent_type: str) -> List[BaseAgent]:
        """Get agents by type"""
        return [
            agent for agent in self.agents.values()
            if agent.agent_type == agent_type
        ]
        
    async def coordinate_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Coordinate multi-agent task"""
        # Find coordinator agents
        coordinators = self.get_agents_by_type("coordinator")
        
        if not coordinators:
            raise ValueError("No coordinator agents available")
            
        # Select coordinator (simple selection - could be enhanced)
        coordinator = coordinators[0]
        
        # Send coordination request
        await coordinator.send_message(
            coordinator.agent_id,  # Self-message for coordination
            "coordination_request",
            task_data
        )
        
        return {"coordinator_id": coordinator.agent_id, "task_data": task_data}
        
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get system-wide agent metrics"""
        total_agents = len(self.agents)
        active_agents = sum(1 for agent in self.agents.values() if agent.state == AgentState.ACTIVE)
        
        agent_metrics = {}
        for agent_id, agent in self.agents.items():
            agent_metrics[agent_id] = agent.get_metrics()
            
        return {
            "total_agents": total_agents,
            "active_agents": active_agents,
            "agent_details": agent_metrics,
            "system_uptime": datetime.now(timezone.utc).isoformat()
        }
```

## Testing Guidelines

### Agent Testing Framework
```python
import pytest
from unittest.mock import AsyncMock, MagicMock
from src.agents.base_agent import BaseAgent, AgentProfile, AgentCapability

class TestAgent(BaseAgent):
    """Test agent for testing purposes"""
    
    def __init__(self, profile: AgentProfile, event_bus=None):
        super().__init__(profile, event_bus)
        self.processed_messages = []
        
    async def _agent_initialize(self):
        self.initialization_completed = True
        
    async def _agent_shutdown(self):
        self.shutdown_completed = True
        
    async def _process_message(self, message):
        self.processed_messages.append(message)

@pytest.fixture
def test_agent_profile():
    """Test agent profile fixture"""
    capabilities = [
        AgentCapability(
            name="test_capability",
            description="Test capability for testing"
        )
    ]
    
    return AgentProfile(
        agent_id="test_agent_001",
        agent_type="test",
        name="Test Agent",
        description="Agent for testing purposes",
        capabilities=capabilities
    )

@pytest.mark.asyncio
async def test_agent_initialization(test_agent_profile):
    """Test agent initialization"""
    event_bus = AsyncMock()
    agent = TestAgent(test_agent_profile, event_bus)
    
    # Test initialization
    success = await agent.initialize()
    
    assert success is True
    assert agent.state == AgentState.ACTIVE
    assert hasattr(agent, 'initialization_completed')
    assert agent.initialization_completed is True

@pytest.mark.asyncio
async def test_agent_message_handling(test_agent_profile):
    """Test agent message handling"""
    event_bus = AsyncMock()
    agent = TestAgent(test_agent_profile, event_bus)
    
    await agent.initialize()
    
    # Create test message
    message = AgentMessage(
        id="msg_001",
        sender_id="sender_agent",
        recipient_id=agent.agent_id,
        message_type="test_message",
        content={"data": "test"},
        timestamp=datetime.now(timezone.utc)
    )
    
    # Send message to agent
    await agent.receive_message(message)
    
    # Verify message was processed
    assert len(agent.processed_messages) == 1
    assert agent.processed_messages[0].id == "msg_001"
    assert agent.metrics["messages_received"] == 1

@pytest.mark.asyncio
async def test_agent_shutdown(test_agent_profile):
    """Test agent shutdown"""
    event_bus = AsyncMock()
    agent = TestAgent(test_agent_profile, event_bus)
    
    await agent.initialize()
    await agent.shutdown()
    
    assert agent.state == AgentState.TERMINATED
    assert hasattr(agent, 'shutdown_completed')
    assert agent.shutdown_completed is True

@pytest.mark.asyncio
async def test_agent_manager():
    """Test agent manager functionality"""
    event_bus = AsyncMock()
    manager = AgentManager(event_bus)
    
    # Create test profile
    profile = AgentProfile(
        agent_id="managed_agent_001",
        agent_type="test",
        name="Managed Test Agent",
        description="Test agent for manager testing"
    )
    
    # Create agent through manager
    agent = await manager.create_agent(TestAgent, profile)
    
    assert agent is not None
    assert agent.agent_id in manager.agents
    assert agent.state == AgentState.ACTIVE
    
    # Test agent retrieval
    retrieved_agent = manager.get_agent(agent.agent_id)
    assert retrieved_agent == agent
    
    # Test agent shutdown
    await manager.shutdown_agent(agent.agent_id)
    assert agent.agent_id not in manager.agents
    assert agent.state == AgentState.TERMINATED
```

### Integration Testing
```python
@pytest.mark.integration
@pytest.mark.asyncio
async def test_multi_agent_communication():
    """Test communication between multiple agents"""
    event_bus = AsyncMock()
    manager = AgentManager(event_bus)
    
    # Create multiple agents
    agent1_profile = AgentProfile(
        agent_id="agent_001",
        agent_type="test",
        name="Agent 1",
        description="First test agent"
    )
    
    agent2_profile = AgentProfile(
        agent_id="agent_002", 
        agent_type="test",
        name="Agent 2",
        description="Second test agent"
    )
    
    agent1 = await manager.create_agent(TestAgent, agent1_profile)
    agent2 = await manager.create_agent(TestAgent, agent2_profile)
    
    # Test message sending
    await agent1.send_message(
        agent2.agent_id,
        "test_communication",
        {"message": "Hello from agent1"}
    )
    
    # Verify message was sent
    assert agent1.metrics["messages_sent"] == 1
    
    # Cleanup
    await manager.shutdown_all_agents()

@pytest.mark.performance
async def test_agent_performance():
    """Test agent performance under load"""
    event_bus = AsyncMock()
    manager = AgentManager(event_bus)
    
    # Create multiple agents
    agents = []
    for i in range(10):
        profile = AgentProfile(
            agent_id=f"perf_agent_{i:03d}",
            agent_type="test",
            name=f"Performance Agent {i}",
            description=f"Performance test agent {i}"
        )
        agent = await manager.create_agent(TestAgent, profile)
        agents.append(agent)
    
    # Send many messages
    start_time = time.time()
    
    for i in range(100):
        sender = agents[i % len(agents)]
        recipient = agents[(i + 1) % len(agents)]
        
        await sender.send_message(
            recipient.agent_id,
            "performance_test",
            {"iteration": i}
        )
    
    processing_time = time.time() - start_time
    
    # Should handle 100 messages quickly
    assert processing_time < 5.0  # Less than 5 seconds
    
    # Cleanup
    await manager.shutdown_all_agents()
```

## Security Guidelines

### Agent Security Framework
```python
class AgentSecurityManager:
    """Security management for agents"""
    
    def __init__(self):
        self.agent_permissions: Dict[str, Set[str]] = {}
        self.security_policies: Dict[str, Dict[str, Any]] = {}
        
    def set_agent_permissions(self, agent_id: str, permissions: Set[str]):
        """Set permissions for agent"""
        self.agent_permissions[agent_id] = permissions
        
    def check_permission(self, agent_id: str, permission: str) -> bool:
        """Check if agent has specific permission"""
        agent_perms = self.agent_permissions.get(agent_id, set())
        return permission in agent_perms
        
    def validate_message(self, sender_id: str, recipient_id: str, message_type: str) -> bool:
        """Validate message based on security policies"""
        # Check if sender has permission to send this message type
        required_permission = f"send_{message_type}"
        if not self.check_permission(sender_id, required_permission):
            return False
            
        # Check if recipient can receive this message type
        required_permission = f"receive_{message_type}"
        if not self.check_permission(recipient_id, required_permission):
            return False
            
        return True
        
    def sanitize_message_content(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize message content"""
        # Remove potentially dangerous fields
        dangerous_keys = ['__class__', 'eval', 'exec', 'import']
        
        sanitized = {}
        for key, value in content.items():
            if key not in dangerous_keys:
                if isinstance(value, dict):
                    sanitized[key] = self.sanitize_message_content(value)
                elif isinstance(value, str):
                    # Basic string sanitization
                    sanitized[key] = value[:1000]  # Limit length
                else:
                    sanitized[key] = value
                    
        return sanitized

class SecureAgent(BaseAgent):
    """Agent with built-in security features"""
    
    def __init__(self, profile: AgentProfile, event_bus=None, security_manager=None):
        super().__init__(profile, event_bus)
        self.security_manager = security_manager or AgentSecurityManager()
        
    async def send_message(self, recipient_id: str, message_type: str, content: Dict[str, Any]) -> bool:
        """Send message with security validation"""
        # Validate permission
        if not self.security_manager.validate_message(self.agent_id, recipient_id, message_type):
            logger.warning(f"Message blocked: {self.agent_id} -> {recipient_id} ({message_type})")
            return False
            
        # Sanitize content
        sanitized_content = self.security_manager.sanitize_message_content(content)
        
        return await super().send_message(recipient_id, message_type, sanitized_content)
```

## Performance Guidelines

### Agent Optimization
```python
class PerformantAgent(BaseAgent):
    """High-performance agent implementation"""
    
    def __init__(self, profile: AgentProfile, event_bus=None):
        super().__init__(profile, event_bus)
        self.message_processor_task = None
        self.batch_size = 10
        
    async def _agent_initialize(self):
        """Initialize with performance optimizations"""
        # Start background message processor
        self.message_processor_task = asyncio.create_task(
            self._background_message_processor()
        )
        
    async def _agent_shutdown(self):
        """Shutdown with cleanup"""
        if self.message_processor_task:
            self.message_processor_task.cancel()
            try:
                await self.message_processor_task
            except asyncio.CancelledError:
                pass
                
    async def _background_message_processor(self):
        """Process messages in background batches"""
        while self.state == AgentState.ACTIVE:
            try:
                # Process messages in batches
                batch = []
                while len(batch) < self.batch_size and self.message_queue:
                    batch.append(self.message_queue.pop(0))
                    
                if batch:
                    await self._process_message_batch(batch)
                else:
                    await asyncio.sleep(0.1)  # Brief pause if no messages
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Background processor error: {e}")
                await asyncio.sleep(1)  # Error recovery pause
                
    async def _process_message_batch(self, messages: List[AgentMessage]):
        """Process batch of messages efficiently"""
        # Group messages by type for efficient processing
        message_groups = {}
        for message in messages:
            if message.message_type not in message_groups:
                message_groups[message.message_type] = []
            message_groups[message.message_type].append(message)
            
        # Process each group
        for message_type, message_group in message_groups.items():
            await self._process_message_group(message_type, message_group)
            
    async def _process_message_group(self, message_type: str, messages: List[AgentMessage]):
        """Process group of messages of same type"""
        # Default implementation - process individually
        for message in messages:
            await self._process_message(message)
```

## Common Patterns

### Agent Behavior Templates
```python
class BehaviorTemplate:
    """Template for agent behaviors"""
    
    @staticmethod
    def reactive_behavior():
        """Reactive behavior pattern"""
        async def behavior(agent: BaseAgent, stimulus: Dict[str, Any]):
            # React to stimulus immediately
            response = await agent._generate_response(stimulus)
            await agent._execute_response(response)
            
        return behavior
        
    @staticmethod  
    def proactive_behavior(interval: int = 60):
        """Proactive behavior pattern"""
        async def behavior(agent: BaseAgent):
            while agent.state == AgentState.ACTIVE:
                # Take proactive action
                action = await agent._plan_action()
                if action:
                    await agent._execute_action(action)
                    
                await asyncio.sleep(interval)
                
        return behavior
        
    @staticmethod
    def collaborative_behavior():
        """Collaborative behavior pattern"""
        async def behavior(agent: BaseAgent, task: Dict[str, Any]):
            # Find collaborators
            collaborators = await agent._find_collaborators(task)
            
            # Coordinate with collaborators
            for collaborator_id in collaborators:
                await agent.send_message(
                    collaborator_id,
                    "collaboration_request",
                    {"task": task, "role": "participant"}
                )
                
        return behavior

class AgentFactory:
    """Factory for creating specialized agents"""
    
    @staticmethod
    def create_task_agent(agent_id: str, capabilities: List[str]) -> TaskAgent:
        """Create task-specialized agent"""
        agent_capabilities = [
            AgentCapability(name=cap, description=f"{cap} capability")
            for cap in capabilities
        ]
        
        profile = AgentProfile(
            agent_id=agent_id,
            agent_type="task",
            name=f"Task Agent {agent_id}",
            description="Specialized task execution agent",
            capabilities=agent_capabilities
        )
        
        return TaskAgent(profile)
        
    @staticmethod
    def create_coordinator_agent(agent_id: str) -> CoordinatorAgent:
        """Create coordination agent"""
        capabilities = [
            AgentCapability(name="coordination", description="Multi-agent coordination"),
            AgentCapability(name="task_decomposition", description="Task breakdown"),
            AgentCapability(name="resource_allocation", description="Resource management")
        ]
        
        profile = AgentProfile(
            agent_id=agent_id,
            agent_type="coordinator",
            name=f"Coordinator Agent {agent_id}",
            description="Multi-agent coordination specialist",
            capabilities=capabilities
        )
        
        return CoordinatorAgent(profile)
```

## Debugging Tips
- **Agent state tracking** - Monitor agent state transitions and lifecycle
- **Message flow tracing** - Trace message flows between agents
- **Performance profiling** - Profile agent message processing performance
- **Capability monitoring** - Monitor agent capability usage and effectiveness
- **Communication debugging** - Debug inter-agent communication protocols