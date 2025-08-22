# Super Alita REUG-Copilot Framework Reference

## Quick Reference for Advanced Techniques

### 🎯 Planning Patterns

#### 1. `script.py` Planning Template
```python
# script.py: Plan for [Task Description]
# 1. Analysis Phase
#    - Understand requirements and constraints
#    - Identify existing patterns and components
#    - Map to Sacred Laws and architecture
# 2. Design Phase  
#    - Choose cognitive pattern (analytical/creative/diagnostic/strategic/exploratory)
#    - Plan event flows and plugin interactions
#    - Consider DTA 2.0 cognitive turn integration
# 3. Implementation Phase
#    - Create/modify core files
#    - Ensure event contract compliance
#    - Add comprehensive error handling
# 4. Validation Phase
#    - Run tests and health checks
#    - Validate cognitive turn processing
#    - Check integration with existing system
# 5. Documentation Phase
#    - Update relevant documentation
#    - Add inline comments and explanations
#    - Record architectural decisions
```

#### 2. Pythonic Chain-of-Thought Template
```python
# Step 1: Analyze the problem and identify key components
problem_components = ["user_input", "cognitive_processing", "event_routing"]

# Step 2: Design the solution architecture  
def cognitive_solution_architecture():
    # Map inputs to cognitive patterns
    # Define event flow and plugin interactions
    # Plan error handling and fallback strategies
    pass

# Step 3: Implement core logic with proper contracts
async def implement_with_contracts(input_data):
    # Process through cognitive turn
    # Emit required events per Sacred Laws
    # Return structured result
    pass

# Final Answer: [Complete working implementation]
```

### 🧠 Cognitive Pattern Selection Guide

| Pattern | Use When | Example Tasks |
|---------|----------|---------------|
| **Analytical** | Need deep understanding, explanation | Code review, debugging, analysis |
| **Creative** | Building new features, solutions | Architecture design, tool creation |
| **Diagnostic** | Fixing problems, errors | Bug fixes, performance issues |
| **Strategic** | Long-term planning, decisions | Roadmap planning, refactoring |
| **Exploratory** | Research, discovery | Technology evaluation, investigation |

### 🔧 Event Contract Patterns

#### Standard Event Flow
```python
# 1. Receive Event
async def handle_tool_call(self, event: ToolCallEvent):
    try:
        # 2. Process the request
        result = await self.process_tool_request(event)
        
        # 3. ALWAYS emit result event (Sacred Law #1)
        result_event = ToolResultEvent(
            source_plugin=self.name,
            tool_name=event.tool_name,
            result=result,
            session_id=event.session_id
        )
        await self.event_bus.publish(result_event)
        
    except Exception as e:
        # 4. Emit error result for contract compliance
        error_event = ToolResultEvent(
            source_plugin=self.name,
            tool_name=event.tool_name,
            result={"error": str(e)},
            session_id=event.session_id
        )
        await self.event_bus.publish(error_event)
```

#### Cognitive Turn Integration
```python
# Enhanced event handling with cognitive processing
async def handle_with_cognitive_turn(self, event: ConversationEvent):
    # 1. Process through cognitive turn
    if self.cognitive_enabled:
        turn_record = await self._process_cognitive_turn(
            event.user_message, event.context, event.session_id
        )
        
        # 2. Emit cognitive turn event
        if turn_record:
            cognitive_event = CognitiveTurnCompletedEvent(
                source_plugin=self.name,
                turn_record=turn_record,
                session_id=event.session_id
            )
            await self.event_bus.publish(cognitive_event)
    
    # 3. Continue with standard processing
    await self._process_standard_flow(event)
```

### 📋 Quality Assurance Checklist

#### Before Committing Code
- [ ] **Sacred Laws Compliance**: Event contracts maintained?
- [ ] **Plugin Integration**: Proper setup/start/shutdown methods?
- [ ] **Error Handling**: Comprehensive exception management?
- [ ] **Cognitive Integration**: DTA 2.0 processing where applicable?
- [ ] **Testing**: Unit tests and integration validation?
- [ ] **Documentation**: Clear comments and explanations?
- [ ] **Performance**: Async patterns and efficient resource usage?

#### Architecture Validation
- [ ] **Event Flow**: Clear pub/sub patterns?
- [ ] **Memory Management**: Proper cleanup and resource handling?
- [ ] **Configuration**: Environment variables and settings?
- [ ] **Logging**: Appropriate log levels and context?
- [ ] **Type Safety**: Proper type hints and validation?

### 🚀 Advanced Prompting Techniques

#### Chain-of-Verification (CoVe)
After generating a solution, ask:
1. Are all steps necessary and sufficient?
2. Do event contracts align with tool calls?
3. Are dependencies and prerequisites clear?
4. Would this integrate cleanly with existing code?

#### ReAct (Reason + Act) Pattern
1. **Reason**: Analyze the current state and requirements
2. **Act**: Take specific implementation step
3. **Observe**: Check results and validate
4. **Reason**: Plan next step based on observations
5. **Repeat**: Until goal achieved

#### Tree-of-Thoughts (ToT) for Complex Decisions
```
Problem: How to implement feature X?

Approach A: Plugin-based
├─ Pros: Modular, event-driven
├─ Cons: More complexity
└─ Evaluation: Good for extensibility

Approach B: Monolithic
├─ Pros: Simple, direct
├─ Cons: Violates architecture
└─ Evaluation: Not suitable

Choice: Approach A with cognitive enhancement
```

### 🎯 Confidence Calibration Guide

| Confidence | When to Use | Characteristics |
|------------|-------------|-----------------|
| **9-10/10** | Tested, proven patterns | Standard CRUD, known APIs |
| **7-8/10** | Well-understood domain | New features with existing patterns |
| **5-6/10** | Some uncertainty | New integrations, complex logic |
| **3-4/10** | Experimental approach | Cutting-edge techniques, research |
| **1-2/10** | High uncertainty | Incomplete requirements, unknown domain |

### 🔍 Debugging and Validation Tools

#### Health Check Commands
```bash
# System health validation
python quick_health.py

# DTA 2.0 cognitive integration test
python test_dta2_cognitive_integration.py

# Comprehensive validation suite
python comprehensive_validation_suite.py

# Event contract compliance check
python debug_event_contracts.py
```

#### Common Debugging Patterns
```python
# Add cognitive debug logging
logger.info(f"Cognitive pattern: {pattern}, confidence: {confidence}")

# Validate event bus connection
assert self.event_bus is not None, "Event bus not initialized"

# Check Redis connectivity
await self.event_bus.publish(TestEvent())

# Monitor cognitive turn processing
logger.debug(f"Turn record: {turn_record.state_readout}")
```

This framework serves as your tactical guide for implementing the comprehensive Co-Architect approach in Super Alita development.
