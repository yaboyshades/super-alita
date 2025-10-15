# Super Alita Research Edition

This directory contains research implementations for advanced AI agent capabilities, building on the production scaffolding of Super Alita.

## Research Components

### 1. Constitutional LLM (`src/research/constitutional_llm.py`)
- Fine-tuned language model for constitutional reasoning
- LoRA-based efficient fine-tuning
- Rule-based fallback for reliability
- Performance tracking and A/B testing capabilities

### 2. Memory Consolidation Pipeline (`src/research/memory_consolidation.py`)
- Transformer-based memory consolidation using attention mechanisms
- Unified memory system bridging working, episodic, and semantic memory
- Importance-based retention and forgetting mechanisms
- Performance evaluation and optimization

### 3. Research Integration Manager (`src/research/research_integration.py`)
- Seamless integration with production Super Alita components
- Event-driven research data collection
- Automated research experiment execution
- Performance monitoring and improvement tracking

## Installation

```bash
# Install research dependencies
pip install -r requirements-research.txt

# For GPU acceleration (optional but recommended)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Usage

### Quick Start
```bash
# Run research demo
python start_research.py

# Or integrate with main launcher
python start.py --mode research
```

### Research Experiments

```python
from src.main_research import SuperAlitaResearchApplication, RESEARCH_CONFIG

# Initialize research application
app = SuperAlitaResearchApplication(RESEARCH_CONFIG)
await app.initialize()

# Run constitutional LLM A/B test
experiment_config = {
    "type": "constitutional_llm_ab_test",
    "test_cases": [...],  # Your test cases
    "human_judgments": [...]  # Ground truth
}

results = await app.research_manager.run_research_experiment(experiment_config)
print("A/B Test Results:", results)

# Run memory retention test
memory_config = {
    "type": "memory_retention_test",
    "test_memories": [...],
    "retrieval_tests": [...]
}

results = await app.research_manager.run_research_experiment(memory_config)
print("Memory Test Results:", results)
```

## Research Roadmap

This implementation delivers:

**Milestone 1: Constitutional LLM Prototype**
- ✅ Fine-tuned model with rule-based fallback
- ✅ Confidence scoring and performance tracking
- ✅ A/B testing framework

**Milestone 2: Memory Consolidation Pipeline** 
- ✅ Transformer-guided consolidation
- ✅ Importance-based retention
- ✅ Cross-temporal memory integration

**Milestone 3: Research Integration**
- ✅ Seamless production integration
- ✅ Event-driven data collection
- ✅ Automated experimentation

## Cross-Pillar Synergy

The research system demonstrates how memory informs constitutional reasoning, enabling adaptive safety that improves through experience while maintaining core principles.

## Next Steps

1. **Data Collection**: Gather constitutional cases from production usage
2. **Model Training**: Fine-tune constitutional LLM on collected data
3. **Performance Optimization**: Optimize memory consolidation parameters
4. **Advanced Research**: Implement remaining research roadmap components

## Configuration

Key environment variables for research mode:

```bash
RESEARCH_ENABLED=true
ALITA_ENABLE_Z3=true
CONSTITUTIONAL_LLM_MODEL=microsoft/DialoGPT-medium
MEMORY_CONSOLIDATION_INTERVAL=24  # hours
```

## Contributing

See `CONTRIBUTING.md` for guidelines on contributing to the research components.