# 🚀 Super Alita: Production-Grade Autonomous Cognitive Agent  
*2025 Edition – Neuro-Symbolic, Evolutionary, & Auditable by Design*

## 1. 📦 Project Structure

```
super-alita/
├── src/
│   ├── core/
│   │   ├── plugin_interface.py      # Plugin contract
│   │   ├── neural_atom.py           # Atoms, NeuralStore, genealogy helpers
│   │   ├── genealogy.py             # Genealogy tracer & export (GraphML)
│   │   ├── event_bus.py             # Async Protobuf event bus
│   │   └── events.py                # All event schemas
│   ├── plugins/
│   │   ├── semantic_memory_plugin.py
│   │   ├── ladder_aog_plugin.py
│   │   ├── semantic_fsm_plugin.py
│   │   ├── skill_discovery_plugin.py
│   │   ├── self_heal_plugin.py
│   │   └── event_bus_plugin.py
│   ├── tools/
│   │   ├── mcts_evolution.py       # MCTS evolutionary engine
│   ├── main.py                     # Plugin/agent orchestrator
│   └── config/
│       └── agent.yaml
├── tests/                          # Full pytest/test suite
│   ├── core/
│   │   ├── test_plugin_interface.py
│   │   ├── test_neural_atom.py
│   │   └── test_genealogy.py
│   ├── plugins/
│   └── integration/
├── data/
│   └── .gitkeep
├── requirements.txt
└── README.md
```

## 2. ⚡️ Quickstart: Local Development

```bash
git clone https://github.com/YOUR_ORG/super-alita.git
cd super-alita
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
# Optionally: pip install . for editable mode if set up as a package
python src/main.py
```

> **Tip:** To run tests:  
> ```bash
> pytest
> ```

## 3. 🧠 Architecture Overview

- **Pluggable, modular neuro-symbolic agent core**
- **Event-driven nervous system (Protobuf bus)**
- **Hierarchical semantic memory (FAISS/Chroma, Gemini, SHIMI)**
- **Atomic, reactive working memory (Jotai/signals pattern)**
- **Differentiable planning (LADDER + neural-symbolic)**
- **Semantic-FSM for workflow/attention focus**
- **Skill evolution (PAE; MCTS evolutionary arena)**
- **Full Darwin-Gödel genealogy—exportable as GraphML**
- **Extensible via `PluginInterface`**

## 4. 🛠️ Plugin/Tool Development Workflow

### 4.1. Create a new plugin (`src/plugins/my_plugin.py`)

```python
from src.core.plugin_interface import PluginInterface

class MyPlugin(PluginInterface):
    @property
    def name(self):
        return "my_plugin"

    async def setup(self, event_bus, store, config):
        # Initialize dependencies
        self.bus = event_bus
        self.store = store

    async def start(self):
        # Register for events, spawn background tasks, etc.
        self.bus.subscribe("my_event", self.handle_event)

    async def handle_event(self, payload):
        # Plugin-specific logic here
        pass

    async def shutdown(self):
        # Cleanup tasks, unsubscribe events
        pass
```

### 4.2. Register your plugin in `src/main.py`

```python
from src.plugins.my_plugin import MyPlugin

class SuperAlita:
    def __init__(self):
        # ...existing plugins...
        self.plugins['my_plugin'] = MyPlugin()
```

## 5. ⚙️ Key Concepts & Agent Lifecycle

### **a. Atoms & State**

- All agent context (goal, tools, FSM state) = atom (see `neural_atom.py`)
- Atoms: Small, reactive state units; changes automatically trigger only dependent computations.

### **b. Event Bus**

- All inter-plugin and tool messaging via async Protobuf event bus (see `event_bus.py`)
- Each event is versioned, typed, and can carry embeddings for semantic routing.

### **c. Reasoning**

- Planning engine (`ladder_aog_plugin.py`) takes symbolic and neural inputs; can learn/adapt via experience.
- Use `SemanticFSM` to navigate operational states flexibly by *embedding similarity*, not just static triggers.

### **d. Skill Discovery/Evolution**

- Add new skills via the PAE pipeline (Proposer LLM → Agent tries/learns → Evaluator LLM/metric scores, see `skill_discovery_plugin.py`)
- For creative leaps, use `MCTSEvolutionArena` from `tools/mcts_evolution.py`.

### **e. Genealogy & Audit**

- Every cognitive primitive has **parents, children, metadata, and traceable birth event** (see `genealogy.py`)
- Export full GraphML (`.graphml`) with `GenealogyTracer` for Gephi/yEd/NetworkX analysis at any time.

## 6. 🔥 Best Practices (as enforced/suggested by Super Alita Copilot)

- **Write every component as a hot-swappable plugin.**
- **Emit events instead of calling modules directly**; subscribe with semantic filters for decoupling.
- **Instrument all new skills/tools with `birth_event`, parentage, and performance metadata.**
- **Test all new atoms/plugins in isolation and as part of system integration.**
- **Export genealogy regularly for reproducibility/audit/scientific analysis.**
- **Never allow cognitive stasis:** trigger `MCTSEvolutionArena` if agent is stuck, bored, or needs innovation.
- **Document all new events in `events.py` and in your plugin docstrings.**
- **Continuous lineage pruning:** Optionally, implement a `NeuralStore.prune_unfit()` method.

## 7. ✅ Example: Registering a New Evolutionary Skill

```python
from src.core.neural_atom import NeuralAtom

# Suppose agent learns a new SQL scanner skill
new_skill_atom = NeuralAtom(
    key='skill:sql_scanner',
    default_value=your_skill_obj,
    vector=embedding,
    parent_keys=['skill:sql_base', 'skill:regex_base'],
    birth_event="skill_proposal",
    lineage_metadata={
        "task": "hunter_bounty_314",
        "fitness": 0.85,
        "proposer": "super_alita_self"
    }
)
store.register_with_lineage(new_skill_atom, parents=[atom1, atom2], birth_event="skill_proposal", lineage_metadata=new_skill_atom.lineage_metadata)
```

## 8. 📈 Export and Visualize Genealogy (for Analysis & Debugging)

```python
from src.core.genealogy import GenealogyTracer

tracer = GenealogyTracer()
tracer.export_to_graphml("data/alita_genealogy.graphml")
# Now open with Gephi/yEd for instant visual debugging.
```

## 9. 🧪 Testing & Coverage

- All base interfaces and major plugins should have **unit tests** in `tests/`.
- Simulate high-load event bursts and skill evolutions in `tests/integration/`.
- Goal: 90%+ coverage on all core files and plugins. Use pytest-cov.

## 10. 🚀 Roadmap for Contributors

- **Start by forking and running the CLI system end-to-end.**
- **Pick a layer:** plugin, memory operator, event, evolutionary tool, or orchestration.
- **Develop and test in a branch. Docs required for plugins/events.**
- **Contribute exported GraphML of your genealogy for CI auto-testing.**
- **Propose bold feature PRs! Use the Proposer-Agent-Evaluator cycle as inspiration.**

## 🏁 Summary Table: Copilot Reference

| **Action** | **Copilot-Optimized Instruction** |
|------------|-----------------------------------|
| Add plugin | Subclass `PluginInterface`, register in main, emit/subscribe to events |
| Add new atom | Extend `NeuralAtom`, ensure parent/child links and metadata, register lineage |
| Publish event | Use Protobuf schemas in `events.py`, always version and semantically annotate |
| Export audit | Use `GenealogyTracer.export_to_graphml`, review in Gephi/yEd |
| Trigger Evo | Use `MCTSEvolutionArena`, store only the fittest atom in production graph |

## 🧩 Copy ✨this structure and adapt aggressively—Super Alita isn't just an agent, it's a living, evolvable AI platform.

If you want a customization for your specific LLM, further code scaffolds, or Instant Advanced Examples (e.g., writing a plugin that evolves its own reasoning policies), **just ask!**
