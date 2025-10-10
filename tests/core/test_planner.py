"""Tests for the Planner component."""

from src.core import planner


def test_plan_composition_and_rationale():
    """Test that the planner can compose a multi-step plan and provide rationale."""
    # Set up a simple graph with three atoms and compatible contracts
    gs = planner.GraphStore()
    # Define atoms and their I/O contracts
    gs.add_atom("AtomA", input_fields=set(), output_fields={"x", "y"})
    gs.add_atom("AtomB", input_fields={"x"}, output_fields={"y", "z"})
    gs.add_atom("AtomC", input_fields={"z"}, output_fields={"out"})
    # Define graph edges for possible chaining
    gs.add_edge("AtomA", "AtomB")  # A -> B
    gs.add_edge("AtomB", "AtomC")  # B -> C

    # Create a dummy VectorIndex that will return AtomA as the best match for the query
    class DummyVectorIndex(planner.VectorIndex):
        def find_similar(self, query: str, top_k: int = 5):
            del top_k  # Avoid unused argument warning
            if query == "test query":
                # AtomA is highly relevant, AtomB less so
                return [("AtomA", 0.95), ("AtomB", 0.5)]
            return []

    vi = DummyVectorIndex()

    plan_mgr = planner.Planner(vector_index=vi, graph_store=gs, max_steps=3)
    event = plan_mgr.propose_plan("test query")
    plan = event.plan

    # The plan should start with AtomA and expand to AtomB and AtomC
    # (since contracts allow full chain)
    step_ids = [step.atom_id for step in plan.steps]
    assert step_ids[0] == "AtomA"
    assert "AtomB" in step_ids  # AtomB should be chained after AtomA
    assert "AtomC" in step_ids  # AtomC should be chained after AtomB
    # Rationale should mention the starting atom and that it expanded
    assert "Selected 'AtomA' as starting atom" in plan.rationale
    assert (
        "Expanded with" in plan.rationale
        or "No further expansion" in plan.rationale
    )


def test_plan_caching_reuse():
    """Test that plans are cached and reused for identical queries."""
    gs = planner.GraphStore()
    gs.add_atom("X", input_fields=set(), output_fields={"foo"})
    # No neighbors or additional steps needed for simplicity
    gs.add_edge("X", "X")  # allow a self-loop (not really used here)

    # VectorIndex always returns X for a given query
    class SingleAtomIndex(planner.VectorIndex):
        def find_similar(self, query: str, top_k: int = 5):
            del top_k  # Avoid unused argument warning
            if query == "do X":
                return [("X", 0.9)]
            return []

    vi = SingleAtomIndex()

    plan_mgr = planner.Planner(
        vector_index=vi, graph_store=gs, max_steps=1, cache_size=10
    )
    # First call will compute and cache the plan
    event1 = plan_mgr.propose_plan("do X")
    plan1 = event1.plan
    assert [step.atom_id for step in plan1.steps] == ["X"]

    # Modify the VectorIndex to ensure if it's called again we can detect it
    called_flag = {"called": False}

    class TrackingIndex(planner.VectorIndex):
        def find_similar(self, query, top_k=5):
            del query, top_k  # Avoid unused argument warnings
            called_flag["called"] = True
            return [("X", 0.5)]

    plan_mgr.vector_index = TrackingIndex()

    # Second call with same query should use cache (and not call the new index)
    event2 = plan_mgr.propose_plan("do X")
    plan2 = event2.plan
    assert [step.atom_id for step in plan2.steps] == ["X"]
    # The cached plan object should be returned (same instance)
    assert plan2 is plan1
    # The TrackingIndex should not have been called due to cache
    assert called_flag["called"] is False


def test_empty_plan_when_no_candidates():
    """Test that an empty plan is returned when no candidates are found."""
    gs = planner.GraphStore()

    # Empty VectorIndex that returns no candidates
    class EmptyVectorIndex(planner.VectorIndex):
        def find_similar(self, query: str, top_k: int = 5):
            del query, top_k  # Avoid unused argument warnings
            return []

    vi = EmptyVectorIndex()
    plan_mgr = planner.Planner(vector_index=vi, graph_store=gs)
    event = plan_mgr.propose_plan("unknown query")

    assert len(event.plan.steps) == 0
    assert "No relevant atoms found" in event.plan.rationale


def test_contract_compatibility():
    """Test that contract compatibility is checked correctly."""
    gs = planner.GraphStore()

    # Test compatible contracts
    gs.add_atom("A", input_fields=set(), output_fields={"x", "y"})
    gs.add_atom("B", input_fields={"x"}, output_fields={"z"})
    assert gs.is_contract_compatible("A", "B") is True

    # Test incompatible contracts (B needs "x" but C only outputs "w")
    gs.add_atom("C", input_fields=set(), output_fields={"w"})
    assert gs.is_contract_compatible("C", "B") is False


def test_plan_step_descriptions():
    """Test that plan steps have appropriate descriptions."""
    gs = planner.GraphStore()
    gs.add_atom("Step1", input_fields=set(), output_fields={"data"})
    gs.add_atom("Step2", input_fields={"data"}, output_fields={"result"})
    gs.add_edge("Step1", "Step2")

    class TestVectorIndex(planner.VectorIndex):
        def find_similar(self, query: str, top_k: int = 5):
            del query, top_k  # Avoid unused argument warnings
            return [("Step1", 0.8)]

    vi = TestVectorIndex()
    plan_mgr = planner.Planner(vector_index=vi, graph_store=gs, max_steps=2)
    event = plan_mgr.propose_plan("test")

    # Check that descriptions are set appropriately
    steps = event.plan.steps
    assert len(steps) == 2
    assert "similarity=" in steps[0].description
    assert "Chained from" in steps[1].description


def test_cache_eviction():
    """Test that the LRU cache evicts oldest entries when full."""
    gs = planner.GraphStore()
    gs.add_atom("TestAtom", input_fields=set(), output_fields={"result"})

    class TestVectorIndex(planner.VectorIndex):
        def find_similar(self, query: str, top_k: int = 5):
            del query, top_k  # Avoid unused argument warnings
            return [("TestAtom", 0.9)]

    vi = TestVectorIndex()
    # Create planner with very small cache size
    plan_mgr = planner.Planner(vector_index=vi, graph_store=gs, cache_size=2)

    # Fill cache to capacity
    plan_mgr.propose_plan("query1")
    plan_mgr.propose_plan("query2")
    assert len(plan_mgr._plan_cache) == 2

    # Add third query - should evict oldest (query1)
    plan_mgr.propose_plan("query3")
    assert len(plan_mgr._plan_cache) == 2
    assert "query1" not in plan_mgr._plan_cache
    assert "query2" in plan_mgr._plan_cache
    assert "query3" in plan_mgr._plan_cache


def test_plan_proposed_event_structure():
    """Test that PlanProposedEvent has correct structure."""
    gs = planner.GraphStore()
    gs.add_atom("EventTest", input_fields=set(), output_fields={"output"})

    class TestVectorIndex(planner.VectorIndex):
        def find_similar(self, query: str, top_k: int = 5):
            del query, top_k  # Avoid unused argument warnings
            return [("EventTest", 0.7)]

    vi = TestVectorIndex()
    plan_mgr = planner.Planner(vector_index=vi, graph_store=gs)
    event = plan_mgr.propose_plan("test event")

    # Check event structure
    assert event.query == "test event"
    assert len(event.plan.steps) == 1
    assert event.plan.steps[0].atom_id == "EventTest"

    # Check event dict representation
    event_dict = event.to_dict()
    assert "query" in event_dict
    assert "plan_steps" in event_dict
    assert "rationale" in event_dict
    assert event_dict["plan_steps"] == ["EventTest"]
