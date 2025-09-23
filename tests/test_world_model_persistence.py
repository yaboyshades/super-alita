#!/usr/bin/env python3
"""Test predictive world model persistence mechanism."""

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, "./src")

from core.predictive_world_model import PredictiveWorldModel


async def test_persistence():
    # Create test model with sample data
    model = PredictiveWorldModel()

    # Set sample data
    model.state_transitions = [
        {
            "initial_state": {"task": "test1"},
            "action_taken": "analyze",
            "final_state": {"task": "test1", "result": "success"},
            "timestamp": "2025-08-31T14:00:00Z",
        }
    ]

    model.learned_patterns = {
        "analyze": {"success_rate": 0.95, "avg_duration": 1.2, "count": 10}
    }

    # Test saving
    print("Saving test data...")
    await model._save_persistent_data()

    # Check if file exists
    path = Path("./data/world_model.json")
    if path.exists():
        print(f"✅ World model saved successfully: {path}")
        data = json.loads(path.read_text(encoding="utf-8"))
        print("📊 Data snapshot:")
        print(f'  - Transitions: {len(data.get("state_transitions", []))}')
        print(f'  - Patterns: {len(data.get("learned_patterns", {}))}')
        print(f'  - Timestamp: {data.get("timestamp", "N/A")}')
    else:
        print(f"❌ World model file not created: {path}")

    # Create a new model instance and load
    new_model = PredictiveWorldModel()
    print("\nLoading world model...")
    await new_model._load_persistent_data()

    # Verify data loaded
    transitions = getattr(new_model, "state_transitions", [])
    patterns = getattr(new_model, "learned_patterns", {})
    print("📥 Loaded data:")
    print(f"  - Transitions: {len(transitions)}")
    print(f"  - Patterns: {len(patterns)}")

    if len(transitions) > 0 and len(patterns) > 0:
        print("✅ Persistence mechanism working correctly!")
    else:
        print("❌ Failed to load data correctly")


if __name__ == "__main__":
    asyncio.run(test_persistence())
