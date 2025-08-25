"""
Test event serialization roundtrip to verify structured events work properly
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.events import AgentReplyEvent, AtomReadyEvent, CognitiveAtomEvent
from src.core.serialization import EventSerializer


def test_agent_reply_roundtrip():
    """Test AgentReplyEvent serialization roundtrip."""
    ser = EventSerializer()
    evt = AgentReplyEvent(
        source_plugin="conversation",
        text="Hello, world!",
        correlation_id="abc123",
        session_id="session_1",
        conversation_id="conv_1",
    )

    payload = ser.encode(evt)
    back = ser.decode(payload)

    assert isinstance(back, AgentReplyEvent)
    assert back.text == "Hello, world!"
    assert back.source_plugin == "conversation"
    assert back.correlation_id == "abc123"
    assert back.session_id == "session_1"


def test_cognitive_atom_roundtrip():
    """Test CognitiveAtomEvent serialization roundtrip."""
    ser = EventSerializer()
    atom = {
        "atom_id": "a1",
        "atom_type": "GAP_DETECTED",
        "content": {"text": "Need fibonacci tool"},
        "meta": {"provenance": {"source_id": "creator"}},
    }

    evt = CognitiveAtomEvent(
        source_plugin="creator",
        atom=atom,
        atom_id="a1",
        atom_type="GAP_DETECTED",
        correlation_id="wf-1",
    )

    payload = ser.encode(evt)
    back = ser.decode(payload)

    assert isinstance(back, CognitiveAtomEvent)
    assert back.atom["atom_id"] == "a1"
    assert back.atom_type == "GAP_DETECTED"
    assert back.correlation_id == "wf-1"


def test_atom_ready_roundtrip():
    """Test AtomReadyEvent serialization roundtrip."""
    ser = EventSerializer()
    atom = {
        "atom_id": "tool_123",
        "atom_type": "RESOURCE",
        "title": "Tool:fibonacci_calculator",
        "content": '{"tool_name":"fibonacci_calculator","registered":true}',
    }

    evt = AtomReadyEvent(
        source_plugin="creator",
        atom=atom,
        tool_name="fibonacci_calculator",
        file_path="/tools/fibonacci_calculator.py",
        session_id="session_1",
        conversation_id="conv_1",
    )

    payload = ser.encode(evt)
    back = ser.decode(payload)

    assert isinstance(back, AtomReadyEvent)
    assert back.tool_name == "fibonacci_calculator"
    assert back.atom["atom_id"] == "tool_123"
    assert back.file_path == "/tools/fibonacci_calculator.py"


if __name__ == "__main__":
    test_agent_reply_roundtrip()
    test_cognitive_atom_roundtrip()
    test_atom_ready_roundtrip()
    print("✅ All event roundtrip tests passed!")
