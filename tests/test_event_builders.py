from src.utils.event_builders import build_tool_call_event


class DummyOriginal:
    conversation_id = "conv-1"
    session_id = "sess-1"
    message_id = "m-1"


def test_build_tool_call_event_minimal():
    ev = build_tool_call_event(
        source_plugin="pythonic_preprocessor_plugin",
        tool_name="conversation",
        parameters={"action": "execute", "message": "hi"},
        conversation_id=DummyOriginal.conversation_id,
        session_id=DummyOriginal.session_id,
        message_id=DummyOriginal.message_id,
    )
    assert ev.conversation_id == "conv-1"
    assert ev.session_id == "sess-1"
    assert ev.tool_name == "conversation"
    assert ev.parameters["message"] == "hi"
    assert ev.parameters["action"] == "execute"
    assert ev.tool_call_id


def test_build_tool_call_event_generates_ids():
    ev = build_tool_call_event(
        source_plugin="pythonic_preprocessor_plugin",
        tool_name="creator",
        parameters={"requirements": "do X"},
        conversation_id="c",
        session_id="s",
    )
    # should have generated tool_call_id
    assert ev.tool_call_id.startswith("tool_")
    assert ev.conversation_id == "c"
    assert ev.session_id == "s"
