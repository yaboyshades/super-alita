from reug_runtime.router import BlockParser, MAX_BUFFER_BYTES


def test_partial_tags():
    parser = BlockParser()
    parser.feed('<tool_call>{"tool":"echo",')
    assert parser.get_tool_call() is None
    parser.feed('"args":{}}</tool_call>')
    assert parser.get_tool_call() == {"tool": "echo", "args": {}}


def test_extract_malformed_json_and_recovery():
    parser = BlockParser()
    parser.feed('<tool_call>{bad}</tool_call>')
    assert parser._extract('tool_call') is None
    parser.feed('<tool_call>{"tool":"ok","args":{}}</tool_call>')
    assert parser.get_tool_call() == {"tool": "ok", "args": {}}


def test_feed_truncates_buffer_on_overflow():
    parser = BlockParser()
    parser.feed('x' * (MAX_BUFFER_BYTES + 10))
    assert len(parser.buffer) == MAX_BUFFER_BYTES
    tag = '<final_answer>{"content":"ok"}</final_answer>'
    parser.feed(tag)
    assert parser.buffer.endswith(tag)
    assert parser.get_final_answer() == {"content": "ok"}


def test_final_answer_fallback_on_invalid_json():
    parser = BlockParser()
    parser.feed('<final_answer>{not:json}</final_answer>')
    assert parser.get_final_answer() == {"content": "{not:json}"}
