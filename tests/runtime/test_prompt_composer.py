from copilot.prompt_composer import compose_chat_prompt, compose_inline_prompt


def test_compose_chat_prompt_deterministic():
    banner = "SYS"
    user = "What can you do?"
    files = ["a.txt: alpha", "b.txt: beta"]
    signals = ["note this", "remember that"]
    result1 = compose_chat_prompt(
        banner, user, files, signals, token_budget=10
    )
    result2 = compose_chat_prompt(
        banner, user, files, signals, token_budget=10
    )
    assert result1 == result2
    assert len(result1.hints["text"]) <= 40  # 10 tokens ≈ 40 chars
    assert result1.hints["content_hash"] == result2.hints["content_hash"]


def test_compose_inline_prompt_wrapper():
    banner = "SYS"
    snippet = "print('hi')"
    files = ["a.py: print"]
    res = compose_inline_prompt(banner, snippet, files, token_budget=5)
    assert res.user == snippet
    assert "Chat" not in res.hints["text"]
