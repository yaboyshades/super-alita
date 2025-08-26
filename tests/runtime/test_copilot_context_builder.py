from __future__ import annotations

from src.copilot.context_builder import (
    ChatContext,
    OpenFile,
    build_copilot_context,
)


def test_build_copilot_context_extracts_components() -> None:
    source = (
        "# module comment\n"
        "\n"
        "import os\n"
        "\n"
        "class Foo:\n"
        "    pass\n"
        "\n"
        "def bar():\n"
        "    pass\n"
    )
    open_file = OpenFile(path="a.py", content=source, selection=(5, 6))
    attachment = OpenFile(path="notes.txt", content="extra")
    chat = ChatContext(open_files=[open_file], attachments=[attachment])

    context = build_copilot_context(chat)

    assert context.top_comments["a.py"] == "module comment"
    assert "os" in context.imports["a.py"]
    assert {"Foo", "bar"} <= set(context.symbols["a.py"])
    assert context.selections["a.py"].splitlines()[0].startswith("class Foo")
    assert context.attachments == ["notes.txt"]
    assert context.quality_mode == "fast"


def test_content_hash_deterministic() -> None:
    f1 = OpenFile(path="a.py", content="print('a')\n")
    f2 = OpenFile(path="b.py", content="print('b')\n")

    ctx1 = ChatContext(open_files=[f1, f2])
    ctx2 = ChatContext(open_files=[f2, f1])

    hash1 = build_copilot_context(ctx1).content_hash
    hash2 = build_copilot_context(ctx2).content_hash

    assert hash1 == hash2
