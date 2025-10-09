"""Tests for formatting normalization helpers."""

from reug_runtime.formatting import normalize_output_contract


def test_normalize_output_contract_in_code_block() -> None:
    """Unicode operators should be normalized inside fenced blocks."""
    text = (
        "Before\n"
        "```python\n"
        "area = 2 × radius · radius\n"
        "print(“done”)\n"
        "```\n"
        "After — dash"
    )

    normalized = normalize_output_contract(text)

    assert "area = 2 * radius * radius" in normalized
    assert 'print("done")' in normalized
    # Content outside fenced blocks should remain untouched
    assert "After — dash" in normalized


def test_normalize_output_contract_handles_unclosed_fence() -> None:
    """Unclosed fences should still have their content normalized."""
    text = "```python\nvalue = 3 × 7"

    normalized = normalize_output_contract(text)

    assert "value = 3 * 7" in normalized


def test_normalize_output_contract_no_fence_passthrough() -> None:
    """If there is no fenced block the text should be returned as-is."""
    text = "Just some prose with – en dash."

    normalized = normalize_output_contract(text)

    assert normalized == text
