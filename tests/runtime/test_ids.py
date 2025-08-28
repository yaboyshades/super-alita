import hashlib
import uuid

from hypothesis import assume, given
from hypothesis import strategies as st

from src.core.ids import NAMESPACE_TOOLS, _content_key, deterministic_tool_id

"""
Property-based tests for ID generation utilities.

Edge cases documented:
- Empty strings for title, code, and extra
- Very long code strings (>160 chars) triggering SHA256 hashing
"""


@given(
    title=st.text(),
    code=st.text(),
    extra=st.one_of(st.text(), st.none()),
)
def test_deterministic_tool_id_stable(title: str, code: str, extra: str | None) -> None:
    """IDs remain stable across repeated calls with identical inputs."""
    first = deterministic_tool_id(title, code, extra=extra)
    second = deterministic_tool_id(title, code, extra=extra)
    assert first == second


@given(code=st.text(max_size=160))
def test_content_key_short_returns_plain_text(code: str) -> None:
    """Codes shorter than threshold are used verbatim."""
    ck = _content_key(code)
    assert ck == (code or "").strip()
    assert not ck.startswith("sha256:")

    title = "t"
    extra = "e"
    expected_name = "|".join(["TOOL", title, ck, extra])
    expected_id = str(uuid.uuid5(NAMESPACE_TOOLS, expected_name))
    assert deterministic_tool_id(title, code, extra=extra) == expected_id


@given(code=st.text(min_size=161, max_size=1000))
def test_content_key_long_uses_sha256(code: str) -> None:
    """Long code strings are hashed to maintain stable IDs."""
    stripped = (code or "").strip()
    assume(len(stripped) > 160)
    ck = _content_key(code)
    expected_hash = hashlib.sha256(stripped.encode("utf-8")).hexdigest()
    assert ck == f"sha256:{expected_hash}"

    title = "t"
    extra = "e"
    expected_name = "|".join(["TOOL", title, ck, extra])
    expected_id = str(uuid.uuid5(NAMESPACE_TOOLS, expected_name))
    assert deterministic_tool_id(title, code, extra=extra) == expected_id


def test_explicit_edge_cases() -> None:
    """Explicit regression for documented edge cases."""
    assert deterministic_tool_id("", "", extra="") == deterministic_tool_id(
        "", "", extra=""
    )

    long_code = "x" * 1000
    ck = _content_key(long_code)
    expected_hash = hashlib.sha256(long_code.encode("utf-8")).hexdigest()
    assert ck == f"sha256:{expected_hash}"
