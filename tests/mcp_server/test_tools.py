import pytest

from mcp_server.tools import refactor_to_result


@pytest.mark.asyncio
async def test_refactor_to_result_handles_missing_function(tmp_path):
    file = tmp_path / "mod.py"
    file.write_text("def foo():\n    return 1\n", encoding="utf-8")
    res = await refactor_to_result(str(file), "missing")
    assert res["applied"] is False
    assert res["error"]
