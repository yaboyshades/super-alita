import pytest

from codeact.sandbox import PythonSandbox


@pytest.mark.asyncio
async def test_exec_basic():
    sb = PythonSandbox()
    result = await sb.run("print('hi')")
    assert result.stdout.strip() == "hi"
    assert result.error is None


@pytest.mark.asyncio
async def test_disallow_import():
    sb = PythonSandbox()
    result = await sb.run("import os")
    assert "not allowed" in (result.error or "")
