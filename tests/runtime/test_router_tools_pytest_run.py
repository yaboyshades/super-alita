import pytest

from reug_runtime.router_tools import pytest_run


@pytest.mark.asyncio
async def test_pytest_run_success(tmp_path):
    test_file = tmp_path / "test_sample.py"
    test_file.write_text("def test_ok():\n    assert 1 == 1\n")
    result = await pytest_run({"target": str(test_file)})
    assert result["ok"]
    assert result["exit_code"] == 0
    assert "1 passed" in result["stdout"]
    assert result["stderr"] == ""


@pytest.mark.asyncio
async def test_pytest_run_failure(tmp_path):
    test_file = tmp_path / "test_fail.py"
    test_file.write_text("def test_fail():\n    assert False\n")
    result = await pytest_run({"target": str(test_file)})
    assert not result["ok"]
    assert result["exit_code"] != 0
    combined = result["stdout"] + result["stderr"]
    assert "1 failed" in combined
