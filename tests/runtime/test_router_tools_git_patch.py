import os
import subprocess

import pytest

from reug_runtime.router_tools import git_apply_patch


@pytest.mark.asyncio
async def test_git_apply_patch(tmp_path):
    repo = tmp_path
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "a@b.c"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Tester"], cwd=repo, check=True)
    file_path = repo / "file.txt"
    file_path.write_text("hello\n")
    subprocess.run(["git", "add", "file.txt"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=repo, check=True, capture_output=True)
    file_path.write_text("hello\nworld\n")
    patch = subprocess.run(
        ["git", "diff", "file.txt"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    file_path.write_text("hello\n")
    cwd = os.getcwd()
    os.chdir(repo)
    try:
        result = await git_apply_patch({"patch": patch})
    finally:
        os.chdir(cwd)
    assert result["ok"], result
    assert file_path.read_text() == "hello\nworld\n"
    os.chdir(repo)
    try:
        result_fail = await git_apply_patch({"patch": patch})
    finally:
        os.chdir(cwd)
    assert not result_fail["ok"]
    assert result_fail["stderr"]
