import pytest
from fastapi import HTTPException
from reug_runtime.router_tools import fs_read, fs_write


@pytest.mark.asyncio
async def test_fs_write_and_read_roundtrip(tmp_path):
    file_path = tmp_path / "sample.txt"
    content = "hello there"
    write_result = await fs_write({"path": str(file_path), "content": content})
    assert write_result == {"ok": True}
    read_result = await fs_read({"path": str(file_path)})
    assert read_result == {"content": content}


@pytest.mark.asyncio
async def test_fs_read_missing_file(tmp_path):
    missing = tmp_path / "missing.txt"
    with pytest.raises(HTTPException) as excinfo:
        await fs_read({"path": str(missing)})
    assert excinfo.value.status_code == 404
