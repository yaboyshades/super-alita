import pytest

from tests.runtime.fakes import FakeAbilityRegistry


@pytest.mark.asyncio
async def test_execute_rejects_unexpected_kwargs() -> None:
    reg = FakeAbilityRegistry()
    with pytest.raises(TypeError, match="Unexpected argument\(s\): extra"):
        await reg.execute("echo", {"payload": "hi", "extra": 1})
