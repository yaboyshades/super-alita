import pytest

from src.generation import ConstitutionalCodeGenerator


class _Generator:
    def __init__(self) -> None:
        self.calls = 0

    async def generate(self, prompt: str) -> str:
        self.calls += 1
        if "Violations" in prompt and self.calls < 3:
            return "print('retry')"
        return "def handler():\n    return 'ok'"


class _Validator:
    def __init__(self) -> None:
        self.calls = 0

    async def validate(self, code: str, spec: dict[str, str]):
        self.calls += 1
        compliant = "return 'ok'" in code
        return {"compliant": compliant, "violations": [] if compliant else ["missing return"]}


@pytest.mark.asyncio
async def test_constitutional_coder_refines_until_compliant():
    generator = _Generator()
    validator = _Validator()
    coder = ConstitutionalCodeGenerator(generator, validator=validator)
    code = await coder.generate_capability_code({"cap": "demo"}, {"language": "python"})
    assert "return 'ok'" in code
    assert generator.calls >= 1
