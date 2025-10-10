import json
import subprocess
from collections.abc import Sequence


def _run_json_object_command(
    pairs: Sequence[str],
) -> subprocess.CompletedProcess[str]:
    command = [
        "bash",
        "-c",
        'source scripts/lib/sdd-common.sh && sdd_json_object_from_kv "$@"',
        "sdd_json_object_from_kv",
        *pairs,
    ]
    return subprocess.run(command, capture_output=True, text=True)


def test_sdd_json_object_from_kv_preserves_spaces_and_quotes() -> None:
    result = _run_json_object_command(
        ["owner=Platform Team", 'quote=He said "Hello"']
    )

    assert result.returncode == 0, result.stderr
    data = json.loads(result.stdout.strip())
    assert data == {
        "owner": "Platform Team",
        "quote": 'He said "Hello"',
    }


def test_sdd_json_object_from_kv_preserves_equals_and_empty_values() -> None:
    result = _run_json_object_command(["alpha=1", "beta=two=2", "gamma="])

    assert result.returncode == 0, result.stderr
    data = json.loads(result.stdout.strip())
    assert list(data.items()) == [
        ("alpha", "1"),
        ("beta", "two=2"),
        ("gamma", ""),
    ]


def test_sdd_json_object_from_kv_handles_no_arguments() -> None:
    result = _run_json_object_command([])

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "{}"
