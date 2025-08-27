import re
import subprocess
from pathlib import Path


def run_cargo_tests(manifest: Path) -> str:
    result = subprocess.run(
        ["cargo", "test", "--manifest-path", str(manifest), "--", "--nocapture"],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout + result.stderr


def test_calculator_memory_metric() -> None:
    output = run_cargo_tests(Path("wasm/calculator/Cargo.toml"))
    match = re.search(r"memory_used_calculator=(\d+)", output)
    assert match, output
    assert int(match.group(1)) > 0


def test_code_radar_memory_metric() -> None:
    output = run_cargo_tests(Path("wasm/code_radar/Cargo.toml"))
    match = re.search(r"memory_used_code_radar=(\d+)", output)
    assert match, output
    assert int(match.group(1)) > 0
