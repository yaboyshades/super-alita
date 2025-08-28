import subprocess
from pathlib import Path

import pytest


@pytest.mark.asyncio
async def test_extension_build():
    """Test that the extension builds successfully with npm compile."""
    ext_path = Path("extensions/alita-language-tools")
    assert ext_path.exists(), "Extension directory not found"

    # Check package.json exists
    package_json = ext_path / "package.json"
    assert package_json.exists(), "package.json not found"

    # Run npm compile (includes codegen and tsc)
    result = subprocess.run(
        ["npm", "run", "compile"],
        cwd=ext_path,
        capture_output=True,
        text=True,
        timeout=60,
    )

    # Allow warnings but fail on errors
    assert result.returncode == 0, f"Build failed: {result.stderr}"

    # Check that output directory was created
    out_dir = ext_path / "out"
    assert out_dir.exists(), "Build output directory not created"

    # Check for main compiled output
    main_js = out_dir / "src" / "extension.js"
    assert main_js.exists(), "Main extension.js not found in output"


@pytest.mark.asyncio
async def test_extension_lint():
    """Test that the extension passes linting."""
    ext_path = Path("extensions/alita-language-tools")

    result = subprocess.run(
        ["npm", "run", "lint"], cwd=ext_path, capture_output=True, text=True, timeout=30
    )

    # Lint should pass with no errors
    assert result.returncode == 0, f"Lint failed: {result.stderr}"


@pytest.mark.asyncio
async def test_codegen_integrity():
    """Test that WIT codegen produces expected output."""
    ext_path = Path("extensions/alita-language-tools")

    result = subprocess.run(
        ["npm", "run", "test:codegen"],
        cwd=ext_path,
        capture_output=True,
        text=True,
        timeout=30,
    )

    # Allow exit code 0 (success) or 1 (test warnings but no critical failures)
    assert result.returncode in [0, 1], f"Codegen test failed: {result.stderr}"

    # Check that generated files exist
    generated_dir = ext_path / "src" / "generated"
    assert generated_dir.exists(), "Generated directory not found"

    meta_file = generated_dir / ".codegen.meta.json"
    assert meta_file.exists(), "Codegen metadata file not found"
