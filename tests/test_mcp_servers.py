#!/usr/bin/env python3
"""
Simple MCP Server Diagnostic Tool

This script tests the MCP servers to ensure they can start properly
and respond to basic requests without requiring VS Code integration.
"""

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


def test_server_import(server_path: str, description: str) -> bool:
    """Test if a server can be imported without errors."""
    print(f"\n🔍 Testing {description}...")

    try:
        # Test import only
        if server_path.endswith(".py"):
            # Pass path/module via env to avoid Windows backslash escaping issues,
            # and use ASCII output to avoid cp1252 Unicode errors.
            env = {
                **os.environ,
                "PYTHONIOENCODING": "utf-8",
                "SERVER_DIR": str(Path(server_path).parent),
                "SERVER_MOD": Path(server_path).stem,
            }
            code = (
                "import sys, os; sys.path.insert(0, os.environ['SERVER_DIR']); "
                "__m=os.environ['SERVER_MOD']; __import__(__m); print('OK')"
            )
            result = subprocess.run(
                [sys.executable, "-c", code],
                capture_output=True,
                text=True,
                timeout=25,
                env=env,
            )
        else:
            # For node servers
            node_env = {**os.environ}
            code = (
                f"try {{ require('{server_path}'); console.log('OK'); }} "
                f"catch(e) {{ console.error('Import failed:', e.message); process.exit(1); }}"
            )
            result = subprocess.run(
                ["node", "-e", code],
                capture_output=True,
                text=True,
                timeout=25,
                env=node_env,
            )

        if result.returncode == 0:
            print(f"  ✅ {description} imports successfully")
            if result.stdout.strip():
                print(f"     Output: {result.stdout.strip()}")
            return True
        else:
            print(f"  ❌ {description} import failed (exit code {result.returncode})")
            if result.stderr:
                print(f"     Error: {result.stderr.strip()}")
            return False

    except subprocess.TimeoutExpired:
        print(f"  ⏰ {description} import timed out")
        return False
    except Exception as e:
        print(f"  ❌ {description} test failed: {e}")
        return False


def test_basic_execution(server_path: str, description: str) -> bool:
    """Test if a server can start without immediate crash."""
    print(f"\n🚀 Testing {description} basic execution...")

    try:
        if server_path.endswith(".py"):
            # For Python servers, test with environment that doesn't expect stdio
            cmd = [sys.executable, server_path]
            env = {
                **os.environ,
                "MCP_TRANSPORT": "test",
                "PYTHONPATH": str(Path(__file__).parent / "src"),
                "PYTHONIOENCODING": "utf-8",
            }
        else:
            # For node servers
            cmd = ["node", server_path]
            env = {**os.environ, "NODE_ENV": "test"}

        # Start process and let it run briefly
        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env
        )

        # Wait a bit to see if it crashes immediately
        time.sleep(2)

        if process.poll() is None:
            print(f"  ✅ {description} started successfully (still running)")
            process.terminate()
            process.wait(timeout=5)
            return True
        else:
            stdout, stderr = process.communicate()
            print(f"  ❌ {description} exited immediately (code {process.returncode})")
            if stdout:
                print(f"     Output: {stdout.strip()}")
            if stderr:
                print(f"     Error: {stderr.strip()}")
            return False

    except Exception as e:
        print(f"  ❌ {description} test failed: {e}")
        return False


def main():
    """Run diagnostic tests on all MCP servers."""
    print("🔧 MCP Server Diagnostic Tool")
    print("=" * 50)

    # Define servers to test
    repo_root = Path(__file__).parent
    servers = [
        {
            "path": str(repo_root / "mcp_server_wrapper.py"),
            "description": "Super Alita MCP Wrapper",
        },
        {
            "path": str(repo_root / "mcp_server" / "src" / "mcp_server" / "server.py"),
            "description": "Standalone MCP Server",
        },
    ]

    # Conditionally include Node-based server if Node.js is available
    if shutil.which("node"):
        servers.append(
            {
                "path": str(repo_root / "agentic-tools-mcp" / "dist" / "index.js"),
                "description": "Agentic Tools MCP Server",
            }
        )
    else:
        print("\nℹ️  Skipping Agentic Tools MCP Server (node not found)")

    results = {}

    for server in servers:
        server_path = server["path"]
        description = server["description"]

        # Check if server file exists
        if not Path(server_path).exists():
            print(f"\n❌ {description} not found at {server_path}")
            results[description] = False
            continue

        # Test import
        import_ok = test_server_import(server_path, description)

        # Test basic execution (only if import works)
        execution_ok = False
        if import_ok:
            execution_ok = test_basic_execution(server_path, description)

        results[description] = import_ok and execution_ok

    # Summary
    print("\n" + "=" * 50)
    print("📊 DIAGNOSTIC SUMMARY")
    print("=" * 50)

    working_servers = []
    broken_servers = []

    for description, is_working in results.items():
        if is_working:
            print(f"✅ {description}: WORKING")
            working_servers.append(description)
        else:
            print(f"❌ {description}: NEEDS ATTENTION")
            broken_servers.append(description)

    print(f"\n📈 Status: {len(working_servers)}/{len(servers)} servers working")

    if broken_servers:
        print(f"\n🔧 Servers needing fixes: {', '.join(broken_servers)}")
        print("\nNext steps:")
        print("1. Check import errors and missing dependencies")
        print("2. Verify server configuration and paths")
        print("3. Test VS Code MCP integration after fixes")
    else:
        print("\n🎉 All servers appear to be working!")
        print("If VS Code MCP still shows errors, check:")
        print("1. VS Code MCP extension is installed")
        print("2. .vscode/settings.json MCP configuration")
        print("3. VS Code logs for connection details")


if __name__ == "__main__":
    main()
