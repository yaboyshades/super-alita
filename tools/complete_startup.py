#!/usr/bin/env python3
"""
Super Alita Complete Startup Script
Orchestrates: Ollama → GPT-OSS:20B → Super Alita → Enhanced Consensus

Follows repository development instructions:
- Installs dependencies with uv/pip and constraints
- Starts Ollama (if not running) and ensures gpt-oss:20b
- Boots Super Alita (uvicorn app:app on port 8080)
- Validates deployment and enhanced consensus tool
- Provides an interactive session (REUG tool-based streaming)
"""

from __future__ import annotations

import atexit
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import requests

# Repo utility for subprocess (no shell=True)
try:
    from src.core import proc
except Exception:  # pragma: no cover - fallback if proc not importable
    proc = None  # type: ignore


class SuperAlitaOrchestrator:
    """Complete system orchestrator following development instructions."""

    def __init__(self) -> None:
        self.processes: list[subprocess.Popen] = []
        self.base_dir = Path.cwd()
        self.timeout = 300  # 5 minutes

        # Register cleanup handler
        atexit.register(self.cleanup_all)
        with contextlib_suppress():
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)

        # Helpful defaults for local dev if not set
        os.environ.setdefault("OLLAMA_HOST", "http://127.0.0.1:11434")

    def _signal_handler(self, signum, frame) -> None:  # type: ignore[no-untyped-def]
        print(f"\n🛑 Received signal {signum}, shutting down...")
        self.cleanup_all()
        sys.exit(0)

    def cleanup_all(self) -> None:
        print("🧹 Cleaning up processes...")
        for process in self.processes:
            try:
                if process.poll() is None:
                    process.terminate()
                    process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                with contextlib_suppress():
                    process.kill()
            except Exception:
                pass
        self.processes.clear()

    def check_prerequisites(self) -> bool:
        print("🔍 Checking Prerequisites...")

        # Ensure .env exists (copy from example if available)
        env_path = self.base_dir / ".env"
        env_example = self.base_dir / ".env.example"
        if not env_path.exists() and env_example.exists():
            print("📝 Creating .env from template...")
            try:
                shutil.copy(env_example, env_path)
            except Exception as e:
                print(f"⚠️  Failed to create .env from template: {e}")

        # Key files per repo layout
        required_files = [
            "app.py",
            "src/main.py",
            "src/reug_runtime/router.py",
            "src/abilities/enhanced_consensus_ability.py",
            "validate_deployment.py",
        ]
        missing = [p for p in required_files if not (self.base_dir / p).exists()]
        if missing:
            print("❌ Required files missing:")
            for m in missing:
                print(f"   - {m}")
            return False

        print("✅ Prerequisites check passed")
        return True

    def _run(self, argv: list[str], *, timeout: int | None = None) -> tuple[int, str, str]:
        """Run a command with repo proc util when available; returns rc, out, err."""
        if proc is not None:
            try:
                out = proc.run(argv, timeout=float(timeout) if timeout else None)
                return 0, out, ""
            except Exception as e:  # includes ProcError
                # Best-effort extract stdout/stderr from ProcError
                rc = getattr(e, "returncode", 1)
                out = getattr(e, "stdout", "")
                err = getattr(e, "stderr", str(e))
                return rc, out, err
        # Fallback to subprocess.run
        p = subprocess.run(
            argv,
            timeout=timeout,
            capture_output=True,
            text=True,
            cwd=self.base_dir,
        )
        return p.returncode, p.stdout, p.stderr

    def install_dependencies(self) -> bool:
        print("📦 Installing Dependencies (5 min timeout)...")

        uv = shutil.which("uv")
        has_constraints = (self.base_dir / "constraints.txt").exists()
        has_req_test = (self.base_dir / "requirements-test.txt").exists()

        try:
            if uv:
                args = [uv, "pip", "install", "-r", "requirements.txt"]
                if has_constraints:
                    args.extend(["-c", "constraints.txt"])
                rc, out, err = self._run(args, timeout=self.timeout)
                if rc != 0:
                    print(f"❌ uv install failed: {err or out}")
                    return False
                if has_req_test:
                    args = [uv, "pip", "install", "-r", "requirements-test.txt"]
                    if has_constraints:
                        args.extend(["-c", "constraints.txt"])
                    rc, out, err = self._run(args, timeout=self.timeout)
                    if rc != 0:
                        print(f"❌ uv test-deps install failed: {err or out}")
                        return False
            else:
                python_exe = sys.executable
                args = [python_exe, "-m", "pip", "install", "-r", "requirements.txt"]
                if has_constraints:
                    args.extend(["-c", "constraints.txt"])
                rc, out, err = self._run(args, timeout=self.timeout)
                if rc != 0:
                    print(f"❌ pip install failed: {err or out}")
                    return False
                if has_req_test:
                    args = [
                        python_exe,
                        "-m",
                        "pip",
                        "install",
                        "-r",
                        "requirements-test.txt",
                    ]
                    if has_constraints:
                        args.extend(["-c", "constraints.txt"])
                    rc, out, err = self._run(args, timeout=self.timeout)
                    if rc != 0:
                        print(f"❌ pip test-deps install failed: {err or out}")
                        return False

            print("✅ Dependencies installed successfully")
            return True
        except Exception as e:
            print(f"❌ Dependency installation error: {e}")
            return False

    def start_ollama_serve(self) -> bool:
        print("🚀 Starting Ollama Server...")
        base = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
        try:
            r = requests.get(f"{base}/api/tags", timeout=2)
            if r.status_code == 200:
                print("✅ Ollama server already running")
                return True
        except requests.RequestException:
            pass

        try:
            process = subprocess.Popen(
                ["ollama", "serve"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            self.processes.append(process)
            for i in range(30):
                try:
                    r = requests.get(f"{base}/api/tags", timeout=2)
                    if r.status_code == 200:
                        print("✅ Ollama server started successfully")
                        return True
                except requests.RequestException:
                    pass
                time.sleep(1)
            print("❌ Ollama server failed to start within 30 seconds")
            return False
        except FileNotFoundError:
            print("❌ Ollama not found. Please install Ollama first")
            return False
        except Exception as e:
            print(f"❌ Failed to start Ollama: {e}")
            return False

    def ensure_gpt_oss_model(self) -> bool:
        print("🤖 Checking GPT-OSS:20B Model...")
        base = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
        try:
            r = requests.get(f"{base}/api/tags", timeout=10)
            if r.status_code != 200:
                print("❌ Cannot connect to Ollama API")
                return False
            models = r.json().get("models", [])
            for m in models:
                if "gpt-oss:20b" in (m.get("name", "").lower()):
                    print("✅ GPT-OSS:20B model found")
                    return True
            print("📥 GPT-OSS:20B not found, pulling model (this can take time)...")
            pull = subprocess.Popen(
                ["ollama", "pull", "gpt-oss:20b"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            self.processes.append(pull)
            while True:
                if pull.stdout is None:
                    break
                line = pull.stdout.readline()
                if not line and pull.poll() is not None:
                    break
                if line:
                    print(f"   {line.strip()}")
            if pull.returncode == 0:
                print("✅ GPT-OSS:20B model pulled successfully")
                return True
            print("❌ Failed to pull GPT-OSS:20B model")
            return False
        except Exception as e:
            print(f"❌ Model check error: {e}")
            return False

    def test_gpt_oss_model(self) -> bool:
        print("🧪 Testing GPT-OSS:20B Model...")
        base = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
        try:
            r = requests.post(
                f"{base}/v1/chat/completions",
                json={
                    "model": "gpt-oss:20b",
                    "messages": [
                        {
                            "role": "user",
                            "content": "Say 'Hello from GPT-OSS' in exactly those words.",
                        }
                    ],
                    "max_tokens": 10,
                    "temperature": 0.3,
                },
                timeout=30,
            )
            if r.status_code != 200:
                print(f"❌ GPT-OSS:20B test failed: HTTP {r.status_code}")
                return False
            data = r.json()
            choices = data.get("choices", [])
            if not choices:
                print("❌ No choices from GPT-OSS:20B")
                return False
            content = choices[0].get("message", {}).get("content", "").strip()
            print(f"✅ GPT-OSS:20B response: {content}")
            return True
        except Exception as e:
            print(f"❌ GPT-OSS:20B test error: {e}")
            return False

    def start_super_alita(self) -> bool:
        print("🚀 Starting Super Alita Server...")
        python_exe = sys.executable
        cmd = [
            python_exe,
            "-m",
            "uvicorn",
            "app:app",
            "--host",
            "127.0.0.1",
            "--port",
            "8080",
            "--reload",
            "--timeout-keep-alive",
            "120",
            "--limit-concurrency",
            "100",
        ]
        try:
            p = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=self.base_dir,
            )
            self.processes.append(p)
            print("   Waiting for Super Alita startup...")
            for i in range(60):
                try:
                    r = requests.get("http://127.0.0.1:8080/healthz", timeout=2)
                    if r.status_code == 200:
                        print("✅ Super Alita started! Health: ok")
                        return True
                except requests.RequestException:
                    pass
                time.sleep(1)
                if i % 10 == 9:
                    print(f"   Still waiting... ({i+1}/60)")
            print("❌ Super Alita failed to start within 60 seconds")
            return False
        except Exception as e:
            print(f"❌ Failed to start Super Alita: {e}")
            return False

    def validate_system(self) -> bool:
        print("🔍 Running System Validation...")
        rc, out, err = self._run([sys.executable, "validate_deployment.py"], timeout=60)
        if rc == 0:
            print("✅ System validation passed")
            if "ALL TESTS PASSED" in out.upper():
                print("🎉 ALL TESTS PASSED - Super Alita is ready for deployment!")
            return True
        print("❌ System validation failed:")
        if out:
            print(out)
        if err:
            print(err)
        return False

    def test_enhanced_consensus(self) -> bool:
        print("🧠 Testing Enhanced Consensus Integration...")
        base_url = "http://127.0.0.1:8080"
        try:
            r = requests.get(f"{base_url}/tools/catalog", timeout=10)
            if r.status_code != 200:
                print(f"❌ Tools catalog failed: HTTP {r.status_code}")
                return False
            tools = r.json()
            tool_names = [t.get("name") for t in tools]
            if "deepconf_consensus" in tool_names:
                print("✅ Enhanced consensus tool registered")
            else:
                print(f"⚠️  Enhanced consensus tool not found in: {tool_names}")
                return False

            payload = {
                "prompt": "What is the capital of France?",
                "method": "simple_vote",
                "num_samples": 2,
                "temperature": 0.4,
                "max_tokens": 50,
            }
            r = requests.post(
                f"{base_url}/ability/execute/deepconf_consensus", json=payload, timeout=60
            )
            if r.status_code != 200:
                print(f"❌ Consensus test failed: HTTP {r.status_code}")
                print(f"   Response: {r.text}")
                return False
            data = r.json()
            print("✅ Consensus test successful:")
            print(f"   Response: {data.get('consensus_text', '')}")
            print(f"   Confidence: {data.get('consensus_confidence', 0)}")
            return True
        except Exception as e:
            print(f"❌ Consensus test error: {e}")
            return False

    def interactive_session(self) -> None:
        print("\n" + "=" * 60)
        print("🎉 Super Alita Ready for User Input!")
        print("=" * 60)
        print("Available endpoints:")
        print("  • Health: http://127.0.0.1:8080/healthz")
        print("  • Tools: http://127.0.0.1:8080/tools/catalog")
        print("  • SSE Chat: POST http://127.0.0.1:8080/v1/chat/stream (SSE)")
        print("  • Tool Chat: POST http://127.0.0.1:8080/tools/reug_start_turn")
        print("  • Stream Next: POST http://127.0.0.1:8080/tools/reug_stream_next")
        print("\nDirect model access:")
        print("  • Ollama: http://127.0.0.1:11434/v1/chat/completions")
        print("\nType 'exit' to shutdown all services")
        print("=" * 60)

        while True:
            try:
                user_input = input("\n💬 Enter your message (or 'exit'): ").strip()
                if user_input.lower() in {"exit", "quit", "bye"}:
                    print("👋 Shutting down all services...")
                    break
                if not user_input:
                    continue

                print("🔄 Starting REUG tool-based turn...")
                start = requests.post(
                    "http://127.0.0.1:8080/tools/reug_start_turn",
                    json={"message": user_input, "session_id": f"interactive_{int(time.time())}"},
                    timeout=15,
                )
                if start.status_code != 200:
                    print(f"⚠️  Start failed: HTTP {start.status_code}")
                    continue
                run_id = start.json().get("run_id")
                print(f"✅ Started run: {run_id}")

                finished = False
                total_chunks = 0
                while not finished:
                    step = requests.post(
                        "http://127.0.0.1:8080/tools/reug_stream_next",
                        json={"run_id": run_id},
                        timeout=20,
                    )
                    if step.status_code != 200:
                        print(f"⚠️  Stream error: HTTP {step.status_code}")
                        break
                    payload = step.json()
                    chunks = payload.get("chunks", [])
                    finished = bool(payload.get("finished", False))
                    for c in chunks:
                        total_chunks += 1
                        out = str(c)
                        print(f"   {total_chunks:02d}: {out[:200] + ('…' if len(out) > 200 else '')}")
                    if not finished and not chunks:
                        time.sleep(0.3)
            except KeyboardInterrupt:
                print("\n👋 Shutting down...")
                break
            except Exception as e:
                print(f"❌ Error processing input: {e}")

    def run_complete_startup(self) -> int:
        print("🚀 Super Alita Complete Startup")
        print("Following Super Alita Development Instructions")
        print("=" * 60)

        steps: list[tuple[str, callable[[], bool]]] = [
            ("Prerequisites", self.check_prerequisites),
            ("Dependencies", self.install_dependencies),
            ("Ollama Server", self.start_ollama_serve),
            ("GPT-OSS:20B Model", self.ensure_gpt_oss_model),
            ("Model Test", self.test_gpt_oss_model),
            ("Super Alita Server", self.start_super_alita),
            ("System Validation", self.validate_system),
            ("Enhanced Consensus", self.test_enhanced_consensus),
        ]

        for name, fn in steps:
            print(f"\n🔄 Step: {name}")
            if not fn():
                print(f"❌ Failed at step: {name}")
                self.cleanup_all()
                return 1

        print("\n✅ All startup steps completed successfully!")
        self.interactive_session()
        self.cleanup_all()
        return 0


class contextlib_suppress:
    """Small helper to avoid importing contextlib for tiny suppression needs."""

    def __init__(self, *exceptions: type[BaseException]) -> None:  # pragma: no cover
        self.exceptions = exceptions or (Exception,)

    def __enter__(self) -> None:  # pragma: no cover
        return None

    def __exit__(self, exc_type, exc, tb) -> bool:  # pragma: no cover
        return exc is not None and issubclass(exc_type, self.exceptions)


def main() -> int:
    orch = SuperAlitaOrchestrator()
    try:
        return orch.run_complete_startup()
    except KeyboardInterrupt:
        print("\n👋 Startup interrupted by user")
        return 0
    except Exception as e:
        print(f"❌ Startup failed: {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

