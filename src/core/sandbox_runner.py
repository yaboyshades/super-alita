import asyncio
import logging
import subprocess
import tempfile
import uuid

logger = logging.getLogger(__name__)


class SandboxRunner:
    """Safely execute code in isolated Docker containers."""

    async def run(self, script: str) -> tuple[bool, str]:
        """
        Execute Python script in a Docker sandbox.

        Args:
            script: Python code to execute

        Returns:
            Tuple of (success: bool, output: str)
        """
        logger.info(f"🏃 Running code in sandbox (length: {len(script)} chars)")

        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
                f.write(script)
                f.flush()

                # Create unique container name
                container_name = f"alita-sandbox-{uuid.uuid4().hex[:8]}"

                # Docker command to run Python script
                cmd = [
                    "docker",
                    "run",
                    "--rm",
                    "-i",
                    "--name",
                    container_name,
                    "--network",
                    "none",  # No network access for security
                    "--memory",
                    "128m",  # Limit memory
                    "--cpus",
                    "0.5",  # Limit CPU
                    "python:3.11-slim",
                    "python",
                    "-c",
                    script,
                ]

                logger.debug(
                    f"🐳 Docker command: {' '.join(cmd[:6])}... (script omitted)"
                )

                # Execute with timeout
                proc = await asyncio.wait_for(
                    asyncio.create_subprocess_exec(
                        *cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
                    ),
                    timeout=30.0,  # 30 second timeout
                )

                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(),
                    timeout=25.0,  # Slightly less than above
                )

                success = proc.returncode == 0
                output = (stdout if success else stderr).decode().strip()

                logger.info(
                    f"✅ Sandbox execution {'succeeded' if success else 'failed'}"
                )
                if not success:
                    logger.warning(f"❌ Sandbox error: {output[:200]}...")

                return success, output

        except TimeoutError:
            logger.error("⏱️ Sandbox execution timed out")
            return False, "Execution timed out after 30 seconds"

        except Exception as e:
            logger.error(f"❌ Sandbox execution failed: {e}")
            return False, f"Sandbox error: {e!s}"
