"""GitHub CLI tool for Super Alita cognitive agent integration.

This tool provides GitHub CLI integration with dry-run support and
cognitive agent shadow mode operation.
"""

import shlex
import subprocess
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field

from src.core.schemas import AttentionLevel, GitHubEventSchema, GitHubEventType


class GitHubCliInput(BaseModel):
    """Input schema for GitHub CLI tool."""

    command: str = Field(..., description="GitHub CLI command to execute")
    dry_run: bool = Field(default=True, description="Execute in dry-run mode")
    repository: str | None = Field(
        default=None, description="Target repository (owner/repo)"
    )
    timeout: float = Field(
        default=30.0, description="Command timeout in seconds"
    )
    capture_output: bool = Field(
        default=True, description="Capture command output"
    )


class GitHubCliOutput(BaseModel):
    """Output schema for GitHub CLI tool."""

    success: bool = Field(..., description="Whether command succeeded")
    command: str = Field(..., description="Executed command")
    output: str = Field(default="", description="Command output")
    error: str | None = Field(
        default=None, description="Error message if failed"
    )
    dry_run: bool = Field(..., description="Whether executed in dry-run mode")
    execution_time: float = Field(..., description="Execution time in seconds")
    exit_code: int | None = Field(
        default=None, description="Command exit code"
    )
    github_event: GitHubEventSchema | None = Field(
        default=None, description="Generated GitHub event if applicable"
    )


class GitHubCliTool:
    """GitHub CLI tool with cognitive agent integration."""

    def __init__(self):
        self.name = "github_cli"
        self.description = (
            "Execute GitHub CLI commands with cognitive agent integration"
        )
        self.version = "1.0.0"
        self.tags = ["github", "cli", "git", "integration"]

        # Safe commands that can be executed without dry-run
        self.safe_commands = {
            "gh repo view",
            "gh issue list",
            "gh pr list",
            "gh workflow list",
            "gh release list",
            "gh api",
            "gh status",
            "gh auth status",
        }

        # Commands that should always be dry-run in cognitive agent mode
        self.cognitive_safe_commands = {
            "gh issue create",
            "gh pr create",
            "gh pr merge",
            "gh pr close",
            "gh issue close",
            "gh release create",
            "gh workflow run",
        }

    async def execute(self, input_data: GitHubCliInput) -> GitHubCliOutput:
        """Execute GitHub CLI command with cognitive agent support."""
        start_time = datetime.now(UTC).timestamp()

        try:
            # Validate and prepare command
            validated_command = self._validate_command(input_data.command)
            if not validated_command:
                return GitHubCliOutput(
                    success=False,
                    command=input_data.command,
                    error="Invalid or unsafe GitHub CLI command",
                    dry_run=input_data.dry_run,
                    execution_time=0.0,
                )

            # Execute command
            if input_data.dry_run or self._requires_dry_run(
                input_data.command
            ):
                result = await self._execute_dry_run(
                    validated_command, input_data
                )
            else:
                result = await self._execute_command(
                    validated_command, input_data
                )

            # Generate GitHub event if applicable
            github_event = self._generate_github_event(input_data, result)
            if github_event:
                result.github_event = github_event

            result.execution_time = datetime.now(UTC).timestamp() - start_time
            return result

        except Exception as e:
            execution_time = datetime.now(UTC).timestamp() - start_time
            return GitHubCliOutput(
                success=False,
                command=input_data.command,
                error=f"GitHub CLI tool execution error: {str(e)}",
                dry_run=input_data.dry_run,
                execution_time=execution_time,
            )

    def _validate_command(self, command: str) -> str | None:
        """Validate GitHub CLI command for security and format."""
        command = command.strip()

        # Must start with 'gh'
        if not command.startswith("gh "):
            return None

        # Basic security validation - no shell injection patterns
        dangerous_patterns = [";", "&&", "||", "|", ">", "<", "`", "$"]
        for pattern in dangerous_patterns:
            if pattern in command:
                return None

        return command

    def _requires_dry_run(self, command: str) -> bool:
        """Check if command requires dry-run mode."""
        # Commands that modify state should use dry-run
        modifying_actions = [
            "create",
            "merge",
            "close",
            "delete",
            "update",
            "run",
        ]

        return any(action in command for action in modifying_actions)

    async def _execute_dry_run(
        self, command: str, input_data: GitHubCliInput
    ) -> GitHubCliOutput:
        """Execute command in dry-run mode."""

        # Simulate command execution based on command type
        if "issue create" in command:
            return GitHubCliOutput(
                success=True,
                command=command,
                output="Would create issue: [DRY RUN] Issue creation simulated",
                dry_run=True,
                execution_time=0.0,
                exit_code=0,
            )
        elif "pr create" in command:
            return GitHubCliOutput(
                success=True,
                command=command,
                output="Would create PR: [DRY RUN] Pull request creation simulated",
                dry_run=True,
                execution_time=0.0,
                exit_code=0,
            )
        elif "pr merge" in command:
            return GitHubCliOutput(
                success=True,
                command=command,
                output="Would merge PR: [DRY RUN] Pull request merge simulated",
                dry_run=True,
                execution_time=0.0,
                exit_code=0,
            )
        else:
            return GitHubCliOutput(
                success=True,
                command=command,
                output=f"Would execute: {command} [DRY RUN]",
                dry_run=True,
                execution_time=0.0,
                exit_code=0,
            )

    async def _execute_command(
        self, command: str, input_data: GitHubCliInput
    ) -> GitHubCliOutput:
        """Execute actual GitHub CLI command."""

        try:
            # Split command safely
            cmd_parts = shlex.split(command)

            # Execute with subprocess
            process = subprocess.run(
                cmd_parts,
                capture_output=input_data.capture_output,
                text=True,
                timeout=input_data.timeout,
                cwd=None,  # Use current directory
            )

            return GitHubCliOutput(
                success=process.returncode == 0,
                command=command,
                output=process.stdout or "",
                error=process.stderr if process.returncode != 0 else None,
                dry_run=False,
                exit_code=process.returncode,
            )

        except subprocess.TimeoutExpired:
            return GitHubCliOutput(
                success=False,
                command=command,
                error=f"Command timed out after {input_data.timeout} seconds",
                dry_run=False,
                exit_code=-1,
            )
        except subprocess.CalledProcessError as e:
            return GitHubCliOutput(
                success=False,
                command=command,
                error=f"Command failed with exit code {e.returncode}: {e.stderr}",
                dry_run=False,
                exit_code=e.returncode,
            )

    def _generate_github_event(
        self, input_data: GitHubCliInput, result: GitHubCliOutput
    ) -> GitHubEventSchema | None:
        """Generate GitHub event based on command execution."""

        if not result.success:
            return None

        # Extract event type from command
        event_type = None
        if "issue create" in input_data.command:
            event_type = GitHubEventType.ISSUE_CREATED
        elif "pr create" in input_data.command:
            event_type = GitHubEventType.PR_OPENED
        elif "pr merge" in input_data.command:
            event_type = GitHubEventType.PR_MERGED

        if not event_type:
            return None

        return GitHubEventSchema(
            event_type=event_type,
            repository=input_data.repository or "unknown/unknown",
            actor="cognitive-agent",
            payload={
                "command": input_data.command,
                "dry_run": result.dry_run,
                "output": result.output,
                "execution_time": result.execution_time,
            },
            event_id=f"cli-{datetime.now(UTC).isoformat()}",
            attention_level=AttentionLevel.MEDIUM,
            processing_status="generated",
        )

    def get_input_schema(self) -> dict[str, Any]:
        """Get tool input schema."""
        return GitHubCliInput.model_json_schema()

    def get_output_schema(self) -> dict[str, Any]:
        """Get tool output schema."""
        return GitHubCliOutput.model_json_schema()

    def get_metadata(self) -> dict[str, Any]:
        """Get tool metadata."""
        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "tags": self.tags,
            "input_schema": self.get_input_schema(),
            "output_schema": self.get_output_schema(),
            "safe_commands": list(self.safe_commands),
            "cognitive_safe_commands": list(self.cognitive_safe_commands),
        }
