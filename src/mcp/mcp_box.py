"""
Simple MCP Box storage for successful tool implementations.

This is a lightweight file-based storage system. GitHub Copilot uses its
existing tools (create_file, read_file, semantic_search) to interact with it.

Storage structure:
    .ai/mcp_box/
        metadata.json          # Index of all stored MCPs
        youtube_subtitles/     # Per-MCP directory
            spec.json          # Original MCP specification
            script.py          # Generated script
            environment.yml    # Conda environment
            requirements.txt   # Pip requirements
            usage_log.json     # Usage telemetry
"""

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class MCPMetadata:
    """Metadata for a stored MCP tool."""

    mcp_id: str
    name: str
    purpose: str
    created_at: str
    constitutional_score: float
    usage_count: int = 0
    last_used: str | None = None
    tags: list[str] = field(default_factory=list)
    success_rate: float = 1.0
    version: str = "1.0.0"


class MCPBox:
    """
    Lightweight MCP storage system.

    Copilot interacts with this using:
    - create_file: Store new MCPs
    - read_file: Retrieve MCPs
    - semantic_search: Find relevant MCPs
    - file_search: List available MCPs
    """

    def __init__(self, storage_path: str = ".ai/mcp_box"):
        """Initialize MCP Box storage."""
        self.storage_path = Path(storage_path)
        self.metadata_file = self.storage_path / "metadata.json"
        self._ensure_storage_exists()

    def _ensure_storage_exists(self) -> None:
        """Create storage directory if it doesn't exist."""
        self.storage_path.mkdir(parents=True, exist_ok=True)
        if not self.metadata_file.exists():
            self.metadata_file.write_text(
                json.dumps({"mcps": [], "version": "1.0.0"}, indent=2)
            )

    def store_mcp(
        self,
        mcp_id: str,
        name: str,
        purpose: str,
        spec: dict[str, Any],
        script: str,
        environment_yml: str,
        requirements_txt: str,
        constitutional_score: float,
        tags: list[str] | None = None,
    ) -> Path:
        """
        Store a new MCP tool.

        Returns:
            Path to the MCP directory
        """
        # Create MCP directory
        mcp_dir = self.storage_path / mcp_id
        mcp_dir.mkdir(exist_ok=True)

        # Write files
        (mcp_dir / "spec.json").write_text(
            json.dumps(spec, indent=2), encoding="utf-8"
        )
        (mcp_dir / "script.py").write_text(script, encoding="utf-8")
        (mcp_dir / "environment.yml").write_text(
            environment_yml, encoding="utf-8"
        )
        (mcp_dir / "requirements.txt").write_text(
            requirements_txt, encoding="utf-8"
        )

        # Initialize usage log
        usage_log = {
            "created_at": datetime.utcnow().isoformat(),
            "executions": [],
        }
        (mcp_dir / "usage_log.json").write_text(
            json.dumps(usage_log, indent=2), encoding="utf-8"
        )

        # Update metadata index
        metadata = MCPMetadata(
            mcp_id=mcp_id,
            name=name,
            purpose=purpose,
            created_at=datetime.utcnow().isoformat(),
            constitutional_score=constitutional_score,
            tags=tags or [],
        )

        self._add_to_index(metadata)

        return mcp_dir

    def _add_to_index(self, metadata: MCPMetadata) -> None:
        """Add MCP to metadata index."""
        index = json.loads(self.metadata_file.read_text())

        # Remove existing entry if updating
        index["mcps"] = [
            m for m in index["mcps"] if m["mcp_id"] != metadata.mcp_id
        ]

        # Add new entry
        index["mcps"].append(asdict(metadata))

        self.metadata_file.write_text(json.dumps(index, indent=2))

    def get_mcp(self, mcp_id: str) -> dict[str, Any] | None:
        """
        Retrieve an MCP by ID.

        Returns:
            Dictionary with all MCP files and metadata
        """
        mcp_dir = self.storage_path / mcp_id
        if not mcp_dir.exists():
            return None

        spec_file = mcp_dir / "spec.json"
        script_file = mcp_dir / "script.py"
        env_file = mcp_dir / "environment.yml"
        req_file = mcp_dir / "requirements.txt"
        log_file = mcp_dir / "usage_log.json"

        return {
            "mcp_id": mcp_id,
            "directory": str(mcp_dir),
            "spec": (
                json.loads(spec_file.read_text())
                if spec_file.exists()
                else None
            ),
            "script": (
                script_file.read_text() if script_file.exists() else None
            ),
            "environment_yml": (
                env_file.read_text() if env_file.exists() else None
            ),
            "requirements_txt": (
                req_file.read_text() if req_file.exists() else None
            ),
            "usage_log": (
                json.loads(log_file.read_text()) if log_file.exists() else None
            ),
        }

    def list_mcps(
        self, tag: str | None = None, min_score: float = 0.0
    ) -> list[MCPMetadata]:
        """
        List all stored MCPs with optional filtering.

        Args:
            tag: Filter by tag
            min_score: Minimum constitutional score

        Returns:
            List of MCP metadata
        """
        index = json.loads(self.metadata_file.read_text())
        mcps = [MCPMetadata(**m) for m in index["mcps"]]

        # Apply filters
        if tag:
            mcps = [m for m in mcps if tag in m.tags]
        if min_score > 0.0:
            mcps = [m for m in mcps if m.constitutional_score >= min_score]

        return mcps

    def search_by_purpose(self, query: str) -> list[MCPMetadata]:
        """
        Search MCPs by purpose description.

        Note: For semantic search, use Copilot's semantic_search tool
        against the metadata.json file.

        Args:
            query: Search query

        Returns:
            List of matching MCPs
        """
        query_lower = query.lower()
        all_mcps = self.list_mcps()

        return [
            m
            for m in all_mcps
            if query_lower in m.purpose.lower()
            or query_lower in m.name.lower()
        ]

    def record_usage(
        self, mcp_id: str, success: bool, execution_time_ms: float
    ) -> None:
        """Record MCP usage for telemetry."""
        mcp_dir = self.storage_path / mcp_id
        log_file = mcp_dir / "usage_log.json"

        if not log_file.exists():
            return

        usage_log = json.loads(log_file.read_text())
        usage_log["executions"].append(
            {
                "timestamp": datetime.utcnow().isoformat(),
                "success": success,
                "execution_time_ms": execution_time_ms,
            }
        )

        log_file.write_text(json.dumps(usage_log, indent=2))

        # Update metadata
        index = json.loads(self.metadata_file.read_text())
        for mcp in index["mcps"]:
            if mcp["mcp_id"] == mcp_id:
                mcp["usage_count"] += 1
                mcp["last_used"] = datetime.utcnow().isoformat()

                # Calculate success rate
                successes = sum(
                    1
                    for e in usage_log["executions"]
                    if e.get("success", False)
                )
                mcp["success_rate"] = successes / len(usage_log["executions"])
                break

        self.metadata_file.write_text(json.dumps(index, indent=2))

    def delete_mcp(self, mcp_id: str) -> bool:
        """Delete an MCP from storage."""
        mcp_dir = self.storage_path / mcp_id

        if not mcp_dir.exists():
            return False

        # Remove directory
        import shutil

        shutil.rmtree(mcp_dir)

        # Update index
        index = json.loads(self.metadata_file.read_text())
        index["mcps"] = [m for m in index["mcps"] if m["mcp_id"] != mcp_id]
        self.metadata_file.write_text(json.dumps(index, indent=2))

        return True

    def export_summary(self) -> dict[str, Any]:
        """Export summary statistics."""
        mcps = self.list_mcps()

        return {
            "total_mcps": len(mcps),
            "avg_constitutional_score": (
                sum(m.constitutional_score for m in mcps) / len(mcps)
                if mcps
                else 0.0
            ),
            "total_usage": sum(m.usage_count for m in mcps),
            "avg_success_rate": (
                sum(m.success_rate for m in mcps) / len(mcps) if mcps else 0.0
            ),
            "tags": list({tag for m in mcps for tag in m.tags}),
            "storage_path": str(self.storage_path),
        }


# Helper functions for Copilot to use via Python execution


def store_mcp_tool(
    mcp_id: str,
    name: str,
    purpose: str,
    spec: dict[str, Any],
    script: str,
    environment_yml: str,
    requirements_txt: str,
    constitutional_score: float,
    tags: list[str] | None = None,
) -> str:
    """
    Store an MCP tool (callable by Copilot via run_in_terminal).

    Example usage by Copilot:
        python -c "from src.mcp.mcp_box import store_mcp_tool; ..."
    """
    box = MCPBox()
    mcp_dir = box.store_mcp(
        mcp_id=mcp_id,
        name=name,
        purpose=purpose,
        spec=spec,
        script=script,
        environment_yml=environment_yml,
        requirements_txt=requirements_txt,
        constitutional_score=constitutional_score,
        tags=tags,
    )
    return f"MCP stored at: {mcp_dir}"


def list_mcp_tools(tag: str | None = None, min_score: float = 0.75) -> str:
    """
    List stored MCPs (callable by Copilot via run_in_terminal).

    Example usage by Copilot:
        python -c "from src.mcp.mcp_box import list_mcp_tools; print(list_mcp_tools())"
    """
    box = MCPBox()
    mcps = box.list_mcps(tag=tag, min_score=min_score)

    output = [f"📦 Found {len(mcps)} MCPs:\n"]
    for mcp in mcps:
        output.append(f"  • {mcp.name} (ID: {mcp.mcp_id})")
        output.append(f"    Purpose: {mcp.purpose}")
        output.append(
            f"    Score: {mcp.constitutional_score:.2f} | "
            f"Usage: {mcp.usage_count} | "
            f"Success: {mcp.success_rate:.0%}"
        )
        if mcp.tags:
            output.append(f"    Tags: {', '.join(mcp.tags)}")
        output.append("")

    return "\n".join(output)


def get_mcp_summary() -> str:
    """
    Get MCP Box summary statistics.

    Example usage by Copilot:
        python -c "from src.mcp.mcp_box import get_mcp_summary; print(get_mcp_summary())"
    """
    box = MCPBox()
    summary = box.export_summary()

    return f"""
📊 MCP Box Summary
==================
Total MCPs: {summary['total_mcps']}
Avg Constitutional Score: {summary['avg_constitutional_score']:.2f}
Total Usage: {summary['total_usage']}
Avg Success Rate: {summary['avg_success_rate']:.0%}
Tags: {', '.join(summary['tags']) if summary['tags'] else 'None'}
Storage: {summary['storage_path']}
"""
