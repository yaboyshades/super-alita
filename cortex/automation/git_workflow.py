import re
import subprocess
from pathlib import Path
from typing import Any


class MergeConflictInfo:
    """Information about a merge conflict in a file."""

    def __init__(self, file_path: str, conflict_sections: list[dict[str, Any]]):
        self.file_path = file_path
        self.conflict_sections = conflict_sections


class GitAutomation:
    def create_feature_branch(self, feature_name: str) -> dict[str, Any]:
        branch = f"feature/{feature_name.replace(' ', '-').lower()}"
        res = subprocess.run(
            ["git", "checkout", "-b", branch], capture_output=True, text=True
        )
        return {
            "success": res.returncode == 0,
            "branch": branch,
            "output": res.stdout + res.stderr,
        }

    def auto_commit(
        self, message: str, files: list[str] | None = None
    ) -> dict[str, Any]:
        if files:
            for f in files:
                subprocess.run(["git", "add", f], capture_output=True)
        else:
            subprocess.run(["git", "add", "."], capture_output=True)
        res = subprocess.run(
            ["git", "commit", "-m", f"feat: {message}"], capture_output=True, text=True
        )
        return {"success": res.returncode == 0, "output": res.stdout + res.stderr}

    def detect_merge_conflicts(self) -> dict[str, Any]:
        """Detect merge conflicts in the repository."""
        # Check if we're in a merge state
        merge_head_path = Path(".git/MERGE_HEAD")
        if not merge_head_path.exists():
            return {"has_conflicts": False, "conflicts": []}

        # Get list of files with conflicts
        res = subprocess.run(
            ["git", "diff", "--name-only", "--diff-filter=U"],
            capture_output=True,
            text=True,
        )

        if res.returncode != 0:
            return {"error": f"Failed to detect conflicts: {res.stderr}"}

        conflict_files = res.stdout.strip().split("\n") if res.stdout.strip() else []

        conflicts = []
        for file_path in conflict_files:
            if file_path:  # Skip empty lines
                conflict_info = self._analyze_conflict_file(file_path)
                if conflict_info:
                    conflicts.append(conflict_info)

        return {
            "has_conflicts": len(conflicts) > 0,
            "conflicts": conflicts,
            "conflict_count": len(conflicts),
        }

    def _analyze_conflict_file(self, file_path: str) -> MergeConflictInfo | None:
        """Analyze a file with merge conflicts."""
        try:
            with open(file_path, encoding="utf-8") as f:
                content = f.read()
        except Exception:
            return None

        # Find conflict markers
        conflict_pattern = r"<<<<<<< .*?\n(.*?)\n=======\n(.*?)\n>>>>>>> .*?\n"
        conflicts = []

        for match in re.finditer(conflict_pattern, content, re.DOTALL):
            conflicts.append(
                {
                    "start_pos": match.start(),
                    "end_pos": match.end(),
                    "current_branch": match.group(1).strip(),
                    "incoming_branch": match.group(2).strip(),
                    "full_match": match.group(0),
                }
            )

        if conflicts:
            return MergeConflictInfo(file_path, conflicts)
        return None

    def resolve_simple_conflicts(self, strategy: str = "auto") -> dict[str, Any]:
        """Automatically resolve simple merge conflicts using various strategies."""
        conflict_info = self.detect_merge_conflicts()

        if not conflict_info.get("has_conflicts"):
            return {"success": True, "message": "No conflicts to resolve"}

        resolved_files = []
        failed_files = []

        for conflict in conflict_info["conflicts"]:
            try:
                if strategy == "auto":
                    success = self._auto_resolve_conflict(conflict)
                elif strategy == "current":
                    success = self._resolve_conflict_take_current(conflict)
                elif strategy == "incoming":
                    success = self._resolve_conflict_take_incoming(conflict)
                else:
                    success = False

                if success:
                    resolved_files.append(conflict.file_path)
                else:
                    failed_files.append(conflict.file_path)

            except Exception as e:
                failed_files.append(f"{conflict.file_path}: {str(e)}")

        # Stage resolved files
        for file_path in resolved_files:
            subprocess.run(["git", "add", file_path], capture_output=True)

        return {
            "success": len(failed_files) == 0,
            "resolved_files": resolved_files,
            "failed_files": failed_files,
            "total_conflicts": len(conflict_info["conflicts"]),
        }

    def _auto_resolve_conflict(self, conflict: MergeConflictInfo) -> bool:
        """Intelligently resolve conflicts based on content analysis."""
        try:
            with open(conflict.file_path, encoding="utf-8") as f:
                content = f.read()

            # Apply intelligent resolution strategies
            resolved_content = content

            for section in conflict.conflict_sections:
                current = section["current_branch"]
                incoming = section["incoming_branch"]

                # Strategy 1: If one side is empty, take the non-empty side
                if not current.strip() and incoming.strip():
                    resolution = incoming
                elif (
                    current.strip()
                    and not incoming.strip()
                    or current.strip() == incoming.strip()
                ):
                    resolution = current
                # Strategy 3: For imports, combine unique imports
                elif self._is_import_section(current, incoming):
                    resolution = self._merge_imports(current, incoming)
                # Strategy 4: For simple additions, combine both
                elif self._is_additive_change(current, incoming):
                    resolution = self._merge_additive_changes(current, incoming)
                else:
                    # Can't auto-resolve, return false
                    return False

                # Replace the conflict section with resolution
                resolved_content = resolved_content.replace(
                    section["full_match"], resolution + "\n"
                )

            # Write resolved content
            with open(conflict.file_path, "w", encoding="utf-8") as f:
                f.write(resolved_content)

            return True

        except Exception:
            return False

    def _resolve_conflict_take_current(self, conflict: MergeConflictInfo) -> bool:
        """Resolve conflicts by taking the current branch's version."""
        try:
            with open(conflict.file_path, encoding="utf-8") as f:
                content = f.read()

            for section in conflict.conflict_sections:
                resolution = section["current_branch"]
                content = content.replace(section["full_match"], resolution + "\n")

            with open(conflict.file_path, "w", encoding="utf-8") as f:
                f.write(content)

            return True
        except Exception:
            return False

    def _resolve_conflict_take_incoming(self, conflict: MergeConflictInfo) -> bool:
        """Resolve conflicts by taking the incoming branch's version."""
        try:
            with open(conflict.file_path, encoding="utf-8") as f:
                content = f.read()

            for section in conflict.conflict_sections:
                resolution = section["incoming_branch"]
                content = content.replace(section["full_match"], resolution + "\n")

            with open(conflict.file_path, "w", encoding="utf-8") as f:
                f.write(content)

            return True
        except Exception:
            return False

    def _is_import_section(self, current: str, incoming: str) -> bool:
        """Check if the conflict is in an import section."""
        import_keywords = ["import ", "from ", "#include", "using ", "require("]
        return any(
            keyword in current or keyword in incoming for keyword in import_keywords
        )

    def _is_additive_change(self, current: str, incoming: str) -> bool:
        """Check if the changes are purely additive (no overlapping modifications)."""
        current_lines = {
            line.strip() for line in current.split("\n") if line.strip()
        }
        incoming_lines = {
            line.strip() for line in incoming.split("\n") if line.strip()
        }

        # If there's no overlap, it's additive
        return len(current_lines.intersection(incoming_lines)) == 0

    def _merge_imports(self, current: str, incoming: str) -> str:
        """Merge import statements, removing duplicates."""
        current_lines = [line.strip() for line in current.split("\n") if line.strip()]
        incoming_lines = [line.strip() for line in incoming.split("\n") if line.strip()]

        # Combine and deduplicate
        all_imports = list(dict.fromkeys(current_lines + incoming_lines))
        return "\n".join(all_imports)

    def _merge_additive_changes(self, current: str, incoming: str) -> str:
        """Merge additive changes by combining both sides."""
        return current.rstrip() + "\n" + incoming.rstrip()

    def create_conflict_resolution_branch(
        self, base_branch: str, target_branch: str
    ) -> dict[str, Any]:
        """Create a new branch for conflict resolution."""
        resolution_branch = f"auto-resolve-{base_branch}-into-{target_branch}"

        # Create and checkout new branch
        res = subprocess.run(
            ["git", "checkout", "-b", resolution_branch, base_branch],
            capture_output=True,
            text=True,
        )

        if res.returncode != 0:
            return {"success": False, "error": f"Failed to create branch: {res.stderr}"}

        # Attempt merge
        merge_res = subprocess.run(
            ["git", "merge", target_branch, "--no-commit"],
            capture_output=True,
            text=True,
        )

        return {
            "success": res.returncode == 0,
            "branch": resolution_branch,
            "merge_attempted": True,
            "merge_success": merge_res.returncode == 0,
            "output": res.stdout + res.stderr + merge_res.stdout + merge_res.stderr,
        }
