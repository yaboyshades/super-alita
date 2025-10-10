"""
DeepCode integration for VS Code extension
"""

import asyncio
import logging
from pathlib import Path
from typing import Any

from .analyzer_simple import AnalysisLevel, create_deepcode_engine

logger = logging.getLogger(__name__)


class ExtensionDeepCodeIntegration:
    """Integration class for VS Code extension"""

    def __init__(self):
        self.engine = create_deepcode_engine()
        self.enabled = True

    async def analyze_current_file(self, file_path: str) -> dict[str, Any]:
        """Analyze the currently active file"""
        if not self.enabled:
            return {"enabled": False}

        try:
            result = await self.engine.analyze_file(
                file_path, AnalysisLevel.SEMANTIC
            )
            return {
                "enabled": True,
                "file_path": file_path,
                "issues_count": len(result.issues),
                "quality_score": self._calculate_file_quality_score(result),
                "issues": [
                    issue.to_dict() for issue in result.issues[:10]
                ],  # Limit for performance
                "metrics": result.metrics,
                "execution_time": result.execution_time,
            }
        except Exception as e:
            logger.exception(f"DeepCode analysis failed for {file_path}: {e}")
            return {"enabled": True, "error": str(e), "file_path": file_path}

    async def analyze_workspace_sample(
        self, workspace_path: str, max_files: int = 20
    ) -> dict[str, Any]:
        """Analyze a sample of files from the workspace"""
        if not self.enabled:
            return {"enabled": False}

        try:
            # Get Python files (limited sample)
            workspace = Path(workspace_path)
            python_files = list(workspace.rglob("*.py"))[:max_files]

            if not python_files:
                return {
                    "enabled": True,
                    "message": "No Python files found in workspace",
                    "workspace_path": workspace_path,
                }

            # Analyze files concurrently
            results = {}
            semaphore = asyncio.Semaphore(3)  # Limit concurrency

            async def analyze_file(file_path: Path) -> tuple[str, Any]:
                async with semaphore:
                    result = await self.engine.analyze_file(
                        str(file_path), AnalysisLevel.SEMANTIC
                    )
                    return str(file_path), result

            tasks = [analyze_file(fp) for fp in python_files]
            completed = await asyncio.gather(*tasks, return_exceptions=True)

            for item in completed:
                if isinstance(item, Exception):
                    logger.error(f"Analysis failed: {item}")
                else:
                    file_path, result = item
                    results[file_path] = result

            # Generate summary report
            report = self.engine.generate_report(results)

            return {
                "enabled": True,
                "workspace_path": workspace_path,
                "files_analyzed": len(results),
                "total_issues": report["summary"]["total_issues"],
                "quality_score": report["summary"]["quality_score"],
                "severity_breakdown": report["summary"]["severity_breakdown"],
                "top_categories": dict(
                    list(report["summary"]["category_breakdown"].items())[:5]
                ),
                "recommendations": report["recommendations"][
                    :3
                ],  # Top 3 recommendations
                "metrics": report["metrics"],
            }

        except Exception as e:
            logger.exception(f"DeepCode workspace analysis failed: {e}")
            return {
                "enabled": True,
                "error": str(e),
                "workspace_path": workspace_path,
            }

    def _calculate_file_quality_score(self, result) -> float:
        """Calculate quality score for a single file"""
        if not result.issues:
            return 100.0

        score = 100.0
        for issue in result.issues:
            if issue.severity.value == "critical":
                score -= 20
            elif issue.severity.value == "error":
                score -= 10
            elif issue.severity.value == "warning":
                score -= 5
            else:
                score -= 1

        return max(0.0, score)

    def set_enabled(self, enabled: bool) -> None:
        """Enable or disable DeepCode analysis"""
        self.enabled = enabled

    def get_supported_extensions(self) -> list[str]:
        """Get list of supported file extensions"""
        extensions = set()
        for analyzer in self.engine.analyzers.values():
            extensions.update(analyzer.get_supported_languages())
        return list(extensions)


# Global instance for the extension
_deepcode_integration = None


def get_deepcode_integration() -> ExtensionDeepCodeIntegration:
    """Get the global DeepCode integration instance"""
    global _deepcode_integration
    if _deepcode_integration is None:
        _deepcode_integration = ExtensionDeepCodeIntegration()
    return _deepcode_integration


# Convenience functions for extension commands
async def analyze_current_file(file_path: str) -> dict[str, Any]:
    """Analyze the current file (convenience function)"""
    integration = get_deepcode_integration()
    return await integration.analyze_current_file(file_path)


async def analyze_workspace_sample(workspace_path: str) -> dict[str, Any]:
    """Analyze workspace sample (convenience function)"""
    integration = get_deepcode_integration()
    return await integration.analyze_workspace_sample(workspace_path)


def is_supported_file(file_path: str) -> bool:
    """Check if file type is supported by DeepCode"""
    integration = get_deepcode_integration()
    path = Path(file_path)
    return path.suffix in integration.get_supported_extensions()
