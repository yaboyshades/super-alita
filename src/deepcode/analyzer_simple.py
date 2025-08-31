"""
DeepCode Integration for Advanced Code Analysis and Intelligence
"""

import ast
import asyncio
import logging
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class AnalysisLevel(Enum):
    """Analysis depth levels"""

    SYNTAX = "syntax"
    SEMANTIC = "semantic"
    SECURITY = "security"
    PERFORMANCE = "performance"
    ARCHITECTURE = "architecture"
    DEEP = "deep"


class SeverityLevel(Enum):
    """Issue severity levels"""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class CodeIssue:
    """Represents a code issue found during analysis"""

    id: str
    severity: SeverityLevel
    category: str
    message: str
    file_path: str
    line_number: int
    column_number: int = 0
    suggestions: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "id": self.id,
            "severity": self.severity.value,
            "category": self.category,
            "message": self.message,
            "file_path": self.file_path,
            "line_number": self.line_number,
            "column_number": self.column_number,
            "suggestions": self.suggestions,
            "metadata": self.metadata,
            "confidence": self.confidence,
        }


@dataclass
class AnalysisResult:
    """Results from code analysis"""

    issues: list[CodeIssue] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    suggestions: list[str] = field(default_factory=list)
    execution_time: float = 0.0
    analysis_level: AnalysisLevel = AnalysisLevel.SYNTAX

    def add_issue(self, issue: CodeIssue) -> None:
        """Add an issue to the results"""
        self.issues.append(issue)

    def get_issues_by_severity(self, severity: SeverityLevel) -> list[CodeIssue]:
        """Get issues filtered by severity"""
        return [issue for issue in self.issues if issue.severity == severity]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "issues": [issue.to_dict() for issue in self.issues],
            "metrics": self.metrics,
            "suggestions": self.suggestions,
            "execution_time": self.execution_time,
            "analysis_level": self.analysis_level.value,
        }


class CodeAnalyzer(ABC):
    """Abstract base class for code analyzers"""

    @abstractmethod
    async def analyze(
        self, file_path: str, content: str, level: AnalysisLevel
    ) -> AnalysisResult:
        """Analyze code and return results"""
        pass

    @abstractmethod
    def get_supported_languages(self) -> set[str]:
        """Get set of supported language file extensions"""
        pass


class PythonDeepAnalyzer(CodeAnalyzer):
    """Deep analysis for Python code"""

    def __init__(self):
        self.patterns = self._load_patterns()

    def _load_patterns(self) -> dict[str, list[dict[str, Any]]]:
        """Load analysis patterns and rules"""
        return {
            "security": [
                {
                    "pattern": r"eval\s*\(",
                    "message": "Use of eval() is dangerous and can lead to code injection",
                    "severity": SeverityLevel.CRITICAL,
                    "category": "security.code-injection",
                },
                {
                    "pattern": r"exec\s*\(",
                    "message": "Use of exec() can be dangerous if user input is involved",
                    "severity": SeverityLevel.WARNING,
                    "category": "security.code-execution",
                },
            ],
            "performance": [
                {
                    "pattern": r"for\s+\w+\s+in\s+range\s*\(\s*len\s*\(",
                    "message": "Use enumerate() instead of range(len()) for better performance",
                    "severity": SeverityLevel.INFO,
                    "category": "performance.iteration",
                }
            ],
        }

    async def analyze(
        self, file_path: str, content: str, level: AnalysisLevel
    ) -> AnalysisResult:
        """Perform deep analysis on Python code"""
        start_time = time.time()
        result = AnalysisResult(analysis_level=level)

        try:
            # Parse AST for semantic analysis
            tree = ast.parse(content)

            # Run different analysis levels
            if level in [
                AnalysisLevel.SYNTAX,
                AnalysisLevel.SEMANTIC,
                AnalysisLevel.DEEP,
            ]:
                await self._analyze_ast(tree, file_path, result)

            if level in [AnalysisLevel.SECURITY, AnalysisLevel.DEEP]:
                await self._analyze_security(content, file_path, result)

            # Calculate metrics
            result.metrics = self._calculate_metrics(tree, content)

        except SyntaxError as e:
            result.add_issue(
                CodeIssue(
                    id="syntax-error",
                    severity=SeverityLevel.ERROR,
                    category="syntax",
                    message=f"Syntax error: {e.msg}",
                    file_path=file_path,
                    line_number=e.lineno or 0,
                    column_number=e.offset or 0,
                )
            )
        except Exception as e:
            logger.exception(f"Analysis failed for {file_path}: {e}")
            result.add_issue(
                CodeIssue(
                    id="analysis-error",
                    severity=SeverityLevel.ERROR,
                    category="internal",
                    message=f"Analysis failed: {str(e)}",
                    file_path=file_path,
                    line_number=0,
                )
            )

        result.execution_time = time.time() - start_time
        return result

    async def _analyze_ast(
        self, tree: ast.AST, file_path: str, result: AnalysisResult
    ) -> None:
        """Analyze AST for semantic issues"""

        class SemanticAnalyzer(ast.NodeVisitor):
            def __init__(self, result: AnalysisResult, file_path: str):
                self.result = result
                self.file_path = file_path

            def visit_FunctionDef(self, node):
                # Check function complexity
                complexity = self._calculate_function_complexity(node)
                if complexity > 10:
                    self.result.add_issue(
                        CodeIssue(
                            id=f"high-complexity-{node.name}",
                            severity=SeverityLevel.WARNING,
                            category="architecture.complexity",
                            message=f"Function '{node.name}' has high complexity ({complexity})",
                            file_path=self.file_path,
                            line_number=node.lineno,
                            suggestions=[
                                "Consider breaking this function into smaller functions"
                            ],
                        )
                    )

                self.generic_visit(node)

            def _calculate_function_complexity(self, node) -> int:
                """Calculate cyclomatic complexity"""
                complexity = 1  # Base complexity

                for child in ast.walk(node):
                    if (
                        isinstance(child, ast.If | ast.While | ast.For | ast.AsyncFor | ast.ExceptHandler | (ast.And | ast.Or))
                    ):
                        complexity += 1

                return complexity

        analyzer = SemanticAnalyzer(result, file_path)
        analyzer.visit(tree)

    async def _analyze_security(
        self, content: str, file_path: str, result: AnalysisResult
    ) -> None:
        """Analyze for security vulnerabilities"""
        lines = content.split("\n")

        for pattern_info in self.patterns.get("security", []):
            pattern = pattern_info["pattern"]
            for line_num, line in enumerate(lines, 1):
                if re.search(pattern, line):
                    result.add_issue(
                        CodeIssue(
                            id=f"security-{line_num}",
                            severity=pattern_info["severity"],
                            category=pattern_info["category"],
                            message=pattern_info["message"],
                            file_path=file_path,
                            line_number=line_num,
                            suggestions=pattern_info.get("suggestions", []),
                        )
                    )

    def _calculate_metrics(self, tree: ast.AST, content: str) -> dict[str, Any]:
        """Calculate code metrics"""
        lines = content.split("\n")

        class MetricsCalculator(ast.NodeVisitor):
            def __init__(self):
                self.functions = 0
                self.classes = 0
                self.imports = 0

            def visit_FunctionDef(self, node):
                self.functions += 1
                self.generic_visit(node)

            def visit_ClassDef(self, node):
                self.classes += 1
                self.generic_visit(node)

            def visit_Import(self, node):
                self.imports += len(node.names)
                self.generic_visit(node)

        calc = MetricsCalculator()
        calc.visit(tree)

        return {
            "lines_of_code": len(lines),
            "functions": calc.functions,
            "classes": calc.classes,
            "imports": calc.imports,
        }

    def get_supported_languages(self) -> set[str]:
        """Get supported file extensions"""
        return {".py", ".pyw"}


class DeepCodeEngine:
    """Main DeepCode analysis engine"""

    def __init__(self):
        self.analyzers: dict[str, CodeAnalyzer] = {"python": PythonDeepAnalyzer()}
        self.cache: dict[str, tuple[AnalysisResult, float]] = {}
        self.cache_ttl = 300  # 5 minutes

    def get_analyzer(self, file_extension: str) -> CodeAnalyzer | None:
        """Get appropriate analyzer for file type"""
        for _, analyzer in self.analyzers.items():
            if file_extension in analyzer.get_supported_languages():
                return analyzer
        return None

    async def analyze_file(
        self, file_path: str, level: AnalysisLevel = AnalysisLevel.SEMANTIC
    ) -> AnalysisResult:
        """Analyze a single file"""
        path = Path(file_path)

        # Check cache first
        cache_key = f"{file_path}:{level.value}"
        if cache_key in self.cache:
            result, timestamp = self.cache[cache_key]
            if time.time() - timestamp < self.cache_ttl:
                return result

        # Get appropriate analyzer
        analyzer = self.get_analyzer(path.suffix)
        if not analyzer:
            result = AnalysisResult(analysis_level=level)
            result.add_issue(
                CodeIssue(
                    id="unsupported-file-type",
                    severity=SeverityLevel.INFO,
                    category="internal",
                    message=f"No analyzer available for {path.suffix} files",
                    file_path=file_path,
                    line_number=0,
                )
            )
            return result

        # Read file content
        try:
            with open(file_path, encoding="utf-8") as f:
                content = f.read()
        except Exception as e:
            result = AnalysisResult(analysis_level=level)
            result.add_issue(
                CodeIssue(
                    id="file-read-error",
                    severity=SeverityLevel.ERROR,
                    category="internal",
                    message=f"Could not read file: {e}",
                    file_path=file_path,
                    line_number=0,
                )
            )
            return result

        # Perform analysis
        result = await analyzer.analyze(file_path, content, level)

        # Cache result
        self.cache[cache_key] = (result, time.time())

        return result

    async def analyze_directory(
        self, directory_path: str, level: AnalysisLevel = AnalysisLevel.SEMANTIC
    ) -> dict[str, AnalysisResult]:
        """Analyze all supported files in a directory"""
        results = {}
        directory = Path(directory_path)

        if not directory.exists():
            return results

        # Get all supported files
        supported_extensions = set()
        for analyzer in self.analyzers.values():
            supported_extensions.update(analyzer.get_supported_languages())

        files_to_analyze = []
        for ext in supported_extensions:
            files_to_analyze.extend(directory.rglob(f"*{ext}"))

        # Analyze files concurrently (but with limits)
        semaphore = asyncio.Semaphore(5)  # Limit concurrent analyses

        async def analyze_single_file(file_path: Path) -> tuple[str, AnalysisResult]:
            async with semaphore:
                result = await self.analyze_file(str(file_path), level)
                return str(file_path), result

        tasks = [analyze_single_file(file_path) for file_path in files_to_analyze]
        completed_analyses = await asyncio.gather(*tasks, return_exceptions=True)

        for item in completed_analyses:
            if isinstance(item, Exception):
                logger.error(f"Analysis failed: {item}")
            else:
                file_path, result = item
                results[file_path] = result

        return results

    def generate_report(self, results: dict[str, AnalysisResult]) -> dict[str, Any]:
        """Generate a comprehensive analysis report"""
        all_issues = []
        total_metrics = {
            "lines_of_code": 0,
            "functions": 0,
            "classes": 0,
            "imports": 0,
            "files_analyzed": len(results),
        }

        severity_counts = {
            SeverityLevel.CRITICAL: 0,
            SeverityLevel.ERROR: 0,
            SeverityLevel.WARNING: 0,
            SeverityLevel.INFO: 0,
        }

        category_counts = {}

        for _file_path, result in results.items():
            all_issues.extend(result.issues)

            # Aggregate metrics
            for key, value in result.metrics.items():
                if key in total_metrics and isinstance(value, int | float):
                    total_metrics[key] += value

            # Count issues by severity and category
            for issue in result.issues:
                severity_counts[issue.severity] += 1
                category_counts[issue.category] = (
                    category_counts.get(issue.category, 0) + 1
                )

        # Calculate quality score (0-100)
        quality_score = self._calculate_quality_score(severity_counts, total_metrics)

        return {
            "summary": {
                "total_issues": len(all_issues),
                "files_analyzed": len(results),
                "quality_score": quality_score,
                "severity_breakdown": {
                    sev.value: count for sev, count in severity_counts.items()
                },
                "category_breakdown": category_counts,
            },
            "metrics": total_metrics,
            "issues": [issue.to_dict() for issue in all_issues],
            "recommendations": self._generate_recommendations(
                all_issues, total_metrics
            ),
        }

    def _calculate_quality_score(
        self, severity_counts: dict[SeverityLevel, int], _metrics: dict[str, Any]
    ) -> float:
        """Calculate a quality score from 0-100"""
        base_score = 100.0

        # Deduct points for issues
        deductions = {
            SeverityLevel.CRITICAL: 20,
            SeverityLevel.ERROR: 10,
            SeverityLevel.WARNING: 5,
            SeverityLevel.INFO: 1,
        }

        for severity, count in severity_counts.items():
            base_score -= count * deductions[severity]

        return max(0.0, min(100.0, base_score))

    def _generate_recommendations(
        self, issues: list[CodeIssue], metrics: dict[str, Any]
    ) -> list[str]:
        """Generate actionable recommendations based on analysis"""
        recommendations = []

        # Group issues by category
        category_counts = {}
        for issue in issues:
            category_counts[issue.category] = category_counts.get(issue.category, 0) + 1

        # Generate recommendations based on most common issues
        if category_counts.get("security.code-injection", 0) > 0:
            recommendations.append(
                "Review and sanitize all user inputs to prevent code injection attacks"
            )

        if category_counts.get("architecture.complexity", 0) > 3:
            recommendations.append(
                "Consider refactoring complex functions to improve maintainability"
            )

        if metrics.get("functions", 0) / max(metrics.get("classes", 1), 1) > 10:
            recommendations.append(
                "Consider organizing functions into classes for better structure"
            )

        return recommendations


# Integration utilities
def create_deepcode_engine() -> DeepCodeEngine:
    """Factory function to create a configured DeepCode engine"""
    return DeepCodeEngine()


async def analyze_workspace(
    workspace_path: str, level: AnalysisLevel = AnalysisLevel.SEMANTIC
) -> dict[str, Any]:
    """Convenience function to analyze an entire workspace"""
    engine = create_deepcode_engine()
    results = await engine.analyze_directory(workspace_path, level)
    return engine.generate_report(results)


if __name__ == "__main__":
    # Example usage
    async def main():
        engine = create_deepcode_engine()

        # Analyze a single file
        result = await engine.analyze_file("example.py", AnalysisLevel.DEEP)
        print(f"Found {len(result.issues)} issues")

        # Analyze a directory
        results = await engine.analyze_directory("./src", AnalysisLevel.SEMANTIC)
        report = engine.generate_report(results)
        print(f"Quality score: {report['summary']['quality_score']}")

    asyncio.run(main())
