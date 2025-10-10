"""
Data models for code reasoning analysis.
"""

from typing import Any

from pydantic import BaseModel, Field


class Finding(BaseModel):
    """Represents a single code analysis finding."""

    rule_name: str = Field(
        ..., description="Name of the rule that triggered this finding"
    )
    symbol: str | None = Field(None, description="Function/class symbol name")
    file: str | None = Field(
        None, description="File path where finding occurred"
    )
    complexity: float | None = Field(
        None, description="Complexity score if applicable"
    )
    indegree: int | None = Field(
        None, description="Number of incoming calls if applicable"
    )
    file_a: str | None = Field(
        None, description="First file in dependency cycle"
    )
    file_b: str | None = Field(
        None, description="Second file in dependency cycle"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional finding metadata"
    )


class CodeAnalysisRequest(BaseModel):
    """Request model for code analysis."""

    repo_path: str = Field(
        ..., description="Path to the repository to analyze"
    )
    include_tests: bool = Field(
        True, description="Whether to include test files in analysis"
    )
    rules_to_run: list[str] | None = Field(
        None, description="Specific rules to run, or None for all"
    )


class CodeAnalysisResponse(BaseModel):
    """Response model for code analysis."""

    repo_path: str = Field(..., description="Path that was analyzed")
    total_files: int = Field(
        ..., description="Total number of Python files analyzed"
    )
    total_symbols: int = Field(
        ..., description="Total number of symbols extracted"
    )
    findings: dict[str, list[Finding]] = Field(
        default_factory=dict, description="Findings grouped by rule"
    )
    summary: dict[str, int] = Field(
        default_factory=dict, description="Summary counts by rule"
    )
    analysis_time: float = Field(
        ..., description="Time taken for analysis in seconds"
    )
    success: bool = Field(
        True, description="Whether analysis completed successfully"
    )
    error_message: str | None = Field(
        None, description="Error message if analysis failed"
    )
