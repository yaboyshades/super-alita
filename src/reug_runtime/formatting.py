"""Output formatting and normalization utilities.

This module provides unified output normalization logic to ensure
consistent formatting across the runtime. It handles Unicode
replacement, code block normalization, and contract enforcement.
"""

from __future__ import annotations


def normalize_output_contract(text: str) -> str:
    """Normalize text to reduce formatting issues.

    This function provides unified normalization logic that:
    - Replaces common unicode operators with ASCII inside fenced blocks
    - Normalizes quotes and dashes to ASCII equivalents
    - Processes only content within fenced code blocks (```)
    - Leaves other content unchanged when enforcement is disabled

    Args:
        text: Input text to normalize

    Returns:
        Normalized text with ASCII operators in code blocks
    """
    # Quick exit if nothing to do
    if "```" not in text:
        return text
        
    # Process fenced blocks only
    parts: list[str] = []
    lines = text.split("\n")
    in_fence = False
    buf: list[str] = []

    def _normalize_line(s: str) -> str:
        """Apply ASCII normalization to a single line."""
        # Replace multiplication operators
        s = s.replace("×", " * ").replace("·", " * ")
        # Replace dashes with hyphens
        s = s.replace("−", "-").replace("–", "-").replace("—", "-")
        # Replace smart quotes with straight quotes
        s = s.replace("“", '"').replace("”", '"').replace("’", "'")
        return s

    for line in lines:
        if line.startswith("```"):
            if in_fence:
                # Close fence: flush normalized buffer
                parts.append("\n".join(buf))
                buf = []
                in_fence = False
                parts.append(line)
            else:
                # Open fence: start buffering
                in_fence = True
                parts.append(line)
            continue
        
        if in_fence:
            # Normalize content inside fenced blocks
            buf.append(_normalize_line(line))
        else:
            # Leave content outside fenced blocks unchanged
            parts.append(line)
    
    # If fence was left open, append remaining normalized content
    if buf:
        parts.append("\n".join(buf))
        
    return "\n".join(parts)
