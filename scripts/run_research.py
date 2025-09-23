#!/usr/bin/env python3
"""Direct runner for research pipeline."""
import os
import sys
from pathlib import Path

# Add src to Python path
src_path = str(Path(__file__).resolve().parent.parent / "src")
sys.path.insert(0, src_path)
from research_agent.pipeline import ResearchPipeline  # noqa: E402


def main(query: str) -> None:
    """Run research pipeline with the given query."""
    os.environ.setdefault("SEARXNG_BASE_URL", "http://localhost:8080")
    pipeline = ResearchPipeline()
    result = pipeline.run(query)
    print(result.to_markdown())


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(
            "Usage: python run_research.py 'your query here'",
            file=sys.stderr,
        )
        sys.exit(1)
    main(" ".join(sys.argv[1:]))
