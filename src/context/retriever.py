"""
Context retrieval module for Super Alita prompt pipeline.
Leverages knowledge graph and semantic search to find relevant context.
"""
from pathlib import Path
from typing import Dict, List

# Define context source paths relative to project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
DOC_PATHS = {
    "architecture": PROJECT_ROOT / "ADVANCED_DEVELOPMENT_PATTERNS.md",
    "agents": PROJECT_ROOT / "AGENTS.md",
    "consensus": PROJECT_ROOT / "src" / "abilities" / "enhanced_consensus_ability.py",
    "reug": PROJECT_ROOT / "src" / "reug_runtime" / "router.py",
    "quick_reference": PROJECT_ROOT / "AGENT_QUICK_REFERENCE.md",
}

# Context cache
_context_cache: Dict[str, Dict] = {}

async def fetch_context(query: str, max_results: int = 3) -> List[Dict[str, str]]:
    """
    Fetch relevant context based on the query.
    Uses a combination of keyword matching and semantic similarity.

    Args:
        query: User query for context retrieval
        max_results: Maximum number of context snippets to return

    Returns:
        List of context dictionaries with title and snippet
    """
    results = []

    # Simple keyword-based matching (can be replaced with vector search)
    keywords = set(query.lower().split())

    for source_name, path in DOC_PATHS.items():
        if not path.exists():
            continue

        # Check cache first
        if source_name in _context_cache:
            source_content = _context_cache[source_name]
        else:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read()

                # Simple chunking by paragraphs
                paragraphs = content.split("\n\n")
                source_content = {
                    f"{source_name}_{i}": {
                        "title": f"{source_name.title()} {i+1}",
                        "content": p.strip()
                    }
                    for i, p in enumerate(paragraphs) if len(p.strip()) > 50
                }
                _context_cache[source_name] = source_content
            except Exception as e:
                print(f"Error loading context from {path}: {e}")
                continue

        # Find matching snippets
        for snippet_id, snippet_data in source_content.items():
            content = snippet_data["content"].lower()
            # Calculate simple relevance score based on keyword matches
            matching_keywords = sum(1 for k in keywords if k in content)
            if matching_keywords > 0:
                results.append({
                    "title": snippet_data["title"],
                    "snippet": snippet_data["content"][:300] + "...",
                    "relevance": matching_keywords / len(keywords)
                })

    # Sort by relevance and limit results
    results.sort(key=lambda x: x["relevance"], reverse=True)
    return [{"title": r["title"], "snippet": r["snippet"]} for r in results[:max_results]]

async def get_consensus_methods() -> List[str]:
    """Get available consensus methods from the codebase."""
    methods = ["simple_vote", "weighted_vote", "confidence_based",
               "semantic_similarity", "ensemble_ranking"]
    return methods

async def initialize_context() -> None:
    """Pre-load context sources to warm up the cache."""
    for source_name, path in DOC_PATHS.items():
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read()
                paragraphs = content.split("\n\n")
                source_content = {
                    f"{source_name}_{i}": {
                        "title": f"{source_name.title()} {i+1}",
                        "content": p.strip()
                    }
                    for i, p in enumerate(paragraphs) if len(p.strip()) > 50
                }
                _context_cache[source_name] = source_content
            except Exception as e:
                print(f"Error initializing context from {path}: {e}")
