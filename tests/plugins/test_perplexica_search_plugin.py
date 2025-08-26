"""Tests for Perplexica search plugin."""
import pytest
from src.tools.perplexica_tool import PerplexicaSearchTool, PerplexicaConfig


@pytest.fixture
def search_tool():
    config = PerplexicaConfig(cache_enabled=False, rerank_enabled=True)
    return PerplexicaSearchTool(config=config)


@pytest.mark.asyncio
async def test_web_search(search_tool):
    result = await search_tool(
        query="what is retrieval augmented generation",
        mode="web",
        max_results=5,
    )

    assert "summary" in result
    assert "citations" in result
    assert len(result["citations"]) <= 5
    assert result["confidence"] > 0


@pytest.mark.asyncio
async def test_academic_search(search_tool):
    result = await search_tool(
        query="attention is all you need transformer architecture",
        mode="academic",
        max_results=3,
    )

    assert result["mode"] == "academic"
    assert "citations" in result
    assert all(
        c["source"] in ["semantic_scholar", "crossref"] for c in result["citations"]
    )


@pytest.mark.asyncio
async def test_news_search(search_tool):
    result = await search_tool(
        query="artificial intelligence regulation",
        mode="news",
        time_range="7d",
        max_results=5,
    )

    assert result["mode"] == "news"
    assert len(result.get("followup_questions", [])) > 0


@pytest.mark.asyncio
async def test_video_search(search_tool):
    result = await search_tool(
        query="how to fine tune llama 3",
        mode="video",
        max_results=3,
    )

    assert result["mode"] == "video"
    assert "summary" in result


@pytest.mark.asyncio
async def test_cache_hit(search_tool):
    search_tool.config.cache_enabled = True

    result1 = await search_tool(query="test query", mode="web")
    assert not result1.get("_cache_hit", False)

    result2 = await search_tool(query="test query", mode="web")
    assert result2.get("_cache_hit", False)


@pytest.mark.asyncio
async def test_deduplication(search_tool):
    duplicates = [
        {
            "title": "Same Title",
            "url": "https://example.com/1",
            "snippet": "A",
            "source": "s1",
            "score": 0.9,
        },
        {
            "title": "Same Title",
            "url": "https://example.com/2",
            "snippet": "B",
            "source": "s2",
            "score": 0.8,
        },
    ]

    deduped = search_tool._dedupe_results(duplicates)
    assert len(deduped) == 1
    assert deduped[0]["score"] == 0.9


@pytest.mark.asyncio
async def test_error_handling(search_tool):
    result = await search_tool(query="", mode="web")
    assert result["confidence"] == 0.0
    assert "No relevant results" in result["summary"]


@pytest.mark.parametrize(
    "mode",
    [
        "web",
        "academic",
        "news",
        "video",
        "images",
        "reddit",
        "shopping",
        "wolfram",
    ],
)
@pytest.mark.asyncio
async def test_all_modes(search_tool, mode):
    queries = {
        "web": "python programming",
        "academic": "machine learning",
        "news": "technology news",
        "video": "tutorial",
        "images": "diagram",
        "reddit": "best practices",
        "shopping": "laptop",
        "wolfram": "integrate x^2",
    }

    result = await search_tool(query=queries[mode], mode=mode, max_results=2)

    assert result["mode"] == mode
    assert "summary" in result
    assert result["confidence"] >= 0
