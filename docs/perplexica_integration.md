# Perplexica Integration for Super Alita

## Overview

The PerplexicaSearchPlugin integrates AI-powered search capabilities similar to Perplexica into the Super Alita agent system. This provides enhanced search functionality with reasoning, summarization, and multi-modal search capabilities.

## Features

### 🧠 AI-Powered Search
- **Intelligent Reasoning**: AI analysis of search results with explanation of findings
- **Summarization**: Concise summaries of complex search results
- **Confidence Scoring**: Automatic confidence assessment for search quality
- **Follow-up Questions**: AI-generated follow-up questions for deeper exploration

### 🔍 Multi-Modal Search Modes
- **Web**: General web search
- **Academic**: Scholarly and research papers
- **Video**: Video content and tutorials
- **News**: Recent news and current events
- **Images**: Visual content search
- **Reddit**: Community discussions and insights
- **Shopping**: Product and price comparisons
- **Wolfram**: Computational and mathematical queries

### 🔗 Integration Benefits
- **WebAgent Compatibility**: Leverages existing WebAgentAtom infrastructure
- **Event-Driven**: Integrates seamlessly with Super Alita's event bus
- **MCP Tools**: Ready for VS Code integration via MCP
- **Plugin Architecture**: Follows Super Alita's plugin interface standards

## Quick Start

### 1. Basic Usage

```python
from src.plugins.perplexica_search_plugin import PerplexicaSearchPlugin, SearchMode

# Create and setup plugin
plugin = PerplexicaSearchPlugin()
await plugin.setup(event_bus, store, config)
await plugin.start()

# Perform AI-powered search
response = await plugin.search(
    query="artificial intelligence trends 2024",
    search_mode=SearchMode.WEB,
    max_results=10,
    include_reasoning=True
)

print(f"Summary: {response.summary}")
print(f"Reasoning: {response.reasoning}")
print(f"Sources: {len(response.sources)}")
print(f"Citations: {response.citations}")
```

### 2. Event-Driven Usage

```python
# Subscribe to search events
await plugin.subscribe("perplexica_search", handle_search_results)

# Emit search request
await event_bus.emit("perplexica_search", 
    query="machine learning frameworks",
    search_mode="academic",
    session_id="user_session_123"
)
```

### 3. Integration with WebAgentAtom

```python
# Configure with existing WebAgent
web_agent = WebAgentAtom()
config = {"web_agent": web_agent}

plugin = PerplexicaSearchPlugin()
await plugin.setup(event_bus, store, config)

# Perplexica will use WebAgent for search backend
response = await plugin.search("Python tutorials", SearchMode.WEB)
```

## Observability

Events emitted by the Perplexica plugin include telemetry fields:

- `source_plugin` – always `perplexica_search`
- `conversation_id` – session identifier
- `correlation_id` – shared ID across related events
- `timestamp` – UTC event creation time

These fields enable end‑to‑end tracing of search requests and results.

## Configuration

### Environment Variables

```bash
# Optional: For enhanced AI reasoning
GEMINI_API_KEY=your_gemini_api_key

# Optional: For GitHub integration (via WebAgent)
GITHUB_TOKEN=your_github_token

# Optional: For SearXNG integration (via WebAgent)
SEARXNG_BASE_URL=http://localhost:4000
```

### Plugin Configuration

```python
config = {
    "web_agent": web_agent_instance,  # Optional: Use existing WebAgent
    "max_search_history": 100,        # Default: 100
    "default_reasoning": True,        # Default: True
    "confidence_threshold": 0.5       # Default: 0.5
}
```

## API Reference

### PerplexicaSearchPlugin

#### Methods

##### `search(query, search_mode, max_results, include_reasoning)`
Perform AI-powered search with reasoning.

**Parameters:**
- `query` (str): Search query
- `search_mode` (SearchMode): Type of search to perform
- `max_results` (int): Maximum number of results (default: 10)
- `include_reasoning` (bool): Include AI reasoning (default: True)

**Returns:**
- `PerplexicaResponse`: Enhanced search response with AI analysis

##### `get_search_history(limit)`
Get recent search history.

**Parameters:**
- `limit` (int): Number of recent searches to return (default: 10)

**Returns:**
- `List[PerplexicaResponse]`: List of recent search responses

##### `health_check()`
Get plugin health status.

**Returns:**
- `Dict[str, Any]`: Health status information

### SearchMode Enum

Available search modes:
- `SearchMode.WEB` - General web search
- `SearchMode.ACADEMIC` - Academic and research papers
- `SearchMode.VIDEO` - Video content
- `SearchMode.NEWS` - News and current events
- `SearchMode.IMAGES` - Image search
- `SearchMode.REDDIT` - Reddit discussions
- `SearchMode.SHOPPING` - Shopping and products
- `SearchMode.WOLFRAM` - Computational queries

### PerplexicaResponse Model

```python
class PerplexicaResponse(BaseModel):
    query: str                          # Original search query
    search_mode: SearchMode             # Search mode used
    summary: str                        # AI-generated summary
    reasoning: str                      # AI reasoning process
    sources: List[SearchResult]         # Search results
    citations: List[str]                # Source citations
    follow_up_questions: List[str]      # Suggested follow-ups
    confidence_score: float             # Confidence (0.0-1.0)
    processing_time: float              # Processing time in seconds
    total_results: int                  # Total number of results
```

### SearchResult Model

```python
class SearchResult(BaseModel):
    title: str                          # Result title
    url: str                            # Result URL
    snippet: str                        # Content snippet
    source: str                         # Source identifier
    relevance_score: float              # Relevance score (0.0-1.0)
    timestamp: Optional[str]            # Result timestamp
    metadata: Dict[str, Any]            # Additional metadata
```

## Event Types

### Input Events

#### `perplexica_search`
Request for AI-powered search.

```python
{
    "query": "search query",
    "search_mode": "web",
    "max_results": 10,
    "include_reasoning": True,
    "session_id": "session_123"
}
```

#### `enhanced_search`
Alternative event name for enhanced search requests.

### Output Events

#### `perplexica_result`
Search results with AI analysis.

```python
{
    "query": "search query",
    "response": PerplexicaResponse,
    "session_id": "session_123",
    "search_mode": "web"
}
```

#### `tool_result`
Standard tool result for MCP integration.

## Integration Examples

### With Main Agent Loop

```python
# In main agent setup
from src.plugins.perplexica_search_plugin import PerplexicaSearchPlugin

async def setup_agent():
    # Create plugins
    web_agent = WebAgentAtom()
    perplexica = PerplexicaSearchPlugin()
    
    # Setup
    await web_agent.setup(event_bus, store, {})
    await perplexica.setup(event_bus, store, {"web_agent": web_agent})
    
    # Start
    await web_agent.start()
    await perplexica.start()
    
    return [web_agent, perplexica]
```

### With MCP Tools

```python
# Register as MCP tool
tools = plugin.get_tools()
for tool in tools:
    mcp_server.register_tool(tool)
```

### With VS Code Integration

The plugin automatically provides tools compatible with the MCP server for VS Code integration:

```json
{
    "name": "perplexica_search",
    "description": "AI-powered search with reasoning and multi-modal capabilities",
    "parameters": {
        "query": {"type": "string", "description": "Search query"},
        "search_mode": {"type": "string", "enum": ["web", "academic", "video", "news", "images", "reddit", "shopping", "wolfram"]},
        "max_results": {"type": "integer", "default": 10},
        "include_reasoning": {"type": "boolean", "default": true}
    }
}
```

## Comparison with Basic Search

| Feature | Basic WebAgent | Perplexica Plugin |
|---------|----------------|-------------------|
| Search Results | ✅ Basic results | ✅ Enhanced results |
| AI Reasoning | ❌ None | ✅ Full reasoning |
| Summarization | ❌ None | ✅ AI summaries |
| Citations | ❌ Basic links | ✅ Formatted citations |
| Follow-ups | ❌ None | ✅ AI-generated questions |
| Confidence | ❌ None | ✅ Confidence scoring |
| Multi-modal | ❌ Web only | ✅ 8 search modes |
| Search History | ❌ None | ✅ Full history |

## Best Practices

### 1. Search Mode Selection
- Use `WEB` for general queries
- Use `ACADEMIC` for research and scholarly content
- Use `VIDEO` for tutorials and demonstrations
- Use `NEWS` for current events and recent developments

### 2. Query Optimization
- Be specific with search terms
- Use natural language for better AI reasoning
- Include context for better results

### 3. Result Processing
- Check confidence scores for result quality
- Use citations for fact-checking
- Leverage follow-up questions for deeper exploration

### 4. Performance
- Use caching for repeated queries
- Limit max_results for faster responses
- Consider disabling reasoning for simple searches

## Troubleshooting

### Common Issues

1. **No AI Reasoning**
   - Check GEMINI_API_KEY is set
   - Verify API key is valid
   - Fallback to simple reasoning automatically

2. **No Search Results**
   - Check SearXNG is running (if using WebAgent)
   - Verify network connectivity
   - Check for search service availability

3. **Event Errors**
   - Ensure proper event structure
   - Check required fields (session_id, tool_call_id)
   - Verify event bus is properly configured

### Debug Mode

Enable debug logging for troubleshooting:

```python
import logging
logging.getLogger("src.plugins.perplexica_search_plugin").setLevel(logging.DEBUG)
```

## Future Enhancements

- [ ] Vector similarity search integration
- [ ] Custom search engine support
- [ ] Advanced citation formatting
- [ ] Multi-language search support
- [ ] Search result caching
- [ ] Custom AI model support
- [ ] Advanced filtering options
- [ ] Real-time search updates

## License

This plugin is part of the Super Alita project and follows the same license terms.