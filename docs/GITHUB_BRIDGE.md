# GitHub Code Search Integration

The ecosystem enhances GitHub Copilot's context by sourcing relevant, real-world code examples directly from GitHub using its Code Search API.

## Components

-   **`GitHubCodeSearchBridge`:** A low-level client that communicates with the GitHub API. It requires a GitHub Personal Access Token (PAT) with `public_repo` scope.
-   **`CopilotContextEnhancerFromGitHub`:** A higher-level class that implements the `ICopilotContextEnhancer` protocol. It uses the bridge to find examples and normalizes them into a structured format for the orchestrator.

## Configuration

To enable this feature, you must provide a GitHub token via the `GITHUB_TOKEN` environment variable.

```bash
export GITHUB_TOKEN="ghp_YourPersonalAccessTokenHere"
```

You can also optionally set `GITHUB_DEFAULT_ORG` to limit searches to a specific GitHub organization.

## How It Works

1.  When a `TODO` is detected, the `EcosystemOrchestrator` calls `find_github_examples` on the enhancer.
2.  The enhancer constructs a search query including the TODO text, language, and optional organization.
3.  The `GitHubCodeSearchBridge` executes the query against the GitHub API.
4.  The results are normalized into `GitHubExample` objects.
5.  The orchestrator synthesizes these examples into the final prompt sent to GitHub Copilot.