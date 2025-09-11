# Spec — DeepCode + GitHub Copilot Integration (Super‑Alita)

## 1. Overview
Integrate a DeepCode Reasoning Provider into the Super‑Alita VS Code extension to enhance inline completions with consensus reasoning, confidence calibration, and semantic insights (architecture, patterns, security, performance). The provider is configurable, resilient, and caches recent analyses.

## 2. Goals
- Real‑time enhanced suggestions for Copilot
- Configurable consensus methods (simple_vote, weighted_vote, confidence_based, semantic_similarity, ensemble_ranking)
- Confidence calibration via DeepConf (fallback available)
- Semantic insights (Mangle‑style heuristics)
- Caching + graceful fallbacks
- Commands: analyze selection/file, toggle reasoning, configure provider

## 3. Architecture
### 3.1 VS Code Extension
- Provider: `DeepCodeReasoningProvider` (InlineCompletionItemProvider) — see `extensions/alita-language-tools/src/reasoning/copilot-context-provider.ts`
- Commands:
  - `alita.reasoning.analyzeCurrentFile` (analysis view)
  - `alita.reasoning.explainSuggestion` (explanation view)
  - Aliases: `alita.deepcode.toggle`, `alita.deepcode.configure`
- Settings (contributes):
  - `alita.reasoning.enabled` (bool, default true)
  - `alita.reasoning.confidenceThreshold` (number, default 0.6)
  - `alita.reasoning.consensusMethod` (enum, default ensemble_ranking)

### 3.2 Backend (FastAPI)
- Endpoints (lightweight, adapter‑aware):
  - `POST /reasoning/consensus` → in‑process tool `deepconf_consensus` if available; fallback response otherwise
  - `POST /reasoning/deepconf` → heuristic calibration fallback (avg + context factors)
  - `POST /reasoning/mangle` → heuristics for architecture/security/performance insights
  - Existing `POST /reasoning/analyze-code` remains available (composed behavior)

## 4. Interfaces & Models
TypeScript (extension): DeepCodeAnalysis, CodePattern, SemanticInsight, RiskAssessment

Python (Pydantic): ConsensusRequest/Response, DeepConfRequest/Response, MangleRequest/Response, CodeAnalysisRequest/Response

## 5. VS Code Integration
- Activation: imports provider and registers inline completions + commands in `extension.ts`
- Status bar: “DeepCode” brain icon for quick analysis
- Command palette entries added via `package.json`

## 6. Caching & Performance
- In‑memory map with size guard (drops oldest entries when >100)
- Minimal payloads; endpoints are stateless

## 7. Test Plan
Unit tests:
- Mock fetch in provider; verify items emitted with thresholds
- Endpoint tests for valid/invalid payloads; fallback paths

Integration tests:
- Launch runtime; verify provider calls endpoints and renders suggestions
- Toggle setting and confirm behavior

Edge cases:
- Cancellation tokens; network timeouts; large/empty contexts; dynamic settings changes

## 8. Notes
- Provider keys use `alita.reasoning.*`; `alita.deepcode.*` are surfaced as commands/UX labels only. Mapping keeps settings minimal and consistent with existing config.

