import * as vscode from 'vscode';

export interface CodePattern {
  pattern: string;
  confidence: number;
  recommendation: string;
  examples: string[];
}

export interface SemanticInsight {
  type: 'architecture' | 'performance' | 'security' | 'maintainability';
  insight: string;
  severity: 'low' | 'medium' | 'high';
  suggestion: string;
}

export interface RiskAssessment {
  overall: number;
  factors: {
    complexity?: number;
    coupling?: number;
    testability?: number;
    security?: number;
  };
}

export interface DeepCodeAnalysis {
  confidence: number;
  reasoning: string[];
  alternatives: string[];
  codePatterns: CodePattern[];
  semanticInsights: SemanticInsight[];
  riskAssessment: RiskAssessment;
}

export class CopilotReasoningProvider implements vscode.InlineCompletionItemProvider {
  private alitaRuntimeBase: string;
  private reasoningCache: Map<string, DeepCodeAnalysis> = new Map();

  constructor() {
    const cfg = vscode.workspace.getConfiguration('alita');
    this.alitaRuntimeBase = (cfg.get<string>('runtime.host', 'http://127.0.0.1:8080') || 'http://127.0.0.1:8080').replace(/\/$/, '');
  }

  async provideInlineCompletionItems(
    document: vscode.TextDocument,
    position: vscode.Position,
    context: vscode.InlineCompletionContext,
    _token: vscode.CancellationToken
  ): Promise<vscode.InlineCompletionItem[]> {
    const enabled = vscode.workspace.getConfiguration('alita').get<boolean>('reasoning.enabled', true);
    if (!enabled) return [];

    const currentLine = document.lineAt(position.line).text;
    const contextBefore = this.getContextBefore(document, position, 20);
    const contextAfter = this.getContextAfter(document, position, 10);

    const analysis = await this.getDeepCodeAnalysis({
      currentLine,
      contextBefore,
      contextAfter,
      language: document.languageId,
      fileName: document.fileName,
    });

    const items: vscode.InlineCompletionItem[] = [];
    if (analysis.confidence > (vscode.workspace.getConfiguration('alita').get<number>('reasoning.confidenceThreshold', 0.6) || 0.6)) {
      const suggestion = this.generateEnhancedSuggestion(analysis) || '';
      if (suggestion) {
        const range = new vscode.Range(position, position);
        const primary = new vscode.InlineCompletionItem(suggestion, range);
        primary.command = {
          title: 'Explain Reasoning',
          command: 'alita.reasoning.explainSuggestion',
          arguments: [analysis],
        };
        items.push(primary);
      }
    }
    for (const alt of analysis.alternatives.slice(0, 3)) {
      const range = new vscode.Range(position, position);
      items.push(new vscode.InlineCompletionItem(alt, range));
    }
    return items;
  }

  async analyzeFullFile(document: vscode.TextDocument): Promise<DeepCodeAnalysis> {
    const text = document.getText();
    return this.callReasoningEndpoint({
      code: text,
      context_before: [],
      context_after: [],
      language: document.languageId,
      file_name: document.fileName,
      consensus_method: vscode.workspace.getConfiguration('alita').get<string>('reasoning.consensusMethod', 'ensemble_ranking') || 'ensemble_ranking',
      include_alternatives: true,
      confidence_threshold: vscode.workspace.getConfiguration('alita').get<number>('reasoning.confidenceThreshold', 0.6) || 0.6,
    });
  }

  async getDeepCodeAnalysis(context: any): Promise<DeepCodeAnalysis> {
    const cacheKey = this.getCacheKey(context);
    if (this.reasoningCache.has(cacheKey)) {
      return this.reasoningCache.get(cacheKey)!;
    }
    try {
      const result = await this.callReasoningEndpoint({
        code: context.currentLine,
        context_before: context.contextBefore,
        context_after: context.contextAfter,
        language: context.language,
        file_name: context.fileName,
        consensus_method: vscode.workspace.getConfiguration('alita').get<string>('reasoning.consensusMethod', 'ensemble_ranking') || 'ensemble_ranking',
        include_alternatives: true,
        confidence_threshold: vscode.workspace.getConfiguration('alita').get<number>('reasoning.confidenceThreshold', 0.6) || 0.6,
      });
      this.reasoningCache.set(cacheKey, result);
      return result;
    } catch (err) {
      console.error('DeepCode analysis failed:', err);
      return this.getDefaultAnalysis();
    }
  }

  private async callReasoningEndpoint(payload: any): Promise<DeepCodeAnalysis> {
    const response = await (globalThis as any).fetch(`${this.alitaRuntimeBase}/reasoning/analyze-code`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    const json = await response.json();
    const analysis: DeepCodeAnalysis = {
      confidence: json.confidence ?? 0.5,
      reasoning: (json.reasoning_steps || []).map((s: any) => s.step || String(s)),
      alternatives: json.alternatives || [],
      codePatterns: json.patterns || [],
      semanticInsights: json.insights || [],
      riskAssessment: json.risk_assessment || { overall: 0.5, factors: {} },
    };
    return analysis;
  }

  private generateEnhancedSuggestion(analysis: DeepCodeAnalysis): string {
    let suggestion = '';
    // Promote high-severity architecture insight as a comment hint
    for (const i of analysis.semanticInsights) {
      if (i.type === 'architecture' && i.severity === 'high') {
        suggestion += `// ${i.suggestion}\n`;
      }
    }
    const best = [...analysis.codePatterns].sort((a, b) => (b.confidence || 0) - (a.confidence || 0))[0];
    if (best && best.recommendation) {
      suggestion += best.recommendation;
    }
    return suggestion.trim();
  }

  private getContextBefore(doc: vscode.TextDocument, pos: vscode.Position, lines: number): string[] {
    const res: string[] = [];
    for (let i = Math.max(0, pos.line - lines); i < pos.line; i++) {
      res.push(doc.lineAt(i).text);
    }
    return res;
  }

  private getContextAfter(doc: vscode.TextDocument, pos: vscode.Position, lines: number): string[] {
    const res: string[] = [];
    for (let i = pos.line + 1; i < Math.min(doc.lineCount, pos.line + lines + 1); i++) {
      res.push(doc.lineAt(i).text);
    }
    return res;
  }

  private getCacheKey(ctx: any): string {
    return `${ctx.fileName}:${ctx.currentLine}:${ctx.language}`;
  }

  private getDefaultAnalysis(): DeepCodeAnalysis {
    return {
      confidence: 0.5,
      reasoning: [],
      alternatives: [],
      codePatterns: [],
      semanticInsights: [],
      riskAssessment: { overall: 0.5, factors: {} },
    };
  }
}

export function registerReasoningFeatures(context: vscode.ExtensionContext) {
  const enabled = vscode.workspace.getConfiguration('alita').get<boolean>('reasoning.enabled', true);
  if (!enabled) return [] as vscode.Disposable[];

  const provider = new CopilotReasoningProvider();
  const disposables: vscode.Disposable[] = [];

  disposables.push(
    vscode.languages.registerInlineCompletionItemProvider(
      ['python', 'typescript', 'javascript', 'java', 'cpp', 'csharp', 'go', 'rust'],
      provider
    )
  );

  disposables.push(
    vscode.commands.registerCommand('alita.reasoning.analyzeCurrentFile', async () => {
      const editor = vscode.window.activeTextEditor;
      if (!editor) return;
      const analysis = await provider.analyzeFullFile(editor.document);
      const explanation = [
        `# DeepCode Reasoning Analysis`,
        ``,
        `Confidence: ${(analysis.confidence * 100).toFixed(1)}%`,
        ``,
        `## Reasoning Steps`,
        ...(analysis.reasoning || []).map((s, i) => `${i + 1}. ${s}`),
        ``,
        `## Patterns`,
        ...(analysis.codePatterns || []).map(p => `- ${p.pattern} (${(p.confidence * 100).toFixed(1)}%): ${p.recommendation}`),
        ``,
        `## Risk`,
        `Overall: ${(analysis.riskAssessment.overall * 100).toFixed(1)}%`,
      ].join('\n');
      const doc = await vscode.workspace.openTextDocument({ content: explanation, language: 'markdown' });
      await vscode.window.showTextDocument(doc, { preview: true });
    })
  );

  disposables.push(
    vscode.commands.registerCommand('alita.reasoning.explainSuggestion', async (analysis: DeepCodeAnalysis) => {
      const explanation = [
        `# DeepCode Reasoning Analysis`,
        `Confidence: ${(analysis.confidence * 100).toFixed(1)}%`,
        ``,
        `## Reasoning Steps`,
        ...(analysis.reasoning || []).map((s, i) => `${i + 1}. ${s}`),
        ``,
        `## Patterns`,
        ...(analysis.codePatterns || []).map(p => `- ${p.pattern} (${(p.confidence * 100).toFixed(1)}%): ${p.recommendation}`),
        ``,
        `## Risk`,
        `Overall: ${(analysis.riskAssessment.overall * 100).toFixed(1)}%`,
      ].join('\n');
      const doc = await vscode.workspace.openTextDocument({ content: explanation, language: 'markdown' });
      await vscode.window.showTextDocument(doc, { preview: true });
    })
  );

  const statusBar = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Right, 100);
  statusBar.text = '$(brain) DeepCode';
  statusBar.tooltip = 'DeepCode reasoning active';
  statusBar.command = 'alita.reasoning.analyzeCurrentFile';
  statusBar.show();
  disposables.push(statusBar);

  context.subscriptions.push(...disposables);
  return disposables;
}
