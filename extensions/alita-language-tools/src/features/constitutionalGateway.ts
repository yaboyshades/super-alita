/**
 * Constitutional Gateway Integration for VS Code
 *
 * Provides direct Copilot capability augmentation through the Constitutional Gateway API.
 * Features:
 * - Context providers for workspace analysis
 * - Code actions for refactoring and suggestions
 * - Inline suggestions and completions
 * - Real-time diagnostics and constitutional enforcement
 */

import * as vscode from 'vscode';

interface ConstitutionalConfig {
  gatewayUrl: string;
  apiKey?: string;
  timeout: number;
  enableDiagnostics: boolean;
  enableCodeActions: boolean;
  enableInlineSuggestions: boolean;
}

interface ContextResponse {
  files: Array<{
    path: string;
    type: string;
    content?: string;
  }>;
  summary: {
    total_files: number;
    total_directories: number;
    file_types: Record<string, number>;
    project_structure: string;
  };
  metadata: Record<string, any>;
}

interface CodeActionResponse {
  actions: Array<{
    title: string;
    kind: string;
    isPreferred?: boolean;
    edit?: any;
    metadata?: any;
  }>;
  suggestions: Array<{
    text: string;
    kind: string;
    insertText: string;
    documentation: string;
  }>;
  diagnostics: Array<{
    range: {
      start: { line: number; character: number };
      end: { line: number; character: number };
    };
    severity: number;
    message: string;
    source: string;
    code: string;
  }>;
}

interface InlineSuggestionResponse {
  suggestions: Array<{
    text: string;
    kind: string;
    detail: string;
    insertText: string;
    confidence: number;
  }>;
  completions: string[];
  confidence: number;
}

interface DiagnosticResponse {
  diagnostics: Array<{
    range: {
      start: { line: number; character: number };
      end: { line: number; character: number };
    };
    severity: number;
    message: string;
    source: string;
    code: string;
    type: string;
  }>;
  summary: {
    total_issues: number;
    by_severity: Record<string, number>;
    by_type: Record<string, number>;
    file_health_score: number;
  };
  recommendations: string[];
}

export class ConstitutionalGateway {
  private config: ConstitutionalConfig;
  private outputChannel: vscode.OutputChannel;
  private diagnosticCollection: vscode.DiagnosticCollection;

  constructor() {
    this.config = this.loadConfig();
    this.outputChannel = vscode.window.createOutputChannel('Constitutional Gateway');
    this.diagnosticCollection = vscode.languages.createDiagnosticCollection('constitutional');
  }

  private loadConfig(): ConstitutionalConfig {
    const cfg = vscode.workspace.getConfiguration('alita');
    return {
      gatewayUrl:
        cfg.get<string>('constitutional.gatewayUrl') || 'http://127.0.0.1:8080/constitutional',
      apiKey: cfg.get<string>('constitutional.apiKey'),
      timeout: cfg.get<number>('constitutional.timeout') || 30000,
      enableDiagnostics: cfg.get<boolean>('constitutional.enableDiagnostics') ?? true,
      enableCodeActions: cfg.get<boolean>('constitutional.enableCodeActions') ?? true,
      enableInlineSuggestions: cfg.get<boolean>('constitutional.enableInlineSuggestions') ?? true,
    };
  }

  private async makeRequest<T>(endpoint: string, body?: any): Promise<T> {
    const url = `${this.config.gatewayUrl}${endpoint}`;
    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
    };

    if (this.config.apiKey) {
      headers['Authorization'] = `Bearer ${this.config.apiKey}`;
    }

    try {
      const response = await fetch(url, {
        method: body ? 'POST' : 'GET',
        headers,
        body: body ? JSON.stringify(body) : undefined,
        signal: AbortSignal.timeout(this.config.timeout),
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      return await response.json();
    } catch (error) {
      this.outputChannel.appendLine(`Constitutional Gateway request failed: ${error}`);
      throw error;
    }
  }

  async getWorkspaceContext(workspacePath: string): Promise<ContextResponse> {
    return this.makeRequest<ContextResponse>('/context/workspace', {
      workspace_path: workspacePath,
      file_pattern: '**/*.py',
      include_content: false,
      max_files: 100,
    });
  }

  async getFileContext(filePath: string): Promise<any> {
    return this.makeRequest('/context/file', {
      file_path: filePath,
    });
  }

  async getRefactorActions(
    document: vscode.TextDocument,
    range?: vscode.Range
  ): Promise<CodeActionResponse> {
    const content = document.getText();
    const position = range ? range.start : new vscode.Position(0, 0);

    return this.makeRequest<CodeActionResponse>('/actions/refactor', {
      file_path: document.uri.fsPath,
      content,
      line: position.line,
      character: position.character,
      selection: range
        ? {
            start: { line: range.start.line, character: range.start.character },
            end: { line: range.end.line, character: range.end.character },
          }
        : undefined,
    });
  }

  async getQuickFixes(document: vscode.TextDocument, range?: vscode.Range): Promise<any> {
    const content = document.getText();
    const position = range ? range.start : new vscode.Position(0, 0);

    return this.makeRequest('/actions/quick-fix', {
      file_path: document.uri.fsPath,
      content,
      line: position.line,
      character: position.character,
      selection: range
        ? {
            start: { line: range.start.line, character: range.start.character },
            end: { line: range.end.line, character: range.end.character },
          }
        : undefined,
    });
  }

  async getInlineSuggestions(
    document: vscode.TextDocument,
    position: vscode.Position
  ): Promise<InlineSuggestionResponse> {
    const content = document.getText();

    return this.makeRequest<InlineSuggestionResponse>('/suggestions/inline', {
      file_path: document.uri.fsPath,
      content,
      cursor_position: {
        line: position.line,
        character: position.character,
      },
      context_lines: 5,
    });
  }

  async analyzeDiagnostics(document: vscode.TextDocument): Promise<DiagnosticResponse> {
    const content = document.getText();

    return this.makeRequest<DiagnosticResponse>('/diagnostics/analyze', {
      file_path: document.uri.fsPath,
      content,
      check_types: ['syntax', 'style', 'security', 'patterns'],
    });
  }

  async validateConstitutionalCompliance(content: string): Promise<any> {
    return this.makeRequest('/enforce/validate', {
      content,
    });
  }

  // Update diagnostics for a document
  async updateDiagnostics(document: vscode.TextDocument): Promise<void> {
    if (!this.config.enableDiagnostics) {
      return;
    }

    try {
      const result = await this.analyzeDiagnostics(document);
      const diagnostics: vscode.Diagnostic[] = result.diagnostics.map(diag => {
        const range = new vscode.Range(
          diag.range.start.line,
          diag.range.start.character,
          diag.range.end.line,
          diag.range.end.character
        );

        const diagnostic = new vscode.Diagnostic(
          range,
          diag.message,
          diag.severity === 1
            ? vscode.DiagnosticSeverity.Error
            : diag.severity === 2
              ? vscode.DiagnosticSeverity.Warning
              : vscode.DiagnosticSeverity.Information
        );

        diagnostic.source = diag.source;
        diagnostic.code = diag.code;
        return diagnostic;
      });

      this.diagnosticCollection.set(document.uri, diagnostics);
    } catch (error) {
      this.outputChannel.appendLine(`Failed to update diagnostics: ${error}`);
    }
  }

  dispose(): void {
    this.outputChannel.dispose();
    this.diagnosticCollection.dispose();
  }
}

/**
 * Context Provider for VS Code
 */
export class ConstitutionalContextProvider {
  constructor(private gateway: ConstitutionalGateway) {}

  async provideWorkspaceContext(): Promise<string> {
    const workspaceFolders = vscode.workspace.workspaceFolders;
    if (!workspaceFolders || workspaceFolders.length === 0) {
      return 'No workspace folder available';
    }

    try {
      const context = await this.gateway.getWorkspaceContext(workspaceFolders[0].uri.fsPath);
      return `Workspace Context:
- Files: ${context.summary.total_files}
- Directories: ${context.summary.total_directories}
- File Types: ${Object.entries(context.summary.file_types)
        .map(([ext, count]) => `${ext}: ${count}`)
        .join(', ')}
- Structure: ${context.summary.project_structure}`;
    } catch (error) {
      return `Failed to get workspace context: ${error}`;
    }
  }

  async provideFileContext(document: vscode.TextDocument): Promise<string> {
    try {
      const context = await this.gateway.getFileContext(document.uri.fsPath);
      return `File Context for ${document.uri.fsPath}:
- Lines: ${context.line_count}
- Size: ${context.metadata.size} bytes
- Git History: ${context.git_history?.length || 0} commits
- Security Issues: ${context.security_scan?.issue_count || 0}`;
    } catch (error) {
      return `Failed to get file context: ${error}`;
    }
  }
}

/**
 * Code Action Provider for VS Code
 */
export class ConstitutionalCodeActionProvider implements vscode.CodeActionProvider {
  constructor(private gateway: ConstitutionalGateway) {}

  async provideCodeActions(
    document: vscode.TextDocument,
    range: vscode.Range | vscode.Selection,
    context: vscode.CodeActionContext,
    token: vscode.CancellationToken
  ): Promise<vscode.CodeAction[]> {
    if (!this.gateway['config'].enableCodeActions) {
      return [];
    }

    try {
      const result = await this.gateway.getRefactorActions(document, range);
      const codeActions: vscode.CodeAction[] = [];

      // Convert refactor actions
      for (const action of result.actions) {
        const codeAction = new vscode.CodeAction(action.title, vscode.CodeActionKind.Refactor);
        codeAction.isPreferred = action.isPreferred;

        if (action.edit?.changes) {
          const edit = new vscode.WorkspaceEdit();
          for (const [filePath, changes] of Object.entries(action.edit.changes)) {
            const uri = vscode.Uri.file(filePath);
            for (const change of changes as any[]) {
              const range = new vscode.Range(
                change.range.start.line,
                change.range.start.character,
                change.range.end.line,
                change.range.end.character
              );
              edit.replace(uri, range, change.newText);
            }
          }
          codeAction.edit = edit;
        }

        codeActions.push(codeAction);
      }

      // Add quick fixes if there are diagnostics
      if (context.diagnostics.length > 0) {
        const quickFixes = await this.gateway.getQuickFixes(document, range);
        for (const fix of quickFixes.fixes || []) {
          const codeAction = new vscode.CodeAction(fix.title, vscode.CodeActionKind.QuickFix);
          codeAction.isPreferred = fix.severity === 'HIGH';
          codeActions.push(codeAction);
        }
      }

      return codeActions;
    } catch (error) {
      return [];
    }
  }
}

/**
 * Inline Completion Provider for VS Code
 */
export class ConstitutionalInlineCompletionProvider implements vscode.InlineCompletionItemProvider {
  constructor(private gateway: ConstitutionalGateway) {}

  async provideInlineCompletionItems(
    document: vscode.TextDocument,
    position: vscode.Position,
    context: vscode.InlineCompletionContext,
    token: vscode.CancellationToken
  ): Promise<vscode.InlineCompletionItem[]> {
    if (!this.gateway['config'].enableInlineSuggestions) {
      return [];
    }

    try {
      const result = await this.gateway.getInlineSuggestions(document, position);

      return result.completions.map(completion => {
        const item = new vscode.InlineCompletionItem(completion);
        item.range = new vscode.Range(position, position);
        return item;
      });
    } catch (error) {
      return [];
    }
  }
}

/**
 * Register all Constitutional Gateway features
 */
export function registerConstitutionalGateway(
  context: vscode.ExtensionContext
): ConstitutionalGateway {
  const gateway = new ConstitutionalGateway();
  const contextProvider = new ConstitutionalContextProvider(gateway);

  // Register providers
  context.subscriptions.push(
    vscode.languages.registerCodeActionsProvider(
      { scheme: 'file', language: 'python' },
      new ConstitutionalCodeActionProvider(gateway)
    )
  );

  context.subscriptions.push(
    vscode.languages.registerInlineCompletionItemProvider(
      { scheme: 'file', language: 'python' },
      new ConstitutionalInlineCompletionProvider(gateway)
    )
  );

  // Register commands
  context.subscriptions.push(
    vscode.commands.registerCommand('alita.constitutional.analyzeWorkspace', async () => {
      const context = await contextProvider.provideWorkspaceContext();
      vscode.window.showInformationMessage(context);
    })
  );

  context.subscriptions.push(
    vscode.commands.registerCommand('alita.constitutional.analyzeFile', async () => {
      const editor = vscode.window.activeTextEditor;
      if (!editor) {
        vscode.window.showWarningMessage('No active editor');
        return;
      }

      const context = await contextProvider.provideFileContext(editor.document);
      vscode.window.showInformationMessage(context);
    })
  );

  context.subscriptions.push(
    vscode.commands.registerCommand('alita.constitutional.validateCompliance', async () => {
      const editor = vscode.window.activeTextEditor;
      if (!editor) {
        vscode.window.showWarningMessage('No active editor');
        return;
      }

      try {
        const result = await gateway.validateConstitutionalCompliance(editor.document.getText());
        const message = `Constitutional Compliance: ${result.compliance_status} (Score: ${result.score}/100)`;

        if (result.compliance_status === 'compliant') {
          vscode.window.showInformationMessage(message);
        } else {
          vscode.window.showWarningMessage(message);
        }
      } catch (error) {
        vscode.window.showErrorMessage(`Compliance validation failed: ${error}`);
      }
    })
  );

  // Auto-update diagnostics on document changes
  context.subscriptions.push(
    vscode.workspace.onDidChangeTextDocument(async event => {
      if (event.document.languageId === 'python') {
        await gateway.updateDiagnostics(event.document);
      }
    })
  );

  context.subscriptions.push(
    vscode.workspace.onDidOpenTextDocument(async document => {
      if (document.languageId === 'python') {
        await gateway.updateDiagnostics(document);
      }
    })
  );

  context.subscriptions.push(gateway);

  return gateway;
}
