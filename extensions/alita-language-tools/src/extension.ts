import * as vscode from 'vscode';
import { registerSemanticTokens } from './features/semanticTokens';
import { registerAlitaTasks } from './features/tasks';
import { registerDebugScaffold } from './features/debug';
import { createTelemetry } from './telemetry';
import { createLspClient } from './lspClient';
import { registerMcpSearch } from './features/mcpSearch';
import { registerSkillsetCommand } from './features/skillset';

let disposables: vscode.Disposable[] = [];

export async function activate(ctx: vscode.ExtensionContext) {
  const telemetry = createTelemetry(ctx);
  telemetry?.send('alita/activated', {});

  // Semantic tokens
  disposables.push(registerSemanticTokens());

  // Custom tasks
  disposables.push(registerAlitaTasks(ctx));

  // Debug config provider
  disposables.push(registerDebugScaffold());

  // LSP Client
  const client = createLspClient(ctx, telemetry);
  await client.start();
  telemetry?.send('alita/lsp/start', { mode: 'ipc' });
  disposables.push({ dispose: () => client.stop() });

  // Commands
  disposables.push(vscode.commands.registerCommand('alita.restart', async () => {
    await client.stop();
    await client.start();
  }));
  disposables.push(registerMcpSearch(telemetry ?? undefined));
  disposables.push(registerSkillsetCommand(telemetry ?? undefined));

  ctx.subscriptions.push(...disposables);
}

export function deactivate() {
  while (disposables.length) {
    try { disposables.pop()?.dispose(); } catch { /* noop */ }
  }
}
