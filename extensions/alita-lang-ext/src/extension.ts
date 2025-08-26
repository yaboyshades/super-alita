import * as vscode from 'vscode';
import { registerSemanticTokens } from './features/semanticTokens';
import { registerAlitaTasks } from './features/tasks';
import { registerDebugScaffold } from './features/debug';
import { createTelemetry } from './telemetry';
import { createLspClient } from './lspClient';

interface OllamaChatChunk { message?: { content?: string }; done?: boolean }

async function invokeOllama(prompt: string): Promise<string> {
  const cfg = vscode.workspace.getConfiguration('alita');
  let host = cfg.get<string>('ollama.host') || 'http://127.0.0.1:11434';
  host = host.replace(/\/$/, '');
  let model = cfg.get<string>('ollama.model');
  if (!model) {
    model = await vscode.window.showInputBox({
      prompt: 'Enter Ollama model tag (e.g. llama3.1:8b)',
      placeHolder: 'llama3.1:8b'
    });
    if (!model) { throw new Error('No model specified'); }
    await cfg.update('ollama.model', model, vscode.ConfigurationTarget.Workspace);
  }
  const resp = await (globalThis as any).fetch(host + '/api/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ model, messages: [{ role: 'user', content: prompt }], stream: false })
  });
  if (!resp.ok) {
    throw new Error(`Ollama error ${resp.status}`);
  }
  const data = await resp.json() as OllamaChatChunk;
  return data?.message?.content || '';
}

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
  // Agent invoke command (Ollama)
  disposables.push(vscode.commands.registerCommand('alita.invokeAgent', async () => {
    const prompt = await vscode.window.showInputBox({ prompt: 'Agent prompt' });
    if (!prompt) { return; }
    const progressOpts: vscode.ProgressOptions = { location: vscode.ProgressLocation.Notification, title: 'Invoking Alita Agent (Ollama)...' };
    await vscode.window.withProgress(progressOpts, async () => {
      try {
        const reply = await invokeOllama(prompt);
        const doc = await vscode.workspace.openTextDocument({ content: reply, language: 'markdown' });
        await vscode.window.showTextDocument(doc, { preview: true });
        telemetry?.send('alita/agent/invoke', { provider: 'ollama', ok: 'true' });
      } catch (err) {
        vscode.window.showErrorMessage('Agent invocation failed: ' + (err as Error).message);
        telemetry?.send('alita/agent/invoke', { provider: 'ollama', ok: 'false' });
      }
    });
  }));

  ctx.subscriptions.push(...disposables);
}

export function deactivate() {
  while (disposables.length) {
    try { disposables.pop()?.dispose(); } catch { /* noop */ }
  }
}
