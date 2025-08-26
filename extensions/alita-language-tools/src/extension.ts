import * as vscode from 'vscode';
import { registerSemanticTokens } from './features/semanticTokens';
import { registerAlitaTasks } from './features/tasks';
import { registerDebugScaffold } from './features/debug';
import { createTelemetry } from './telemetry';
import { createLspClient } from './lspClient';
import { registerMcpSearch } from './features/mcpSearch';
import { registerSkillsetCommand } from './features/skillset';
import { PredictiveManager } from './predictiveManager';
import { FeedbackManager } from './feedbackManager';

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

async function runWasmCalculator(a: number, b: number): Promise<number> {
  try {
    const ext = vscode.extensions.getExtension('super-alita.alita-language-tools');
    if (!ext) {
      vscode.window.showWarningMessage('Extension context unavailable.');
      return 0;
    }
    const wasmUri = vscode.Uri.joinPath(ext.extensionUri, 'out', 'src', 'calculator.wasm');
    let bytes: Uint8Array;
    try {
      bytes = await vscode.workspace.fs.readFile(wasmUri);
    } catch {
      vscode.window.showWarningMessage('Calculator WASM not found. Build the WASM module.');
      return 0;
    }
    const mod = await (globalThis as any).WebAssembly.instantiate(bytes, {});
    const addExport = (mod.instance as any).exports?.add;
    if (typeof addExport !== 'function') {
      vscode.window.showWarningMessage('add export not found in WASM module.');
      return 0;
    }
    return addExport(a, b) ?? 0;
  } catch (err) {
    vscode.window.showErrorMessage('WASM calc failed: ' + (err as Error).message);
    return 0;
  }
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
  disposables.push(registerMcpSearch(telemetry ?? undefined));
  disposables.push(registerSkillsetCommand(telemetry ?? undefined));

  // Predictive + feedback managers (clairvoyant scaffolding)
  const predictive = new PredictiveManager(telemetry ?? undefined);
  const feedback = new FeedbackManager();
  disposables.push(predictive, feedback);

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

  // WASM calculator command
  disposables.push(vscode.commands.registerCommand('alita.calc', async () => {
    const aStr = await vscode.window.showInputBox({ prompt: 'First integer', value: '2' });
    const bStr = await vscode.window.showInputBox({ prompt: 'Second integer', value: '2' });
    if (!aStr || !bStr) { return; }
    const result = await runWasmCalculator(parseInt(aStr, 10), parseInt(bStr, 10));
    vscode.window.showInformationMessage(`WASM Result: ${result}`);
  }));

  // Refactor selection with predictive cache
  disposables.push(vscode.commands.registerCommand('alita.agent.refactorSelection', async () => {
    const editor = vscode.window.activeTextEditor;
    if (!editor || editor.selection.isEmpty) { return; }
    const sel = editor.selection;
    const original = editor.document.getText(sel);
    const cached = predictive.getCachedRefactor(editor.document.uri, sel, original);
    await vscode.window.withProgress({ location: vscode.ProgressLocation.Notification, title: 'Alita Refactor' }, async () => {
      let replacement: string | null = null;
      if (cached) {
        replacement = cached.patch;
      } else {
        // Build simple prompt (future: add context server snippets)
        const prompt = `Refactor the following code for clarity, keep semantics identical.\n\n\n\n\n\n\n\n\nCode:\n\n\n\n\n\n\n\n\n\n\n${original}`;
        try {
          replacement = await invokeOllama(prompt);
        } catch (err) {
          vscode.window.showErrorMessage('Refactor failed: ' + (err as Error).message);
          return;
        }
      }
      if (!replacement) { return; }
      const cleaned = replacement.replace(/```[a-zA-Z]*\n([\s\S]+?)```/, '$1').trim();
  await editor.edit((b: vscode.TextEditorEdit) => b.replace(sel, cleaned));
      feedback.logFeedback('refactor-selection', original, cleaned, 'accepted');
    });
  }));

  // Index workspace (stub)
  disposables.push(vscode.commands.registerCommand('alita.context.indexWorkspace', async () => {
    const folders = vscode.workspace.workspaceFolders;
    if (!folders) { return; }
    const alitaFiles: { uri: vscode.Uri; content: string }[] = [];
    for (const f of folders) {
      const files = await vscode.workspace.findFiles(new vscode.RelativePattern(f, '**/*.{py,ts,rs,alita}'), '**/node_modules/**', 300);
      for (const file of files) {
        try {
          const buf = await vscode.workspace.fs.readFile(file);
          const decoder = new TextDecoder('utf-8');
          alitaFiles.push({ uri: file, content: decoder.decode(buf) });
        } catch { /* ignore */ }
      }
    }
    const endpoint = vscode.workspace.getConfiguration('alita').get<string>('context.serverEndpoint');
    const body = { files: Object.fromEntries(alitaFiles.map(f => [f.uri.toString(), f.content])) };
    try {
      await (globalThis as any).fetch(endpoint + '/index', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
      vscode.window.showInformationMessage(`Indexed ${alitaFiles.length} files for context.`);
    } catch (err) {
      vscode.window.showWarningMessage('Indexing failed: ' + (err as Error).message);
    }
  }));

  ctx.subscriptions.push(...disposables);
}

export function deactivate() {
  while (disposables.length) {
    try { disposables.pop()?.dispose(); } catch { /* noop */ }
  }
}
