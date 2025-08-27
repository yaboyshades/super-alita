import * as vscode from 'vscode';
import { registerSemanticTokens } from './features/semanticTokens';
import { registerAlitaTasks } from './features/tasks';
import { registerDebugScaffold } from './features/debug';
import { createTelemetry } from './telemetry';
import { createLspClient } from './lspClient';
import { registerMcpSearch } from './features/mcpSearch';
import { PredictiveManager } from './predictiveManager';
import { FeedbackManager } from './feedbackManager';
import { WasmPredictiveAnalyzer } from './predictive/wasmAnalyzer';

interface OllamaChatMessage { content?: string }
interface OllamaChatChunk { message?: OllamaChatMessage; done?: boolean }

async function invokeOllama(prompt: string): Promise<string> {
  const cfg = vscode.workspace.getConfiguration('alita');
  let host = cfg.get<string>('ollama.host') || 'http://127.0.0.1:11434';
  host = host.replace(/\/$/, '');
  let model: string | undefined = cfg.get<string>('ollama.model');
  if (!model) {
    model = await vscode.window.showInputBox({
      prompt: 'Enter Ollama model tag (e.g. llama3.1:8b)',
      placeHolder: 'llama3.1:8b'
    });
    if (!model) { throw new Error('No model specified'); }
    await cfg.update('ollama.model', model, vscode.ConfigurationTarget.Workspace);
  }
    const resp = await (globalThis as unknown as { fetch: typeof fetch }).fetch(host + '/api/chat', {
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
    const g = globalThis as unknown as { WebAssembly?: { instantiate: (b: Uint8Array, i: unknown) => Promise<{ instance: { exports: Record<string, unknown> } }> } };
    if (!g.WebAssembly) {
      vscode.window.showWarningMessage('WebAssembly API unavailable in this environment.');
      return 0;
    }
    const mod = await g.WebAssembly.instantiate(bytes, {});
    const addExport = mod.instance.exports.add as ((x: number, y: number) => number | undefined) | undefined;
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

const disposables: vscode.Disposable[] = [];

export async function activate(ctx: vscode.ExtensionContext) {
  const telemetry = createTelemetry(ctx);
  telemetry?.send('alita/activated', {});

  // ---- DeepCode Diff Results Tree (lightweight) ----
  interface DeepCodeFileNode { type: 'file'; path: string; diff: string }
  class DeepCodeDiffProvider implements vscode.TreeDataProvider<DeepCodeFileNode> {
    private _onDidChangeTreeData = new vscode.EventEmitter<void>();
    readonly onDidChangeTreeData = this._onDidChangeTreeData.event;
    private latest: { diffs?: any[] } | null = null;
    constructor(private readonly ctx: vscode.ExtensionContext, private readonly telemetry?: ReturnType<typeof createTelemetry>) {}
    refresh() { this._onDidChangeTreeData.fire(); }
    setLatest(obj: any) { this.latest = obj || null; this.refresh(); }
    getTreeItem(el: DeepCodeFileNode): vscode.TreeItem {
      const t = new vscode.TreeItem(el.path, vscode.TreeItemCollapsibleState.None);
      t.contextValue = 'deepcodeFile';
      t.description = 'diff';
      t.tooltip = 'Unified diff';
      return t;
    }
    getChildren(): DeepCodeFileNode[] {
      if (!this.latest) return [];
      const diffs = Array.isArray(this.latest.diffs) ? this.latest.diffs : [];
      return diffs.map(d => ({ type: 'file', path: String(d.path || d.file || 'unknown'), diff: String(d.diff || d.patch || '') }));
    }
    getLatestJson() { return this.latest; }
  }
  const dcProvider = new DeepCodeDiffProvider(ctx, telemetry ?? undefined);
  vscode.window.registerTreeDataProvider('alitaDeepCodeView', dcProvider);

  // Connectivity status bar & periodic poll
  const connItem = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Left, 50);
  connItem.text = 'Alita: $(sync~spin)';
  connItem.tooltip = 'Checking Alita connectivity...';
  connItem.command = 'alita.connectivity.ping';
  connItem.show();
  ctx.subscriptions.push(connItem);

  async function checkConnectivity(silent = false) {
    const cfg = vscode.workspace.getConfiguration('alita');
    const runtime = (cfg.get<string>('runtime.host') || 'http://127.0.0.1:8080').replace(/\/$/, '');
    const ollama = (cfg.get<string>('ollama.host') || 'http://127.0.0.1:11434').replace(/\/$/, '');
    let rtOk = false; let llmOk = false;
    try {
      const r = await (globalThis as any).fetch(runtime + '/health/simple');
      rtOk = r.ok;
    } catch { /* ignore */ }
    try {
      const o = await (globalThis as any).fetch(ollama + '/api/tags');
      if (o.ok) {
        const tags = await o.json();
        const target = cfg.get<string>('ollama.model') || 'gpt-oss:20b';
        llmOk = Array.isArray(tags?.models) && tags.models.some((m: any) => m.name === target);
      }
    } catch { /* ignore */ }
    const state = rtOk && llmOk ? '$(pass) Ready' : rtOk ? '$(server) LLM?' : llmOk ? '$(warning) Runtime?' : '$(circle-slash) Down';
    connItem.text = `Alita: ${state}`;
    connItem.tooltip = `Runtime: ${rtOk ? 'OK' : 'FAIL'} | LLM: ${llmOk ? 'OK' : 'FAIL'}`;
    if (!silent) telemetry?.send('alita/connectivity', { runtime: String(rtOk), llm: String(llmOk) });
    return { rtOk, llmOk };
  }
  void checkConnectivity(true);
  const poll = setInterval(() => { void checkConnectivity(true); }, 30000);
  ctx.subscriptions.push({ dispose: () => clearInterval(poll) });

  ctx.subscriptions.push(vscode.commands.registerCommand('alita.connectivity.ping', async () => {
    const { rtOk, llmOk } = await checkConnectivity(false);
    vscode.window.showInformationMessage(`Alita connectivity - Runtime: ${rtOk ? 'OK' : 'FAIL'} | LLM: ${llmOk ? 'OK' : 'FAIL'}`);
  }));

  // Report codegen metadata (if present)
  try {
    const ext = vscode.extensions.getExtension('super-alita.alita-language-tools');
    if (ext) {
      const metaUri = vscode.Uri.joinPath(ext.extensionUri, 'out', 'src', 'generated', '.codegen.meta.json');
      let exists = false;
      try { await vscode.workspace.fs.stat(metaUri); exists = true; } catch { /* no file */ }
      if (exists) {
        const raw = await vscode.workspace.fs.readFile(metaUri);
        const txt = Buffer.from(raw).toString('utf8');
        const meta = JSON.parse(txt);
        telemetry?.send('alita/codegen/meta', {
          mode: String(meta.mode || ''),
          jco: String(meta.toolchain?.jco || false),
          wasmTools: String(meta.toolchain?.wasmTools || false)
        });
        const lines = Number(meta.metrics?.lines || 0);
        const exports = Array.isArray(meta.metrics?.exports) ? meta.metrics.exports.length : 0;
        telemetry?.send('alita/codegen/metrics', { lines: String(lines), exports: String(exports) });
        const servicesUsed = Array.isArray(meta.servicesUsed) ? meta.servicesUsed.length : 0;
        telemetry?.send('alita/codegen/host', { count: String(servicesUsed) });

        // Optional status bar indicator
        if (vscode.workspace.getConfiguration('alita').get('codegen.showBindingStatus')) {
          const item = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Right, 100);
          item.text = `WIT: ${meta.mode === 'generated' ? 'generated' : 'stub'}`;
          item.tooltip = `Alita WIT bindings (${meta.mode}). Exports: ${exports}, Lines: ${lines}`;
          item.command = 'alita.codegen.status';
          item.show();
          ctx.subscriptions.push(item);
        }
      }
    }
  } catch {
    // ignore meta read errors
  }

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

  // Predictive + feedback managers (clairvoyant scaffolding)
  const predictive = new PredictiveManager(telemetry ?? undefined);
  const feedback = new FeedbackManager();
  
  // Initialize WASM analyzer
  const wasmAnalyzer = new WasmPredictiveAnalyzer(ctx);
  wasmAnalyzer.initialize().catch((err: unknown) => {
    console.warn('Failed to initialize WASM analyzer:', err);
  });
  
  disposables.push(predictive, feedback, wasmAnalyzer);

  // Bridge worker host-call telemetry: listen for posted messages tagged __alitaHost
  try {
  (globalThis as unknown as { addEventListener?: (t: string, h: (e: MessageEvent) => void) => void }).addEventListener?.('message', (ev: MessageEvent) => {
      const data = ev.data;
      if (data && data.__alitaHost && vscode.workspace.getConfiguration('alita').get('codegen.hostTelemetry')) {
        const evt = data.__alitaHost;
        telemetry?.send('alita/hostCall', {
          name: String(evt.name || ''),
          ok: String(evt.ok),
          dur: String(evt.dur || 0)
        });
      }
    });
  } catch { /* ignore addEventListener absence */ }

  // Kick off a lightweight WASM-prefetch integration if bindings exist
  try {
    const ext = vscode.extensions.getExtension('super-alita.alita-language-tools');
    if (ext) {
      const metaUri = vscode.Uri.joinPath(ext.extensionUri, 'out', 'src', 'generated', '.codegen.meta.json');
      const raw = await vscode.workspace.fs.readFile(metaUri);
      const meta = JSON.parse(Buffer.from(raw).toString('utf8'));
      await predictive.prefetchUsingWasmAnalysis(meta);
    }
  } catch { /* optional */ }

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

  // Chat via local runtime (stream)
  disposables.push(vscode.commands.registerCommand('alita.chatRuntime', async () => {
    const prompt = await vscode.window.showInputBox({ prompt: 'Message to runtime (streamed)' });
    if (!prompt) { return; }
    const cfg = vscode.workspace.getConfiguration('alita');
    const base = (cfg.get<string>('runtime.host') || 'http://127.0.0.1:8080').replace(/\/$/, '');
    const url = `${base}/v1/chat/stream`;
    const out = vscode.window.createOutputChannel('Alita Runtime Chat');
    out.clear(); out.show(true);
    out.appendLine(`[POST] ${url}`);
    try {
      const resp = await (globalThis as any).fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: 'vscode', message: prompt })
      });
      if (!resp.ok || !resp.body) {
        throw new Error(`HTTP ${resp.status}`);
      }
      const reader = (resp.body as ReadableStream<Uint8Array>).getReader();
      const decoder = new TextDecoder();
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        out.append(decoder.decode(value));
      }
      telemetry?.send('alita/runtime/chat', { ok: 'true' });
    } catch (err) {
      vscode.window.showErrorMessage('Runtime chat failed: ' + (err as Error).message);
      telemetry?.send('alita/runtime/chat', { ok: 'false' });
    }
  }));

  // Codegen status command (smoke/status)
  disposables.push(vscode.commands.registerCommand('alita.codegen.status', async () => {
    try {
      const ext = vscode.extensions.getExtension('super-alita.alita-language-tools');
      if (!ext) return;
      const metaUri = vscode.Uri.joinPath(ext.extensionUri, 'out', 'src', 'generated', '.codegen.meta.json');
      const buf = await vscode.workspace.fs.readFile(metaUri);
      const meta = JSON.parse(Buffer.from(buf).toString('utf8'));
      const lines = Number(meta.metrics?.lines || 0);
      const exports = Array.isArray(meta.metrics?.exports) ? meta.metrics.exports.length : 0;
      let smoke = '';
      if (meta.mode === 'generated') {
        try {
          const worldJs = vscode.Uri.joinPath(ext.extensionUri, 'out', 'src', 'generated', 'alita-world.generated.js');
          const pathStr = worldJs.fsPath;
          // eslint-disable-next-line @typescript-eslint/no-var-requires
          const mod: Record<string, unknown> = require(pathStr);
          const candidate = Object.entries(mod).find(([, val]) => typeof val === 'function' && ((val as (...a: unknown[]) => unknown).length <= 1));
          if (candidate) {
            const [name, fn] = candidate as [string, (...args: []) => unknown];
            // Attempt a benign call with no args
            try { void fn(); smoke = `; called ${name}()`; } catch { /* ignore */ }
          }
        } catch { /* ignore dynamic import errors */ }
      }
      vscode.window.showInformationMessage(`WIT bindings: ${meta.mode}. Exports: ${exports}, Lines: ${lines}${smoke}`);
    } catch (err) {
      vscode.window.showWarningMessage('Codegen status unavailable: ' + (err as Error).message);
    }
  }));

  // DeepCode: Analyze workspace (via runtime)
  disposables.push(vscode.commands.registerCommand('alita.deepcode.analyze', async () => {
    const cfg = vscode.workspace.getConfiguration('alita');
    const base = (cfg.get<string>('runtime.host') || 'http://127.0.0.1:8080').replace(/\/$/, '');
    const url = `${base}/deepcode/request`;
    const folders = vscode.workspace.workspaceFolders;
    const repo_path = folders && folders.length ? folders[0].uri.fsPath : '.';
    try {
      const resp = await (globalThis as any).fetch(url, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ task_kind: 'analyze', repo_path })
      });
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      vscode.window.showInformationMessage('DeepCode analyze request sent.');
      telemetry?.send('alita/deepcode/analyze', { ok: 'true' });
    } catch (err) {
      vscode.window.showErrorMessage('DeepCode analyze failed: ' + (err as Error).message);
      telemetry?.send('alita/deepcode/analyze', { ok: 'false' });
    }
  }));

  // DeepCode: Generate from prompt (via runtime)
  disposables.push(vscode.commands.registerCommand('alita.deepcode.generate', async () => {
    const prompt = await vscode.window.showInputBox({ prompt: 'DeepCode requirements (e.g., implement feature X)' });
    if (!prompt) return;
    const cfg = vscode.workspace.getConfiguration('alita');
    const base = (cfg.get<string>('runtime.host') || 'http://127.0.0.1:8080').replace(/\/$/, '');
    const url = `${base}/deepcode/request`;
    const folders = vscode.workspace.workspaceFolders;
    const repo_path = folders && folders.length ? folders[0].uri.fsPath : '.';
    try {
      const resp = await (globalThis as any).fetch(url, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ task_kind: 'text2backend', requirements: prompt, repo_path })
      });
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      vscode.window.showInformationMessage('DeepCode generate request sent.');
      telemetry?.send('alita/deepcode/generate', { ok: 'true' });
    } catch (err) {
      vscode.window.showErrorMessage('DeepCode generate failed: ' + (err as Error).message);
      telemetry?.send('alita/deepcode/generate', { ok: 'false' });
    }
  }));

  // DeepCode: Refresh results
  disposables.push(vscode.commands.registerCommand('alita.deepcode.refreshResults', async () => {
    const cfg = vscode.workspace.getConfiguration('alita');
    const base = (cfg.get<string>('runtime.host') || 'http://127.0.0.1:8080').replace(/\/$/, '');
    const url = `${base}/deepcode/latest`;
    try {
      const resp = await (globalThis as any).fetch(url);
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const json = await resp.json();
      dcProvider.setLatest(json);
      if (!Array.isArray(json.diffs) || json.diffs.length === 0) {
        vscode.window.showInformationMessage('DeepCode: No diffs in latest proposal.');
      }
      telemetry?.send('alita/deepcode/refresh', { ok: 'true' });
    } catch (err) {
      vscode.window.showWarningMessage('DeepCode refresh failed: ' + (err as Error).message);
      telemetry?.send('alita/deepcode/refresh', { ok: 'false' });
    }
  }));

  // DeepCode: Show results (focus view)
  disposables.push(vscode.commands.registerCommand('alita.deepcode.showResults', async () => {
    await vscode.commands.executeCommand('alita.deepcode.refreshResults');
    await vscode.commands.executeCommand('workbench.view.explorer');
    // Try reveal by creating dummy selection (API limitations w/out view API)
  }));

  // DeepCode: Open diff document for selected file
  disposables.push(vscode.commands.registerCommand('alita.deepcode.openDiffDoc', async (node?: any) => {
    const target = node as { diff?: string; path?: string } | undefined;
    if (!target || !target.diff) { vscode.window.showWarningMessage('No diff available.'); return; }
    const doc = await vscode.workspace.openTextDocument({ content: target.diff, language: 'diff' });
    await vscode.window.showTextDocument(doc, { preview: true });
  }));

  // DeepCode: Apply selected file diff (delegates to runtime orchestrator apply endpoint)
  disposables.push(vscode.commands.registerCommand('alita.deepcode.applySelected', async (node?: any) => {
    const target = node as { path?: string } | undefined;
    if (!target || !target.path) { vscode.window.showWarningMessage('No file selected.'); return; }
    const confirm = await vscode.window.showWarningMessage(`Apply proposed changes for ${target.path}?`, { modal: true }, 'Apply');
    if (confirm !== 'Apply') return;
    const cfg = vscode.workspace.getConfiguration('alita');
    const base = (cfg.get<string>('runtime.host') || 'http://127.0.0.1:8080').replace(/\/$/, '');
    const url = `${base}/deepcode/apply`;
    try {
      const resp = await (globalThis as any).fetch(url, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ paths: [target.path] }) });
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const result = await resp.json();
      vscode.window.showInformationMessage(`Apply request sent (files considered: ${result.file_count ?? '?'})`);
      telemetry?.send('alita/deepcode/apply', { ok: 'true', count: String(result.file_count || 0) });
    } catch (err) {
      vscode.window.showErrorMessage('DeepCode apply failed: ' + (err as Error).message);
      telemetry?.send('alita/deepcode/apply', { ok: 'false' });
    }
  }));

  // Refactor selection with predictive cache
  disposables.push(vscode.commands.registerCommand('alita.agent.refactorSelection', async () => {
    const editor = vscode.window.activeTextEditor;
    if (!editor || editor.selection.isEmpty) { return; }
    const sel = editor.selection;
    const original = editor.document.getText(sel);
    const cached = predictive.getCachedRefactor(editor.document.uri);
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
      await (globalThis as unknown as { fetch: typeof fetch }).fetch(endpoint + '/index', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
      vscode.window.showInformationMessage(`Indexed ${alitaFiles.length} files for context.`);
    } catch (err) {
      vscode.window.showWarningMessage('Indexing failed: ' + (err as Error).message);
    }
  }));

  ctx.subscriptions.push(...disposables);
}

export function deactivate() {
  for (const d of [...disposables]) {
    try { d.dispose(); } catch { /* noop */ }
  }
}
