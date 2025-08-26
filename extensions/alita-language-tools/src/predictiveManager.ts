import * as vscode from 'vscode';
import { createHash } from 'crypto';

type Action = 'refactor-selection' | 'generate-tests' | 'fix-error';
interface CachedResult { key: string; patch: string; prompt: string; createdAt: number; action: Action; confidence: number; }

export class PredictiveManager implements vscode.Disposable {
  private cache = new Map<string, CachedResult>();
  private disposables: vscode.Disposable[] = [];
  private queue: any[] = [];
  private busy = false;
  private maxEntries = 5;

  constructor(private telemetry?: { send: (e: string, p?: Record<string, any>) => void }) {
    this.disposables.push(
  vscode.workspace.onDidChangeTextDocument((e: vscode.TextDocumentChangeEvent) => {
        if (!vscode.workspace.getConfiguration('alita').get('predictive.enabled')) return;
        if (e.contentChanges.length === 0) return;
        this.queue.push({ type: 'edit', doc: e.document, ts: Date.now() });
      }),
  vscode.languages.onDidChangeDiagnostics((e: vscode.DiagnosticChangeEvent) => {
        for (const uri of e.uris) {
          const diags = vscode.languages.getDiagnostics(uri).filter((d: vscode.Diagnostic) => d.severity === vscode.DiagnosticSeverity.Error);
          if (diags.length) this.queue.push({ type: 'error', uri, diags, ts: Date.now() });
        }
      })
    );
  globalThis.setInterval(() => { if (!this.busy) void this.process(); }, 1200);
  }

  private async process() {
    if (!this.queue.length) return;
    this.busy = true;
    const batch = this.queue.splice(0, this.queue.length);
    const docEvents = batch.filter(b => b.doc).slice(-1); // latest doc event only
    for (const evt of docEvents) {
      const pred = this.predict(evt.doc);
      if (pred && pred.confidence >= 0.55) await this.generate(pred, evt.doc);
    }
    const errEvt = batch.find(b => b.type === 'error');
    if (errEvt) {
  const open = vscode.workspace.textDocuments.find((d: vscode.TextDocument) => d.uri.toString() === errEvt.uri.toString());
      if (open) await this.generate({ action: 'fix-error', confidence: 0.7 }, open, errEvt.diags);
    }
    this.busy = false;
  }

  private predict(doc: vscode.TextDocument): { action: Action; confidence: number } | null {
    const lines = doc.lineCount;
    if (lines > 120) return { action: 'refactor-selection', confidence: 0.65 };
    return null;
  }

  private async generate(pred: { action: Action; confidence: number }, doc: vscode.TextDocument, diags?: vscode.Diagnostic[]) {
    const content = doc.getText();
    const key = createHash('sha256').update(pred.action + '|' + doc.uri.toString() + '|' + String(content.length)).digest('hex');
    if (this.cache.has(key)) return;
    // Build prompt (speculative heuristic)
    let prompt: string;
    if (pred.action === 'fix-error' && diags) {
      prompt = `Fix the following errors:\n${diags.map(d => d.message).join('\n')}\n\nCode:\n${content}`;
    } else if (pred.action === 'refactor-selection') {
      prompt = `Speculatively refactor the following code for clarity. Maintain identical behavior.\n\n${content}`;
    } else {
      prompt = `Perform action ${pred.action} on code:\n${content}`;
    }
    // Placeholder: no background model call yet (future: integrate endpoint)
    const patch = content; // identity placeholder
    this.insert({ key, patch, prompt, createdAt: Date.now(), action: pred.action, confidence: pred.confidence });
  }

  private insert(res: CachedResult) {
    if (this.cache.size >= this.maxEntries) {
      // Evict oldest
      const oldest = [...this.cache.values()].sort((a, b) => a.createdAt - b.createdAt)[0];
      if (oldest) this.cache.delete(oldest.key);
    }
    this.cache.set(res.key, res);
    this.telemetry?.send('predictive/cache/store', { action: res.action, confidence: String(res.confidence) });
  }

  getCachedRefactor(uri: vscode.Uri): { patch: string; prompt: string } | null {
    for (const v of this.cache.values()) {
      if (v.key.includes(uri.toString()) && v.action === 'refactor-selection') {
        this.cache.delete(v.key);
        this.telemetry?.send('predictive/cache/hit');
        return { patch: v.patch, prompt: v.prompt };
      }
    }
    return null;
  }

  dispose() { this.disposables.forEach(d => d.dispose()); }
}