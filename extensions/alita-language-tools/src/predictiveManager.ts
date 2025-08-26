import * as vscode from 'vscode';
import { createHash } from 'crypto';

interface CacheEntry { key: string; patch: string; createdAt: number; prompt: string; }

export class PredictiveManager implements vscode.Disposable {
  private cache = new Map<string, CacheEntry>();
  private disposables: vscode.Disposable[] = [];
  constructor(private telemetry?: { send: (e: string, p?: Record<string, any>) => void }) {
    let timer: NodeJS.Timeout | undefined;
    this.disposables.push(vscode.workspace.onDidChangeTextDocument(e => {
      if (!vscode.workspace.getConfiguration('alita').get('predictive.enabled')) { return; }
      if (e.contentChanges.length === 0) { return; }
      clearTimeout(timer);
      timer = setTimeout(() => this.analyze(e.document), 1200);
    }));
  }

  private async analyze(doc: vscode.TextDocument) {
    const text = doc.getText();
    // Heuristic: large file or long function presence triggers speculative refactor of entire selection (simplified)
    if (text.split('\n').length < 80) { return; }
    const key = this.computeKey('refactor-selection', doc.uri, text);
    if (this.cache.has(key)) { return; }
    const prompt = `Speculative refactor for potential future request. Improve readability and keep behavior.\n\n\n\n${text}`;
    try {
      // Placeholder: We do not invoke background model here to avoid resource cost; future hook.
      this.cache.set(key, { key, patch: text, createdAt: Date.now(), prompt });
      this.telemetry?.send('predictive/cache/store', { size: String(text.length) });
    } catch (err) {
      this.telemetry?.send('predictive/cache/error', { error: (err as Error).message });
    }
  }

  private computeKey(action: string, uri: vscode.Uri, content: string): string {
    return createHash('sha256').update(action + '|' + uri.toString() + '|' + content).digest('hex');
  }

  getCachedRefactor(uri: vscode.Uri, range: vscode.Range, original: string): CacheEntry | null {
    // Very naive lookup; future will use range fingerprint
    for (const c of this.cache.values()) {
      if (c.key.includes(uri.toString())) {
        this.cache.delete(c.key);
        this.telemetry?.send('predictive/cache/hit');
        return c;
      }
    }
    return null;
  }

  dispose() { this.disposables.forEach(d => d.dispose()); }
}