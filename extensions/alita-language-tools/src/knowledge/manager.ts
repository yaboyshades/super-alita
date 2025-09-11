import * as vscode from 'vscode';
import * as fs from 'fs';
import * as path from 'path';

import { KnowledgePattern, KnowledgeLedgerEntry } from './types';

function ensureDirSync(dir: string) {
  if (!fs.existsSync(dir)) {
    fs.mkdirSync(dir, { recursive: true });
  }
}

function readJsonSafe<T>(filePath: string, fallback: T): T {
  try {
    if (!fs.existsSync(filePath)) return fallback;
    const txt = fs.readFileSync(filePath, 'utf8');
    return JSON.parse(txt) as T;
  } catch {
    return fallback;
  }
}

function writeJsonSafe(filePath: string, data: unknown): void {
  ensureDirSync(path.dirname(filePath));
  fs.writeFileSync(filePath, JSON.stringify(data, null, 2), 'utf8');
}

export class KnowledgeManager {
  private baseDir: string;

  constructor(private readonly context: vscode.ExtensionContext) {
    const workspace = vscode.workspace.workspaceFolders?.[0]?.uri.fsPath;
    // Prefer repository workspace; fallback to extension global storage
    this.baseDir = workspace ?? context.globalStorageUri.fsPath;
  }

  private patternsPath(): string {
    return path.join(this.baseDir, '.alita', 'knowledge', 'patterns.json');
  }

  private ledgerPath(): string {
    return path.join(this.baseDir, '.alita', 'sessions', 'ledger.json');
  }

  async upsertPattern(pattern: KnowledgePattern): Promise<void> {
    const file = this.patternsPath();
    type Store = { schema_version: string; generated_by: string; patterns: KnowledgePattern[] };
    const store = readJsonSafe<Store>(file, {
      schema_version: '1.0.0',
      generated_by: 'alita-language-tools',
      patterns: [],
    });

    const idx = store.patterns.findIndex(
      (p) => p.name === pattern.name && p.version === pattern.version
    );
    if (idx >= 0) {
      store.patterns[idx] = pattern;
    } else {
      store.patterns.push(pattern);
    }
    writeJsonSafe(file, store);
    await this.appendLedger({ kind: 'knowledge_pattern', timestamp: new Date().toISOString(), data: pattern });
  }

  async recordDecision(data: unknown): Promise<void> {
    await this.appendLedger({
      kind: 'knowledge_decision',
      timestamp: new Date().toISOString(),
      data,
    });
  }

  async snapshotMetrics(data: unknown): Promise<void> {
    await this.appendLedger({
      kind: 'knowledge_metrics',
      timestamp: new Date().toISOString(),
      data,
    });
  }

  async showPatterns(): Promise<void> {
    const file = this.patternsPath();
    ensureDirSync(path.dirname(file));
    if (!fs.existsSync(file)) {
      writeJsonSafe(file, { schema_version: '1.0.0', generated_by: 'alita-language-tools', patterns: [] });
    }
    const doc = await vscode.workspace.openTextDocument(vscode.Uri.file(file));
    await vscode.window.showTextDocument(doc, { preview: true });
  }

  private async appendLedger(entry: KnowledgeLedgerEntry): Promise<void> {
    const file = this.ledgerPath();
    const records = readJsonSafe<KnowledgeLedgerEntry[]>(file, []);
    records.push(entry);
    writeJsonSafe(file, records);
  }
}

