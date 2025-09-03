import * as vscode from 'vscode';

export interface AmplifyContext {
  repoName?: string;
  filePath?: string;
  languageId?: string;
  selectionPreview?: string;
}

export function optimizePrompt(input: string): string {
  let s = input || '';
  s = s.replace(/[\t ]+/g, ' ').replace(/\s+\n/g, '\n').trim();
  // Normalize punctuation spacing
  s = s.replace(/\s*([.,;:!?])\s*/g, '$1 ');
  // Ensure imperative clarity (heuristic): prefix with "Please" if looks like a command
  if (/^(add|create|explain|optimize|rewrite|refactor|fix|summarize|implement)\b/i.test(s)) {
    s = `Please ${s[0].toLowerCase()}${s.slice(1)}`;
  }
  return s.trim();
}

export function gatherAmplifyContext(maxChars = 500): AmplifyContext {
  const editor = vscode.window.activeTextEditor;
  const repoName = vscode.workspace.name;
  const ctx: AmplifyContext = { repoName };
  if (editor) {
    ctx.filePath = editor.document.uri.fsPath;
    ctx.languageId = editor.document.languageId;
    const sel = editor.selection;
    let text = sel && !sel.isEmpty ? editor.document.getText(sel) : '';
    if (!text) {
      // Fallback: small window around cursor
      const start = Math.max(0, editor.selection.active.line - 10);
      const end = Math.min(editor.document.lineCount - 1, editor.selection.active.line + 10);
      const range = new vscode.Range(start, 0, end, editor.document.lineAt(end).range.end.character);
      text = editor.document.getText(range);
    }
    if (text.length > maxChars) text = text.slice(0, maxChars) + '\n…';
    ctx.selectionPreview = text;
  }
  return ctx;
}

export function amplifyPrompt(content: string, context: AmplifyContext): string {
  const parts: string[] = [];
  if (context.repoName) parts.push(`Repository: ${context.repoName}`);
  if (context.filePath) parts.push(`File: ${context.filePath}`);
  if (context.languageId) parts.push(`Language: ${context.languageId}`);
  if (context.selectionPreview) {
    parts.push('Context snippet:\n```\n' + context.selectionPreview + '\n```');
  }
  const header = parts.length ? parts.join('\n') + '\n\n' : '';
  const role = 'You are a precise, helpful coding assistant.';
  return `${role}\n\n${header}${content}`.trim();
}

