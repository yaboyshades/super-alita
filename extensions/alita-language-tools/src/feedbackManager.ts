import * as vscode from 'vscode';

export class FeedbackManager implements vscode.Disposable {
  private disposables: vscode.Disposable[] = [];
  logFeedback(kind: string, original: string, final: string, outcome: 'accepted' | 'modified' | 'rejected') {
    // Placeholder for future persistence / context server logging
    void kind; void original; void final; void outcome;
  }
  dispose() { this.disposables.forEach(d => d.dispose()); }
}