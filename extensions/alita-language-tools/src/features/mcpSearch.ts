import * as vscode from 'vscode';
import { Telemetry } from '../telemetry';

export function registerMcpSearch(telemetry?: Telemetry): vscode.Disposable {
  return vscode.commands.registerCommand('alita.search', async () => {
    const query = await vscode.window.showInputBox({ prompt: 'Search MCP' });
    if (!query) {
      return;
    }
    telemetry?.send('alita/mcp/search', { length: String(query.length) });
    await vscode.commands.executeCommand('workbench.action.findInFiles', { query });
  });
}
