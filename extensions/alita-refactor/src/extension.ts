import * as vscode from 'vscode';
import { execFile } from 'child_process';
import * as path from 'path';

function pickPython(): string {
  // Best-effort Python runner
  const pySetting = vscode.workspace.getConfiguration('python').get<string>('defaultInterpreterPath');
  if (pySetting && pySetting.trim().length > 0) return pySetting;
  return process.platform === 'win32' ? 'python' : 'python3';
}

function runRefactorScan(targetPath: string): Promise<string> {
  return new Promise((resolve, reject) => {
    const py = pickPython();
    const cwd = vscode.workspace.workspaceFolders?.[0]?.uri.fsPath || process.cwd();
    const tool = path.join(cwd, 'tools', 'refactor_hotspots.py');
    const args = [tool, '--scan', targetPath, '--output', 'refactor_report_vscode.json'];
    const child = execFile(py, args, { cwd }, (err, stdout, stderr) => {
      if (err) return reject(new Error(stderr || err.message));
      resolve(stdout || 'Scan complete');
    });
  });
}

async function scanWorkspace() {
  const folder = vscode.workspace.workspaceFolders?.[0]?.uri.fsPath;
  if (!folder) {
    vscode.window.showErrorMessage('No workspace folder open');
    return;
  }
  const channel = vscode.window.createOutputChannel('Alita Refactor');
  channel.show(true);
  channel.appendLine(`Scanning: ${folder}`);
  try {
    const out = await runRefactorScan(folder);
    channel.appendLine(out);
    channel.appendLine('Wrote refactor_report_vscode.json');
    vscode.window.showInformationMessage('Alita Refactor: scan complete');
  } catch (e: any) {
    channel.appendLine(`Error: ${e?.message || e}`);
    vscode.window.showErrorMessage(`Alita Refactor failed: ${e?.message || e}`);
  }
}

async function scanFolder() {
  const uri = await vscode.window.showOpenDialog({ canSelectFolders: true, canSelectFiles: false, canSelectMany: false });
  if (!uri || uri.length === 0) return;
  const target = uri[0].fsPath;
  const channel = vscode.window.createOutputChannel('Alita Refactor');
  channel.show(true);
  channel.appendLine(`Scanning: ${target}`);
  try {
    const out = await runRefactorScan(target);
    channel.appendLine(out);
    channel.appendLine('Wrote refactor_report_vscode.json');
    vscode.window.showInformationMessage('Alita Refactor: scan complete');
  } catch (e: any) {
    channel.appendLine(`Error: ${e?.message || e}`);
    vscode.window.showErrorMessage(`Alita Refactor failed: ${e?.message || e}`);
  }
}

export async function activate(context: vscode.ExtensionContext) {
  context.subscriptions.push(
    vscode.commands.registerCommand('alitaRefactor.scanWorkspace', scanWorkspace),
    vscode.commands.registerCommand('alitaRefactor.scanFolder', scanFolder),
  );

  // Chat participant (proposed API) – guard at runtime
  const anyVscode = vscode as any;
  if (anyVscode?.chat?.createChatParticipant) {
    const participant = anyVscode.chat.createChatParticipant('constitutional.refactor', async (request: any, _context: any, stream: any) => {
      const folder = vscode.workspace.workspaceFolders?.[0]?.uri.fsPath || process.cwd();
      stream.markdown(`🔎 Scanning ${folder} for refactor hotspots...`);
      try {
        await runRefactorScan(folder);
        stream.markdown('✅ Scan complete. See refactor_report_vscode.json in workspace root.');
      } catch (e: any) {
        stream.markdown(`❌ Scan failed: ${e?.message || e}`);
      }
    });
    context.subscriptions.push(participant);
  }
}

export function deactivate() {}

