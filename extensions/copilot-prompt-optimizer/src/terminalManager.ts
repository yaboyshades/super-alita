import * as vscode from 'vscode';

type ReusePolicy = 'single' | 'perCommand' | 'new';

let sharedTerminal: vscode.Terminal | undefined;

function cfg<T>(key: string, fallback: T): T {
  const c = vscode.workspace.getConfiguration('copilotPromptOptimizer');
  return (c.get<T>(key) ?? fallback) as T;
}

function detectLongRunning(cmd: string): boolean {
  const s = cmd.toLowerCase();
  return /\b(watch|serve|dev|start)\b/.test(s) || /\b(uvicorn|gunicorn|nodemon)\b/.test(s);
}

function addNoWatchFlags(cmd: string): string {
  if (!cfg('terminal.addNoWatchFlags', true)) return cmd;
  let out = cmd;
  if (/\bjest(\b|\.)/.test(out) && !/--watchall=false/i.test(out)) {
    out += ' --watchAll=false';
  }
  if (/\bvitest\b/.test(out) && !/--run/.test(out)) {
    out += ' --run --watch=false';
  }
  if (/\bpytest\b/.test(out) && !/\b--maxfail\b/.test(out)) {
    out += ' -q --maxfail=1';
  }
  return out;
}

function chooseTerminal(nameHint: string, reuse: ReusePolicy): vscode.Terminal {
  if (reuse === 'single') {
    if (!sharedTerminal) {
      sharedTerminal = vscode.window.createTerminal({ name: 'SafeRun: Shared' });
    }
    return sharedTerminal;
  }
  if (reuse === 'perCommand') {
    // try reuse an existing SafeRun terminal that is idle (best-effort)
    const existing = vscode.window.terminals.find(t => t.name.startsWith('SafeRun: '));
    if (existing) return existing;
  }
  return vscode.window.createTerminal({ name: `SafeRun: ${nameHint}` });
}

export async function safeRunCommandInteractive() {
  const input = await vscode.window.showInputBox({
    prompt: 'Enter command to run safely in terminal',
    placeHolder: 'e.g., npm test -- -u'
  });
  if (!input) return;
  await safeRunCommand(input);
}

export async function safeRunCommand(rawCmd: string, opts?: { cwd?: string; timeoutMs?: number }) {
  const reuse = cfg<ReusePolicy>('terminal.reusePolicy', 'single');
  const detect = cfg('terminal.detectWatch', true);
  const timeoutMs = opts?.timeoutMs ?? cfg('terminal.timeoutMs', 60000);

  const longRun = detect ? detectLongRunning(rawCmd) : false;
  const cmd = addNoWatchFlags(rawCmd);
  const term = chooseTerminal(longRun ? 'Long-Running' : 'One-Off', reuse);
  term.show(true);

  if (opts?.cwd) {
    term.sendText(`cd ${quotePath(opts.cwd)}`);
  }
  term.sendText(cmd);

  if (!longRun && timeoutMs > 0) {
    const tokenSource = new vscode.CancellationTokenSource();
    const timer = setTimeout(async () => {
      const choice = await vscode.window.showWarningMessage(
        `Command may be hanging after ${Math.round(timeoutMs/1000)}s.`,
        'Stop', 'Let it run'
      );
      if (choice === 'Stop') {
        // Best-effort: kill the terminal
        try { term.dispose(); } catch {}
      }
      tokenSource.cancel();
    }, timeoutMs);
    tokenSource.token.onCancellationRequested(() => clearTimeout(timer));
  } else if (longRun) {
    vscode.window.setStatusBarMessage('SafeRun: long-running command started (use Stop Long-Running to terminate).', 5000);
  }
}

function quotePath(p: string): string { return p.includes(' ') ? `"${p}"` : p; }

export async function stopLongRunningTerminals() {
  const toKill = vscode.window.terminals.filter(t => t.name.startsWith('SafeRun:'));
  if (toKill.length === 0) {
    vscode.window.showInformationMessage('No SafeRun terminals found.');
    return;
  }
  for (const t of toKill) {
    try { t.dispose(); } catch {}
  }
  vscode.window.showInformationMessage(`Stopped ${toKill.length} SafeRun terminal(s).`);
}

