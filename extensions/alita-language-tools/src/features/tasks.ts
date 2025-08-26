import * as vscode from 'vscode';

export function registerAlitaTasks(ctx: vscode.ExtensionContext) {
  const type = 'alita';
  const provider: vscode.TaskProvider = {
    provideTasks: async () => [
      new vscode.Task(
        { type, label: 'alita: validate' },
        vscode.TaskScope.Workspace,
        'alita: validate',
        type,
        new vscode.ShellExecution('echo "Validating Alita workspace..."')
      )
    ],
    resolveTask: task => task
  };
  const d = vscode.tasks.registerTaskProvider(type, provider);
  ctx.subscriptions.push(d);
  return d;
}
