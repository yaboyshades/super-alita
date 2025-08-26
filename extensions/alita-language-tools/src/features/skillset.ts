import * as vscode from 'vscode';
import { Telemetry } from '../telemetry';

export function registerSkillsetCommand(telemetry?: Telemetry): vscode.Disposable {
  return vscode.commands.registerCommand('alita.skillset', async () => {
    telemetry?.send('alita/skillset/open', {});
    await vscode.window.showInformationMessage('Alita skills: search, telemetry, language server.');
  });
}
