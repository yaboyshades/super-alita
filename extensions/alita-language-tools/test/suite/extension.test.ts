import * as assert from 'assert';
import * as vscode from 'vscode';

suite('Alita Language Tools', () => {
  test('activates on alita language', async () => {
    const ext = vscode.extensions.getExtension('super-alita.alita-language-tools');
    await ext?.activate();
    assert.ok(ext?.isActive);
  });

  test('registers search and skillset commands', async () => {
    const ext = vscode.extensions.getExtension('super-alita.alita-language-tools');
    await ext?.activate();
    const cmds = await vscode.commands.getCommands(true);
    assert.ok(cmds.includes('alita.search'));
    assert.ok(cmds.includes('alita.skillset'));
  });
});
