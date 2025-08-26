import * as assert from 'assert';
import * as vscode from 'vscode';

suite('Alita Extension', () => {
  test('activates on alita language', async () => {
    const ext = vscode.extensions.getExtension('super-alita.alita-lang-ext');
    await ext?.activate();
    assert.ok(ext?.isActive);
  });
});
