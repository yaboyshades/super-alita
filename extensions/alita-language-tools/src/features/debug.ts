import * as vscode from 'vscode';

export function registerDebugScaffold() {
  return vscode.debug.registerDebugConfigurationProvider('alita-debug', {
    resolveDebugConfiguration(_folder, config) {
      if (!config.type) {
        config.type = 'alita-debug';
        config.request = 'launch';
        config.name = 'Alita: Launch Script';
        config.program = '${file}';
      }
      return config;
    }
  });
}
