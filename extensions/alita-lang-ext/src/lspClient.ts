import * as path from 'path';
import * as vscode from 'vscode';
import { LanguageClient, LanguageClientOptions, ServerOptions, TransportKind } from 'vscode-languageclient/node';
import type { Telemetry } from './telemetry';

export function createLspClient(ctx: vscode.ExtensionContext, telemetry: Telemetry | null) {
  const serverModule = ctx.asAbsolutePath(path.join('server', 'out', 'src', 'server.js'));
  const debugOptions = { execArgv: ['--nolazy', '--inspect=6009'] };

  const serverOptions: ServerOptions = {
    run:    { module: serverModule, transport: TransportKind.ipc },
    debug:  { module: serverModule, transport: TransportKind.ipc, options: debugOptions }
  };

  const clientOptions: LanguageClientOptions = {
    documentSelector: [
      { language: 'alita', scheme: 'file' },
      { language: 'alita', scheme: 'vscode-vfs' },  // virtual
      { language: 'alita', scheme: 'untitled' }     // new unsaved
    ],
    synchronize: { fileEvents: vscode.workspace.createFileSystemWatcher('**/*.alita') },
    middleware: {
      handleDiagnostics: (uri, diagnostics, next) => {
        telemetry?.send('alita/diagnostics', { count: String(diagnostics.length) });
        return next(uri, diagnostics);
      }
    }
  };

  const client = new LanguageClient('alitaLangServer', 'Alita Language Server', serverOptions, clientOptions);
  return client;
}
