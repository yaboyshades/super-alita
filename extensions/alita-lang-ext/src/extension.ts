import * as vscode from "vscode";
import {
  LanguageClient,
  LanguageClientOptions,
  ServerOptions,
  TransportKind,
} from "vscode-languageclient/node";

let client: LanguageClient;

export async function activate(
  context: vscode.ExtensionContext,
): Promise<void> {
  const serverModule = context.asAbsolutePath("server/out/server.js");
  const serverOptions: ServerOptions = {
    run: { module: serverModule, transport: TransportKind.ipc },
    debug: {
      module: serverModule,
      transport: TransportKind.ipc,
      options: { execArgv: ["--nolazy", "--inspect=6009"] },
    },
  };
  const clientOptions: LanguageClientOptions = {
    documentSelector: [{ scheme: "file", language: "alita" }],
  };
  client = new LanguageClient(
    "alitaLangServer",
    "Alita Language Server",
    serverOptions,
    clientOptions,
  );
  context.subscriptions.push(client.start());

  // semantic tokens provider
  const legend = new vscode.SemanticTokensLegend(["keyword", "comment"]);
  const semanticProvider: vscode.DocumentSemanticTokensProvider = {
    provideDocumentSemanticTokens(document) {
      const builder = new vscode.SemanticTokensBuilder(legend);
      const text = document.getText();
      const match = /^(\w+)/.exec(text);
      if (match) {
        builder.push(0, 0, match[0].length, 0, 0);
      }
      return builder.build();
    },
  };
  context.subscriptions.push(
    vscode.languages.registerDocumentSemanticTokensProvider(
      { language: "alita" },
      semanticProvider,
      legend,
    ),
  );

  // task provider
  const taskProvider: vscode.TaskProvider = {
    provideTasks: () => {
      const task = new vscode.Task(
        { type: "alita", task: "sample" },
        vscode.TaskScope.Workspace,
        "sample",
        "alita",
        new vscode.ShellExecution("echo Alita task"),
      );
      return [task];
    },
    resolveTask: (_task: vscode.Task) => undefined,
  };
  context.subscriptions.push(
    vscode.tasks.registerTaskProvider("alita", taskProvider),
  );

  // debug adapter
  class InlineDebugAdapter implements vscode.DebugAdapter {
    handleMessage(_message: vscode.DebugProtocolMessage): void {}
    dispose(): void {}
    readonly onDidSendMessage = new vscode.EventEmitter<vscode.DebugProtocolMessage>().event;
  }
  class AlitaDebugFactory implements vscode.DebugAdapterDescriptorFactory {
    createDebugAdapterDescriptor(_session: vscode.DebugSession) {
      return new vscode.DebugAdapterInlineImplementation(
        new InlineDebugAdapter(),
      );
    }
  }
  context.subscriptions.push(
    vscode.debug.registerDebugAdapterDescriptorFactory(
      "alita",
      new AlitaDebugFactory(),
    ),
  );
}

export function deactivate(): Thenable<void> | undefined {
  return client ? client.stop() : undefined;
}
