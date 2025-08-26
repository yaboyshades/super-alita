import {
  createConnection, ProposedFeatures, InitializeParams, InitializeResult,
  TextDocuments, TextDocumentSyncKind, CompletionItem, CompletionItemKind, Diagnostic, DiagnosticSeverity
} from 'vscode-languageserver/node';
import { TextDocument } from 'vscode-languageserver-textdocument';

const connection = createConnection(ProposedFeatures.all);
const documents = new TextDocuments(TextDocument);

connection.onInitialize((_params: InitializeParams): InitializeResult => ({
  capabilities: {
    textDocumentSync: TextDocumentSyncKind.Incremental,
    completionProvider: { resolveProvider: false }
  }
}));

documents.onDidChangeContent(change => validate(change.document));
async function validate(doc: TextDocument) {
  const text = doc.getText();
  const diags: Diagnostic[] = [];
  const UPPER = /\b[A-Z]{2,}\b/g;
  let match: RegExpExecArray | null;
  while ((match = UPPER.exec(text))) {
    diags.push({
      severity: DiagnosticSeverity.Warning,
      range: { start: doc.positionAt(match.index), end: doc.positionAt(match.index + match[0].length) },
      message: `${match[0]} is all uppercase.`,
      source: 'alita-lsp'
    });
  }
  connection.sendDiagnostics({ uri: doc.uri, diagnostics: diags });
}

connection.onCompletion((_pos): CompletionItem[] => [
  { label: 'ability', kind: CompletionItemKind.Keyword },
  { label: 'atom', kind: CompletionItemKind.Class }
]);

documents.listen(connection);
connection.listen();
