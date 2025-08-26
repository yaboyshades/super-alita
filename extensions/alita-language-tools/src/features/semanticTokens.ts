import * as vscode from 'vscode';

const tokenTypes = ['class','interface','enum','function','variable','property','ability','atom'] as const;
const tokenModifiers = ['declaration','readonly','static','telemetry'] as const;

const legend = new vscode.SemanticTokensLegend(tokenTypes as unknown as string[], tokenModifiers as unknown as string[]);

export function registerSemanticTokens() {
  const selector = { language: 'alita', scheme: '*' };

  const provider: vscode.DocumentSemanticTokensProvider = {
    provideDocumentSemanticTokens(document) {
      const builder = new vscode.SemanticTokensBuilder(legend);
      const text = document.getText();
      const regex = /\bability\s+([A-Za-z_]\w*)/g;
      for (const match of text.matchAll(regex)) {
        const start = document.positionAt(match.index! + 'ability '.length);
        const end   = document.positionAt(match.index! + 'ability '.length + match[1].length);
        builder.push(new vscode.Range(start, end), 'ability', ['declaration']);
      }
      return builder.build();
    }
  };

  return vscode.languages.registerDocumentSemanticTokensProvider(selector, provider, legend);
}
