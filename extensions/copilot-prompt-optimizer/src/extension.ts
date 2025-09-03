// vscode import already present at top of file
import { amplifyPrompt, gatherAmplifyContext, optimizePrompt } from './pipeline';
import { safeRunCommandInteractive, stopLongRunningTerminals } from './terminalManager';
import * as vscode from 'vscode';
import { SnippetBrowser } from './snippetBrowser';

interface ExtensionBridge {
  windsurf?: vscode.Extension<any>;
  cursor?: vscode.Extension<any>;
  tabnine?: vscode.Extension<any>;
}

class ExtensionIntegrationManager {
  public bridges: ExtensionBridge = {};

  constructor() {
    this.initializeBridges();
  }

  private initializeBridges() {
    // Known AI extension IDs
    const knownExtensions = {
      windsurf: ['windsurf.windsurf', 'windsurf-ai.windsurf'],
      cursor: ['cursor.cursor-ai', 'cursor.cursor'],
      tabnine: ['tabnine.tabnine-vscode']
    };

    for (const [name, ids] of Object.entries(knownExtensions)) {
      for (const id of ids) {
        const ext = vscode.extensions.getExtension(id);
        if (ext) {
          this.bridges[name as keyof ExtensionBridge] = ext;
          console.log(`Found ${name} extension: ${id}`);
          break;
        }
      }
    }
  }

  async activateExtension(name: keyof ExtensionBridge): Promise<any> {
    const extension = this.bridges[name];
    if (!extension) return null;

    if (!extension.isActive) {
      await extension.activate();
    }
    return extension.exports;
  }

  async sendToWindsurf(prompt: string, options?: any): Promise<string | null> {
    try {
      const windsurfAPI = await this.activateExtension('windsurf');
      if (!windsurfAPI) return null;

      // Try common API patterns for AI extensions
      if (windsurfAPI.generateResponse) {
        return await windsurfAPI.generateResponse(prompt, options);
      }

      if (windsurfAPI.chat) {
        return await windsurfAPI.chat(prompt, options);
      }

      // Fallback: use commands if available
      await vscode.commands.executeCommand('windsurf.chat', prompt);
      return "Sent to Windsurf via command";
    } catch (error) {
      console.warn('Failed to communicate with Windsurf:', error);
      return null;
    }
  }

  async orchestrateAIWorkflow(prompt: string): Promise<string> {
    // Step 1: Optimize with our extension
    const optimized = optimizePrompt(prompt);
    const ctx = gatherAmplifyContext();
    const amplified = amplifyPrompt(optimized, ctx);

    // Step 2: Try Windsurf first
    const windsurfResult = await this.sendToWindsurf(amplified);
    if (windsurfResult) {
      return `Windsurf Response:\n${windsurfResult}`;
    }

    // Step 3: Fallback to Copilot LM API
    const copilotResult = await sendViaLM(amplified);
    if (copilotResult) {
      return "Sent to Copilot LM API";
    }

    // Step 4: Final fallback
    await vscode.env.clipboard.writeText(amplified);
    return "Optimized prompt copied to clipboard";
  }
}

let extensionManager: ExtensionIntegrationManager;
let snippetBrowser: SnippetBrowser;

async function sendViaLM(optimized: string): Promise<boolean> {
  // Best-effort use of VS Code LM Chat API if present.
  try {
    const lm: any = (vscode as any).lm;
    if (!lm || !lm.selectChatModels) return false;

    const models = await lm.selectChatModels({
      // Prefer GitHub Copilot provider when available
      vendor: 'github',
    });
    const model = models?.[0] ?? (await lm.selectChatModels({}))[0];
    if (!model) return false;

    const stream = await model.sendRequest([{ role: 'user', content: optimized }], {});
    let text = '';
    for await (const part of stream.text) {
      text += part;
    }
    const doc = await vscode.workspace.openTextDocument({ language: 'markdown', content: text });
    await vscode.window.showTextDocument(doc, { preview: true });
    return true;
  } catch (err) {
    console.warn('LM routing failed:', err);
    return false;
  }
}

async function startOptimizedChat() {
  const input = await vscode.window.showInputBox({
    prompt: 'Copilot Prompt Optimizer - Enter your prompt to optimize and send',
    placeHolder: 'e.g., explain why my tests are flaky'
  });
  if (!input) return;

  // Enhanced workflow with extension orchestration
  const result = await extensionManager.orchestrateAIWorkflow(input);

  if (result.startsWith("Windsurf Response:")) {
    // Show Windsurf response in new document
    const doc = await vscode.workspace.openTextDocument({
      language: 'markdown',
      content: result
    });
    await vscode.window.showTextDocument(doc, { preview: true });
  } else {
    vscode.window.showInformationMessage(result);
  }
}

async function optimizeSelection() {
  const editor = vscode.window.activeTextEditor;
  if (!editor) {
    vscode.window.showWarningMessage('No active editor to optimize selection.');
    return;
  }
  const sel = editor.selection;
  const text = sel && !sel.isEmpty ? editor.document.getText(sel) : editor.document.getText();
  const optimized = optimizePrompt(text);
  const ctx = gatherAmplifyContext();
  const amplified = amplifyPrompt(optimized, ctx);

  const pick = await vscode.window.showQuickPick([
    { label: 'Replace selection with optimized', action: 'replace' },
    { label: 'Copy optimized to clipboard', action: 'copy' },
    { label: 'Send via LM (preview response)', action: 'lm' }
  ], { placeHolder: 'What would you like to do?' });
  if (!pick) return;

  if (pick.action === 'replace') {
    await editor.edit(edit => {
      if (sel && !sel.isEmpty) {
        edit.replace(sel, amplified);
      } else {
        const full = new vscode.Range(0, 0, editor.document.lineCount, 0);
        edit.replace(full, amplified);
      }
    });
  } else if (pick.action === 'copy') {
    await vscode.env.clipboard.writeText(amplified);
    vscode.window.showInformationMessage('Optimized prompt copied to clipboard.');
  } else if (pick.action === 'lm') {
    const ok = await sendViaLM(amplified);
    if (!ok) {
      await vscode.env.clipboard.writeText(amplified);
      vscode.window.showInformationMessage('LM not available. Prompt copied to clipboard instead.');
    }
  }
}

async function bridgeToExtension() {
  const availableExtensions = [];

  if (extensionManager.bridges.windsurf) {
    availableExtensions.push({ label: 'Windsurf Plugin', id: 'windsurf' });
  }
  if (extensionManager.bridges.cursor) {
    availableExtensions.push({ label: 'Cursor AI', id: 'cursor' });
  }
  if (extensionManager.bridges.tabnine) {
    availableExtensions.push({ label: 'Tabnine', id: 'tabnine' });
  }

  if (availableExtensions.length === 0) {
    vscode.window.showWarningMessage('No compatible AI extensions found.');
    return;
  }

  const selected = await vscode.window.showQuickPick(availableExtensions, {
    placeHolder: 'Select AI extension to bridge with Copilot'
  });

  if (!selected) return;

  const prompt = await vscode.window.showInputBox({
    prompt: `Enter prompt to send to ${selected.label}`,
    placeHolder: 'Your prompt will be optimized before sending'
  });

  if (!prompt) return;

  if (selected.id === 'windsurf') {
    const result = await extensionManager.sendToWindsurf(prompt);
    if (result) {
      const doc = await vscode.workspace.openTextDocument({
        language: 'markdown',
        content: `Windsurf Response:\n${result}`
      });
      await vscode.window.showTextDocument(doc, { preview: true });
    } else {
      vscode.window.showInformationMessage('Sent to Windsurf (no direct response available).');
    }
  } else {
    // For other extensions, reuse our optimization + LM pipeline
    const optimized = optimizePrompt(prompt);
    const ctx = gatherAmplifyContext();
    const amplified = amplifyPrompt(optimized, ctx);
    const ok = await sendViaLM(amplified);
    if (!ok) {
      await vscode.env.clipboard.writeText(amplified);
      vscode.window.showInformationMessage('Bridged prompt copied to clipboard.');
  }
}

}

async function insertSnippet() {
  await snippetBrowser.browseSnippets();
}

async function browseSnippets() {
  await snippetBrowser.browseSnippetsByCategory();
}

async function searchSnippets() {
  await snippetBrowser.searchSnippets();
}

async function insertSnippetByPrefix() {
  await snippetBrowser.insertSnippetByPrefix();
}

export function activate(context: vscode.ExtensionContext) {
  extensionManager = new ExtensionIntegrationManager();
  snippetBrowser = new SnippetBrowser(context);

  context.subscriptions.push(
    vscode.commands.registerCommand('copilotPromptOptimizer.startChat', startOptimizedChat),
    vscode.commands.registerCommand('copilotPromptOptimizer.optimizeSelection', optimizeSelection),
    vscode.commands.registerCommand('copilotPromptOptimizer.bridgeExtensions', bridgeToExtension),
    vscode.commands.registerCommand('copilotPromptOptimizer.safeRunCommand', safeRunCommandInteractive),
    vscode.commands.registerCommand('copilotPromptOptimizer.stopLongRunning', stopLongRunningTerminals),
    vscode.commands.registerCommand('copilotPromptOptimizer.githubSearchSelection', async () => {
      const editor = vscode.window.activeTextEditor;
      const sel = editor && !editor.selection.isEmpty ? editor.document.getText(editor.selection) : '';
      const q = await vscode.window.showInputBox({ prompt: 'GitHub code search query', value: sel || '' });
      if (!q) return;
      const url = `https://github.com/search?q=${encodeURIComponent(q)}&type=code`;
      await vscode.env.openExternal(vscode.Uri.parse(url));
    }),
    vscode.commands.registerCommand('copilotPromptOptimizer.reugPipeline', async () => {
      const goal = await vscode.window.showInputBox({ prompt: 'Goal/spec to implement (REUG pipeline)' });
      if (!goal) return;
      const filePath = await vscode.window.showInputBox({ prompt: 'Target file path (where to write code)' });
      if (!filePath) return;
      const language = await vscode.window.showInputBox({ prompt: 'Language (default: python)', value: 'python' });
      const useDiscoveryPick = await vscode.window.showQuickPick(['Yes (default)', 'No'], { placeHolder: 'Use GitHub discovery before coding?' });
      const testFirstPick = await vscode.window.showQuickPick(['Yes (default)', 'No'], { placeHolder: 'Generate tests first?' });

      const use_github_discovery = useDiscoveryPick !== 'No';
      const test_first = testFirstPick !== 'No';

      try {
        const body = {
          tool_id: 'ladder_reug_generate',
          args: {
            goal,
            file_path: filePath,
            language: language || 'python',
            use_github_discovery,
            test_first
          }
        };
        const resp = await fetch('http://127.0.0.1:8080/tools/execute', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(body)
        });
        if (!resp.ok) {
          const txt = await resp.text();
          vscode.window.showErrorMessage(`REUG pipeline failed: ${resp.status} ${txt}`);
          return;
        }
        const data: any = await resp.json();
        const cg = data.codegen || {};
        const wrote = cg.wrote ? 'yes' : 'no';
        const issues = cg.issue_count ?? 0;
        const summary = `REUG pipeline: wrote=${wrote}, issues=${issues}, file=${cg.file_path || filePath}`;
        vscode.window.showInformationMessage(summary);
        if (cg.file_path) {
          try {
            const doc = await vscode.workspace.openTextDocument(vscode.Uri.file(cg.file_path));
            await vscode.window.showTextDocument(doc, { preview: false });
          } catch {}
        }
      } catch (err: any) {
        vscode.window.showErrorMessage(`REUG pipeline error: ${err?.message || String(err)}`);
      }
    }),
    vscode.commands.registerCommand('copilotPromptOptimizer.insertSnippet', insertSnippet),
    vscode.commands.registerCommand('copilotPromptOptimizer.browseSnippets', browseSnippets),
    vscode.commands.registerCommand('copilotPromptOptimizer.searchSnippets', searchSnippets),
    vscode.commands.registerCommand('copilotPromptOptimizer.insertSnippetByPrefix', insertSnippetByPrefix)
  );
}

export function deactivate() { /* noop */ }
