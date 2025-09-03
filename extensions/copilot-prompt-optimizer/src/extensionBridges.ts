import * as vscode from 'vscode';

export interface AIExtensionAPI {
  generateResponse?(prompt: string, options?: any): Promise<string>;
  chat?(prompt: string, options?: any): Promise<string>;
  complete?(prompt: string, options?: any): Promise<string>;
}

export class ExtensionBridgeRegistry {
  private static bridges: Map<string, (prompt: string) => Promise<string | null>> = new Map();

  static registerBridge(extensionId: string, handler: (prompt: string) => Promise<string | null>) {
    this.bridges.set(extensionId, handler);
  }

  static async sendToExtension(extensionId: string, prompt: string): Promise<string | null> {
    const handler = this.bridges.get(extensionId);
    if (handler) {
      return await handler(prompt);
    }

    // Fallback to command-based communication
    try {
      await vscode.commands.executeCommand(`${extensionId}.chat`, prompt);
      return `Sent to ${extensionId} via command`;
    } catch (error) {
      console.warn(`Failed to send to ${extensionId}:`, error);
      return null;
    }
  }

  // Pre-register known extension bridges
  static initialize() {
    // Windsurf bridge
    this.registerBridge('windsurf', async (prompt: string) => {
      try {
        // Try various Windsurf command patterns
        const commands = [
          'windsurf.chat',
          'windsurf.generateCode',
          'windsurf.askQuestion'
        ];

        for (const cmd of commands) {
          try {
            await vscode.commands.executeCommand(cmd, prompt);
            return `Windsurf executed: ${cmd}`;
          } catch {}
        }

        return null;
      } catch (error) {
        return null;
      }
    });

    // Cursor bridge
    this.registerBridge('cursor', async (prompt: string) => {
      try {
        await vscode.commands.executeCommand('cursor.chat', prompt);
        return 'Sent to Cursor';
      } catch (error) {
        return null;
      }
    });

    // Super Alita integration
    this.registerBridge('super-alita', async (prompt: string) => {
      try {
        // Use Super Alita's enhanced consensus
        const response = await fetch('http://127.0.0.1:8080/ability/execute/deepconf_consensus', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            prompt,
            method: 'weighted_vote',
            num_samples: 3,
            temperature: 0.7
          })
        });

        if (response.ok) {
          const result: any = await response.json();
          return (result && (result.best_response || result.consensus_result || result.consensus_text)) as string | null;
        }
        return null;
      } catch (error) {
        return null;
      }
    });
  }
}
