/**
 * Built-in MCP Server Provider for Super Alita Agent
 * 
 * This extension leverages VS Code's built-in MCP support instead of external servers.
 * It registers the super-alita agent as a native MCP server definition provider
 * that VS Code can automatically discover and use.
 */

import * as vscode from 'vscode';
import * as path from 'path';
import * as fs from 'fs';

export class SuperAlitaMcpProvider implements vscode.McpServerDefinitionProvider<vscode.McpStdioServerDefinition> {
    private _onDidChangeMcpServerDefinitions = new vscode.EventEmitter<void>();
    readonly onDidChangeMcpServerDefinitions = this._onDidChangeMcpServerDefinitions.event;

    private workspaceFolder: vscode.WorkspaceFolder | undefined;
    private pythonPath: string | undefined;

    constructor() {
        this.workspaceFolder = vscode.workspace.workspaceFolders?.[0];
        this.discoverPythonPath();
    }

    private async discoverPythonPath(): Promise<void> {
        if (!this.workspaceFolder) {
            return;
        }

        // Try to find the virtual environment Python executable
        const venvPaths = [
            path.join(this.workspaceFolder.uri.fsPath, '.venv', 'Scripts', 'python.exe'),
            path.join(this.workspaceFolder.uri.fsPath, '.venv', 'bin', 'python'),
            path.join(this.workspaceFolder.uri.fsPath, 'venv', 'Scripts', 'python.exe'),
            path.join(this.workspaceFolder.uri.fsPath, 'venv', 'bin', 'python')
        ];

        for (const pythonPath of venvPaths) {
            if (fs.existsSync(pythonPath)) {
                this.pythonPath = pythonPath;
                console.log(`Found Python executable: ${pythonPath}`);
                break;
            }
        }

        // Fallback to system Python if no venv found
        if (!this.pythonPath) {
            this.pythonPath = 'python';
            console.log('Using system Python as fallback');
        }
    }

    async provideMcpServerDefinitions(token: vscode.CancellationToken): Promise<vscode.McpStdioServerDefinition[]> {
        if (!this.workspaceFolder || !this.pythonPath) {
            return [];
        }

        const agentIntegrationPath = path.join(
            this.workspaceFolder.uri.fsPath,
            'src',
            'vscode_integration',
            'agent_mcp_server.py'
        );

        // Check if the agent integration file exists
        if (!fs.existsSync(agentIntegrationPath)) {
            console.warn(`Super Alita agent integration not found at: ${agentIntegrationPath}`);
            return [];
        }

        const server = new vscode.McpStdioServerDefinition(
            'Super Alita Agent',
            this.pythonPath,
            [agentIntegrationPath],
            {
                'PYTHONPATH': this.workspaceFolder.uri.fsPath,
                'WORKSPACE_FOLDER': this.workspaceFolder.uri.fsPath,
                'VSCODE_EXTENSION_MODE': 'mcp-provider'
            },
            '1.0.0'
        );

        console.log('Providing Super Alita MCP server definition:', server);
        return [server];
    }

    async resolveMcpServerDefinition(server: vscode.McpStdioServerDefinition, token: vscode.CancellationToken): Promise<vscode.McpStdioServerDefinition> {
        // Ensure the server is properly configured before starting
        if (!this.workspaceFolder) {
            throw new Error('No workspace folder available for Super Alita agent');
        }

        // Validate that required files exist
        const agentPath = path.join(this.workspaceFolder.uri.fsPath, 'src', 'vscode_integration', 'agent_mcp_server.py');
        if (!fs.existsSync(agentPath)) {
            throw new Error(`Super Alita agent integration not found: ${agentPath}`);
        }

        console.log('Resolved Super Alita MCP server:', server);
        return server;
    }

    refreshServers(): void {
        this._onDidChangeMcpServerDefinitions.fire();
    }

    dispose(): void {
        this._onDidChangeMcpServerDefinitions.dispose();
    }
}

export class SuperAlitaBuiltinExtension {
    private mcpProvider: SuperAlitaMcpProvider;
    private disposables: vscode.Disposable[] = [];

    constructor(private context: vscode.ExtensionContext) {
        this.mcpProvider = new SuperAlitaMcpProvider();
    }

    async activate(): Promise<void> {
        try {
            // Register the MCP server definition provider with VS Code's built-in LM API
            const registration = vscode.lm.registerMcpServerDefinitionProvider(
                'super-alita-agent-provider',
                this.mcpProvider
            );

            this.disposables.push(registration);

            // Register commands for manual server management
            const refreshCommand = vscode.commands.registerCommand(
                'superAlita.refreshMcpServers',
                () => {
                    this.mcpProvider.refreshServers();
                    vscode.window.showInformationMessage('Super Alita MCP servers refreshed');
                }
            );

            const statusCommand = vscode.commands.registerCommand(
                'superAlita.showMcpStatus',
                async () => {
                    const servers = await this.mcpProvider.provideMcpServerDefinitions(
                        new vscode.CancellationTokenSource().token
                    );
                    
                    if (servers.length > 0) {
                        vscode.window.showInformationMessage(
                            `Super Alita Agent MCP server is available (${servers[0].label})`
                        );
                    } else {
                        vscode.window.showWarningMessage(
                            'Super Alita Agent MCP server is not available'
                        );
                    }
                }
            );

            this.disposables.push(refreshCommand, statusCommand);

            // Show activation message
            console.log('Super Alita built-in MCP provider activated');
            vscode.window.showInformationMessage(
                'Super Alita Agent is now available as a built-in MCP server'
            );

        } catch (error) {
            console.error('Failed to activate Super Alita built-in MCP provider:', error);
            vscode.window.showErrorMessage(
                `Failed to activate Super Alita MCP provider: ${error}`
            );
        }
    }

    deactivate(): void {
        this.disposables.forEach(d => d.dispose());
        this.mcpProvider.dispose();
        console.log('Super Alita built-in MCP provider deactivated');
    }
}

// Extension entry points for VS Code
export function activate(context: vscode.ExtensionContext): Promise<void> {
    const extension = new SuperAlitaBuiltinExtension(context);
    return extension.activate();
}

export function deactivate(): void {
    // Cleanup handled by SuperAlitaBuiltinExtension.deactivate()
}