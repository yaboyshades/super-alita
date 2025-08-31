const vscode = require('vscode');
const { exec } = require('child_process');
const path = require('path');

/**
 * @param {vscode.ExtensionContext} context
 */
function activate(context) {
    console.log('Prompt Pipeline extension is active!');

    // Register the command to optimize a prompt
    let disposable = vscode.commands.registerCommand('prompt-pipeline.optimizePrompt', async function () {
        try {
            // Get user input for prompt
            const promptInput = await vscode.window.showInputBox({
                placeHolder: 'Enter your prompt for optimization',
                prompt: 'Super Alita Prompt Optimization',
                value: 'How can I implement DeepConf consensus?'
            });

            if (!promptInput) return;

            // Show progress
            vscode.window.withProgress({
                location: vscode.ProgressLocation.Notification,
                title: "Optimizing prompt with Super Alita...",
                cancellable: false
            }, async (progress) => {
                progress.report({ increment: 0 });

                // Create or show output channel
                const channel = vscode.window.createOutputChannel('Super Alita Prompt Pipeline');
                channel.show(true);
                channel.appendLine('🔍 Optimizing prompt: ' + promptInput);
                progress.report({ increment: 20 });

                // Get workspace path
                const workspaceFolder = vscode.workspace.workspaceFolders?.[0];
                if (!workspaceFolder) {
                    channel.appendLine('❌ No workspace folder found');
                    return;
                }

                const pipelinePath = path.join(workspaceFolder.uri.fsPath, 'src', 'pipeline.py');
                progress.report({ increment: 40 });

                // Execute the pipeline script
                const pythonPath = path.join(workspaceFolder.uri.fsPath, '.venv', 'Scripts', 'python.exe');
                const command = `"${pythonPath}" "${pipelinePath}" "${promptInput.replace(/"/g, '\\"')}"`;

                exec(command, { cwd: workspaceFolder.uri.fsPath }, (error, stdout, stderr) => {
                    progress.report({ increment: 100 });

                    if (error) {
                        channel.appendLine('❌ Error: ' + error.message);
                        channel.appendLine(stderr);
                        return;
                    }

                    if (stderr) {
                        channel.appendLine('⚠️ Stderr: ' + stderr);
                    }

                    channel.appendLine('\n' + stdout);
                    channel.appendLine('\n✅ Prompt optimization complete!');
                });
            });
        } catch (error) {
            vscode.window.showErrorMessage('Error optimizing prompt: ' + error.message);
        }
    });

    context.subscriptions.push(disposable);
}

function deactivate() {}

module.exports = {
    activate,
    deactivate
};
