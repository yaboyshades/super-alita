import * as vscode from 'vscode';
import { SuperAlitaGuardian } from './guardian';

let guardian: SuperAlitaGuardian;

export function activate(context: vscode.ExtensionContext) {
    console.log('Super Alita Guardian extension is now active!');
    
    // Initialize guardian
    guardian = new SuperAlitaGuardian(context);
    
    // Register chat participant
    const chatParticipant = vscode.chat.createChatParticipant(
        'super-alita.guardian',
        async (request, context, stream, token) => {
            await guardian.handleChatRequest(request, context, stream, token);
        }
    );
    
    // Set participant properties
    chatParticipant.iconPath = new vscode.ThemeIcon('shield');
    chatParticipant.followupProvider = {
        provideFollowups(result, context, token) {
            return [
                {
                    prompt: 'Show telemetry dashboard',
                    label: '📊 View Telemetry',
                    command: 'super-alita.showTelemetry'
                },
                {
                    prompt: 'Show Copilot performance feedback',
                    label: '🤖 Copilot Feedback',
                    command: 'super-alita.showCopilotFeedback'
                },
                {
                    prompt: 'Rate this response',
                    label: '⭐ Rate Response',
                    command: 'super-alita.rateCopilotResponse'
                },
                {
                    prompt: 'Run architectural compliance check',
                    label: '🔍 Check Compliance',
                    command: 'super-alita.runCompliance'
                },
                {
                    prompt: 'Audit workspace architecture',
                    label: '📋 Audit Workspace',
                    command: 'super-alita.auditWorkspace'
                }
            ];
        }
    };
    
    // Register commands
    const commands = [
        vscode.commands.registerCommand('super-alita.showTelemetry', async () => {
            await guardian.showTelemetryDashboard();
        }),
        
        vscode.commands.registerCommand('super-alita.runCompliance', async () => {
            // Trigger compliance check via chat
            const message = "Run architectural compliance check on current workspace";
            vscode.window.showInformationMessage(
                "Starting compliance check... Check the chat for results.",
                { detail: "Use @alita in chat to interact with the guardian" }
            );
        }),
        
        vscode.commands.registerCommand('super-alita.auditWorkspace', async () => {
            // Trigger workspace audit via chat
            const message = "Perform comprehensive architectural audit of workspace";
            vscode.window.showInformationMessage(
                "Starting workspace audit... Check the chat for results.",
                { detail: "Use @alita audit workspace in chat for detailed results" }
            );
        }),
        
        vscode.commands.registerCommand('super-alita.toggleGuardianMode', async () => {
            const config = vscode.workspace.getConfiguration('super-alita.guardian');
            const isEnabled = config.get<boolean>('enabled', true);
            
            await config.update('enabled', !isEnabled, vscode.ConfigurationTarget.Workspace);
            
            vscode.window.showInformationMessage(
                `Super Alita Guardian ${!isEnabled ? 'enabled' : 'disabled'}`
            );
        }),
        
        vscode.commands.registerCommand('super-alita.showCopilotFeedback', async () => {
            await guardian.showCopilotFeedbackDashboard();
        }),
        
        vscode.commands.registerCommand('super-alita.rateCopilotResponse', async () => {
            const rating = await vscode.window.showQuickPick([
                { label: '⭐⭐⭐⭐⭐ Excellent (5)', value: 5 },
                { label: '⭐⭐⭐⭐☆ Good (4)', value: 4 },
                { label: '⭐⭐⭐☆☆ Average (3)', value: 3 },
                { label: '⭐⭐☆☆☆ Poor (2)', value: 2 },
                { label: '⭐☆☆☆☆ Very Poor (1)', value: 1 }
            ], { placeHolder: 'Rate the last Copilot response' });
            
            if (rating) {
                const feedback = await vscode.window.showInputBox({
                    prompt: 'Optional: Provide additional feedback',
                    placeHolder: 'What worked well or could be improved?'
                });
                
                await guardian.rateCopilotResponse(rating.value, feedback);
                vscode.window.showInformationMessage(`Thank you for rating! (${rating.value}/5)`);
            }
        }),
        
        vscode.commands.registerCommand('super-alita.exportTelemetryData', async () => {
            await guardian.exportTelemetryData();
        })
    ];
    
    // Register status bar item
    const statusBarItem = vscode.window.createStatusBarItem(
        vscode.StatusBarAlignment.Right,
        100
    );
    statusBarItem.text = "$(shield) Alita Guardian";
    statusBarItem.tooltip = "Super Alita Architectural Guardian - Click for telemetry";
    statusBarItem.command = 'super-alita.showTelemetry';
    statusBarItem.show();
    
    // Add all disposables to context
    context.subscriptions.push(
        chatParticipant,
        statusBarItem,
        ...commands
    );
    
    // Show activation message
    vscode.window.showInformationMessage(
        "🛡️ Super Alita Guardian v2.0 activated! Use @alita in chat for architectural guidance.",
        "Show Dashboard"
    ).then(selection => {
        if (selection === "Show Dashboard") {
            guardian.showTelemetryDashboard();
        }
    });
}

export function deactivate() {
    console.log('Super Alita Guardian extension deactivated');
}