import * as vscode from 'vscode';
import { SuperAlitaGuardian } from './guardian';

const LLM_KEY_SECRET = 'super-alita.llm-key';
const RUNTIME_HOST_SECRET = 'super-alita.runtime-host';

let guardian: SuperAlitaGuardian;

export function activate(context: vscode.ExtensionContext) {
    console.log('Super Alita Guardian extension is now active!');

    guardian = new SuperAlitaGuardian(context);

    const formatHostLabel = (value: string | undefined): string => {
        if (!value) {
            return '(runtime?)';
        }

        try {
            const url = new URL(value);
            return url.host || value;
        } catch (error) {
            return value.replace(/^https?:\/\//, '');
        }
    };

    const statusBarItem = vscode.window.createStatusBarItem(
        vscode.StatusBarAlignment.Right,
        100
    );
    statusBarItem.command = 'super-alita.openAgentDashboard';
    statusBarItem.show();

    const refreshStatusBar = async () => {
        const runtimeHost = await context.secrets.get(RUNTIME_HOST_SECRET);
        const llmKey = await context.secrets.get(LLM_KEY_SECRET);

        const hostLabel = formatHostLabel(runtimeHost ?? undefined);
        const secretSuffix = llmKey ? '' : ' !';

        statusBarItem.text = `$(shield) Alita ${hostLabel}${secretSuffix}`;
        statusBarItem.tooltip = runtimeHost
            ? `Super Alita Architectural Guardian\nRuntime host: ${runtimeHost}${llmKey ? '' : '\nSet an LLM key for tool access.'}`
            : 'Set a Super Alita runtime host to enable quick actions.';
    };

    const chatParticipant = vscode.chat.createChatParticipant(
        'super-alita.guardian',
        async (request, participantContext, stream, token) => {
            await guardian.handleChatRequest(request, participantContext, stream, token);
        }
    );

    chatParticipant.iconPath = new vscode.ThemeIcon('shield');
    chatParticipant.followupProvider = {
        provideFollowups() {
            return [
                {
                    prompt: 'Show telemetry dashboard',
                    label: '?? View Telemetry',
                    command: 'super-alita.showTelemetry'
                },
                {
                    prompt: 'Show Copilot performance feedback',
                    label: '?? Copilot Feedback',
                    command: 'super-alita.showCopilotFeedback'
                },
                {
                    prompt: 'Rate this response',
                    label: '? Rate Response',
                    command: 'super-alita.rateCopilotResponse'
                },
                {
                    prompt: 'Run architectural compliance check',
                    label: '?? Check Compliance',
                    command: 'super-alita.runCompliance'
                },
                {
                    prompt: 'Audit workspace architecture',
                    label: '?? Audit Workspace',
                    command: 'super-alita.auditWorkspace'
                },
                {
                    prompt: 'Open agent operations dashboard',
                    label: '?? Agent Dashboard',
                    command: 'super-alita.openAgentDashboard'
                },
                {
                    prompt: 'Configure Super Alita runtime host',
                    label: '?? Configure Runtime Host',
                    command: 'super-alita.setRuntimeHost'
                }
            ];
        }
    };

    const commands = [
        vscode.commands.registerCommand('super-alita.showTelemetry', async () => {
            await guardian.showTelemetryDashboard();
        }),
        vscode.commands.registerCommand('super-alita.runCompliance', async () => {
            vscode.window.showInformationMessage(
                'Starting compliance check… check the chat for results.',
                { detail: 'Use @alita in chat to interact with the guardian.' }
            );
        }),
        vscode.commands.registerCommand('super-alita.auditWorkspace', async () => {
            vscode.window.showInformationMessage(
                'Starting workspace audit… check the chat for results.',
                { detail: 'Use @alita audit workspace in chat for detailed results.' }
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
                { label: '????? Excellent (5)', value: 5 },
                { label: '????? Good (4)', value: 4 },
                { label: '????? Average (3)', value: 3 },
                { label: '????? Poor (2)', value: 2 },
                { label: '????? Very Poor (1)', value: 1 }
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
        }),
        vscode.commands.registerCommand('super-alita.setLlmKey', async () => {
            const key = await vscode.window.showInputBox({
                prompt: 'Enter LLM API key (stored securely in VS Code Secret Storage)',
                ignoreFocusOut: true,
                password: true,
                placeHolder: 'sk-…'
            });

            if (!key) {
                return;
            }

            await context.secrets.store(LLM_KEY_SECRET, key.trim());
            await refreshStatusBar();
            vscode.window.showInformationMessage('Stored Super Alita LLM key in Secret Storage.');
        }),
        vscode.commands.registerCommand('super-alita.setRuntimeHost', async () => {
            const currentHost = await context.secrets.get(RUNTIME_HOST_SECRET);
            const host = await vscode.window.showInputBox({
                prompt: 'Enter Super Alita runtime host',
                value: currentHost ?? 'http://127.0.0.1:8080',
                ignoreFocusOut: true
            });

            if (!host) {
                return;
            }

            await context.secrets.store(RUNTIME_HOST_SECRET, host.trim());
            await refreshStatusBar();
            vscode.window.showInformationMessage(`Runtime host set to ${host.trim()}.`);
        }),
        vscode.commands.registerCommand('super-alita.openAgentDashboard', async () => {
            const runtimeHost = await context.secrets.get(RUNTIME_HOST_SECRET);
            const llmKey = await context.secrets.get(LLM_KEY_SECRET);

            await guardian.showAgentDashboard({
                runtimeHost: runtimeHost ?? undefined,
                llmConfigured: Boolean(llmKey),
                hostConfigured: Boolean(runtimeHost)
            });
        })
    ];

    const secretWatcher = context.secrets.onDidChange(async (event) => {
        if (event.key === LLM_KEY_SECRET || event.key === RUNTIME_HOST_SECRET) {
            await refreshStatusBar();
        }
    });

    context.subscriptions.push(
        chatParticipant,
        statusBarItem,
        secretWatcher,
        ...commands
    );

    refreshStatusBar().catch((error) => {
        console.error('Failed to refresh Super Alita status bar details', error);
    });

    vscode.window.showInformationMessage(
        '??? Super Alita Guardian v2.0 activated! Use @alita in chat for architectural guidance.',
        'Open Agent Dashboard'
    ).then((selection) => {
        if (selection === 'Open Agent Dashboard') {
            vscode.commands.executeCommand('super-alita.openAgentDashboard');
        }
    });
}

export function deactivate() {
    console.log('Super Alita Guardian extension deactivated');
}
