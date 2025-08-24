/**
 * VS Code → Cortex Event Bridge
 * Emits structured events to JSONL for Cortex consumption
 */

import * as vscode from 'vscode';
import * as fs from 'fs';
import * as path from 'path';
import * as os from 'os';
import { randomUUID } from 'crypto';

export interface CortexEvent {
    id: string;
    kind: string;
    ts: number;
    actor: string;
    payload: {
        tool?: string;
        args?: any;
        ok?: boolean;
        meta?: {
            PROMPT_VERSION?: string;
            ARCHITECTURE_HASH?: string;
            VERIFICATION_MODE?: string;
            [key: string]: any;
        };
        [key: string]: any;
    };
    schema_version: string;
}

export class CortexEventEmitter {
    private telemetryPath: string;
    private context: vscode.ExtensionContext;

    constructor(context: vscode.ExtensionContext) {
        this.context = context;
        this.telemetryPath = path.join(os.homedir(), '.super-alita', 'telemetry.jsonl');
        this.ensureTelemetryDir();
    }

    private ensureTelemetryDir(): void {
        const dir = path.dirname(this.telemetryPath);
        if (!fs.existsSync(dir)) {
            fs.mkdirSync(dir, { recursive: true });
        }
    }

    private getCurrentMeta(): CortexEvent['payload']['meta'] {
        return {
            PROMPT_VERSION: '2.0.0',
            ARCHITECTURE_HASH: 'sha256:a3f4b5c6d7e8', // TODO: compute from actual architecture
            VERIFICATION_MODE: 'ACTIVE',
            extension_version: this.context.extension.packageJSON.version,
            vscode_version: vscode.version,
            workspace_folder: vscode.workspace.workspaceFolders?.[0]?.uri.fsPath || 'unknown'
        };
    }

    public emitEvent(kind: string, actor: string, payload: Partial<CortexEvent['payload']>): void {
        const event: CortexEvent = {
            id: randomUUID(),
            kind,
            ts: Date.now() / 1000,
            actor,
            payload: {
                ...payload,
                meta: {
                    ...this.getCurrentMeta(),
                    ...payload.meta
                }
            },
            schema_version: 'v1'
        };

        try {
            fs.appendFileSync(this.telemetryPath, JSON.stringify(event) + '\n', 'utf8');
        } catch (error) {
            console.error('Failed to emit Cortex event:', error);
        }
    }

    // Guardian-specific event emitters
    public emitGuardianAudit(file: string, rule: string, findings: any[], success: boolean): void {
        this.emitEvent('TOOL_RUN', 'agent', {
            tool: 'super_alita_guardian',
            args: {
                file,
                rule,
                version: '2.0.0',
                findings_count: findings.length
            },
            ok: success,
            findings
        });
    }

    public emitCopilotQuery(text: string, language?: string, filePath?: string): void {
        this.emitEvent('COPILOT_QUERY', 'user', {
            text,
            language,
            file_path: filePath,
            timestamp: Date.now()
        });
    }

    public emitTerminalCommand(command: string, exitCode: number, stdout?: string, stderr?: string): void {
        this.emitEvent('TERMINAL_COMMAND', 'agent', {
            command,
            exit_code: exitCode,
            stdout: stdout?.substring(0, 1000), // Truncate long output
            stderr: stderr?.substring(0, 1000),
            success: exitCode === 0
        });
    }

    public emitUserFeedback(rating: number, comment?: string, context?: any): void {
        this.emitEvent('USER_FEEDBACK', 'user', {
            rating,
            comment,
            context,
            timestamp: Date.now()
        });
    }

    public emitArchitecturalCompliance(file: string, compliance: any): void {
        this.emitEvent('ARCHITECTURAL_AUDIT', 'agent', {
            tool: 'super_alita_guardian',
            args: {
                file,
                audit_type: 'architectural_compliance'
            },
            ok: compliance.passed,
            compliance_score: compliance.score,
            violations: compliance.violations,
            suggestions: compliance.suggestions
        });
    }
}