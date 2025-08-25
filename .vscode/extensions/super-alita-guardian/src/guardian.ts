import * as vscode from 'vscode';
import { CortexEventEmitter } from './vscode_listener';

interface TelemetryMetrics {
    totalInteractions: number;
    complianceScore: number;
    guidelineReferences: Record<string, number>;
    modeUsage: Record<string, number>;
    copilotPerformance: {
        responsesGenerated: number;
        userRatings: { rating: number; timestamp: string; context: string }[];
        responseAcceptanceRate: number;
        averageResponseTime: number;
        architecturalComplianceRate: number;
        feedbackPatterns: Record<string, number>;
        contextualAccuracy: number;
        codeQualityScore: number;
    };
    ideInteraction: {
        editsTracked: number;
        filesModified: string[];
        commandsUsed: Record<string, number>;
        errorPatterns: Record<string, number>;
        sessionDuration: number;
        productivityMetrics: {
            linesGenerated: number;
            functionsCreated: number;
            bugsIntroduced: number;
            testsCovered: number;
        };
    };
}

interface ArchitecturalGuideline {
    id: number;
    title: string;
    patterns: string[];
    violations: string[];
}

export class SuperAlitaGuardian {
    private static readonly GUIDELINES: ArchitecturalGuideline[] = [
        {
            id: 1,
            title: "Plugin Architecture",
            patterns: ["PluginInterface", "async def setup", "async def shutdown"],
            violations: ["standalone plugin", "missing interface"]
        },
        {
            id: 2,
            title: "Tool Registry Management", 
            patterns: ["SimpleAbilityRegistry", "register(contract)", "schema validation"],
            violations: ["separate registry", "no validation"]
        },
        {
            id: 3,
            title: "REUG State Machine Patterns",
            patterns: ["async def handle_state", "TransitionTrigger", "ExecutionPlan"],
            violations: ["sync handlers", "manual transitions"]
        },
        {
            id: 4,
            title: "Event Bus Patterns",
            patterns: ["create_event()", "source_plugin", "event schemas"],
            violations: ["direct event creation", "missing source"]
        },
        {
            id: 5,
            title: "Component Integration",
            patterns: ["DecisionPolicyEngine", "app.state", "REUG streaming"],
            violations: ["tight coupling", "direct imports"]
        }
    ];

    private telemetryMetrics: TelemetryMetrics = {
        totalInteractions: 0,
        complianceScore: 0,
        guidelineReferences: {},
        modeUsage: {},
        copilotPerformance: {
            responsesGenerated: 0,
            userRatings: [],
            responseAcceptanceRate: 0,
            averageResponseTime: 0,
            architecturalComplianceRate: 0,
            feedbackPatterns: {},
            contextualAccuracy: 0,
            codeQualityScore: 0
        },
        ideInteraction: {
            editsTracked: 0,
            filesModified: [],
            commandsUsed: {},
            errorPatterns: {},
            sessionDuration: 0,
            productivityMetrics: {
                linesGenerated: 0,
                functionsCreated: 0,
                bugsIntroduced: 0,
                testsCovered: 0
            }
        }
    };

    private sessionStartTime: number = Date.now();
    private lastCopilotResponse: { content: string; timestamp: number; context: string } | null = null;
    private cortexEmitter: CortexEventEmitter;

    constructor(private context: vscode.ExtensionContext) {
        this.cortexEmitter = new CortexEventEmitter(context);
        this.initializeTelemetryTracking();
    }

    private initializeTelemetryTracking() {
        // Track document changes for Copilot feedback
        vscode.workspace.onDidChangeTextDocument((event) => {
            this.trackDocumentChanges(event);
        });

        // Track command execution
        vscode.commands.registerCommand('super-alita.internal.trackCommand', (commandId: string) => {
            this.trackCommandUsage(commandId);
        });

        // Track editor actions
        vscode.window.onDidChangeActiveTextEditor((editor) => {
            if (editor) {
                this.trackEditorActivity(editor);
            }
        });

        // Track diagnostics (errors/warnings)
        vscode.languages.onDidChangeDiagnostics((event) => {
            this.trackDiagnosticsChanges(event);
        });
    }

    async handleChatRequest(
        request: vscode.ChatRequest,
        context: vscode.ChatContext,
        stream: vscode.ChatResponseStream,
        token: vscode.CancellationToken
    ): Promise<void> {
        
        const responseStartTime = Date.now();
        
        // Track interaction
        this.telemetryMetrics.totalInteractions++;
        this.telemetryMetrics.copilotPerformance.responsesGenerated++;
        
        // Emit Cortex event for Copilot query
        this.cortexEmitter.emitCopilotQuery(
            request.prompt,
            vscode.window.activeTextEditor?.document.languageId,
            vscode.window.activeTextEditor?.document.fileName
        );
        
        try {
            // Acknowledge Guardian role
            stream.markdown("🛡️ **Super Alita Architectural Guardian v2.0** - Expert guidance for agent architecture\\n\\n");
            
            const userMessage = request.prompt;
            const mode = this.detectMode(userMessage);
            
            // Track mode usage
            this.telemetryMetrics.modeUsage[mode] = (this.telemetryMetrics.modeUsage[mode] || 0) + 1;
            
            stream.markdown(`**Mode:** ${mode.charAt(0).toUpperCase() + mode.slice(1)}\\n\\n`);
            
            let responseContent = "";
            
            switch (mode) {
                case 'audit':
                    await this.handleAuditMode(userMessage, stream, token);
                    responseContent = "Workspace audit completed";
                    break;
                case 'refactor':
                    await this.handleRefactorMode(userMessage, stream, token);
                    responseContent = "Refactoring suggestions provided";
                    break;
                case 'generator':
                    await this.handleGeneratorMode(userMessage, stream, token);
                    responseContent = "Code generation completed";
                    break;
                default:
                    await this.handleGuardianMode(userMessage, stream, token);
                    responseContent = "Architectural guidance provided";
            }
            
            // Track response time
            const responseTime = Date.now() - responseStartTime;
            this.telemetryMetrics.copilotPerformance.averageResponseTime = 
                (this.telemetryMetrics.copilotPerformance.averageResponseTime + responseTime) / 2;
            
            // Store last response for rating
            this.lastCopilotResponse = {
                content: responseContent,
                timestamp: Date.now(),
                context: `${mode} mode: ${userMessage.substring(0, 100)}...`
            };
            
            // Add rating prompt
            stream.markdown("\\n---\\n💭 **How was this response?** Use `Super Alita: Rate Last Copilot Response` to provide feedback\\n");
            
            // Update telemetry
            await this.updateTelemetryDashboard();
            
        } catch (error) {
            stream.markdown(`❌ **Error:** ${error instanceof Error ? error.message : 'Unknown error occurred'}\\n`);
        }
    }

    private detectMode(message: string): string {
        const lowerMessage = message.toLowerCase();
        
        if (lowerMessage.includes('audit') || lowerMessage.includes('review')) {
            return 'audit';
        } else if (lowerMessage.includes('refactor') || lowerMessage.includes('fix')) {
            return 'refactor';
        } else if (lowerMessage.includes('generate') || lowerMessage.includes('create')) {
            return 'generator';
        }
        
        return 'guardian';
    }

    private async handleGuardianMode(
        message: string,
        stream: vscode.ChatResponseStream,
        token: vscode.CancellationToken
    ): Promise<void> {
        stream.markdown("**Guardian Mode Active** - Reviewing architectural compliance\\n\\n");
        
        // Analyze message for architectural patterns
        const violations = this.analyzeArchitecturalCompliance(message);
        
        // Emit Cortex event for guardian audit
        this.cortexEmitter.emitGuardianAudit(
            'chat_message', 
            'ArchitecturalCompliance', 
            violations, 
            violations.length === 0
        );
        
        if (violations.length > 0) {
            stream.markdown("### 🚨 Architectural Violations Detected\\n\\n");
            
            for (const violation of violations) {
                stream.markdown(`**Guideline #${violation.guideline}**: ${violation.description}\\n`);
                stream.markdown(`- **Issue**: ${violation.issue}\\n`);
                stream.markdown(`- **Fix**: ${violation.fix}\\n\\n`);
                
                // Track guideline reference
                const guidelineKey = `guideline_${violation.guideline}`;
                this.telemetryMetrics.guidelineReferences[guidelineKey] = 
                    (this.telemetryMetrics.guidelineReferences[guidelineKey] || 0) + 1;
            }
        } else {
            stream.markdown("✅ **Architecture looks good!** No violations detected.\\n\\n");
        }
        
        // Provide architectural guidance
        stream.markdown("### 📋 Architectural Guidelines Reminder\\n\\n");
        for (const guideline of SuperAlitaGuardian.GUIDELINES) {
            stream.markdown(`**${guideline.id}. ${guideline.title}**\\n`);
            stream.markdown(`- Required patterns: ${guideline.patterns.join(', ')}\\n\\n`);
        }
    }

    private async handleAuditMode(
        message: string,
        stream: vscode.ChatResponseStream,
        token: vscode.CancellationToken
    ): Promise<void> {
        stream.markdown("**Audit Mode Active** - Performing comprehensive architectural review\\n\\n");
        
        try {
            // Get workspace files
            const pythonFiles = await vscode.workspace.findFiles('**/*.py', '**/node_modules/**', 100);
            
            stream.markdown(`### 📊 Workspace Audit Results\\n`);
            stream.markdown(`**Files analyzed**: ${pythonFiles.length}\\n\\n`);
            
            let totalViolations = 0;
            let compliantFiles = 0;
            
            for (const file of pythonFiles.slice(0, 10)) { // Limit to prevent timeout
                const document = await vscode.workspace.openTextDocument(file);
                const content = document.getText();
                
                const violations = this.analyzeArchitecturalCompliance(content);
                
                if (violations.length === 0) {
                    compliantFiles++;
                } else {
                    totalViolations += violations.length;
                    stream.markdown(`**${file.path}**: ${violations.length} violations\\n`);
                }
            }
            
            const compliancePercentage = (compliantFiles / Math.min(pythonFiles.length, 10)) * 100;
            
            // Emit comprehensive architectural compliance event
            this.cortexEmitter.emitArchitecturalCompliance('workspace_audit', {
                passed: compliancePercentage >= 80,
                score: compliancePercentage,
                violations: totalViolations,
                suggestions: [`Analyzed ${pythonFiles.length} files`, `${compliantFiles} compliant`, `${totalViolations} violations found`]
            });
            
            stream.markdown(`\\n### 📈 Compliance Summary\\n`);
            stream.markdown(`- **Compliance Rate**: ${compliancePercentage.toFixed(1)}%\\n`);
            stream.markdown(`- **Total Violations**: ${totalViolations}\\n`);
            stream.markdown(`- **Compliant Files**: ${compliantFiles}\\n`);
            
        } catch (error) {
            stream.markdown(`Error during audit: ${error instanceof Error ? error.message : 'Unknown error'}\\n`);
        }
    }

    private async handleRefactorMode(
        message: string,
        stream: vscode.ChatResponseStream,
        token: vscode.CancellationToken
    ): Promise<void> {
        stream.markdown("**Refactor Mode Active** - Transforming code to follow architectural patterns\\n\\n");
        
        const violations = this.analyzeArchitecturalCompliance(message);
        
        if (violations.length > 0) {
            stream.markdown("### 🔧 Refactoring Suggestions\\n\\n");
            
            for (const violation of violations) {
                stream.markdown(`**Fix for Guideline #${violation.guideline}:**\\n`);
                const refactorCode = this.generateRefactorCode(violation);
                stream.markdown("```python\\n" + refactorCode + "\\n```\\n\\n");
            }
        } else {
            stream.markdown("✅ No refactoring needed - code follows architectural patterns!\\n");
        }
    }

    private async handleGeneratorMode(
        message: string,
        stream: vscode.ChatResponseStream,
        token: vscode.CancellationToken
    ): Promise<void> {
        stream.markdown("**Generator Mode Active** - Creating compliant architectural components\\n\\n");
        
        // Detect what to generate based on message
        const lowerMessage = message.toLowerCase();
        
        if (lowerMessage.includes('plugin')) {
            stream.markdown("### 🔌 Generating Plugin Template\\n\\n");
            stream.markdown("```python\\nfrom src.core.plugin_interface import PluginInterface\\nfrom src.core.events import create_event\\n\\nclass NewPlugin(PluginInterface):\\n    @property\\n    def name(self) -> str:\\n        return 'new_plugin'\\n    \\n    async def setup(self, event_bus, store, config):\\n        self.event_bus = event_bus\\n        await self.register_capabilities()\\n    \\n    async def shutdown(self):\\n        pass\\n```\\n\\n");
        } else if (lowerMessage.includes('event')) {
            stream.markdown("### 📢 Generating Event Handler\\n\\n");
            stream.markdown("```python\\nfrom src.core.events import create_event\\n\\nasync def handle_custom_event(event_data):\\n    response_event = create_event(\\n        'custom_event_processed',\\n        source_plugin='event_handler',\\n        result=event_data\\n    )\\n    return response_event\\n```\\n\\n");
        } else {
            stream.markdown("Please specify what to generate (plugin, event handler, tool, etc.)\\n");
        }
    }

    private analyzeArchitecturalCompliance(content: string): Array<{
        guideline: number;
        description: string;
        issue: string;
        fix: string;
    }> {
        const violations = [];
        
        // Check each guideline
        for (const guideline of SuperAlitaGuardian.GUIDELINES) {
            const hasPatterns = guideline.patterns.some(pattern => 
                content.includes(pattern)
            );
            
            const hasViolations = guideline.violations.some(violation => 
                content.toLowerCase().includes(violation.toLowerCase())
            );
            
            if (hasViolations || (!hasPatterns && content.includes('class') && content.includes('def'))) {
                violations.push({
                    guideline: guideline.id,
                    description: guideline.title,
                    issue: hasViolations ? "Contains violation patterns" : "Missing required patterns",
                    fix: `Use: ${guideline.patterns.join(', ')}`
                });
            }
        }
        
        return violations;
    }

    private generateRefactorCode(violation: any): string {
        switch (violation.guideline) {
            case 1:
                return "# Refactor to use PluginInterface\\nfrom src.core.plugin_interface import PluginInterface\\n\\nclass YourPlugin(PluginInterface):\\n    @property\\n    def name(self) -> str:\\n        return 'your_plugin'\\n    \\n    async def setup(self, event_bus, store, config):\\n        pass\\n    \\n    async def shutdown(self):\\n        pass";
            
            case 2:
                return "# Use unified registry\\nfrom src.main import app\\n\\n# Register tool via unified registry\\nawait app.state.registry.register(tool_contract)";
            
            case 4:
                return "# Use event creation helper\\nfrom src.core.events import create_event\\n\\nevent = create_event(\\n    'your_event_type',\\n    source_plugin='your_plugin',\\n    data=your_data\\n)";
            
            default:
                return "# Apply appropriate architectural pattern";
        }
    }

    private async updateTelemetryDashboard(): Promise<void> {
        // Calculate compliance score
        const totalGuidelines = SuperAlitaGuardian.GUIDELINES.length;
        const referencedGuidelines = Object.keys(this.telemetryMetrics.guidelineReferences).length;
        this.telemetryMetrics.complianceScore = (referencedGuidelines / totalGuidelines) * 100;
        
        // Update telemetry file
        try {
            const telemetryData = {
                timestamp: new Date().toISOString(),
                metrics: this.telemetryMetrics,
                version: "2.0.0"
            };
            
            const workspaceFolder = vscode.workspace.workspaceFolders?.[0];
            if (workspaceFolder) {
                const telemetryFile = vscode.Uri.joinPath(
                    workspaceFolder.uri, 
                    '.vscode', 
                    'telemetry.json'
                );
                
                const content = JSON.stringify(telemetryData, null, 2);
                await vscode.workspace.fs.writeFile(telemetryFile, Buffer.from(content));
            }
        } catch (error) {
            console.error('Failed to update telemetry:', error);
        }
    }

    public async showTelemetryDashboard(): Promise<void> {
        const panel = vscode.window.createWebviewPanel(
            'super-alita-telemetry',
            'Super Alita Telemetry Dashboard',
            vscode.ViewColumn.Two,
            {
                enableScripts: true,
                retainContextWhenHidden: true
            }
        );

        panel.webview.html = this.getTelemetryDashboardHtml();
        
        // Refresh every 30 seconds
        const interval = setInterval(() => {
            if (panel.visible) {
                panel.webview.html = this.getTelemetryDashboardHtml();
            } else {
                clearInterval(interval);
            }
        }, 30000);
    }

    // Enhanced Bidirectional Telemetry Tracking Methods with Cortex Integration
    private trackDocumentChanges(event: vscode.TextDocumentChangeEvent): void {
        // Track IDE changes that might be Copilot-influenced
        if (event.contentChanges.length > 0) {
            this.telemetryMetrics.ideInteraction.editsTracked++;
            
            const fileName = event.document.fileName;
            if (!this.telemetryMetrics.ideInteraction.filesModified.includes(fileName)) {
                this.telemetryMetrics.ideInteraction.filesModified.push(fileName);
            }

            // Analyze content for architectural patterns
            const content = event.document.getText();
            this.analyzeCopilotGeneratedContent(content, fileName);
            
            // Count lines and functions
            const lines = event.contentChanges.reduce((acc, change) => 
                acc + (change.text.split('\\n').length - 1), 0);
            this.telemetryMetrics.ideInteraction.productivityMetrics.linesGenerated += lines;
            
            // Detect function creation
            const functionMatches = content.match(/(def |function |async def |class )/g);
            if (functionMatches) {
                this.telemetryMetrics.ideInteraction.productivityMetrics.functionsCreated += functionMatches.length;
            }

            // Emit Cortex event for document changes
            this.cortexEmitter.emitEvent('IDE_INTERACTION', 'user', {
                action: 'document_change',
                file: fileName,
                lines_changed: event.contentChanges.length,
                lines_generated: lines,
                functions_detected: functionMatches?.length || 0,
                timestamp: Date.now()
            });
        }
    }

    private trackCommandUsage(commandId: string): void {
        this.telemetryMetrics.ideInteraction.commandsUsed[commandId] = 
            (this.telemetryMetrics.ideInteraction.commandsUsed[commandId] || 0) + 1;

        // Emit Cortex event for command usage
        this.cortexEmitter.emitEvent('COMMAND_EXECUTION', 'user', {
            command: commandId,
            count: this.telemetryMetrics.ideInteraction.commandsUsed[commandId],
            timestamp: Date.now()
        });
    }

    private trackEditorActivity(editor: vscode.TextEditor): void {
        // Track which files are being worked on
        const fileName = editor.document.fileName;
        if (!this.telemetryMetrics.ideInteraction.filesModified.includes(fileName)) {
            this.telemetryMetrics.ideInteraction.filesModified.push(fileName);
        }

        // Emit Cortex event for editor activity
        this.cortexEmitter.emitEvent('EDITOR_ACTIVITY', 'user', {
            action: 'editor_focus',
            file: fileName,
            language: editor.document.languageId,
            timestamp: Date.now()
        });
    }

    private trackDiagnosticsChanges(event: vscode.DiagnosticChangeEvent): void {
        // Track errors introduced (potential quality issues)
        for (const uri of event.uris) {
            const diagnostics = vscode.languages.getDiagnostics(uri);
            const errors = diagnostics.filter(d => d.severity === vscode.DiagnosticSeverity.Error);
            const warnings = diagnostics.filter(d => d.severity === vscode.DiagnosticSeverity.Warning);
            
            if (errors.length > 0) {
                this.telemetryMetrics.ideInteraction.productivityMetrics.bugsIntroduced += errors.length;
                
                // Categorize error patterns
                for (const error of errors) {
                    const errorType = this.categorizeError(error.message);
                    this.telemetryMetrics.ideInteraction.errorPatterns[errorType] = 
                        (this.telemetryMetrics.ideInteraction.errorPatterns[errorType] || 0) + 1;
                }
            }

            // Emit Cortex event for diagnostics
            this.cortexEmitter.emitEvent('DIAGNOSTIC_UPDATE', 'ide', {
                file: uri.fsPath,
                errors: errors.length,
                warnings: warnings.length,
                diagnostics: diagnostics.map(d => ({ message: d.message, severity: d.severity })),
                timestamp: Date.now()
            });
        }
    }

    private analyzeCopilotGeneratedContent(content: string, fileName: string) {
        // Analyze if content follows architectural guidelines
        let complianceScore = 0;
        const totalChecks = SuperAlitaGuardian.GUIDELINES.length;
        
        for (const guideline of SuperAlitaGuardian.GUIDELINES) {
            const hasRequiredPatterns = guideline.patterns.some(pattern => 
                content.includes(pattern)
            );
            if (hasRequiredPatterns) {
                complianceScore++;
            }
        }
        
        const complianceRate = (complianceScore / totalChecks) * 100;
        this.telemetryMetrics.copilotPerformance.architecturalComplianceRate = 
            (this.telemetryMetrics.copilotPerformance.architecturalComplianceRate + complianceRate) / 2;
    }

    private categorizeError(errorMessage: string): string {
        if (errorMessage.includes('syntax')) return 'syntax_error';
        if (errorMessage.includes('import')) return 'import_error'; 
        if (errorMessage.includes('undefined')) return 'undefined_reference';
        if (errorMessage.includes('type')) return 'type_error';
        if (errorMessage.includes('indentation')) return 'indentation_error';
        return 'other_error';
    }

    public async rateCopilotResponse(rating: number, feedback?: string) {
        if (this.lastCopilotResponse) {
            this.telemetryMetrics.copilotPerformance.userRatings.push({
                rating,
                timestamp: new Date().toISOString(),
                context: this.lastCopilotResponse.context + (feedback ? ` | ${feedback}` : '')
            });
            
            // Update acceptance rate
            const totalRatings = this.telemetryMetrics.copilotPerformance.userRatings.length;
            const positiveRatings = this.telemetryMetrics.copilotPerformance.userRatings
                .filter(r => r.rating >= 4).length;
            this.telemetryMetrics.copilotPerformance.responseAcceptanceRate = 
                (positiveRatings / totalRatings) * 100;
                
            await this.updateTelemetryDashboard();
        }
    }

    public async showCopilotFeedbackDashboard() {
        const panel = vscode.window.createWebviewPanel(
            'super-alita-copilot-feedback',
            'Copilot Performance Analysis',
            vscode.ViewColumn.Two,
            { enableScripts: true, retainContextWhenHidden: true }
        );

        panel.webview.html = this.getCopilotFeedbackHtml();
    }

    public async exportTelemetryData(): Promise<void> {
        try {
            const exportData = {
                timestamp: new Date().toISOString(),
                sessionDuration: Date.now() - this.sessionStartTime,
                metrics: this.telemetryMetrics,
                version: "2.0.0",
                exportType: "comprehensive_bidirectional_telemetry"
            };
            
            const workspaceFolder = vscode.workspace.workspaceFolders?.[0];
            if (workspaceFolder) {
                const exportFile = vscode.Uri.joinPath(
                    workspaceFolder.uri, 
                    '.vscode', 
                    `telemetry-export-${Date.now()}.json`
                );
                
                const content = JSON.stringify(exportData, null, 2);
                await vscode.workspace.fs.writeFile(exportFile, Buffer.from(content));
                
                vscode.window.showInformationMessage(
                    `Telemetry data exported to ${exportFile.fsPath}`,
                    'Open File'
                ).then(selection => {
                    if (selection === 'Open File') {
                        vscode.window.showTextDocument(exportFile);
                    }
                });
            }
        } catch (error) {
            vscode.window.showErrorMessage(`Failed to export telemetry: ${error}`);
        }
    }

    private getCopilotFeedbackHtml(): string {
        const metrics = this.telemetryMetrics;
        const sessionDuration = Math.round((Date.now() - this.sessionStartTime) / 1000 / 60); // minutes
        
        const avgRating = metrics.copilotPerformance.userRatings.length > 0 
            ? metrics.copilotPerformance.userRatings.reduce((sum, r) => sum + r.rating, 0) / metrics.copilotPerformance.userRatings.length
            : 0;
            
        return `<!DOCTYPE html>
<html>
<head>
    <title>Copilot Performance Analysis</title>
    <style>
        body { 
            font-family: var(--vscode-font-family); 
            color: var(--vscode-foreground); 
            padding: 20px;
        }
        .metric-grid { 
            display: grid; 
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); 
            gap: 20px; 
            margin: 20px 0; 
        }
        .metric-card { 
            padding: 15px; 
            background: var(--vscode-editor-background); 
            border: 1px solid var(--vscode-panel-border);
            border-radius: 5px;
        }
        .metric-title { 
            font-size: 18px; 
            font-weight: bold; 
            margin-bottom: 10px; 
        }
        .metric-value { 
            font-size: 24px; 
            font-weight: bold; 
            color: var(--vscode-terminal-ansiGreen); 
        }
        .rating-history {
            max-height: 200px;
            overflow-y: auto;
            border: 1px solid var(--vscode-panel-border);
            padding: 10px;
            margin: 10px 0;
        }
        .rating-item {
            margin: 5px 0;
            padding: 5px;
            background: var(--vscode-list-hoverBackground);
        }
        table { 
            width: 100%; 
            border-collapse: collapse; 
            margin: 10px 0;
        }
        th, td { 
            border: 1px solid var(--vscode-panel-border); 
            padding: 8px; 
            text-align: left; 
        }
        .warning { color: var(--vscode-terminal-ansiYellow); }
        .error { color: var(--vscode-terminal-ansiRed); }
        .success { color: var(--vscode-terminal-ansiGreen); }
    </style>
</head>
<body>
    <h1>🤖 Copilot Performance Analysis Dashboard</h1>
    <p><strong>Session Duration:</strong> ${sessionDuration} minutes</p>
    
    <div class="metric-grid">
        <div class="metric-card">
            <div class="metric-title">Response Quality</div>
            <div class="metric-value">${avgRating.toFixed(1)}/5.0</div>
            <p>Average user rating from ${metrics.copilotPerformance.userRatings.length} responses</p>
        </div>
        
        <div class="metric-card">
            <div class="metric-title">Acceptance Rate</div>
            <div class="metric-value">${metrics.copilotPerformance.responseAcceptanceRate.toFixed(1)}%</div>
            <p>Percentage of responses rated 4+ stars</p>
        </div>
        
        <div class="metric-card">
            <div class="metric-title">Architectural Compliance</div>
            <div class="metric-value">${metrics.copilotPerformance.architecturalComplianceRate.toFixed(1)}%</div>
            <p>Code follows Super Alita patterns</p>
        </div>
        
        <div class="metric-card">
            <div class="metric-title">Productivity Impact</div>
            <div class="metric-value">${metrics.ideInteraction.productivityMetrics.linesGenerated}</div>
            <p>Lines of code generated</p>
        </div>
    </div>
    
    <div class="metric-grid">
        <div class="metric-card">
            <div class="metric-title">📝 Recent Ratings</div>
            <div class="rating-history">
                ${metrics.copilotPerformance.userRatings.slice(-5).reverse().map(rating => 
                    `<div class="rating-item">
                        <strong>${'⭐'.repeat(rating.rating)}${'☆'.repeat(5-rating.rating)}</strong> 
                        ${new Date(rating.timestamp).toLocaleTimeString()}
                        <br><small>${rating.context}</small>
                    </div>`
                ).join('')}
            </div>
        </div>
        
        <div class="metric-card">
            <div class="metric-title">🏗️ Code Quality Metrics</div>
            <table>
                <tr><th>Metric</th><th>Count</th></tr>
                <tr><td>Functions Created</td><td>${metrics.ideInteraction.productivityMetrics.functionsCreated}</td></tr>
                <tr><td>Files Modified</td><td>${metrics.ideInteraction.filesModified.length}</td></tr>
                <tr><td>Edits Tracked</td><td>${metrics.ideInteraction.editsTracked}</td></tr>
                <tr><td class="error">Bugs Introduced</td><td>${metrics.ideInteraction.productivityMetrics.bugsIntroduced}</td></tr>
            </table>
        </div>
    </div>
    
    <div class="metric-card">
        <div class="metric-title">🐛 Error Pattern Analysis</div>
        <table>
            <tr><th>Error Type</th><th>Frequency</th><th>Impact</th></tr>
            ${Object.entries(metrics.ideInteraction.errorPatterns).map(([type, count]) => 
                `<tr>
                    <td>${type.replace('_', ' ')}</td>
                    <td>${count}</td>
                    <td class="${count > 5 ? 'error' : count > 2 ? 'warning' : 'success'}">
                        ${count > 5 ? 'High' : count > 2 ? 'Medium' : 'Low'}
                    </td>
                </tr>`
            ).join('')}
        </table>
    </div>
    
    <div class="metric-card">
        <div class="metric-title">💡 Performance Insights</div>
        <ul>
            <li>${avgRating >= 4 ? '✅' : '⚠️'} Response Quality: ${avgRating >= 4 ? 'Excellent' : 'Needs Improvement'}</li>
            <li>${metrics.copilotPerformance.architecturalComplianceRate >= 70 ? '✅' : '⚠️'} Architectural Alignment: ${metrics.copilotPerformance.architecturalComplianceRate >= 70 ? 'Good' : 'Review Guidelines'}</li>
            <li>${metrics.ideInteraction.productivityMetrics.bugsIntroduced < 5 ? '✅' : '⚠️'} Code Quality: ${metrics.ideInteraction.productivityMetrics.bugsIntroduced < 5 ? 'Stable' : 'Monitor Errors'}</li>
            <li>${metrics.ideInteraction.editsTracked > 0 ? '✅' : '⚠️'} Activity Level: ${metrics.ideInteraction.editsTracked > 0 ? 'Active' : 'Low Activity'}</li>
        </ul>
    </div>
    
    <div style="margin-top: 20px; text-align: center;">
        <p><em>This data helps improve Copilot's architectural guidance and code quality</em></p>
    </div>
</body>
</html>`;
    }

    private getTelemetryDashboardHtml(): string {
        const metrics = this.telemetryMetrics;
        
        return `<!DOCTYPE html>
<html>
<head>
    <title>Super Alita Telemetry</title>
    <style>
        body { font-family: var(--vscode-font-family); color: var(--vscode-foreground); }
        .metric { margin: 10px 0; padding: 10px; background: var(--vscode-editor-background); }
        .score { font-size: 24px; font-weight: bold; color: var(--vscode-terminal-ansiGreen); }
        table { width: 100%; border-collapse: collapse; }
        th, td { border: 1px solid var(--vscode-panel-border); padding: 8px; text-align: left; }
    </style>
</head>
<body>
    <h1>🛡️ Super Alita Telemetry Dashboard</h1>
    <div class="metric">
        <h3>Compliance Score</h3>
        <div class="score">${metrics.complianceScore.toFixed(1)}%</div>
    </div>
    <div class="metric">
        <h3>Total Interactions</h3>
        <div>${metrics.totalInteractions}</div>
    </div>
    <div class="metric">
        <h3>Mode Usage</h3>
        <table>
            <tr><th>Mode</th><th>Count</th></tr>
            ${Object.entries(metrics.modeUsage).map(([mode, count]) => 
                `<tr><td>${mode}</td><td>${count}</td></tr>`
            ).join('')}
        </table>
    </div>
    <div class="metric">
        <h3>Guideline References</h3>
        <table>
            <tr><th>Guideline</th><th>References</th></tr>
            ${Object.entries(metrics.guidelineReferences).map(([guideline, count]) => 
                `<tr><td>${guideline}</td><td>${count}</td></tr>`
            ).join('')}
        </table>
    </div>
</body>
</html>`;
    }

    // Public method to track user feedback with Cortex emission
    trackUserFeedback(rating: number, comment?: string, context?: string): void {
        this.telemetryMetrics.copilotPerformance.userRatings.push({
            rating,
            timestamp: Date.now().toString(),
            context: context || ''
        });

        // Emit Cortex event for user feedback
        this.cortexEmitter.emitUserFeedback(rating, comment, { context, timestamp: Date.now() });
    }
}