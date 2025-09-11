/**
 * APE (Automatic Prompt Engineering) Engine for VS Code Copilot Integration
 * Implements a lightweight Prompt Optimization Engine.
 */
import * as vscode from 'vscode';

interface PromptQualityScore {
  taskClarity: number;
  roleAssignment: number;
  context: number;
  outputFormat: number;
  toneConstraints: number;
  reasoningRequest: number;
  ambiguity: number; // higher is better (low ambiguity)
  totalScore: number;
}

interface OptimizedPrompt {
  originalPrompt: string;
  optimizedPrompt: string;
  qualityScore: PromptQualityScore;
  optimizationReasoning: string[];
  variations: string[];
  selectedVariation: number;
}

export class APEEngine {
  async optimizePrompt(originalPrompt: string, _context?: unknown): Promise<OptimizedPrompt> {
    const variations = this.generatePromptVariations(originalPrompt);
    const evaluations = variations.map(v => this.evaluatePromptQuality(v));
    const bestIndex = this.selectBestVariation(evaluations);
    const best = variations[bestIndex];
    const reasoning = this.generateOptimizationReasoning(originalPrompt, best, evaluations[bestIndex]);
    return {
      originalPrompt,
      optimizedPrompt: best,
      qualityScore: evaluations[bestIndex],
      optimizationReasoning: reasoning,
      variations,
      selectedVariation: bestIndex,
    };
  }

  private generatePromptVariations(prompt: string): string[] {
    return [
      this.applyZeroShot(prompt),
      this.applyFewShot(prompt),
      this.applyRolePrompting(prompt),
      this.applyChainOfThought(prompt),
      this.applyConstitutionalAI(prompt),
    ];
  }

  private applyZeroShot(prompt: string): string {
    return `Task: ${prompt}\n\nRequirements:\n- Provide a clear, step-by-step solution\n- Include specific examples where applicable\n- Explain your reasoning process\n- Ensure the output is actionable and complete\n\nPlease proceed with the task above.`;
  }

  private applyFewShot(prompt: string): string {
    const examples = `Example 1: [High-quality response matching a similar request]\nExample 2: [Another best-practice example]\nExample 3: [Clear, concise, actionable output]`;
    return `I need help with: ${prompt}\n\nHere are examples of high-quality responses:\n\n${examples}\n\nFollowing the pattern above, please provide a comprehensive response that:\n1. Addresses the specific request clearly\n2. Includes relevant examples and context\n3. Provides actionable next steps\n4. Maintains high quality standards\n\nYour response:`;
  }

  private applyRolePrompting(prompt: string): string {
    const role = this.inferRole(prompt);
    return `You are ${role}.\n\nUser Request: ${prompt}\n\nAs ${role}, please:\n1. Analyze the request from your professional perspective\n2. Provide expert-level recommendations\n3. Include best practices and pitfalls\n4. Structure your response for clarity\n\nYour expert response:`;
  }

  private applyChainOfThought(prompt: string): string {
    return `Request: ${prompt}\n\nPlease work through this step-by-step:\n\nStep 1: Understanding — clarify scope, context, constraints\nStep 2: Analysis — candidate approaches with pros/cons\nStep 3: Implementation — concrete steps, resources, validation\nStep 4: Verification — tests, risks, follow-ups\n\nProvide a systematic response.`;
  }

  private applyConstitutionalAI(prompt: string): string {
    return `Request: ${prompt}\n\nPlease ensure your response adheres to these principles:\n1) Library-First  2) Test-First  3) Simplicity  4) Integration-first\n5) Clarity  6) Counterfactual Justification\n\nProvide a constitutionally-aligned response with rationale and alternatives.`;
  }

  private inferRole(prompt: string): string {
    const mapping: Array<{ rx: RegExp; role: string }> = [
      { rx: /(code|software|program|refactor)/i, role: 'a Senior Software Engineer and Technical Architect' },
      { rx: /(security|vulnerab|auth)/i, role: 'a Cybersecurity Expert' },
      { rx: /(data|ml|analytics|model)/i, role: 'a Senior Data Scientist and ML Engineer' },
      { rx: /(devops|deploy|infra)/i, role: 'a DevOps and SRE Engineer' },
      { rx: /(design|ui|ux)/i, role: 'a Senior UX/UI Designer' },
    ];
    for (const m of mapping) if (m.rx.test(prompt)) return m.role;
    return 'an Expert Consultant';
  }

  private evaluatePromptQuality(prompt: string): PromptQualityScore {
    const score = (cond: boolean, inc = 1) => (cond ? inc : 0);
    const s: PromptQualityScore = {
      taskClarity: 1 + score(/Task:|Request:/i.test(prompt)) + score(/Please|Requirements:/i.test(prompt)) + score(prompt.length > 50) + score(/specific|clear/i.test(prompt)),
      roleAssignment: 1 + score(/You are|As .*\b/i.test(prompt), 2) + score(/expert|professional/i.test(prompt)) + score(/experience|expertise/i.test(prompt)),
      context: 1 + score(/context|background/i.test(prompt)) + score(/example|similar/i.test(prompt)) + score(/constraint|requirement/i.test(prompt)) + score(prompt.length > 200),
      outputFormat: 1 + score(/step|list/i.test(prompt)) + score(/format|structure/i.test(prompt)) + score(/\n\d+\.|\n- /i.test(prompt)) + score(/provide|include/i.test(prompt)),
      toneConstraints: 1 + score(/professional|expert/i.test(prompt)) + score(/clear|concise/i.test(prompt)) + score(/comprehensive|detailed/i.test(prompt)) + score(/actionable|practical/i.test(prompt)),
      reasoningRequest: 1 + score(/explain|reason/i.test(prompt)) + score(/why|how/i.test(prompt)) + score(/step-by-step|systematic/i.test(prompt)) + score(/analysis|consider/i.test(prompt)),
      ambiguity: Math.max(1, 5 - (score(/maybe|perhaps|might|could/i.test(prompt)) + score(prompt.length < 30) + score(!/[\?\.!]/.test(prompt)))) ,
      totalScore: 0,
    };
    s.totalScore = s.taskClarity + s.roleAssignment + s.context + s.outputFormat + s.toneConstraints + s.reasoningRequest + s.ambiguity;
    return s;
  }

  private selectBestVariation(evals: PromptQualityScore[]): number {
    let best = 0;
    for (let i = 1; i < evals.length; i++) if (evals[i].totalScore > evals[best].totalScore) best = i;
    return best;
  }

  private generateOptimizationReasoning(_orig: string, _opt: string, score: PromptQualityScore): string[] {
    const notes: string[] = [`Selected optimized prompt with quality score: ${score.totalScore}/35`];
    if (score.taskClarity >= 4) notes.push('✅ High task clarity'); else notes.push('💡 Improve task clarity');
    if (score.roleAssignment >= 4) notes.push('✅ Strong role assignment'); else notes.push('💡 Add explicit role');
    if (score.context >= 4) notes.push('✅ Rich context'); else notes.push('💡 Provide more context');
    if (score.outputFormat >= 4) notes.push('✅ Clear output structure'); else notes.push('💡 Specify structure');
    if (score.reasoningRequest >= 4) notes.push('✅ Reasoning requested'); else notes.push('💡 Ask for reasoning');
    return notes;
  }
}

export class OptimizedChatProvider {
  private ape: APEEngine;
  constructor() {
    this.ape = new APEEngine();
  }
  async openOptimizedChat(): Promise<void> {
    const userPrompt = await vscode.window.showInputBox({
      prompt: 'Enter your prompt to optimize with APE engine',
      placeHolder: 'What would you like help with?',
      ignoreFocusOut: true,
    });
    if (!userPrompt) return;
    await vscode.window.withProgress({ location: vscode.ProgressLocation.Notification, title: 'APE Engine: Optimizing your prompt...' }, async () => {
      const result = await this.ape.optimizePrompt(userPrompt);
      await this.displayOptimizationResults(result);
      await vscode.env.clipboard.writeText(result.optimizedPrompt);
      const action = await vscode.window.showInformationMessage('Optimized prompt copied. Open Copilot Chat?', 'Open Chat', 'Dismiss');
      if (action === 'Open Chat') {
        await vscode.commands.executeCommand('github.copilot.openCopilotChat');
      }
    });
  }

  private async displayOptimizationResults(result: OptimizedPrompt): Promise<void> {
    const panel = vscode.window.createWebviewPanel('apeOptimization', 'APE Engine — Prompt Optimization', vscode.ViewColumn.Beside, { enableScripts: true });
    panel.webview.html = this.renderReport(result);
  }

  private renderReport(r: OptimizedPrompt): string {
    const sc = r.qualityScore; const col = sc.totalScore >= 30 ? '#4CAF50' : sc.totalScore >= 25 ? '#FF9800' : '#F44336';
    const li = (xs: string[]) => xs.map(x => `<li>${this.escape(x)}</li>`).join('');
    return `<!doctype html><html><head><meta charset="utf-8"><style>
      body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;background:var(--vscode-editor-background);color:var(--vscode-editor-foreground);padding:20px}
      .box{border:1px solid var(--vscode-panel-border);border-radius:8px;padding:16px;margin:12px 0}
      .score{font-size:28px;color:${col};font-weight:700}
      .grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}
      .code{white-space:pre-wrap;background:var(--vscode-textCodeBlock-background);padding:12px;border-radius:6px}
    </style></head><body>
      <h1>🧠 APE Engine Optimization</h1>
      <div class="box"><div class="score">${sc.totalScore}/35</div><div>Quality Score</div></div>
      <div class="grid">
        <div class="box"><h3>Original Prompt</h3><div class="code">${this.escape(r.originalPrompt)}</div></div>
        <div class="box"><h3>Optimized Prompt ⭐</h3><div class="code">${this.escape(r.optimizedPrompt)}</div></div>
      </div>
      <div class="box"><h3>Optimization Reasoning</h3><ul>${li(r.optimizationReasoning)}</ul></div>
    </body></html>`;
  }

  private escape(s: string): string { return s.replace(/[&<>"]g/, (m) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[m] as string)); }
}

