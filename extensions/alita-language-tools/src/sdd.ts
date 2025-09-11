// SDD (Spec-Driven Development) Integration for Super-Alita
import * as vscode from 'vscode';

// SDD Data Structures
interface SDDSpecification {
  id: string;
  title: string;
  description: string;
  requirements: string[];
  constraints: string[];
  createdAt: Date;
}

interface SDDTechnicalPlan {
  id: string;
  specificationId: string;
  techStack: string[];
  architecture: string;
  dependencies: string[];
  implementation: {
    phases: string[];
    timeline: string;
  };
}

interface SDDTask {
  id: string;
  planId: string;
  title: string;
  description: string;
  priority: 'low' | 'medium' | 'high';
  status: 'not-started' | 'in-progress' | 'completed';
  dependencies: string[];
}

// SDD Workspace State Manager
class SDDWorkspaceManager {
  private context: vscode.ExtensionContext;
  private specifications: Map<string, SDDSpecification> = new Map();
  private plans: Map<string, SDDTechnicalPlan> = new Map();
  private tasks: Map<string, SDDTask> = new Map();

  constructor(context: vscode.ExtensionContext) {
    this.context = context;
    this.loadState();
  }

  private loadState() {
    const specs = this.context.workspaceState.get<Record<string, SDDSpecification>>('sdd.specifications', {});
    const plans = this.context.workspaceState.get<Record<string, SDDTechnicalPlan>>('sdd.plans', {});
    const tasks = this.context.workspaceState.get<Record<string, SDDTask>>('sdd.tasks', {});

    this.specifications = new Map(Object.entries(specs));
    this.plans = new Map(Object.entries(plans));
    this.tasks = new Map(Object.entries(tasks));
  }

  private async saveState() {
    await this.context.workspaceState.update('sdd.specifications', Object.fromEntries(this.specifications));
    await this.context.workspaceState.update('sdd.plans', Object.fromEntries(this.plans));
    await this.context.workspaceState.update('sdd.tasks', Object.fromEntries(this.tasks));
  }

  async addSpecification(spec: Omit<SDDSpecification, 'id' | 'createdAt'>): Promise<string> {
    const id = `spec-${Date.now()}`;
    const specification: SDDSpecification = {
      ...spec,
      id,
      createdAt: new Date(),
    };
    this.specifications.set(id, specification);
    await this.saveState();
    return id;
  }

  async addPlan(plan: Omit<SDDTechnicalPlan, 'id'>): Promise<string> {
    const id = `plan-${Date.now()}`;
    const technicalPlan: SDDTechnicalPlan = { ...plan, id };
    this.plans.set(id, technicalPlan);
    await this.saveState();
    return id;
  }

  async addTask(task: Omit<SDDTask, 'id'>): Promise<string> {
    const id = `task-${Date.now()}`;
    const sddTask: SDDTask = { ...task, id };
    this.tasks.set(id, sddTask);
    await this.saveState();
    return id;
  }

  getSpecifications(): SDDSpecification[] {
    return Array.from(this.specifications.values());
  }

  getPlans(): SDDTechnicalPlan[] {
    return Array.from(this.plans.values());
  }

  getTasks(): SDDTask[] {
    return Array.from(this.tasks.values());
  }
}

// SDD Commands
export class SDDCommands {
  private workspaceManager: SDDWorkspaceManager;
  private alitaRuntimeBase: string;

  constructor(context: vscode.ExtensionContext) {
    this.workspaceManager = new SDDWorkspaceManager(context);
    this.alitaRuntimeBase = (vscode.workspace.getConfiguration('alita')
      .get<string>('runtime.host', 'http://127.0.0.1:8080') || 'http://127.0.0.1:8080')
      .replace(/\/$/, '');
  }

  // /specify command - Capture high-level intent
  async specify(): Promise<void> {
    const title = await vscode.window.showInputBox({
      prompt: 'Specification Title',
      placeHolder: 'e.g., Photo Album Organizer',
    });

    if (!title) return;

    const description = await vscode.window.showInputBox({
      prompt: 'Describe what you want to build (focus on what and why, not tech details)',
      placeHolder: 'Build an application that can help me organize my photos...'
    });

    if (!description) return;

    // Multi-step requirement gathering
    const requirements: string[] = [];
    while (true) {
      const requirement = await vscode.window.showInputBox({
        prompt: `Add requirement (${requirements.length + 1}) - Leave empty to finish`,
        placeHolder: 'Albums are grouped by date...'
      });
      if (!requirement) break;
      requirements.push(requirement);
    }

    // Constraints gathering
    const constraints: string[] = [];
    while (true) {
      const constraint = await vscode.window.showInputBox({
        prompt: `Add constraint (${constraints.length + 1}) - Leave empty to finish`,
        placeHolder: 'No external uploads, local SQLite database...'
      });
      if (!constraint) break;
      constraints.push(constraint);
    }

    const specId = await this.workspaceManager.addSpecification({
      title,
      description,
      requirements,
      constraints,
    });

    // Send to Alita runtime for processing
    await this.sendToAlitaRuntime('/sdd/specify', {
      spec_id: specId,
      title,
      description,
      requirements,
      constraints,
    });

    vscode.window.showInformationMessage(`Specification "${title}" created with ID: ${specId}`);
  }

  // /plan command - Technical implementation planning
  async plan(): Promise<void> {
    const specs = this.workspaceManager.getSpecifications();
    if (specs.length === 0) {
      vscode.window.showWarningMessage('No specifications found. Run /specify first.');
      return;
    }

    const specItems = specs.map(s => ({ label: s.title, description: s.id, spec: s }));
    const selectedSpec = await vscode.window.showQuickPick(specItems, {
      placeHolder: 'Select specification to plan',
    });

    if (!selectedSpec) return;

    const architecture = await vscode.window.showInputBox({
      prompt: 'Describe your architecture approach',
      placeHolder: 'Vite with minimal libraries, vanilla HTML/CSS/JS...'
    });

    if (!architecture) return;

    // Tech stack selection
    const techStack: string[] = [];
    while (true) {
      const tech = await vscode.window.showInputBox({
        prompt: `Add technology (${techStack.length + 1}) - Leave empty to finish`,
        placeHolder: 'Vite, SQLite, TypeScript...'
      });
      if (!tech) break;
      techStack.push(tech);
    }

    // Dependencies
    const dependencies: string[] = [];
    while (true) {
      const dep = await vscode.window.showInputBox({
        prompt: `Add dependency (${dependencies.length + 1}) - Leave empty to finish`,
        placeHolder: 'sqlite3, vite, typescript...'
      });
      if (!dep) break;
      dependencies.push(dep);
    }

    const planId = await this.workspaceManager.addPlan({
      specificationId: selectedSpec.spec.id,
      techStack,
      architecture,
      dependencies,
      implementation: {
        phases: ['Setup', 'Core Implementation', 'Testing', 'Deployment'],
        timeline: '2-3 weeks',
      },
    });

    // Send to Alita runtime for processing
    await this.sendToAlitaRuntime('/sdd/plan', {
      plan_id: planId,
      specification_id: selectedSpec.spec.id,
      tech_stack: techStack,
      architecture,
      dependencies,
    });

    vscode.window.showInformationMessage(`Technical plan created with ID: ${planId}`);
  }

  // /tasks command - Break down into actionable tasks
  async tasks(): Promise<void> {
    const plans = this.workspaceManager.getPlans();
    if (plans.length === 0) {
      vscode.window.showWarningMessage('No plans found. Run /plan first.');
      return;
    }

    const planItems = plans.map(p => ({
      label: `Plan for ${p.specificationId}`,
      description: p.architecture.substring(0, 100),
      plan: p,
    }));
    const selectedPlan = await vscode.window.showQuickPick(planItems, {
      placeHolder: 'Select plan to break down into tasks',
    });

    if (!selectedPlan) return;

    // Auto-generate initial tasks based on plan
    const defaultTasks = [
      { title: 'Project Setup', description: 'Initialize project structure and dependencies' },
      { title: 'Core Architecture', description: 'Implement main application architecture' },
      { title: 'Feature Implementation', description: 'Build core features' },
      { title: 'Testing', description: 'Add comprehensive tests' },
      { title: 'Documentation', description: 'Create user and developer documentation' },
    ];

    for (const defaultTask of defaultTasks) {
      await this.workspaceManager.addTask({
        planId: selectedPlan.plan.id,
        title: defaultTask.title,
        description: defaultTask.description,
        priority: 'medium',
        status: 'not-started',
        dependencies: [],
      });
    }

    // Allow custom task addition
    while (true) {
      const taskTitle = await vscode.window.showInputBox({
        prompt: 'Add custom task (leave empty to finish)',
        placeHolder: 'Implement photo upload feature...'
      });
      if (!taskTitle) break;

      const taskDescription = await vscode.window.showInputBox({
        prompt: 'Task description',
        placeHolder: 'Detailed description of the task...'
      });
      if (!taskDescription) continue;

      const priority = await vscode.window.showQuickPick(['low', 'medium', 'high'], {
        placeHolder: 'Select task priority',
      });
      if (!priority) continue;

      await this.workspaceManager.addTask({
        planId: selectedPlan.plan.id,
        title: taskTitle,
        description: taskDescription,
        priority: priority as 'low' | 'medium' | 'high',
        status: 'not-started',
        dependencies: [],
      });
    }

    // Send to Alita runtime for processing
    const allTasks = this.workspaceManager.getTasks();
    const tasksForPlan = allTasks.filter(t => t.planId === selectedPlan.plan.id);
    await this.sendToAlitaRuntime('/sdd/tasks', {
      plan_id: selectedPlan.plan.id,
      tasks: tasksForPlan,
    });

    vscode.window.showInformationMessage('Tasks created and sent to Alita runtime for processing');
  }

  // Send SDD data to Super-Alita runtime for processing
  private async sendToAlitaRuntime(endpoint: string, data: any): Promise<void> {
    try {
      const response = await (globalThis as any).fetch(`${this.alitaRuntimeBase}${endpoint}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data),
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }

      const result = await response.json().catch(() => ({}));
      console.log('Alita runtime response:', result);
    } catch (error) {
      console.error('Failed to send to Alita runtime:', error);
      vscode.window.showWarningMessage(`Failed to send to Alita runtime: ${(error as Error).message}`);
    }
  }

  // View current SDD state
  async viewState(): Promise<void> {
    const specs = this.workspaceManager.getSpecifications();
    const plans = this.workspaceManager.getPlans();
    const tasks = this.workspaceManager.getTasks();

    const content = `# Spec-Driven Development State

## Specifications (${specs.length})
${specs
  .map(
    s => `- **${s.title}** (${s.id})\n  ${s.description}\n  Requirements: ${s.requirements.length}\n  Constraints: ${s.constraints.length}`
  )
  .join('\n\n')}

## Technical Plans (${plans.length})
${plans
  .map(
    p => `- **Plan for ${p.specificationId}** (${p.id})\n  Architecture: ${p.architecture}\n  Tech Stack: ${p.techStack.join(', ')}\n  Dependencies: ${p.dependencies.join(', ')}`
  )
  .join('\n\n')}

## Tasks (${tasks.length})
${tasks
  .map(
    t => `- **${t.title}** [${t.status}] (${t.priority})\n  ${t.description}\n  Plan: ${t.planId}`
  )
  .join('\n\n')}
`;

    const doc = await vscode.workspace.openTextDocument({
      content,
      language: 'markdown',
    });
    await vscode.window.showTextDocument(doc, { preview: true });
  }
}

// Register SDD commands
export function registerSDDCommands(context: vscode.ExtensionContext): vscode.Disposable[] {
  const sddCommands = new SDDCommands(context);
  return [
    vscode.commands.registerCommand('alita.sdd.specify', () => sddCommands.specify()),
    vscode.commands.registerCommand('alita.sdd.plan', () => sddCommands.plan()),
    vscode.commands.registerCommand('alita.sdd.tasks', () => sddCommands.tasks()),
    vscode.commands.registerCommand('alita.sdd.viewState', () => sddCommands.viewState()),
  ];
}

