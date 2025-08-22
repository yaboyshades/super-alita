import * as vscode from 'vscode';
import * as path from 'path';
import * as fs from 'fs';
import { exec } from 'child_process';
import { promisify } from 'util';

const execAsync = promisify(exec);

interface LadderTask {
    id: string;
    title: string;
    description: string;
    completed: boolean;
    priority: string;
    createdAt: string;
    updatedAt: string;
    tags: string[];
    context: any;
    ladder_task_id?: string;
    ladder_stage?: string;
    energy_required?: number;
    depends_on?: string[];
}

interface TodosData {
    todoList: Array<{
        id: number;
        title: string;
        description: string;
        status: string;
    }>;
    lastModified: string;
    version: string;
}

class LadderTaskProvider implements vscode.TaskProvider {
    private tasks: LadderTask[] = [];
    private workspaceFolder: vscode.WorkspaceFolder;
    private _onDidChange: vscode.EventEmitter<void> = new vscode.EventEmitter<void>();
    readonly onDidChange: vscode.Event<void> = this._onDidChange.event;

    constructor(workspaceFolder: vscode.WorkspaceFolder) {
        this.workspaceFolder = workspaceFolder;
    }

    async provideTasks(): Promise<vscode.Task[]> {
        // Get tasks from Python task provider
        await this.refreshTasks();
        
        return this.tasks.map(task => {
            const definition: vscode.TaskDefinition = {
                type: 'ladder',
                task: task.id,
                title: task.title,
                stage: task.ladder_stage || 'not_started'
            };

            const taskItem = new vscode.Task(
                definition,
                this.workspaceFolder,
                task.title,
                'ladder',
                new vscode.ShellExecution('echo', [`"LADDER Task: ${task.title} (${task.ladder_stage || 'not_started'})"`])
            );

            taskItem.detail = task.description;
            taskItem.group = task.completed ? vscode.TaskGroup.Test : vscode.TaskGroup.Build;
            
            // Add task context
            taskItem.source = 'ladder';
            if (task.priority) {
                taskItem.detail += ` [Priority: ${task.priority}]`;
            }
            
            return taskItem;
        });
    }

    async resolveTask(task: vscode.Task): Promise<vscode.Task | undefined> {
        const definition = task.definition;
        if (definition.type === 'ladder' && definition.task) {
            // Find the task data
            const taskData = this.tasks.find(t => t.id === definition.task);
            if (taskData) {
                // Create enhanced execution for the task
                const pythonPath = vscode.workspace.getConfiguration('ladder').get<string>('pythonPath') 
                    || path.join(this.workspaceFolder.uri.fsPath, '.venv', 'Scripts', 'python.exe');
                
                const execution = new vscode.ShellExecution(pythonPath, [
                    path.join(this.workspaceFolder.uri.fsPath, 'src', 'vscode_integration', 'task_runner.py'),
                    '--task-id', definition.task,
                    '--action', 'execute'
                ]);
                
                task.execution = execution;
                task.detail = `${taskData.description} [Stage: ${taskData.ladder_stage || 'not_started'}]`;
            }
        }
        return task;
    }

    async refreshTasks(): Promise<void> {
        try {
            this.tasks = await this.getTasksFromPython();
            this._onDidChange.fire();
        } catch (error) {
            console.error('Error refreshing LADDER tasks:', error);
            vscode.window.showErrorMessage(`Failed to refresh LADDER tasks: ${error}`);
        }
    }

    private async getTasksFromPython(): Promise<LadderTask[]> {
        try {
            // First, try to get tasks from the Python task provider
            const pythonPath = vscode.workspace.getConfiguration('ladder').get<string>('pythonPath') 
                || path.join(this.workspaceFolder.uri.fsPath, '.venv', 'Scripts', 'python.exe');
            
            const taskProviderScript = path.join(
                this.workspaceFolder.uri.fsPath, 
                'src', 
                'vscode_integration', 
                'task_provider_cli.py'
            );
            
            if (fs.existsSync(taskProviderScript)) {
                const { stdout } = await execAsync(`"${pythonPath}" "${taskProviderScript}" --action get_tasks`);
                const pythonTasks = JSON.parse(stdout.trim());
                return pythonTasks;
            }
            
            // Fallback: read from todos.json
            return await this.getTasksFromTodos();
            
        } catch (error) {
            console.warn('Error getting tasks from Python, falling back to todos.json:', error);
            return await this.getTasksFromTodos();
        }
    }

    private async getTasksFromTodos(): Promise<LadderTask[]> {
        const todosPath = path.join(this.workspaceFolder.uri.fsPath, '.vscode', 'todos.json');
        
        if (!fs.existsSync(todosPath)) {
            return [];
        }

        try {
            const todosData: TodosData = JSON.parse(fs.readFileSync(todosPath, 'utf8'));
            return todosData.todoList?.map((todo, index) => ({
                id: todo.id.toString(),
                title: todo.title,
                description: todo.description,
                completed: todo.status === 'completed',
                priority: 'medium',
                createdAt: new Date().toISOString(),
                updatedAt: todosData.lastModified || new Date().toISOString(),
                tags: [],
                context: {
                    source: 'vscode_todos',
                    original_status: todo.status
                }
            })) || [];
        } catch (error) {
            console.error('Error reading todos.json:', error);
            return [];
        }
    }
}

class LadderTaskTreeDataProvider implements vscode.TreeDataProvider<LadderTask> {
    private _onDidChangeTreeData: vscode.EventEmitter<LadderTask | undefined | null | void> = new vscode.EventEmitter<LadderTask | undefined | null | void>();
    readonly onDidChangeTreeData: vscode.Event<LadderTask | undefined | null | void> = this._onDidChangeTreeData.event;

    private tasks: LadderTask[] = [];
    private taskProvider: LadderTaskProvider;

    constructor(taskProvider: LadderTaskProvider) {
        this.taskProvider = taskProvider;
        this.taskProvider.onDidChange(() => this.refresh());
    }

    refresh(): void {
        this.taskProvider.refreshTasks().then(() => {
            this.tasks = this.taskProvider['tasks']; // Access private property
            this._onDidChangeTreeData.fire();
        });
    }

    getTreeItem(element: LadderTask): vscode.TreeItem {
        const item = new vscode.TreeItem(element.title, vscode.TreeItemCollapsibleState.None);
        
        item.description = element.ladder_stage || 'not_started';
        item.tooltip = `${element.title}\n${element.description}\nStage: ${element.ladder_stage || 'not_started'}`;
        
        // Set icon based on completion status and stage
        if (element.completed) {
            item.iconPath = new vscode.ThemeIcon('pass', new vscode.ThemeColor('testing.iconPassed'));
        } else {
            switch (element.ladder_stage) {
                case 'execution':
                    item.iconPath = new vscode.ThemeIcon('play', new vscode.ThemeColor('testing.iconQueued'));
                    break;
                case 'review':
                    item.iconPath = new vscode.ThemeIcon('eye', new vscode.ThemeColor('testing.iconQueued'));
                    break;
                case 'decomposition':
                case 'action_identification':
                    item.iconPath = new vscode.ThemeIcon('gear', new vscode.ThemeColor('testing.iconQueued'));
                    break;
                default:
                    item.iconPath = new vscode.ThemeIcon('circle-outline');
            }
        }

        // Add commands
        item.command = {
            command: 'ladder.showTaskDetails',
            title: 'Show Task Details',
            arguments: [element]
        };

        // Context value for context menu
        item.contextValue = element.completed ? 'completedTask' : 'activeTask';

        return item;
    }

    getChildren(element?: LadderTask): Thenable<LadderTask[]> {
        if (!element) {
            // Return root tasks (grouped by stage or priority)
            return Promise.resolve(this.tasks);
        }
        return Promise.resolve([]);
    }
}

export function activate(context: vscode.ExtensionContext) {
    console.log('LADDER Planner extension is now active');

    // Check if workspace has LADDER configuration
    const workspaceFolder = vscode.workspace.workspaceFolders?.[0];
    if (!workspaceFolder) {
        console.log('No workspace folder found');
        return;
    }

    // Set context for views
    const hasLadderConfig = fs.existsSync(path.join(workspaceFolder.uri.fsPath, 'cortex', 'planner'));
    vscode.commands.executeCommand('setContext', 'workspaceHasLadderConfig', hasLadderConfig);

    if (!hasLadderConfig) {
        console.log('LADDER planner not found in workspace');
        return;
    }

    // Initialize task provider
    const taskProvider = new LadderTaskProvider(workspaceFolder);
    const taskProviderDisposable = vscode.tasks.registerTaskProvider('ladder', taskProvider);
    context.subscriptions.push(taskProviderDisposable);

    // Initialize tree view
    const treeDataProvider = new LadderTaskTreeDataProvider(taskProvider);
    const treeView = vscode.window.createTreeView('ladderTasks', { 
        treeDataProvider,
        showCollapseAll: true
    });
    context.subscriptions.push(treeView);

    // Register commands
    const createTaskCommand = vscode.commands.registerCommand('ladder.createTask', async () => {
        const title = await vscode.window.showInputBox({
            prompt: 'Enter task title',
            placeHolder: 'Task title',
            validateInput: (value) => {
                return value.trim().length === 0 ? 'Task title cannot be empty' : null;
            }
        });
        
        if (!title) return;

        const description = await vscode.window.showInputBox({
            prompt: 'Enter task description (optional)',
            placeHolder: 'Task description'
        });

        const priority = await vscode.window.showQuickPick(
            ['low', 'medium', 'high', 'critical'],
            { placeHolder: 'Select task priority' }
        );

        try {
            // Create task via Python task provider
            const pythonPath = vscode.workspace.getConfiguration('ladder').get<string>('pythonPath') 
                || path.join(workspaceFolder.uri.fsPath, '.venv', 'Scripts', 'python.exe');
            
            const createTaskScript = path.join(
                workspaceFolder.uri.fsPath, 
                'src', 
                'vscode_integration', 
                'task_provider_cli.py'
            );

            if (fs.existsSync(createTaskScript)) {
                const taskData = JSON.stringify({
                    title: title,
                    description: description || '',
                    priority: priority || 'medium'
                });

                await execAsync(`"${pythonPath}" "${createTaskScript}" --action create_task --data '${taskData}'`);
                
                vscode.window.showInformationMessage(`Created LADDER task: ${title}`);
                await taskProvider.refreshTasks();
            } else {
                vscode.window.showErrorMessage('LADDER task provider not found');
            }
        } catch (error) {
            vscode.window.showErrorMessage(`Failed to create task: ${error}`);
        }
    });

    const refreshTasksCommand = vscode.commands.registerCommand('ladder.refreshTasks', async () => {
        await taskProvider.refreshTasks();
        vscode.window.showInformationMessage('LADDER tasks refreshed');
    });

    const showPlannerStatusCommand = vscode.commands.registerCommand('ladder.showPlannerStatus', async () => {
        try {
            const pythonPath = vscode.workspace.getConfiguration('ladder').get<string>('pythonPath') 
                || path.join(workspaceFolder.uri.fsPath, '.venv', 'Scripts', 'python.exe');
            
            const statusScript = path.join(
                workspaceFolder.uri.fsPath, 
                'src', 
                'vscode_integration', 
                'task_provider_cli.py'
            );

            if (fs.existsSync(statusScript)) {
                const { stdout } = await execAsync(`"${pythonPath}" "${statusScript}" --action get_status`);
                const status = JSON.parse(stdout.trim());
                
                const message = `LADDER Planner Status:
• Tasks: ${status.task_count || 0}
• Active Planner: ${status.planner_active ? 'Yes' : 'No'}
• Last Sync: ${status.last_sync || 'Never'}`;

                vscode.window.showInformationMessage(message);
            } else {
                vscode.window.showWarningMessage('LADDER planner status unavailable');
            }
        } catch (error) {
            vscode.window.showErrorMessage(`Failed to get planner status: ${error}`);
        }
    });

    const showTaskDetailsCommand = vscode.commands.registerCommand('ladder.showTaskDetails', (task: LadderTask) => {
        const panel = vscode.window.createWebviewPanel(
            'ladderTaskDetails',
            `LADDER Task: ${task.title}`,
            vscode.ViewColumn.One,
            { enableScripts: true }
        );

        panel.webview.html = getTaskDetailsWebviewContent(task);
    });

    // Register all commands
    context.subscriptions.push(
        createTaskCommand,
        refreshTasksCommand,
        showPlannerStatusCommand,
        showTaskDetailsCommand
    );

    // Watch for changes to todos.json and refresh tasks
    const todosWatcher = vscode.workspace.createFileSystemWatcher('**/.vscode/todos.json');
    todosWatcher.onDidChange(async () => {
        console.log('todos.json changed, refreshing tasks');
        await taskProvider.refreshTasks();
    });
    todosWatcher.onDidCreate(async () => {
        console.log('todos.json created, refreshing tasks');
        await taskProvider.refreshTasks();
    });
    context.subscriptions.push(todosWatcher);

    // Auto-refresh tasks periodically
    const refreshInterval = vscode.workspace.getConfiguration('ladder').get<number>('syncInterval') || 30;
    const intervalId = setInterval(async () => {
        await taskProvider.refreshTasks();
    }, refreshInterval * 1000);

    context.subscriptions.push({
        dispose: () => clearInterval(intervalId)
    });

    // Initial refresh
    taskProvider.refreshTasks();

    console.log('LADDER Planner extension initialized successfully');
}

function getTaskDetailsWebviewContent(task: LadderTask): string {
    return `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LADDER Task Details</title>
    <style>
        body {
            font-family: var(--vscode-font-family);
            padding: 20px;
            color: var(--vscode-foreground);
            background-color: var(--vscode-editor-background);
        }
        .task-header {
            border-bottom: 1px solid var(--vscode-panel-border);
            padding-bottom: 16px;
            margin-bottom: 20px;
        }
        .task-title {
            font-size: 24px;
            font-weight: bold;
            margin-bottom: 8px;
        }
        .task-meta {
            display: flex;
            gap: 16px;
            flex-wrap: wrap;
        }
        .meta-item {
            display: flex;
            align-items: center;
            gap: 4px;
        }
        .status-badge {
            padding: 2px 8px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: bold;
        }
        .status-completed {
            background-color: var(--vscode-testing-iconPassed);
            color: white;
        }
        .status-active {
            background-color: var(--vscode-testing-iconQueued);
            color: white;
        }
        .section {
            margin: 20px 0;
        }
        .section-title {
            font-size: 16px;
            font-weight: bold;
            margin-bottom: 8px;
            color: var(--vscode-textLink-foreground);
        }
        .description {
            background-color: var(--vscode-textBlockQuote-background);
            border-left: 4px solid var(--vscode-textBlockQuote-border);
            padding: 12px;
            margin: 8px 0;
        }
        .context-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 12px;
        }
        .context-item {
            background-color: var(--vscode-input-background);
            border: 1px solid var(--vscode-input-border);
            padding: 8px;
            border-radius: 4px;
        }
        .context-key {
            font-weight: bold;
            color: var(--vscode-textLink-foreground);
        }
    </style>
</head>
<body>
    <div class="task-header">
        <div class="task-title">${task.title}</div>
        <div class="task-meta">
            <div class="meta-item">
                <span class="status-badge ${task.completed ? 'status-completed' : 'status-active'}">
                    ${task.completed ? 'Completed' : task.ladder_stage || 'Not Started'}
                </span>
            </div>
            <div class="meta-item">
                <strong>Priority:</strong> ${task.priority}
            </div>
            <div class="meta-item">
                <strong>Created:</strong> ${new Date(task.createdAt).toLocaleDateString()}
            </div>
            ${task.energy_required ? `<div class="meta-item"><strong>Energy:</strong> ${task.energy_required}</div>` : ''}
        </div>
    </div>

    ${task.description ? `
    <div class="section">
        <div class="section-title">Description</div>
        <div class="description">${task.description}</div>
    </div>
    ` : ''}

    ${task.tags && task.tags.length > 0 ? `
    <div class="section">
        <div class="section-title">Tags</div>
        <div>${task.tags.map(tag => `<span class="status-badge status-active">${tag}</span>`).join(' ')}</div>
    </div>
    ` : ''}

    ${task.depends_on && task.depends_on.length > 0 ? `
    <div class="section">
        <div class="section-title">Dependencies</div>
        <div>${task.depends_on.map(dep => `<div class="context-item">${dep}</div>`).join('')}</div>
    </div>
    ` : ''}

    ${task.context && Object.keys(task.context).length > 0 ? `
    <div class="section">
        <div class="section-title">Context</div>
        <div class="context-grid">
            ${Object.entries(task.context).map(([key, value]) => `
                <div class="context-item">
                    <div class="context-key">${key}:</div>
                    <div>${typeof value === 'object' ? JSON.stringify(value, null, 2) : value}</div>
                </div>
            `).join('')}
        </div>
    </div>
    ` : ''}
</body>
</html>`;
}

export function deactivate() {
    console.log('LADDER Planner extension deactivated');
}