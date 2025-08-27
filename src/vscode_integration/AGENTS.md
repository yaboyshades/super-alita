# VS Code Integration - Agent Instructions

## Overview
The `src/vscode_integration/` directory contains VS Code integration components:
- **Agent Integration** - VS Code agent integration and communication
- **MCP Server** - MCP server components for VS Code integration
- **Task Providers** - VS Code task provider implementations
- **Extension Components** - TypeScript extension components

## Key Files & Responsibilities

### Core Integration Components
- `agent_integration.py` - Main VS Code agent integration logic
- `agent_mcp_server.py` - MCP server for VS Code communication
- `agent_mcp_tool.py` - MCP tools for VS Code integration
- `task_provider.py` - VS Code task provider implementation
- `task_runner.py` - Task execution and management
- `extension.ts` - TypeScript VS Code extension entry point
- `builtin_mcp_provider.ts` - Built-in MCP provider implementation

## Development Guidelines

### VS Code Agent Integration
```python
import asyncio
import json
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
from pathlib import Path
import vscode  # VS Code Python API

@dataclass
class VSCodeTask:
    """VS Code task representation"""
    id: str
    name: str
    description: str
    command: str
    args: List[str] = None
    cwd: str = None
    env: Dict[str, str] = None
    
    def __post_init__(self):
        if self.args is None:
            self.args = []
        if self.env is None:
            self.env = {}

@dataclass
class AgentCommand:
    """Agent command for VS Code"""
    command_id: str
    title: str
    category: str
    handler: Callable
    parameters: List[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.parameters is None:
            self.parameters = []

class VSCodeAgentIntegration:
    """Main VS Code agent integration"""
    
    def __init__(self, workspace_root: str = None):
        self.workspace_root = Path(workspace_root) if workspace_root else Path.cwd()
        self.commands: Dict[str, AgentCommand] = {}
        self.task_providers: Dict[str, Callable] = {}
        self.diagnostics_collection = None
        self.status_bar_items: Dict[str, Any] = {}
        
    async def initialize(self, context: Any) -> bool:
        """Initialize VS Code integration"""
        try:
            # Register commands
            await self._register_commands(context)
            
            # Register task providers
            await self._register_task_providers(context)
            
            # Setup diagnostics
            self._setup_diagnostics(context)
            
            # Setup status bar
            self._setup_status_bar(context)
            
            # Initialize MCP server
            await self._initialize_mcp_server()
            
            return True
            
        except Exception as e:
            logger.error(f"VS Code integration initialization failed: {e}")
            return False
            
    def register_command(self, command: AgentCommand):
        """Register VS Code command"""
        self.commands[command.command_id] = command
        
    def register_task_provider(self, provider_id: str, provider: Callable):
        """Register task provider"""
        self.task_providers[provider_id] = provider
        
    async def execute_agent_task(self, task: VSCodeTask) -> Dict[str, Any]:
        """Execute agent task in VS Code context"""
        try:
            # Create VS Code terminal for task execution
            terminal = vscode.window.createTerminal({
                'name': task.name,
                'cwd': task.cwd or str(self.workspace_root),
                'env': task.env
            })
            
            # Build command
            command_line = f"{task.command} {' '.join(task.args)}"
            
            # Execute command
            terminal.sendText(command_line)
            terminal.show()
            
            return {
                "success": True,
                "task_id": task.id,
                "terminal_name": task.name
            }
            
        except Exception as e:
            return {
                "success": False,
                "task_id": task.id,
                "error": str(e)
            }
            
    async def show_agent_progress(self, title: str, task_func: Callable) -> Any:
        """Show progress for agent operation"""
        return await vscode.window.withProgress({
            'location': vscode.ProgressLocation.Notification,
            'title': title,
            'cancellable': True
        }, task_func)
        
    def update_status_bar(self, item_id: str, text: str, tooltip: str = None):
        """Update status bar item"""
        if item_id not in self.status_bar_items:
            self.status_bar_items[item_id] = vscode.window.createStatusBarItem(
                vscode.StatusBarAlignment.Left
            )
            
        status_item = self.status_bar_items[item_id]
        status_item.text = text
        if tooltip:
            status_item.tooltip = tooltip
        status_item.show()
        
    def show_diagnostic(self, file_path: str, diagnostics: List[Dict[str, Any]]):
        """Show diagnostics for file"""
        if not self.diagnostics_collection:
            return
            
        uri = vscode.Uri.file(file_path)
        vscode_diagnostics = []
        
        for diag in diagnostics:
            vscode_diag = vscode.Diagnostic(
                range=vscode.Range(
                    vscode.Position(diag['line'], diag['column']),
                    vscode.Position(diag['end_line'], diag['end_column'])
                ),
                message=diag['message'],
                severity=getattr(vscode.DiagnosticSeverity, diag.get('severity', 'Error'))
            )
            vscode_diagnostics.append(vscode_diag)
            
        self.diagnostics_collection.set(uri, vscode_diagnostics)
```

### MCP Server Integration
```python
from mcp import MCPServer, MCPTool
from mcp.types import ToolResult

class VSCodeMCPServer(MCPServer):
    """MCP Server for VS Code integration"""
    
    def __init__(self, workspace_root: str):
        super().__init__("vscode-agent-server")
        self.workspace_root = Path(workspace_root)
        self.agent_integration = VSCodeAgentIntegration(workspace_root)
        
    async def initialize(self):
        """Initialize MCP server"""
        await super().initialize()
        
        # Register MCP tools
        await self._register_tools()
        
        # Initialize agent integration
        await self.agent_integration.initialize(None)
        
    async def _register_tools(self):
        """Register MCP tools"""
        
        @self.tool("execute_task")
        async def execute_task(
            task_name: str,
            command: str,
            args: List[str] = None,
            cwd: str = None
        ) -> ToolResult:
            """Execute task in VS Code"""
            task = VSCodeTask(
                id=f"task_{int(time.time())}",
                name=task_name,
                description=f"Execute {command}",
                command=command,
                args=args or [],
                cwd=cwd
            )
            
            result = await self.agent_integration.execute_agent_task(task)
            
            return ToolResult(
                success=result["success"],
                result=result,
                error=result.get("error")
            )
            
        @self.tool("open_file")
        async def open_file(file_path: str, line: int = None) -> ToolResult:
            """Open file in VS Code editor"""
            try:
                abs_path = (self.workspace_root / file_path).resolve()
                
                # Validate file is within workspace
                if not abs_path.is_relative_to(self.workspace_root):
                    return ToolResult(
                        success=False,
                        error="File outside workspace"
                    )
                    
                # Open file in VS Code
                uri = vscode.Uri.file(str(abs_path))
                document = await vscode.workspace.openTextDocument(uri)
                editor = await vscode.window.showTextDocument(document)
                
                # Navigate to line if specified
                if line is not None:
                    position = vscode.Position(line - 1, 0)
                    editor.selection = vscode.Selection(position, position)
                    editor.revealRange(vscode.Range(position, position))
                    
                return ToolResult(
                    success=True,
                    result={"file_path": str(abs_path), "line": line}
                )
                
            except Exception as e:
                return ToolResult(
                    success=False,
                    error=str(e)
                )
                
        @self.tool("show_message")
        async def show_message(message: str, message_type: str = "info") -> ToolResult:
            """Show message in VS Code"""
            try:
                if message_type == "error":
                    vscode.window.showErrorMessage(message)
                elif message_type == "warning":
                    vscode.window.showWarningMessage(message)
                else:
                    vscode.window.showInformationMessage(message)
                    
                return ToolResult(
                    success=True,
                    result={"message": message, "type": message_type}
                )
                
            except Exception as e:
                return ToolResult(
                    success=False,
                    error=str(e)
                )
                
        @self.tool("get_workspace_files")
        async def get_workspace_files(pattern: str = "**/*") -> ToolResult:
            """Get files in workspace"""
            try:
                files = []
                for file_path in self.workspace_root.rglob(pattern):
                    if file_path.is_file():
                        relative_path = file_path.relative_to(self.workspace_root)
                        files.append(str(relative_path))
                        
                return ToolResult(
                    success=True,
                    result={"files": files, "count": len(files)}
                )
                
            except Exception as e:
                return ToolResult(
                    success=False,
                    error=str(e)
                )
```

### Task Provider Implementation
```python
class AgentTaskProvider:
    """VS Code task provider for agent tasks"""
    
    def __init__(self, workspace_root: str):
        self.workspace_root = Path(workspace_root)
        self.task_definitions: List[Dict[str, Any]] = []
        
    def register_task_definition(self, task_def: Dict[str, Any]):
        """Register task definition"""
        self.task_definitions.append(task_def)
        
    async def provide_tasks(self) -> List[VSCodeTask]:
        """Provide available tasks"""
        tasks = []
        
        # Load task definitions from workspace
        await self._load_workspace_tasks()
        
        # Create VS Code tasks
        for task_def in self.task_definitions:
            task = VSCodeTask(
                id=task_def["id"],
                name=task_def["name"],
                description=task_def.get("description", ""),
                command=task_def["command"],
                args=task_def.get("args", []),
                cwd=task_def.get("cwd"),
                env=task_def.get("env", {})
            )
            tasks.append(task)
            
        return tasks
        
    async def _load_workspace_tasks(self):
        """Load task definitions from workspace"""
        tasks_file = self.workspace_root / ".vscode" / "agent_tasks.json"
        
        if tasks_file.exists():
            try:
                with open(tasks_file) as f:
                    workspace_tasks = json.load(f)
                    
                if "tasks" in workspace_tasks:
                    self.task_definitions.extend(workspace_tasks["tasks"])
                    
            except Exception as e:
                logger.error(f"Failed to load workspace tasks: {e}")
                
    def create_default_tasks(self) -> List[Dict[str, Any]]:
        """Create default agent tasks"""
        return [
            {
                "id": "analyze_code",
                "name": "Analyze Code",
                "description": "Analyze code with Super Alita",
                "command": "python",
                "args": ["-m", "src.deepcode.analyzer", "${file}"]
            },
            {
                "id": "generate_docs",
                "name": "Generate Documentation", 
                "description": "Generate documentation for current file",
                "command": "python",
                "args": ["-m", "src.tools.doc_generator", "${file}"]
            },
            {
                "id": "run_tests",
                "name": "Run Tests",
                "description": "Run tests for current file",
                "command": "python",
                "args": ["-m", "pytest", "${fileDirname}/test_${fileBasenameNoExtension}.py"]
            },
            {
                "id": "agent_chat",
                "name": "Chat with Agent",
                "description": "Start chat session with Super Alita",
                "command": "python",
                "args": ["-m", "src.main", "--mode", "chat"]
            }
        ]
```

### TypeScript Extension Components
```typescript
// extension.ts
import * as vscode from 'vscode';
import { MCPProvider } from './builtin_mcp_provider';

export async function activate(context: vscode.ExtensionContext) {
    console.log('Super Alita VS Code extension is now active!');
    
    // Initialize MCP provider
    const mcpProvider = new MCPProvider(context);
    await mcpProvider.initialize();
    
    // Register commands
    registerCommands(context, mcpProvider);
    
    // Register task provider
    registerTaskProvider(context);
    
    // Setup status bar
    setupStatusBar(context);
}

function registerCommands(context: vscode.ExtensionContext, mcpProvider: MCPProvider) {
    // Chat with agent command
    const chatCommand = vscode.commands.registerCommand('superalita.chat', async () => {
        const panel = vscode.window.createWebviewPanel(
            'superalitaChat',
            'Super Alita Chat',
            vscode.ViewColumn.Two,
            {
                enableScripts: true,
                retainContextWhenHidden: true
            }
        );
        
        panel.webview.html = getChatWebviewContent();
        
        // Handle messages from webview
        panel.webview.onDidReceiveMessage(
            async message => {
                switch (message.command) {
                    case 'sendMessage':
                        const response = await mcpProvider.sendMessage(message.text);
                        panel.webview.postMessage({
                            command: 'receiveMessage',
                            text: response
                        });
                        break;
                }
            }
        );
    });
    
    // Analyze code command
    const analyzeCommand = vscode.commands.registerCommand('superalita.analyzeCode', async () => {
        const editor = vscode.window.activeTextEditor;
        if (!editor) {
            vscode.window.showErrorMessage('No active editor');
            return;
        }
        
        const document = editor.document;
        const code = document.getText();
        
        try {
            const result = await mcpProvider.analyzeCode(code, document.fileName);
            
            // Show results in new document
            const resultDoc = await vscode.workspace.openTextDocument({
                content: JSON.stringify(result, null, 2),
                language: 'json'
            });
            
            await vscode.window.showTextDocument(resultDoc, vscode.ViewColumn.Beside);
            
        } catch (error) {
            vscode.window.showErrorMessage(`Analysis failed: ${error}`);
        }
    });
    
    context.subscriptions.push(chatCommand, analyzeCommand);
}

function registerTaskProvider(context: vscode.ExtensionContext) {
    const taskProvider = vscode.tasks.registerTaskProvider('superalita', {
        provideTasks: async () => {
            // Load tasks from agent task provider
            return await loadAgentTasks();
        },
        resolveTask: async (task: vscode.Task) => {
            // Resolve task execution
            return task;
        }
    });
    
    context.subscriptions.push(taskProvider);
}

async function loadAgentTasks(): Promise<vscode.Task[]> {
    const tasks: vscode.Task[] = [];
    
    // Default agent tasks
    const taskDefinitions = [
        {
            type: 'superalita',
            task: 'analyze',
            label: 'Analyze Code',
            command: 'python',
            args: ['-m', 'src.deepcode.analyzer', '${file}']
        },
        {
            type: 'superalita', 
            task: 'test',
            label: 'Run Tests',
            command: 'python',
            args: ['-m', 'pytest']
        }
    ];
    
    for (const def of taskDefinitions) {
        const task = new vscode.Task(
            def,
            vscode.TaskScope.Workspace,
            def.label,
            'superalita',
            new vscode.ShellExecution(def.command, def.args)
        );
        tasks.push(task);
    }
    
    return tasks;
}

function setupStatusBar(context: vscode.ExtensionContext) {
    const statusBarItem = vscode.window.createStatusBarItem(
        vscode.StatusBarAlignment.Left,
        100
    );
    
    statusBarItem.text = "$(robot) Super Alita";
    statusBarItem.tooltip = "Super Alita Agent Status";
    statusBarItem.command = 'superalita.showStatus';
    statusBarItem.show();
    
    // Register status command
    const statusCommand = vscode.commands.registerCommand('superalita.showStatus', () => {
        vscode.window.showInformationMessage('Super Alita is active and ready!');
    });
    
    context.subscriptions.push(statusBarItem, statusCommand);
}

function getChatWebviewContent(): string {
    return `
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Super Alita Chat</title>
        <style>
            body { font-family: var(--vscode-font-family); padding: 10px; }
            .chat-container { display: flex; flex-direction: column; height: 80vh; }
            .messages { flex: 1; overflow-y: auto; border: 1px solid var(--vscode-panel-border); padding: 10px; margin-bottom: 10px; }
            .message { margin-bottom: 10px; padding: 8px; border-radius: 4px; }
            .user-message { background: var(--vscode-button-background); color: var(--vscode-button-foreground); align-self: flex-end; }
            .agent-message { background: var(--vscode-editor-background); border: 1px solid var(--vscode-panel-border); }
            .input-container { display: flex; gap: 10px; }
            #messageInput { flex: 1; padding: 8px; }
            #sendButton { padding: 8px 16px; }
        </style>
    </head>
    <body>
        <div class="chat-container">
            <div class="messages" id="messages"></div>
            <div class="input-container">
                <input type="text" id="messageInput" placeholder="Type your message..." />
                <button id="sendButton">Send</button>
            </div>
        </div>
        
        <script>
            const vscode = acquireVsCodeApi();
            const messagesDiv = document.getElementById('messages');
            const messageInput = document.getElementById('messageInput');
            const sendButton = document.getElementById('sendButton');
            
            function addMessage(text, isUser) {
                const messageDiv = document.createElement('div');
                messageDiv.className = 'message ' + (isUser ? 'user-message' : 'agent-message');
                messageDiv.textContent = text;
                messagesDiv.appendChild(messageDiv);
                messagesDiv.scrollTop = messagesDiv.scrollHeight;
            }
            
            function sendMessage() {
                const text = messageInput.value.trim();
                if (text) {
                    addMessage(text, true);
                    vscode.postMessage({
                        command: 'sendMessage',
                        text: text
                    });
                    messageInput.value = '';
                }
            }
            
            sendButton.addEventListener('click', sendMessage);
            messageInput.addEventListener('keypress', (e) => {
                if (e.key === 'Enter') {
                    sendMessage();
                }
            });
            
            // Listen for messages from extension
            window.addEventListener('message', event => {
                const message = event.data;
                if (message.command === 'receiveMessage') {
                    addMessage(message.text, false);
                }
            });
            
            // Welcome message
            addMessage('Hello! I\'m Super Alita. How can I help you today?', false);
        </script>
    </body>
    </html>
    `;
}

export function deactivate() {
    console.log('Super Alita VS Code extension is now deactivated');
}
```

## Testing Guidelines

### VS Code Integration Testing
```python
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from src.vscode_integration.agent_integration import VSCodeAgentIntegration, VSCodeTask

@pytest.fixture
def mock_vscode():
    """Mock VS Code API"""
    with patch('vscode.window') as mock_window, \
         patch('vscode.workspace') as mock_workspace:
        
        mock_window.createTerminal = MagicMock()
        mock_window.showInformationMessage = MagicMock()
        mock_workspace.openTextDocument = AsyncMock()
        
        yield {
            'window': mock_window,
            'workspace': mock_workspace
        }

@pytest.mark.asyncio
async def test_vscode_task_execution(mock_vscode, tmp_path):
    """Test VS Code task execution"""
    integration = VSCodeAgentIntegration(str(tmp_path))
    
    task = VSCodeTask(
        id="test_task",
        name="Test Task",
        description="Test task execution",
        command="echo",
        args=["hello", "world"]
    )
    
    result = await integration.execute_agent_task(task)
    
    assert result["success"] is True
    assert result["task_id"] == "test_task"
    mock_vscode['window'].createTerminal.assert_called_once()

@pytest.mark.asyncio
async def test_mcp_server_tools():
    """Test MCP server tool registration"""
    with patch('vscode.window'), patch('vscode.workspace'):
        server = VSCodeMCPServer("/tmp/test_workspace")
        await server.initialize()
        
        # Test that tools are registered
        assert "execute_task" in server.tools
        assert "open_file" in server.tools
        assert "show_message" in server.tools

def test_task_provider():
    """Test task provider functionality"""
    provider = AgentTaskProvider("/tmp/test_workspace")
    
    # Test default tasks creation
    default_tasks = provider.create_default_tasks()
    
    assert len(default_tasks) > 0
    assert any(task["id"] == "analyze_code" for task in default_tasks)
    assert any(task["id"] == "run_tests" for task in default_tasks)
```

### Extension Testing
```typescript
// test/extension.test.ts
import * as assert from 'assert';
import * as vscode from 'vscode';
import * as extension from '../src/extension';

suite('Extension Test Suite', () => {
    vscode.window.showInformationMessage('Start all tests.');

    test('Extension activation', async () => {
        const ext = vscode.extensions.getExtension('superalita.super-alita-vscode');
        assert.ok(ext);
        
        await ext.activate();
        assert.ok(ext.isActive);
    });

    test('Chat command registration', async () => {
        const commands = await vscode.commands.getCommands(true);
        assert.ok(commands.includes('superalita.chat'));
    });

    test('Analyze code command registration', async () => {
        const commands = await vscode.commands.getCommands(true);
        assert.ok(commands.includes('superalita.analyzeCode'));
    });

    test('Task provider registration', async () => {
        const tasks = await vscode.tasks.fetchTasks({ type: 'superalita' });
        assert.ok(tasks.length > 0);
    });
});
```

## Security Guidelines

### VS Code Security Patterns
```python
class VSCodeSecurityManager:
    """Security management for VS Code integration"""
    
    def __init__(self, workspace_root: str):
        self.workspace_root = Path(workspace_root).resolve()
        self.allowed_commands = {
            'python', 'node', 'npm', 'git', 'code',
            'pytest', 'mypy', 'black', 'ruff'
        }
        
    def validate_task_command(self, task: VSCodeTask) -> bool:
        """Validate task command for security"""
        # Check if command is in allowlist
        command_name = task.command.split()[0] if ' ' in task.command else task.command
        if command_name not in self.allowed_commands:
            return False
            
        # Validate file paths
        if task.cwd:
            cwd_path = Path(task.cwd).resolve()
            if not cwd_path.is_relative_to(self.workspace_root):
                return False
                
        # Check for dangerous arguments
        dangerous_patterns = ['rm -rf', 'del /f', 'format', 'shutdown']
        full_command = f"{task.command} {' '.join(task.args)}"
        
        for pattern in dangerous_patterns:
            if pattern in full_command.lower():
                return False
                
        return True
        
    def validate_file_access(self, file_path: str) -> bool:
        """Validate file access is within workspace"""
        try:
            abs_path = Path(file_path).resolve()
            return abs_path.is_relative_to(self.workspace_root)
        except Exception:
            return False
            
    def sanitize_user_input(self, user_input: str) -> str:
        """Sanitize user input"""
        # Remove potentially dangerous characters
        dangerous_chars = ['&', '|', ';', '`', '$', '(', ')']
        sanitized = user_input
        
        for char in dangerous_chars:
            sanitized = sanitized.replace(char, '')
            
        return sanitized[:1000]  # Limit length
```

## Performance Guidelines

### Efficient VS Code Integration
```python
class OptimizedVSCodeIntegration(VSCodeAgentIntegration):
    """Performance-optimized VS Code integration"""
    
    def __init__(self, workspace_root: str = None):
        super().__init__(workspace_root)
        self.command_cache: Dict[str, Any] = {}
        self.file_watcher = None
        
    async def initialize(self, context: Any) -> bool:
        """Initialize with performance optimizations"""
        success = await super().initialize(context)
        
        if success:
            # Setup file watching for efficient updates
            await self._setup_file_watcher()
            
            # Preload common operations
            await self._preload_cache()
            
        return success
        
    async def _setup_file_watcher(self):
        """Setup efficient file watching"""
        if vscode.workspace.workspaceFolders:
            pattern = vscode.RelativePattern(
                vscode.workspace.workspaceFolders[0],
                "**/*.{py,ts,js,json}"
            )
            
            self.file_watcher = vscode.workspace.createFileSystemWatcher(pattern)
            
            self.file_watcher.onDidChange(self._on_file_changed)
            self.file_watcher.onDidCreate(self._on_file_created)
            self.file_watcher.onDidDelete(self._on_file_deleted)
            
    async def _on_file_changed(self, uri: vscode.Uri):
        """Handle file change events"""
        # Invalidate relevant cache entries
        file_path = uri.fsPath
        cache_keys_to_remove = [
            key for key in self.command_cache
            if file_path in str(key)
        ]
        
        for key in cache_keys_to_remove:
            del self.command_cache[key]
            
    async def execute_cached_command(self, command_id: str, *args) -> Any:
        """Execute command with caching"""
        cache_key = f"{command_id}:{hash(args)}"
        
        if cache_key in self.command_cache:
            return self.command_cache[cache_key]
            
        # Execute command
        result = await self._execute_command(command_id, *args)
        
        # Cache result if appropriate
        if self._should_cache_result(command_id):
            self.command_cache[cache_key] = result
            
        return result
```

## Common Patterns

### VS Code Extension Patterns
```typescript
// Common patterns for VS Code extension development

export class ExtensionManager {
    private static instance: ExtensionManager;
    private context: vscode.ExtensionContext;
    
    static getInstance(context?: vscode.ExtensionContext): ExtensionManager {
        if (!ExtensionManager.instance) {
            ExtensionManager.instance = new ExtensionManager(context!);
        }
        return ExtensionManager.instance;
    }
    
    private constructor(context: vscode.ExtensionContext) {
        this.context = context;
    }
    
    registerDisposable(disposable: vscode.Disposable) {
        this.context.subscriptions.push(disposable);
    }
    
    async showProgress<T>(
        title: string,
        task: (progress: vscode.Progress<{ message?: string; increment?: number }>) => Promise<T>
    ): Promise<T> {
        return vscode.window.withProgress({
            location: vscode.ProgressLocation.Notification,
            title,
            cancellable: true
        }, task);
    }
    
    async showQuickPick<T extends vscode.QuickPickItem>(
        items: T[],
        options?: vscode.QuickPickOptions
    ): Promise<T | undefined> {
        return vscode.window.showQuickPick(items, options);
    }
}

export interface AgentWebviewProvider {
    createWebviewPanel(title: string, viewType: string): vscode.WebviewPanel;
    handleMessage(message: any): Promise<void>;
    updateContent(content: string): void;
}

export class AgentWebviewManager implements AgentWebviewProvider {
    private panels: Map<string, vscode.WebviewPanel> = new Map();
    
    createWebviewPanel(title: string, viewType: string): vscode.WebviewPanel {
        const panel = vscode.window.createWebviewPanel(
            viewType,
            title,
            vscode.ViewColumn.Two,
            {
                enableScripts: true,
                retainContextWhenHidden: true,
                localResourceRoots: []
            }
        );
        
        this.panels.set(viewType, panel);
        
        panel.onDidDispose(() => {
            this.panels.delete(viewType);
        });
        
        return panel;
    }
    
    async handleMessage(message: any): Promise<void> {
        // Handle webview messages
        switch (message.command) {
            case 'ready':
                await this.onWebviewReady(message);
                break;
            case 'action':
                await this.onWebviewAction(message);
                break;
        }
    }
    
    updateContent(content: string): void {
        for (const panel of this.panels.values()) {
            panel.webview.html = content;
        }
    }
    
    private async onWebviewReady(message: any): Promise<void> {
        // Webview is ready to receive data
    }
    
    private async onWebviewAction(message: any): Promise<void> {
        // Handle webview actions
    }
}
```

## Debugging Tips
- **Extension debugging** - Use VS Code extension development host for debugging
- **Webview debugging** - Enable webview developer tools for UI debugging
- **MCP communication** - Monitor MCP message flow between VS Code and agent
- **Task execution** - Monitor task execution in VS Code terminal
- **File watching** - Debug file system watcher performance and accuracy