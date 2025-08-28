# Super Alita Multi-Terminal Development Setup

This document describes the comprehensive multi-terminal orchestration and command safety system implemented for Super Alita.

## 🚀 Quick Start

1. **Start the enhanced development environment:**
   ```bash
   # Opens 4 parallel terminals automatically
   Ctrl+Shift+P → "Tasks: Run Task" → "🚀 Start Super Alita Development Environment"
   ```

2. **Use GitHub Copilot CLI for safe commands:**
   ```bash
   ghcs "start Super Alita development server"  # Get command suggestions
   ghce "curl http://127.0.0.1:8080/healthz"   # Explain commands before running
   ```

3. **Load PowerShell helpers:**
   ```powershell
   . .\scripts\copilot-helpers.ps1
   ```

## 🎯 Features Implemented

### Multi-Terminal Orchestration
- **4 parallel terminals** launch automatically on folder open:
  - `terminal:super-alita-server` - Main application server with hot reload
  - `terminal:mcp-server` - MCP server for tool integration
  - `terminal:monitoring` - Health checks and system monitoring
  - `terminal:interactive-shell` - Development commands with Copilot CLI

### Command Safety with GitHub Copilot CLI
- **Safe command generation**: `ghcs "task description"` 
- **Command explanation**: `ghce "command"` explains before execution
- **Confirmation required** for all generated commands
- **Super Alita specific helpers** for common development tasks

### Pre-Commit Safety Hooks
- **Husky + lint-staged** integration blocks bad commits
- **Automatic linting** with Ruff for Python code
- **Formatting checks** with Black
- **Critical file validation** for core system files
- **Pre-push hooks** prevent broken code from reaching repository

### Cross-Platform Scripts
- **PowerShell and Bash versions** of all development scripts
- **cross-env and shx** for unified command syntax across Windows/macOS/Linux
- **Automatic environment detection** and setup

## 🛠️ Available Commands

### VS Code Tasks
- `🚀 Start Super Alita Development Environment` - Multi-terminal startup (default)
- `🔍 Full System Validation` - Complete deployment validation
- `🏥 Health Check` - Quick server health check
- `🛠️ Tools Catalog Check` - Verify MCP tools are available

### NPM Scripts (Cross-Platform)
```bash
npm run dev          # Start development server
npm run validate     # Run deployment validation
npm run health       # Check server health
npm run lint         # Run all linting
npm run format       # Format code
npm run test         # Run tests
```

### PowerShell Scripts
```powershell
.\scripts\start-dev.ps1      # Start development server
.\scripts\monitor-dev.ps1    # Continuous monitoring
.\scripts\quick-test.ps1     # Quick validation
```

### GitHub Copilot CLI Aliases
```bash
ghcs "task description"      # Safe command suggestions
ghce "command to explain"    # Explain commands
Start-SuperAlitaDev         # Quick development startup
Test-SuperAlitaHealth       # Health check
Get-SuperAlitaTools         # List available tools
```

## 🔒 Safety Features

### Real-Time Linting
- **ShellCheck extension** installed and configured
- **Real-time shell script validation** in VS Code
- **Python linting** with Ruff integration

### Git Hooks Protection
- **Pre-commit hooks** run linting and validation
- **Pre-push hooks** ensure system stability before remote push
- **Critical file monitoring** for core system components
- **Automatic formatting** enforcement

### Command Confirmation
- **GitHub Copilot CLI** configured to require confirmation
- **Safe command generation** with explanations
- **No automatic execution** of suggested commands

## 📊 Monitoring and Validation

### Continuous Monitoring
- **Automatic health checks** every 30 seconds
- **Interactive command suggestions** via Copilot CLI
- **Log monitoring** and error detection
- **Failure alerting** after consecutive issues

### Validation Levels
1. **Quick validation** - Health and tools catalog
2. **Full validation** - Complete deployment check + linting
3. **Pre-commit validation** - Staged file checks
4. **Pre-push validation** - Complete system validation

## 🎉 Getting Started

1. The system auto-starts when you open the workspace
2. Use `ghcs` for any command you're unsure about
3. All terminals are pre-configured and ready to use
4. Git hooks protect against committing broken code

**Example workflow:**
```bash
# System starts automatically with 4 terminals
# Want to check something? Ask Copilot:
ghcs "check if Super Alita is running properly"

# Want to understand a command?
ghce "python validate_deployment.py"

# Make changes, then commit (automatically validated):
git add .
git commit -m "feature: new functionality"  # Runs pre-commit hooks
git push  # Runs pre-push validation
```

This setup ensures safe, efficient development with multiple safety nets!