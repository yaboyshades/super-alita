# Super Alita Guardian VS Code Extension

This extension integrates the Super Alita Architectural Guardian directly into VS Code's Agent Mode, providing real-time architectural guidance and telemetry tracking.

## Features

### 🛡️ Chat Participant (@alita)
- **Guardian Mode**: Architectural compliance checking and guidance
- **Audit Mode**: Comprehensive workspace architectural review  
- **Refactor Mode**: Code transformation suggestions
- **Generator Mode**: Create compliant architectural components

### 📊 Telemetry Dashboard
- Real-time compliance scoring
- Guideline reference tracking
- Mode usage analytics
- Interactive web-based dashboard

### 🔧 Commands
- `Super Alita: Show Telemetry Dashboard` - Opens telemetry webview
- `Super Alita: Run Architectural Compliance Check` - Triggers compliance audit
- `Super Alita: Audit Workspace Architecture` - Full workspace analysis
- `Super Alita: Toggle Guardian Mode` - Enable/disable guardian functionality

## Installation

### Option 1: Install from VSIX
```bash
# Navigate to extension directory
cd .vscode/extensions/super-alita-guardian

# Install dependencies
npm install

# Compile TypeScript
npm run compile

# Package extension
vsce package

# Install in VS Code
code --install-extension super-alita-guardian-2.0.0.vsix
```

### Option 2: Development Mode
```bash
# Open in VS Code
code .vscode/extensions/super-alita-guardian

# Press F5 to launch Extension Development Host
# The extension will be active in the new window
```

## Usage

### Basic Chat Interaction
```
@alita review this code for architectural compliance
@alita audit my workspace
@alita refactor this plugin to follow patterns
@alita generate a new plugin template
```

### Status Bar
- Click the "🛡️ Alita Guardian" status bar item for quick telemetry access

### Command Palette
- `Ctrl+Shift+P` → Search "Super Alita" for available commands

## Configuration

Configure the extension via VS Code settings:

```json
{
  "super-alita.guardian.enabled": true,
  "super-alita.telemetry.enabled": true,
  "super-alita.compliance.level": "moderate",
  "super-alita.dashboard.refreshInterval": 30000
}
```

## Architectural Guidelines

The guardian enforces these 5 core guidelines:

1. **Plugin Architecture** - Inherit from `PluginInterface`
2. **Tool Registry Management** - Use `SimpleAbilityRegistry`
3. **REUG State Machine Patterns** - Async handlers with `TransitionTrigger`
4. **Event Bus Patterns** - Use `create_event()` helper
5. **Component Integration** - Integrate via `DecisionPolicyEngine`

## Telemetry

The extension tracks:
- **Compliance Score**: Overall architectural adherence percentage
- **Guideline References**: Which guidelines are being followed
- **Mode Usage**: How different operational modes are used
- **Interaction Count**: Total guardian interactions

All telemetry is stored locally in `.vscode/telemetry.json`.

## Development

### File Structure
```
.vscode/extensions/super-alita-guardian/
├── package.json          # Extension manifest
├── tsconfig.json         # TypeScript configuration
├── src/
│   ├── extension.ts      # Main extension entry point
│   └── guardian.ts       # Guardian logic and chat handling
└── out/                  # Compiled JavaScript output
```

### Key Classes
- `SuperAlitaGuardian`: Main guardian logic, chat handling, telemetry
- Extension activation: Chat participant registration, command setup

### Building
```bash
npm run compile    # Compile TypeScript
npm run watch      # Watch mode for development
```

## Integration Points

### With Super Alita System
- Reads architectural guidelines from `.github/copilot/prompt.md`
- Integrates with existing telemetry scripts in `scripts/`
- Follows same architectural patterns as the main agent system

### With VS Code
- Chat participant for seamless conversation
- Command palette integration
- Status bar indicator
- Webview dashboard for rich telemetry display

## Troubleshooting

### Extension Not Loading
1. Check VS Code version compatibility (>= 1.90.0)
2. Verify TypeScript compilation: `npm run compile`
3. Check extension host console for errors

### Chat Participant Not Available
1. Ensure VS Code has chat/copilot features enabled
2. Restart VS Code after installation
3. Check if @alita appears in chat participant suggestions

### Telemetry Dashboard Empty
1. Interact with @alita in chat to generate telemetry data
2. Check if `.vscode/telemetry.json` is being created
3. Verify dashboard refresh interval in settings

## License

Part of the Super Alita Agent System - see main project license.