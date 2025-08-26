## 🚀 Quick Start Guide

### VS Code Insiders + GPT-OSS Setup

1. **Pull the GPT-OSS model**:
   `ash
   ollama pull gpt-oss:20b
   ollama serve
   `

2. **Start the Super Alita runtime** (PowerShell):
   `powershell
   $env:LLM_MODEL=\
ollama:gpt-oss:20b\
   $env:OLLAMA_HOST=\http://127.0.0.1:11434\
   python -m src.main
   `

3. **Install the extension**:
   - Open xtensions/alita-language-tools/ in VS Code
   - Press F5 to launch Extension Development Host
   - In the new window, open any Python/TypeScript project

4. **Try the commands**:
   - Alita: Invoke Agent (Ollama) - Direct Ollama integration
   - Alita: Chat via Runtime (Stream) - Streaming via local runtime
   - Alita: Generate WIT - WASM component interface generation

### Troubleshooting

- **\NEEDS
RUNTIME\ error**: The extension couldn't reach the runtime server
  - Check: curl http://127.0.0.1:8080/health
  - Start server: Follow step 2 above
- **Timeout errors**: GPT-OSS 20B model needs 60-90s for complex requests
- **Extension not loading**: Ensure you're in VS Code Insiders with Extension Development Host

### Key Features
- 🔄 **Fallback LLM**: Auto-switches between Gemini, OpenAI, and local models
- 📡 **Streaming Chat**: Real-time token streaming from runtime
- 🧮 **WASM Calculator**: Hot-reloadable WebAssembly components
- 🎯 **Predictive Execution**: Context-aware code suggestions
- 📊 **Telemetry**: Structured event tracking for agent behavior

