# 🧠 GitHub Copilot + Mangle Integration

**Native Code Knowledge Graph reasoning for GitHub Copilot**

Transform your GitHub Copilot experience with automatic deductive reasoning over your codebase using Mangle integration. Every interaction now includes constitutional compliance checking, quality analysis, and specification traceability.

## ✨ Features

### 🔮 **Automatic Enhancement**
- **Every GitHub Copilot question** is automatically enhanced with code knowledge graph analysis
- **Constitutional compliance checking** built into every response
- **Quality analysis** and recommendations included contextually
- **Specification traceability** for understanding code relationships

### 🎯 **Natural Language Queries**
Ask GitHub Copilot questions like:
- *"What functions are untested?"*
- *"What violates the constitution?"*
- *"Show me quality issues"*
- *"What features are incomplete?"*
- *"Trace this function to its specification"*

### 📊 **Intelligent Analysis**
- **Constitutional Article Compliance**: Automatic validation against all 6 articles
- **Code Quality Assessment**: Complexity, test coverage, architecture analysis
- **Dependency Analysis**: Circular dependencies, hotspots, library usage
- **Specification Mapping**: Bidirectional code-to-spec traceability

## 🚀 Quick Start

### 1. **One-Command Setup**
```bash
python setup_copilot_mangle.py
```

This automatically:
- ✅ Enables Mangle reasoning in GitHub Copilot
- ✅ Sets up constitutional compliance monitoring
- ✅ Configures automatic code analysis
- ✅ Installs VS Code extension (if available)

### 2. **Start Using GitHub Copilot**
No changes needed! Just use GitHub Copilot normally:

```
You: "How can I improve the quality of this function?"

GitHub Copilot (Enhanced): "I can help improve this function!
🧠 Mangle Analysis shows 3 quality issues:
- Function complexity: 12 (recommendation: <10)
- Missing test coverage (violates Constitutional Article II)
- Could use existing library instead of custom implementation

Here's how to improve it..."
```

### 3. **Advanced Commands** (Optional)
Use dedicated commands for specific analysis:
- `Ctrl+Shift+M Q` - Ask Mangle question
- `Ctrl+Shift+M C` - Check constitutional compliance
- `Ctrl+Shift+M Q` - Analyze code quality

## 🏗️ Architecture

The integration works through multiple layers:

```
┌─────────────────────────────────────────┐
│            GitHub Copilot               │
├─────────────────────────────────────────┤
│      Mangle Enhanced Agent              │ ← Automatic enhancement
├─────────────────────────────────────────┤
│         Mangle Reasoner                 │ ← Deductive reasoning
├─────────────────────────────────────────┤
│       Code Knowledge Graph             │ ← Facts + Rules
├─────────────────────────────────────────┤
│    Constitutional Framework            │ ← Compliance checking
└─────────────────────────────────────────┘
```

### **Native Integration Points**
1. **Chat Enhancement**: Every Copilot interaction includes Mangle context
2. **Command Palette**: Dedicated Mangle commands in VS Code
3. **Context Menus**: Right-click for code analysis
4. **Status Bar**: Constitutional compliance indicators
5. **Hover Tooltips**: Quality insights on hover

## ⚙️ Configuration

### **VS Code Settings**
```json
{
  "copilot.mangle.enabled": true,
  "copilot.mangle.autoAnalyze": true,
  "copilot.mangle.constitutionalMode": true,
  "copilot.mangle.workspaceRoot": "${workspaceFolder}"
}
```

### **Environment Variables**
```bash
export COPILOT_MANGLE_MODE=true
export COPILOT_AUTO_ENHANCE=true
export SDD_CONSTITUTIONAL_MODE=true
```

## 📖 Example Interactions

### **Code Quality Analysis**
```
You: "Is this code good quality?"

Copilot: "Let me analyze this with Mangle reasoning...

🔍 Quality Analysis:
- ✅ Function length: 23 lines (within 50-line limit)
- ⚠️  Complexity: 8 (approaching 10 limit)
- ❌ Missing tests (violates Article II)
- ✅ Uses existing libraries (follows Article I)

Recommendations:
1. Add test coverage for constitutional compliance
2. Consider extracting complex logic into helper functions
3. Add documentation for clarity (Article V)"
```

### **Constitutional Compliance**
```
You: "Check if my code follows the constitutional framework"

Copilot: "🏛️ Constitutional Analysis Results:

Article I (Library-First): ✅ COMPLIANT
- Using requests library instead of custom HTTP
- Leveraging existing parsers

Article II (Test-First): ❌ 3 VIOLATIONS
- Functions without tests: calculate_score, validate_input, format_output

Article III (Simplicity): ⚠️  2 WARNINGS
- process_data() has 12 complexity (limit: 10)
- UserManager class has 25 methods (consider splitting)

Overall Score: 0.73 (threshold: 0.75)
Action needed: Add tests and reduce complexity"
```

### **Specification Traceability**
```
You: "What specification does this function implement?"

Copilot: "🔗 Specification Trace:

Function: send_message()
├── Feature: 001-real-time-chat-system
├── Acceptance Criteria: AC-03 "Message delivery confirmation"
├── Tests: test_message_delivery_confirmation()
└── Related Functions: validate_message(), queue_message()

This function is fully traced to specification and has test coverage!"
```

## 🛠️ Development

### **API Endpoints**
The integration exposes several API endpoints:
- `POST /sdd/ask` - Natural language queries
- `GET /sdd/validate` - Constitutional validation
- `GET /sdd/analyze/quality` - Quality analysis
- `POST /sdd/trace` - Code-to-spec tracing

### **Extension Development**
The VS Code extension provides:
- Command registration for Mangle operations
- Webview panels for analysis results
- Status bar integration
- Context menu enhancements

### **Custom Rules**
Add custom Mangle rules in `src/sdd/mangle_rules.py`:
```prolog
// Custom rule for API endpoint documentation
undocumented_api_endpoint(Endpoint) :-
  api_endpoint(Endpoint, Method, Path),
  !endpoint_documentation(Endpoint, _).
```

## 🔧 Troubleshooting

### **Mangle Binary Not Found**
If you see "Mangle analysis unavailable":
1. Install Mangle: `go install github.com/google/mangle/cmd/mangle@latest`
2. Ensure it's in your PATH
3. Restart VS Code

### **Constitutional Scorer Issues**
If constitutional analysis fails:
1. Check Python environment: `python -c "import constitutional.scorer"`
2. Verify SDD framework: `python -m src.sdd.router`
3. Check server status: `curl http://localhost:8080/healthz`

### **Extension Not Loading**
If VS Code extension doesn't activate:
1. Check extension is enabled in Extensions panel
2. Reload VS Code: `Ctrl+Shift+P` → "Developer: Reload Window"
3. Check output panel for errors

## 🤝 Contributing

The Mangle integration follows the constitutional framework:
1. **Library-First**: Use existing Mangle libraries
2. **Test-First**: Write tests before implementation
3. **Simplicity**: Keep functions focused and simple
4. **Integration**: Test with real Mangle queries
5. **Clarity**: Document all reasoning patterns
6. **Counterfactual**: Justify approach over alternatives

## 📄 License

This integration is part of the Super-Alita project and follows the same licensing terms.

---

**🎉 Ready to experience GitHub Copilot with native deductive reasoning?**

Run `python setup_copilot_mangle.py` and start asking better questions!
