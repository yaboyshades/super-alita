import * as vscode from "vscode";
import axios from "axios";

interface MangleAnalysisResult {
  question: string;
  answer: string;
  query_used?: string;
  insights: string[];
  actions: string[];
}

export function activate(context: vscode.ExtensionContext) {
  console.log("GitHub Copilot Mangle Integration activated");

  // Register commands
  registerCommands(context);

  // Hook into Copilot API if available
  hookIntoCopilot(context);

  // Show activation message
  if (getConfig("enabled")) {
    vscode.window.showInformationMessage(
      "🧠 GitHub Copilot enhanced with Mangle reasoning! Ask questions about your code quality and constitutional compliance."
    );
  }
}

function registerCommands(context: vscode.ExtensionContext) {
  // Ask Mangle Question command
  const askQuestion = vscode.commands.registerCommand(
    "copilot.mangle.askQuestion",
    async () => {
      const question = await vscode.window.showInputBox({
        prompt: "Ask a question about your codebase",
        placeHolder:
          'e.g., "what functions are untested?" or "what violates constitution?"',
      });

      if (question) {
        await executeMangleQuery(question);
      }
    }
  );

  // Analyze Constitutional Compliance
  const analyzeConstitutional = vscode.commands.registerCommand(
    "copilot.mangle.analyzeConstitutional",
    async () => {
      await executeMangleQuery("constitutional violations");
    }
  );

  // Check Code Quality
  const checkQuality = vscode.commands.registerCommand(
    "copilot.mangle.checkQuality",
    async () => {
      await executeMangleQuery("quality issues");
    }
  );

  // Trace to Specification
  const traceToSpec = vscode.commands.registerCommand(
    "copilot.mangle.traceToSpec",
    async () => {
      const editor = vscode.window.activeTextEditor;
      if (!editor) {
        vscode.window.showWarningMessage("No active editor found");
        return;
      }

      const selection = editor.selection;
      const selectedText = editor.document.getText(selection);

      if (selectedText) {
        await executeMangleTrace(selectedText);
      } else {
        vscode.window.showWarningMessage(
          "Please select a function or code element to trace"
        );
      }
    }
  );

  // Toggle Mangle Mode
  const toggleMode = vscode.commands.registerCommand(
    "copilot.mangle.toggleMode",
    async () => {
      const currentState = getConfig("enabled");
      await vscode.workspace
        .getConfiguration("copilot.mangle")
        .update("enabled", !currentState, true);

      const newState = !currentState;
      vscode.window.showInformationMessage(
        `Mangle integration ${newState ? "enabled" : "disabled"}`
      );
    }
  );

  context.subscriptions.push(
    askQuestion,
    analyzeConstitutional,
    checkQuality,
    traceToSpec,
    toggleMode
  );
}

async function executeMangleQuery(question: string) {
  const workspaceRoot = getWorkspaceRoot();

  try {
    vscode.window.withProgress(
      {
        location: vscode.ProgressLocation.Notification,
        title: "Analyzing with Mangle reasoning...",
        cancellable: false,
      },
      async () => {
        const result = await callMangleAPI("/sdd/ask", { question });
        await showMangleResults(question, result);
      }
    );
  } catch (error) {
    vscode.window.showErrorMessage(`Mangle analysis failed: ${error}`);
  }
}

async function executeMangleTrace(codeElement: string) {
  try {
    vscode.window.withProgress(
      {
        location: vscode.ProgressLocation.Notification,
        title: "Tracing code to specifications...",
        cancellable: false,
      },
      async () => {
        const result = await callMangleAPI("/sdd/trace", {
          code_element: codeElement,
        });
        await showTraceResults(codeElement, result);
      }
    );
  } catch (error) {
    vscode.window.showErrorMessage(`Trace analysis failed: ${error}`);
  }
}

async function callMangleAPI(
  endpoint: string,
  data: any
): Promise<MangleAnalysisResult> {
  const baseUrl = "http://127.0.0.1:8080"; // Default SDD server

  const response = await axios.post(`${baseUrl}${endpoint}`, data, {
    headers: { "Content-Type": "application/json" },
    timeout: 30000,
  });

  return response.data;
}

async function showMangleResults(
  question: string,
  result: MangleAnalysisResult
) {
  const panel = vscode.window.createWebviewPanel(
    "mangleResults",
    "Mangle Analysis Results",
    vscode.ViewColumn.Two,
    { enableScripts: true }
  );

  panel.webview.html = generateResultsHTML(question, result);
}

async function showTraceResults(codeElement: string, result: any) {
  const panel = vscode.window.createWebviewPanel(
    "mangleTrace",
    "Code to Specification Trace",
    vscode.ViewColumn.Two,
    { enableScripts: true }
  );

  panel.webview.html = generateTraceHTML(codeElement, result);
}

function generateResultsHTML(
  question: string,
  result: MangleAnalysisResult
): string {
  const insights =
    result.insights?.map((insight) => `<li>${insight}</li>`).join("") || "";
  const actions =
    result.actions?.map((action) => `<li>${action}</li>`).join("") || "";

  return `
    <!DOCTYPE html>
    <html>
    <head>
        <title>Mangle Analysis Results</title>
        <style>
            body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; padding: 20px; }
            .header { border-bottom: 2px solid #007acc; padding-bottom: 10px; margin-bottom: 20px; }
            .question { font-size: 18px; font-weight: bold; color: #007acc; }
            .answer { background: #f8f9fa; padding: 15px; border-radius: 8px; margin: 15px 0; }
            .insights, .actions { margin: 15px 0; }
            .insights h3, .actions h3 { color: #333; margin-bottom: 10px; }
            ul { padding-left: 20px; }
            li { margin: 5px 0; }
            .query-info { font-family: monospace; background: #e9ecef; padding: 10px; border-radius: 4px; font-size: 12px; }
        </style>
    </head>
    <body>
        <div class="header">
            <div class="question">Question: ${question}</div>
        </div>

        <div class="answer">
            <h3>🧠 Mangle Analysis Result:</h3>
            <p>${result.answer}</p>
        </div>

        ${
          result.query_used
            ? `
        <div class="query-info">
            <strong>Query executed:</strong> ${result.query_used}
        </div>
        `
            : ""
        }

        ${
          insights
            ? `
        <div class="insights">
            <h3>💡 Insights:</h3>
            <ul>${insights}</ul>
        </div>
        `
            : ""
        }

        ${
          actions
            ? `
        <div class="actions">
            <h3>⚡ Actions:</h3>
            <ul>${actions}</ul>
        </div>
        `
            : ""
        }
    </body>
    </html>`;
}

function generateTraceHTML(codeElement: string, result: any): string {
  return `
    <!DOCTYPE html>
    <html>
    <head>
        <title>Code to Specification Trace</title>
        <style>
            body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; padding: 20px; }
            .header { border-bottom: 2px solid #28a745; padding-bottom: 10px; margin-bottom: 20px; }
            .code-element { font-family: monospace; background: #e9ecef; padding: 5px 10px; border-radius: 4px; }
            .trace-result { background: #f8f9fa; padding: 15px; border-radius: 8px; margin: 15px 0; }
        </style>
    </head>
    <body>
        <div class="header">
            <h2>🔍 Code to Specification Trace</h2>
            <div>Tracing: <span class="code-element">${codeElement}</span></div>
        </div>

        <div class="trace-result">
            <h3>Trace Results:</h3>
            <pre>${JSON.stringify(result, null, 2)}</pre>
        </div>
    </body>
    </html>`;
}

function hookIntoCopilot(context: vscode.ExtensionContext) {
  // Try to hook into Copilot's chat interface
  if (getConfig("enabled") && getConfig("autoAnalyze")) {
    console.log("Hooking into GitHub Copilot for automatic Mangle enhancement");

    // This would require access to Copilot's internal APIs
    // For now, we provide the commands and UI integration

    // Monitor file changes for automatic analysis
    const fileWatcher = vscode.workspace.onDidSaveTextDocument(
      async (document) => {
        if (
          document.languageId === "python" ||
          document.languageId === "javascript"
        ) {
          // Optionally run quick analysis on save
          console.log(
            `File saved: ${document.fileName} - Mangle analysis could be triggered`
          );
        }
      }
    );

    context.subscriptions.push(fileWatcher);
  }
}

function getConfig(key: string): any {
  return vscode.workspace.getConfiguration("copilot.mangle").get(key);
}

function getWorkspaceRoot(): string {
  const workspaceFolder = vscode.workspace.workspaceFolders?.[0];
  return workspaceFolder?.uri.fsPath || ".";
}

export function deactivate() {
  console.log("GitHub Copilot Mangle Integration deactivated");
}
