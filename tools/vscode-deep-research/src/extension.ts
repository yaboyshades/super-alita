import * as vscode from "vscode";
import { execFile } from "node:child_process";
import { promisify } from "node:util";
import * as path from "node:path";
import { promises as fs } from "node:fs";

const execFileAsync = promisify(execFile);

export function activate(context: vscode.ExtensionContext) {
  const participant = vscode.chat.createChatParticipant(
    "deep-research",
    async (request, _ctx, stream) => {
      const query = request.prompt.trim();
      if (!query) {
        stream.markdown("⚠️ Provide a research query.");
        return;
      }

      const folders = vscode.workspace.workspaceFolders;
      if (!folders || folders.length === 0) {
        stream.markdown("⚠️ Open a workspace to run the research pipeline.");
        return;
      }

      const workspaceRoot = folders[0].uri.fsPath;
      const script = path.join(workspaceRoot, "scripts", "run_research_agent.py");
      const config = vscode.workspace.getConfiguration("deepResearch");
      const baseUrl = config.get<string>("searx.baseUrl", "http://localhost:8080");

      const outputDir = path.join(workspaceRoot, "specs", "research-agent");
      const outputPath = path.join(outputDir, "latest.md");

      try {
        await fs.mkdir(outputDir, { recursive: true });
        stream.markdown("🔎 running CMA research pipeline...");
        const { stdout } = await execFileAsync("python", [
          script,
          query,
          "--output",
          outputPath,
        ], {
          cwd: workspaceRoot,
          env: { ...process.env, SEARXNG_BASE_URL: baseUrl },
        });

        const trimmed = stdout.trim();
        if (trimmed) {
          stream.markdown("```json\n" + trimmed + "\n```");
        }
        stream.markdown(
          "✅ Research results saved to `specs/research-agent/latest.md`. Use `/plan` to incorporate evidence."
        );
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        stream.markdown("❌ Research pipeline failed: " + message);
      }
    }
  );

  context.subscriptions.push(participant);
}

export function deactivate() {}
