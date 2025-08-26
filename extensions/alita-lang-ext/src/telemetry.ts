import * as vscode from 'vscode';
import TelemetryReporter from 'vscode-extension-telemetry';

export type Telemetry = { send: (name: string, props?: Record<string,string>) => void; dispose: () => void };

export function createTelemetry(ctx: vscode.ExtensionContext): Telemetry | null {
  // Respect global VS Code telemetry gate
  const isVsEnabled = vscode.env.isTelemetryEnabled; // mirrors isTelemetryEnabled API
  const enableExt = vscode.workspace.getConfiguration('alita').get<boolean>('telemetry.enable', true);
  if (!isVsEnabled || !enableExt) return null;

  // Read from secrets storage or env (do NOT hardcode or commit)
  const key = process.env.ALITA_APPINSIGHTS_KEY || (ctx.secrets.get('ALITA_APPINSIGHTS_KEY') as unknown as string);
  if (!key) return null;

  const reporter = new TelemetryReporter('super-alita.alita-lang-ext', '0.1.0', key);
  ctx.subscriptions.push(reporter);
  return {
    send: (name, props) => reporter.sendTelemetryEvent(name, props),
    dispose: () => reporter.dispose()
  };
}
