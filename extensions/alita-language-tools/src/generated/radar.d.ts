// world root:component/root
export type PerformanceMetric = import('./interfaces/vscode-example-telemetry.js').PerformanceMetric;
export interface Diagnostic {
  line: number,
  col: number,
  severity: number,
  code: string,
  message: string,
  suggestion?: string,
}
export interface SmellAnalysis {
  complexityScore: number,
  maintainabilityIndex: number,
  debtMinutes: number,
  smellTypes: Array<string>,
}
export type * as VscodeExampleHostApi from './interfaces/vscode-example-host-api.js'; // import vscode:example/host-api
export type * as VscodeExampleTelemetry from './interfaces/vscode-example-telemetry.js'; // import vscode:example/telemetry
export function analyze(source: string): Array<Diagnostic>;
export function analyzeFile(path: string): Array<Diagnostic>;
export function detectSmells(source: string): SmellAnalysis;
export function predictIssues(source: string, history: Array<string>): Array<Diagnostic>;
export function getPerformanceStats(): Array<PerformanceMetric>;
