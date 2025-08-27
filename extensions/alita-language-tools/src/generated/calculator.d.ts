// world root:component/root
export type PerformanceMetric = import('./interfaces/vscode-example-telemetry.js').PerformanceMetric;
export interface PerformanceStat {
  operation: string,
  durationMs: number,
  memoryUsed: number,
  timestamp: bigint,
}
export type * as VscodeExampleTelemetry from './interfaces/vscode-example-telemetry.js'; // import vscode:example/telemetry
export function add(a: number, b: number): number;
export function multiply(a: number, b: number): number;
export function divide(a: number, b: number): number;
export function getPerformanceStats(): Array<PerformanceStat>;
