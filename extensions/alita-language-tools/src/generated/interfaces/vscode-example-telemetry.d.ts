/** @module Interface vscode:example/telemetry **/
export function emitMetric(metric: PerformanceMetric): void;
export interface PerformanceMetric {
  operation: string,
  durationMs: number,
  memoryUsed: number,
  timestamp: bigint,
}
