import { Connection, RAL } from '@vscode/wasm-component-model';
import { world } from './generated/alita-world.generated';
import * as vscode from 'vscode';

// WebWorker scaffold for WASM component model integration.
// Enhanced with telemetry bridge and host API services for WASM components
// Generated WIT bindings (placeholder) are produced by scripts/generate-wit-stubs.cjs
// and exposed via ./generated/alita-world.generated.ts. Replace with real
// codegen output (jco / componentize-js) when available.

interface PerformanceMetric {
  operation: string;
  durationMs: number;
  memoryUsed: number;
  timestamp: number;
}

interface FileInfo {
  path: string;
  size: number;
  modified: number;
}

async function main(): Promise<void> {
  // Host services (typed): populate when bindings define host imports.
  interface AlitaHost {
    [k: string]: (...args: unknown[]) => unknown;
  }

  const baseServices: Record<string, (...args: unknown[]) => unknown> = {
    // Telemetry interface implementation
    'telemetry#emit-metric': (metric: unknown) => {
      try {
        const typedMetric = metric as PerformanceMetric;
        // Emit telemetry to extension host
        const telemetryEvent = {
          type: 'wasm-telemetry',
          timestamp: Date.now(),
          component: 'wasm-worker',
          metric: {
            operation: typedMetric.operation,
            duration: typedMetric.durationMs,
            memory: typedMetric.memoryUsed,
            wasmTimestamp: typedMetric.timestamp
          }
        };

        // Log for debugging
        RAL().console.info(`[WASM Telemetry] ${typedMetric.operation}: ${typedMetric.durationMs}ms`);

        // Post to extension host if available
        const canPost = typeof (globalThis as any).postMessage === 'function';
        if (canPost) {
          try {
            (globalThis as any).postMessage({ __alitaTelemetry: telemetryEvent });
          } catch (e) {
            RAL().console.warn('Failed to post telemetry:', e);
          }
        }
      } catch (e) {
        RAL().console.error('Telemetry emission failed:', e);
      }
    },

    // Host API interface implementation
    'host-api#get-file-info': async (path: unknown): Promise<unknown> => {
      try {
        const typedPath = path as string;
        // Request file info from extension host
        return new Promise((resolve) => {
          const requestId = Math.random().toString(36).substr(2, 9);
          const request = {
            type: 'file-info-request',
            requestId,
            path: typedPath
          };

          // Set up response handler
          const handleResponse = (event: MessageEvent) => {
            if (event.data?.__alitaFileInfo?.requestId === requestId) {
              self.removeEventListener('message', handleResponse);
              resolve(event.data.__alitaFileInfo.result);
            }
          };
          self.addEventListener('message', handleResponse);

          // Send request
          const canPost = typeof (globalThis as any).postMessage === 'function';
          if (canPost) {
            (globalThis as any).postMessage({ __alitaFileRequest: request });
          } else {
            resolve({ error: 'No communication channel available' });
          }

          // Timeout after 5 seconds
          setTimeout(() => {
            self.removeEventListener('message', handleResponse);
            resolve({ error: 'Request timeout' });
          }, 5000);
        });
      } catch (e) {
        return { error: `Failed to get file info: ${e}` };
      }
    },

    'host-api#read-file-snippet': async (path: unknown, startLine: unknown, endLine: unknown): Promise<unknown> => {
      try {
        const typedPath = path as string;
        const typedStartLine = startLine as number;
        const typedEndLine = endLine as number;

        // Request file content from extension host
        return new Promise((resolve) => {
          const requestId = Math.random().toString(36).substr(2, 9);
          const request = {
            type: 'file-read-request',
            requestId,
            path: typedPath,
            startLine: typedStartLine,
            endLine: typedEndLine
          };

          // Set up response handler
          const handleResponse = (event: MessageEvent) => {
            if (event.data?.__alitaFileRead?.requestId === requestId) {
              self.removeEventListener('message', handleResponse);
              resolve(event.data.__alitaFileRead.result);
            }
          };
          self.addEventListener('message', handleResponse);

          // Send request
          const canPost = typeof (globalThis as any).postMessage === 'function';
          if (canPost) {
            (globalThis as any).postMessage({ __alitaFileRequest: request });
          } else {
            resolve({ error: 'No communication channel available' });
          }

          // Timeout after 10 seconds
          setTimeout(() => {
            self.removeEventListener('message', handleResponse);
            resolve({ error: 'Request timeout' });
          }, 10000);
        });
      } catch (e) {
        return { error: `Failed to read file: ${e}` };
      }
    },

    'host-api#emit-diagnostic': (path: unknown, line: unknown, message: unknown) => {
      try {
        const typedPath = path as string;
        const typedLine = line as number;
        const typedMessage = message as string;

        const diagnostic = {
          type: 'wasm-diagnostic',
          timestamp: Date.now(),
          path: typedPath,
          line: typedLine,
          message: typedMessage
        };

        RAL().console.info(`[WASM Diagnostic] ${typedPath}:${typedLine} - ${typedMessage}`);

        // Post to extension host if available
        const canPost = typeof (globalThis as any).postMessage === 'function';
        if (canPost) {
          try {
            (globalThis as any).postMessage({ __alitaDiagnostic: diagnostic });
          } catch (e) {
            RAL().console.warn('Failed to post diagnostic:', e);
          }
        }
      } catch (e) {
        RAL().console.error('Diagnostic emission failed:', e);
      }
    }
  };

  // Wrap services to emit granular timing + error logs
  const hostServices: Record<string, unknown> = {};
  // Bridge helper: postMessage host-call telemetry so the extension main thread can pick it up.
  // We do a feature-detect for postMessage to avoid crashes in non-worker test contexts.
  const maybeGlobal = globalThis as unknown as { postMessage?: (msg: unknown) => void };
  const canPost = typeof maybeGlobal.postMessage === 'function';

  (Object.entries(baseServices) as Array<[string, (...a: unknown[]) => unknown]>).forEach(([name, fn]) => {
    if (typeof fn !== 'function') return;
    hostServices[name] = async (...args: unknown[]) => {
      const start = Date.now();
      try {
        const res = await (fn as (...inner: unknown[]) => Promise<unknown> | unknown)(...args);
        const dur = Date.now() - start;
        const evt = { t: 'host-call', name, dur, ok: true };
        try { RAL().console.info(JSON.stringify(evt)); } catch (_e) { /* ignore console info failure */ }
        if (canPost) { try { maybeGlobal.postMessage?.({ __alitaHost: evt }); } catch { /* ignore */ } }
        return res;
      } catch (err) {
        const dur = Date.now() - start;
        const errMsg = err instanceof Error ? err.message : String(err);
        const evt = { t: 'host-call', name, dur, ok: false, err: errMsg };
        try { RAL().console.error(JSON.stringify(evt)); } catch (_e) { /* ignore console error failure */ }
        if (canPost) { try { maybeGlobal.postMessage?.({ __alitaHost: evt }); } catch { /* ignore */ } }
        throw err;
      }
    };
  });

  type CreateWorkerWithServices = (
    w: unknown,
    opts?: { services?: Record<string, unknown> }
  ) => Promise<{ listen(): void }>;

  const create = Connection.createWorker as unknown as CreateWorkerWithServices;
  const connection = await create(world, { services: hostServices });
  connection.listen();
}

main().catch(err => {
  try {
    RAL().console.error('Worker initialization failed', err);
  } catch {
    // swallow if RAL console unavailable early
  }
});
