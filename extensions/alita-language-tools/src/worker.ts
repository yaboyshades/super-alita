import { Connection, RAL } from '@vscode/wasm-component-model';
import { world } from './generated/alita-world.generated';
// WebWorker scaffold for WASM component model integration.
// Generated WIT bindings (placeholder) are produced by scripts/generate-wit-stubs.cjs
// and exposed via ./generated/alita-world.generated.ts. Replace with real
// codegen output (jco / componentize-js) when available.
// If the generated world requires host functions, wire them via `services` below.

async function main(): Promise<void> {
  // Host services (typed): populate when bindings define host imports.
  interface AlitaHost { [k: string]: (...args: unknown[]) => unknown }
  const baseServices: Partial<AlitaHost> = {};
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
