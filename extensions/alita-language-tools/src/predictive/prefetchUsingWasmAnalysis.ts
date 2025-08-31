import { isEnabled } from '../utils/featureFlags';

export interface AnalysisAction { kind: string; [k: string]: unknown }
export interface AnalyzeResult { actions: AnalysisAction[]; ms?: number }

export async function prefetchUsingWasmAnalysis(opts: {
  source: string;
  cache: { put: (item: any) => void };
  emit: (name: string, payload?: Record<string, unknown>) => void;
  analyzeFn: (source: string) => Promise<AnalyzeResult>;
}): Promise<void> {
  if (!isEnabled('alita.predictive.wasm.enabled')) return;
  const { source, cache, emit, analyzeFn } = opts;
  try {
    const res = await analyzeFn(source);
    const fp = await fingerprint(source);
    cache.put({ sourceFingerprint: fp, actions: res.actions, tookMs: res.ms ?? null });
    emit('predictive_wasm_analysis', { ok: true, actions: res.actions.length, tookMs: res.ms ?? null });
  } catch (err: any) {
    emit('predictive_wasm_analysis', { ok: false, error: String(err?.message || err) });
  }
}

async function fingerprint(s: string): Promise<string> {
  // Simple stable hash (djb2)
  let h = 5381;
  for (let i = 0; i < s.length; i++) h = ((h << 5) + h) ^ s.charCodeAt(i);
  return (h >>> 0).toString(16);
}
