import { describe, it, expect, vi, beforeEach } from 'vitest';
import { prefetchUsingWasmAnalysis } from '../prefetchUsingWasmAnalysis';
import * as flags from '../../utils/featureFlags';

describe('prefetchUsingWasmAnalysis', () => {
  const cache = { put: vi.fn() };
  const emit = vi.fn();
  const fakeWorkerAnalyze = vi.fn();
  beforeEach(() => {
    vi.spyOn(flags, 'isEnabled').mockImplementation((k) => k === 'alita.predictive.wasm.enabled');
    (cache.put as any).mockReset();
    emit.mockReset();
    fakeWorkerAnalyze.mockReset();
  });

  it('skips when flag disabled', async () => {
    vi.spyOn(flags, 'isEnabled').mockReturnValue(false as any);
    await prefetchUsingWasmAnalysis({ source: 'let a=1', cache, emit, analyzeFn: fakeWorkerAnalyze });
    expect(fakeWorkerAnalyze).not.toHaveBeenCalled();
    expect(cache.put).not.toHaveBeenCalled();
  });

  it('invokes analyzer and stores into cache', async () => {
    fakeWorkerAnalyze.mockResolvedValue({ actions: [{ kind: 'fix-error', range: [0, 5] }], ms: 8 });
    await prefetchUsingWasmAnalysis({ source: 'bad();', cache, emit, analyzeFn: fakeWorkerAnalyze });
    expect(fakeWorkerAnalyze).toHaveBeenCalled();
    expect(cache.put).toHaveBeenCalledWith(expect.objectContaining({ sourceFingerprint: expect.any(String), actions: expect.any(Array) }));
    expect(emit).toHaveBeenCalledWith('predictive_wasm_analysis', expect.objectContaining({ ok: true }));
  });

  it('reports failure telemetry gracefully', async () => {
    fakeWorkerAnalyze.mockRejectedValue(new Error('boom'));
    await prefetchUsingWasmAnalysis({ source: 'x', cache, emit, analyzeFn: fakeWorkerAnalyze });
    expect(emit).toHaveBeenCalledWith('predictive_wasm_analysis', expect.objectContaining({ ok: false }));
  });
});
