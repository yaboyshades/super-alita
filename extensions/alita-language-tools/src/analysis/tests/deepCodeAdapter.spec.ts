import { describe, it, expect, vi } from 'vitest';
import { runDeepCodeAnalysis } from '../deepCodeAdapter';

describe('DeepCode adapter', () => {
  it('surfaces findings and computes metrics', async () => {
    const emit = vi.fn();
    const mockClient = { analyze: vi.fn().mockResolvedValue([
      { rule: 'no-dead-code', message: 'Remove unused var', accepted: true },
      { rule: 'prefer-const', message: 'Use const', accepted: false, falsePositive: true },
    ]) };
    const res = await runDeepCodeAnalysis({ path: 'testdata/deepcode/sample.ts', client: mockClient as any, emit });
    expect(res.findings).toHaveLength(2);
    expect(res.metrics.acceptanceRate).toBeCloseTo(0.5);
    expect(res.metrics.falsePositiveRate).toBeCloseTo(0.5);
    expect(emit).toHaveBeenCalledWith('deepcode_analysis', expect.objectContaining({ count: 2 }));
  });
});

