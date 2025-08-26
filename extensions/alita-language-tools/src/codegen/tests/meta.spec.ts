import { describe, it, expect, vi } from 'vitest';
import { enrichCodegenMeta } from '../meta';

describe('codegen meta enrichment', () => {
  it('parses wasm-tools output into sizes/sections', async () => {
    const out = await enrichCodegenMeta(
      { components: [{ path: 'testdata/sample-wasm/add.wasm' }], servicesUsed: ['fs', 'net'] },
      () => Buffer.from('File: add.wasm\nSections:\n - Type (2)\n - Code (1)\nSize: 1024\n'),
    );
    expect(out.components[0]).toMatchObject({ path: expect.stringContaining('add.wasm'), sizeBytes: 1024 });
    expect(out.components[0].sections).toEqual(expect.arrayContaining(['Type', 'Code']));
  });

  it('handles missing wasm-tools gracefully', async () => {
    const out = await enrichCodegenMeta(
      { components: [{ path: 'x.wasm' }], servicesUsed: [] },
      () => { throw new Error('not found'); },
    );
    expect(out.components[0]).toHaveProperty('sizeBytes', null);
    expect(out.warnings.length).toBeGreaterThan(0);
  });
});
