import { describe, it, expect } from 'vitest';
import { enrichCodegenMeta } from '../codegen/meta';
import { StatusController } from '../commands/status';

describe('E2E smoke', () => {
  it('enriches meta and sets status ready', async () => {
    const meta = await enrichCodegenMeta({ components: [{ path: 'testdata/sample-wasm/add.wasm' }], servicesUsed: [] });
    expect(meta.components.length).toBe(1);
    let text = '';
    const ctrl = new StatusController({ setText: (t) => (text = t), setTooltip: () => {} });
    ctrl.setAnalyzerReady(true);
    expect(text).toMatch(/WASM: Ready/);
  });
});
