import { describe, it, expect } from 'vitest';
import { isEnabled } from '../featureFlags';

describe('feature flags', () => {
  it('reads boolean-ish env vars', () => {
    process.env['alita.predictive.wasm.enabled'] = 'true';
    expect(isEnabled('alita.predictive.wasm.enabled')).toBe(true);
    process.env['alita.predictive.wasm.enabled'] = '0';
    expect(isEnabled('alita.predictive.wasm.enabled')).toBe(false);
  });
});
