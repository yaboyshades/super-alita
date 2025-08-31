import { describe, it, expect } from 'vitest';
import { validateRealWitOutputs } from '../componentize';

describe('componentization gates', () => {
  it('fails when REQUIRE_REAL_WIT=1 and no components', () => {
    process.env.REQUIRE_REAL_WIT = '1';
    expect(() => validateRealWitOutputs({ components: [] })).toThrowError(/0 components/i);
  });

  it('passes with components present', () => {
    process.env.REQUIRE_REAL_WIT = '1';
    expect(() => validateRealWitOutputs({ components: [{ path: 'a.wasm' }] })).not.toThrow();
  });
});
