export function validateRealWitOutputs(meta: { components: Array<{ path: string }> }) {
  const requireReal = process.env.REQUIRE_REAL_WIT === '1';
  if (requireReal && meta.components.length === 0) {
    throw new Error('REQUIRE_REAL_WIT=1 but meta shows 0 components');
  }
}
