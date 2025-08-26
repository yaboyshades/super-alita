#!/usr/bin/env node
/*
 Validate codegen output and optionally require real (non-stub) bindings.
 - Reads src/generated/.codegen.meta.json
 - Fails if REQUIRE_REAL_WIT=1 and mode is 'stub' or 'skip'
 - Emits a concise report to stdout
*/
const fs = require('fs');
const path = require('path');

const extRoot = path.resolve(__dirname, '..');
const metaPath = path.join(extRoot, 'src', 'generated', '.codegen.meta.json');
if (!fs.existsSync(metaPath)) {
  console.error('[test:codegen] meta file missing:', metaPath);
  process.exit(1);
}
const meta = JSON.parse(fs.readFileSync(metaPath, 'utf8'));
const requireReal = process.env.REQUIRE_REAL_WIT === '1' || process.env.REQUIRE_REAL_WIT === 'true';

const lines = meta.metrics?.lines ?? 0;
const exportsCount = Array.isArray(meta.metrics?.exports) ? meta.metrics.exports.length : 0;
console.log(`[test:codegen] mode=${meta.mode} lines=${lines} exports=${exportsCount}`);

if (requireReal) {
  if (meta.mode !== 'generated') {
    console.error('[test:codegen] REQUIRE_REAL_WIT=1 and mode is not generated');
    process.exit(2);
  }
  if (!exportsCount || lines <= 1) {
    console.error('[test:codegen] REQUIRE_REAL_WIT=1 but no real exports/lines detected');
    process.exit(3);
  }
}

process.exit(0);
