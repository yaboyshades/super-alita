#!/usr/bin/env node
// Verify componentization produced a usable artifact and optionally validate it
const fs = require('fs');
const path = require('path');
const cp = require('child_process');

const extRoot = path.resolve(__dirname, '..');
const compDir = path.join(extRoot, 'src', 'generated', 'components');

if (!fs.existsSync(compDir)) {
  console.warn('[test:component] components dir not found:', compDir);
  process.exit(0);
}
const files = fs.readdirSync(compDir).filter(f => f.endsWith('.component.wasm'));
if (!files.length) {
  console.error('[test:component] no component artifacts found');
  process.exit(2);
}
const full = path.join(compDir, files[0]);
const sz = fs.statSync(full).size;
console.log('[test:component] found', full, 'size=', sz);
if (sz <= 8) {
  console.error('[test:component] component too small');
  process.exit(3);
}

function which(cmd) {
  try {
    const res = process.platform === 'win32'
      ? cp.spawnSync('where', [cmd], { stdio: 'pipe' })
      : cp.spawnSync('which', [cmd], { stdio: 'pipe' });
    return res.status === 0;
  } catch { return false; }
}

if (which('wasm-tools')) {
  const r = cp.spawnSync('wasm-tools', ['validate', full], { stdio: 'inherit' });
  if (r.status !== 0) {
    console.error('[test:component] wasm-tools validate failed');
    process.exit(r.status || 4);
  }
}

process.exit(0);

