#!/usr/bin/env node
// Build sample-wasm/add.wasm from add.wat using wasm-tools (if present)
const fs = require('fs');
const path = require('path');
const cp = require('child_process');

const extRoot = path.resolve(__dirname, '..');
const wat = path.join(extRoot, 'sample-wasm', 'add.wat');
const wasm = path.join(extRoot, 'sample-wasm', 'add.wasm');

function which(cmd) {
  try {
    const res = process.platform === 'win32'
      ? cp.spawnSync('where', [cmd], { stdio: 'pipe' })
      : cp.spawnSync('which', [cmd], { stdio: 'pipe' });
    return res.status === 0;
  } catch { return false; }
}

if (!fs.existsSync(wat)) {
  console.error('[build-sample-wasm] missing WAT:', wat);
  process.exit(1);
}

if (!which('wasm-tools')) {
  console.warn('[build-sample-wasm] wasm-tools not found; skipping build.');
  process.exit(0);
}

const args = ['parse', wat, '-o', wasm];
console.log('[build-sample-wasm] wasm-tools', args.join(' '));
const res = cp.spawnSync('wasm-tools', args, { stdio: 'inherit', cwd: extRoot });
if (res.status !== 0) {
  console.error('[build-sample-wasm] failed to parse WAT');
  process.exit(res.status || 1);
}
if (!fs.existsSync(wasm) || fs.statSync(wasm).size === 0) {
  console.error('[build-sample-wasm] output wasm missing or empty:', wasm);
  process.exit(2);
}
console.log('[build-sample-wasm] wrote', wasm);

