#!/usr/bin/env node
/*
 Real WIT -> TS codegen driver.
 - Prefers local `jco` if available; falls back to stub if missing.
 - Optional componentization via `wasm-tools` if configured.
 - Creates an adapter that re-exports a `world` symbol if the generator
   emits a different name.
 - Uses input hashing to skip redundant work in CI.
*/
const fs = require('fs');
const path = require('path');
const cp = require('child_process');
const crypto = require('crypto');

const repoRoot = path.resolve(__dirname, '../../..');
const extRoot = path.resolve(__dirname, '..');
const outDir = path.join(extRoot, 'src', 'generated');
fs.mkdirSync(outDir, { recursive: true });

const DEFAULT_WIT = path.join(repoRoot, 'wasm', 'code_radar', 'radar.wit');
const WIT_INPUT = process.env.WIT_INPUT || DEFAULT_WIT;
const FORCE = process.env.FORCE_REGEN === '1' || process.env.FORCE_REGEN === 'true';

function sha256File(file) {
  const h = crypto.createHash('sha256');
  h.update(fs.readFileSync(file));
  return h.digest('hex');
}

function which(cmd) {
  try {
    const res = process.platform === 'win32'
      ? cp.spawnSync('where', [cmd], { stdio: 'pipe' })
      : cp.spawnSync('which', [cmd], { stdio: 'pipe' });
    return res.status === 0;
  } catch { return false; }
}

function run(cmd, args, opts = {}) {
  const res = cp.spawnSync(cmd, args, { stdio: 'inherit', ...opts });
  return res.status === 0;
}

function writeAdapterIfNeeded() {
  const files = fs.readdirSync(outDir).filter(f => f.endsWith('.ts'));
  let hasWorld = false;
  let candidate = null;
  for (const f of files) {
    const full = path.join(outDir, f);
    const txt = fs.readFileSync(full, 'utf8');
    if (/export\s+const\s+world\b/.test(txt)) {
      hasWorld = true;
      break;
    }
    const m = txt.match(/export\s+const\s+(\w+)\b/);
    if (m) { candidate = { file: f.replace(/\.ts$/, ''), name: m[1] }; }
  }
  if (hasWorld) return true;
  if (candidate) {
    const adapter = `// AUTO-GENERATED ADAPTER: re-export generated symbol as 'world'\n` +
      `export { ${candidate.name} as world } from './${candidate.file}';\n`;
    fs.writeFileSync(path.join(outDir, 'alita-world.generated.ts'), adapter, 'utf8');
    console.log('[codegen] wrote adapter to expose world from', candidate.file, 'as', candidate.name);
    return true;
  }
  return false;
}

function writeStub() {
  const banner = `// AUTO-GENERATED STUB (fallback).\n` +
    `// Replace via real WIT codegen when tools are available.\n` +
    `/* eslint-disable @typescript-eslint/no-explicit-any */`;
  const content = `${banner}\nexport const world: any = { id: 'alita-world', witName: 'alita:world' };\n`;
  const target = path.join(outDir, 'alita-world.generated.ts');
  fs.writeFileSync(target, content, 'utf8');
  console.warn('[codegen] using stub world; install jco to generate real bindings.');
}

function writeMeta(meta) {
  fs.writeFileSync(path.join(outDir, '.codegen.meta.json'), JSON.stringify(meta, null, 2));
}

function collectWitInputs() {
  const inputs = new Set();
  if (fs.existsSync(WIT_INPUT)) inputs.add(WIT_INPUT);
  const wasmRoot = path.join(repoRoot, 'wasm');
  const stack = [wasmRoot];
  while (stack.length) {
    const dir = stack.pop();
    if (!dir || !fs.existsSync(dir)) continue;
    for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
      const p = path.join(dir, entry.name);
      if (entry.isDirectory()) stack.push(p);
      else if (entry.isFile() && entry.name.endsWith('.wit')) inputs.add(p);
    }
  }
  return Array.from(inputs);
}

function summarizeGenerated() {
  const files = fs.readdirSync(outDir).filter(f => f.endsWith('.ts'));
  let lines = 0;
  const exports = [];
  for (const f of files) {
    const txt = fs.readFileSync(path.join(outDir, f), 'utf8');
    lines += txt.split(/\r?\n/).length;
    const re = /export\s+const\s+(\w+)/g;
    let m;
    while ((m = re.exec(txt))) exports.push(m[1]);
  }
  return { lines, exports };
}

(function main() {
  const inputs = collectWitInputs();
  if (inputs.length === 0) {
    console.warn('[codegen] No WIT inputs found');
    writeStub();
    writeMeta({ mode: 'stub', toolchain: {}, inputs, metrics: { lines: 1, exports: ['world'] }, ts: new Date().toISOString() });
    return;
  }

  const toolchain = {
    jco: which('jco'),
    wasmTools: which('wasm-tools')
  };

  const hashPath = path.join(outDir, '.codegen.hash');
  const currentHash = inputs.map(sha256File).join('|') + JSON.stringify(toolchain);
  const previousHash = fs.existsSync(hashPath) ? fs.readFileSync(hashPath, 'utf8') : '';
  if (!FORCE && previousHash === currentHash) {
    console.log('[codegen] up-to-date; skipping');
    writeMeta({ mode: 'skip', toolchain, inputs, metrics: summarizeGenerated(), ts: new Date().toISOString() });
    return;
  }

  // Clean output directory except preserved files
  for (const f of fs.readdirSync(outDir)) {
    if (f.startsWith('.codegen')) continue;
    fs.unlinkSync(path.join(outDir, f));
  }

  let ok = false;
  // Optional componentization via wasm-tools when input .wasm + adapter provided
  const COMPONENTIZE = process.env.COMPONENTIZE === '1' || process.env.COMPONENTIZE === 'true';
  if (COMPONENTIZE && toolchain.wasmTools) {
    const inWasm = process.env.COMPONENT_INPUT_WASM;
    const adapter = process.env.ADAPTER_WASM;
    if (inWasm && fs.existsSync(inWasm)) {
      const compDir = path.join(outDir, 'components');
      fs.mkdirSync(compDir, { recursive: true });
      const outComp = path.join(compDir, path.basename(inWasm).replace(/\.wasm$/, '.component.wasm'));
      const args = ['component', 'new', inWasm, '-o', outComp];
      if (adapter) args.push('--adapt', adapter);
      console.log('[codegen] wasm-tools', args.join(' '));
      const okComp = run('wasm-tools', args, { cwd: extRoot });
      if (!okComp) console.warn('[codegen] componentization failed');
      else console.log('[codegen] componentized ->', outComp);
    } else {
      console.log('[codegen] componentization requested but no input wasm configured; skipping');
    }
  }

  if (toolchain.jco) {
    ok = true;
    for (const inp of inputs) {
      console.log('[codegen] jco transpile', path.relative(repoRoot, inp));
      const okOne = run('jco', ['transpile', inp, '--out', outDir], { cwd: extRoot });
      if (!okOne) { ok = false; break; }
    }
  } else {
    console.warn('[codegen] jco not found in PATH');
  }

  if (ok) {
    const adapted = writeAdapterIfNeeded();
    if (!adapted) {
      // If no recognizable export, fall back to stub so worker can import world
      writeStub();
    }
    fs.writeFileSync(hashPath, currentHash, 'utf8');
    const metrics = summarizeGenerated();
    writeMeta({ mode: adapted ? 'generated' : 'stub', toolchain, inputs, metrics, ts: new Date().toISOString() });
    console.log('[codegen] completed', 'mode=', adapted ? 'generated' : 'stub', 'lines=', metrics.lines, 'exports=', metrics.exports.length);
    return;
  }

  // Fallback
  writeStub();
  fs.writeFileSync(hashPath, currentHash, 'utf8');
  writeMeta({ mode: 'stub', toolchain, inputs, metrics: summarizeGenerated(), ts: new Date().toISOString() });
})();
