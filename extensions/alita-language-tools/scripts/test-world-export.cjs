#!/usr/bin/env node
/* Simple assertion that generated world export exists */
const path = require('path');
const fs = require('fs');

const genTs = path.join(__dirname, '..', 'src', 'generated', 'alita-world.generated.ts');
if (!fs.existsSync(genTs)) {
  console.error('[test:codegen] missing generated TS file:', genTs);
  process.exit(1);
}

// We rely on TS compile to JS under out/ before this is meaningful; still check source symbol textually.
const txt = fs.readFileSync(genTs, 'utf8');
if (!/export\s+const\s+world\b/.test(txt)) {
  console.error('[test:codegen] no world export found in generated file');
  process.exit(1);
}
console.log('[test:codegen] world export present');
