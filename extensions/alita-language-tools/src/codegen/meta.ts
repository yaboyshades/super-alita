import { execSync } from 'child_process';

export interface ComponentMeta { path: string; sizeBytes: number | null; sections: string[] }
export interface CodegenMeta { components: ComponentMeta[]; servicesUsed: string[]; warnings: string[] }

export function __execShell(cmd: string): Buffer {
  return execSync(cmd, { stdio: ['ignore', 'pipe', 'pipe'] });
}

export async function enrichCodegenMeta(
  input: { components: { path: string }[]; servicesUsed: string[] },
  execFn: (cmd: string) => Buffer = __execShell,
): Promise<CodegenMeta> {
  const warnings: string[] = [];
  const components: ComponentMeta[] = [];
  for (const c of input.components) {
    let sizeBytes: number | null = null;
    const sections: string[] = [];
    try {
      const out = execFn(`wasm-tools objdump ${c.path}`).toString('utf8');
      const sizeMatch = out.match(/Size:\s*(\d+)/i);
      if (sizeMatch) sizeBytes = parseInt(sizeMatch[1], 10);
      const secLines = out.split(/\r?\n/).filter(l => /Sections:/i.test(l) || /^\s*-\s+/.test(l));
      for (const l of secLines) {
        const m = l.match(/-\s+([A-Za-z]+)/);
        if (m) sections.push(m[1]);
      }
    } catch (e: any) {
      warnings.push(`wasm-tools failed for ${c.path}: ${String(e.message || e)}`);
    }
    components.push({ path: c.path, sizeBytes, sections });
  }
  return { components, servicesUsed: input.servicesUsed, warnings };
}
