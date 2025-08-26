export interface DeepCodeFinding {
  rule: string;
  message: string;
  accepted?: boolean;
  falsePositive?: boolean;
}

export async function runDeepCodeAnalysis(opts: {
  path: string;
  client: { analyze: (path: string) => Promise<DeepCodeFinding[]> };
  emit: (name: string, payload?: Record<string, unknown>) => void;
}) {
  const { path, client, emit } = opts;
  const findings = await client.analyze(path);
  const total = findings.length || 1;
  const accepted = findings.filter(f => f.accepted).length;
  const falsePos = findings.filter(f => f.falsePositive).length;
  const metrics = {
    acceptanceRate: accepted / total,
    falsePositiveRate: falsePos / total,
  };
  emit('deepcode_analysis', { count: findings.length, path });
  return { findings, metrics };
}

