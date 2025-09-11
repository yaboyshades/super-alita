export interface LessonDigest {
  id: string;
  timestamp: string; // ISO string
  pattern_type: string;
  trigger_conditions: string[];
  solution_template: string;
  success_indicators: string[];
  anti_patterns: string[];
  notes?: string;
}

export interface PerformanceMetrics {
  success_rate: number; // 0..1
  avg_completion_time: number; // ms
  constitutional_compliance: number; // 0..1
}

export interface KnowledgePattern {
  name: string;
  version: string;
  triggers: string[];
  context_cues: string[];
  automation_script: string;
  performance_metrics: PerformanceMetrics;
  lessons_learned: LessonDigest[];
}

export interface KnowledgeLedgerEntry {
  kind: 'knowledge_decision' | 'knowledge_metrics' | 'knowledge_pattern';
  timestamp: string; // ISO string
  data: unknown;
}

