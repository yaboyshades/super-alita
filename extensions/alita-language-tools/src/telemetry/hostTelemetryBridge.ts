export interface HostTelemetryEvent {
  host_fn: string;
  duration_ms?: number;
  ok: boolean;
  error_str?: string | null;
  timestamp?: number;
  workspace_hash?: string;
}

type EmitFn = (name: string, payload?: Record<string, unknown>) => void;

export class HostTelemetryBridge {
  private enabled: boolean;
  private sampleRate: number;
  private sanitize: (s: unknown) => string | unknown;
  private emit: EmitFn;

  constructor(opts: { enabled: boolean; sampleRate: number; sanitize: (s: unknown) => string | unknown; emit: EmitFn }) {
    this.enabled = opts.enabled;
    this.sampleRate = opts.sampleRate;
    this.sanitize = opts.sanitize;
    this.emit = opts.emit;
  }

  setEnabled(v: boolean) { this.enabled = v; }
  setSampleRate(v: number) { this.sampleRate = v; }

  onWorkerConsole(line: string) {
    if (!this.enabled) return;
    if (this.sampleRate <= 0) return;
    try {
      const evt = JSON.parse(line) as HostTelemetryEvent;
      if (!evt || typeof evt !== 'object') return;
      if (!('host_fn' in evt) || !('ok' in evt)) return;
      if (!this._sample()) return;
      const payload: Record<string, unknown> = {
        host_fn: evt.host_fn,
        ok: !!evt.ok,
      };
      if (typeof evt.duration_ms === 'number') payload.duration_ms = evt.duration_ms;
      payload.error_str = evt.error_str == null ? null : (this.sanitize(evt.error_str) as string);
      if (typeof evt.timestamp === 'number') payload.timestamp = evt.timestamp;
      if (evt.workspace_hash) payload.workspace_hash = evt.workspace_hash;
      this.emit('host_telemetry_event', payload);
    } catch {
      // ignore malformed lines
    }
  }

  private _sample(): boolean {
    if (this.sampleRate >= 1) return true;
    return Math.random() < this.sampleRate;
  }
}
