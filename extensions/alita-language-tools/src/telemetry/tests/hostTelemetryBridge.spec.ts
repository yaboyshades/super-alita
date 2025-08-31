import { describe, it, expect, vi, beforeEach } from 'vitest';
import { HostTelemetryBridge } from '../hostTelemetryBridge';

function mkLine(payload: any) { return JSON.stringify(payload) + '\n'; }

describe('HostTelemetryBridge', () => {
  let emit: any; let bridge: HostTelemetryBridge;
  beforeEach(() => {
    emit = vi.fn();
    bridge = new HostTelemetryBridge({ enabled: true, sampleRate: 1.0, sanitize: (s) => (s ? String(s).slice(0, 256) : s), emit });
  });

  it('forwards valid JSON events with required fields', async () => {
    bridge.onWorkerConsole(mkLine({ host_fn: 'fs.readFile', duration_ms: 12.4, ok: true, error_str: null, timestamp: 1700000000000, workspace_hash: 'abc' }));
    expect(emit).toHaveBeenCalledTimes(1);
    expect(emit.mock.calls[0][0]).toBe('host_telemetry_event');
    expect(emit.mock.calls[0][1]).toMatchObject({ host_fn: 'fs.readFile', duration_ms: 12.4, ok: true, error_str: null });
  });

  it('drops when disabled', async () => {
    bridge.setEnabled(false);
    bridge.onWorkerConsole(mkLine({ host_fn: 'net.request', ok: true }));
    expect(emit).not.toHaveBeenCalled();
  });

  it('applies sampling', async () => {
    bridge.setSampleRate(0);
    bridge.onWorkerConsole(mkLine({ host_fn: 'x', ok: true }));
    expect(emit).not.toHaveBeenCalled();
  });

  it('sanitizes error strings', async () => {
    bridge.onWorkerConsole(mkLine({ host_fn: 'x', ok: false, error_str: 'A'.repeat(999) }));
    const evt = emit.mock.calls[0][1];
    expect(evt.error_str.length).toBeLessThanOrEqual(256);
  });

  it('ignores malformed JSON', async () => {
    bridge.onWorkerConsole('not-json\n');
    expect(emit).not.toHaveBeenCalled();
  });
});
