import { describe, it, expect, vi } from 'vitest';
import { StatusController } from '../status';

describe('StatusController', () => {
  it('toggles host telemetry indicator', () => {
    const setText = vi.fn(); const setTooltip = vi.fn();
    const ctrl = new StatusController({ setText, setTooltip });
    ctrl.setHostTelemetry(true);
    expect(setText).toHaveBeenCalledWith(expect.stringMatching(/Host: ON/));
    ctrl.setHostTelemetry(false);
    expect(setText).toHaveBeenCalledWith(expect.stringMatching(/Host: OFF/));
  });

  it('reflects WASM analyzer readiness', () => {
    const setText = vi.fn(); const setTooltip = vi.fn();
    const ctrl = new StatusController({ setText, setTooltip });
    ctrl.setAnalyzerReady(true);
    expect(setText).toHaveBeenCalledWith(expect.stringMatching(/WASM: Ready/));
  });
});

