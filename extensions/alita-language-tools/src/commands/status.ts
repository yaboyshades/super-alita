type SetText = (t: string) => void;
type SetTooltip = (t: string) => void;

export class StatusController {
  constructor(private ui: { setText: SetText; setTooltip: SetTooltip }) {}

  setHostTelemetry(on: boolean) {
    this.ui.setText(`Alita Status — Host: ${on ? 'ON' : 'OFF'}`);
    this.ui.setTooltip(`Host telemetry ${on ? 'enabled' : 'disabled'}`);
  }

  setAnalyzerReady(ready: boolean) {
    this.ui.setText(`Alita Status — WASM: ${ready ? 'Ready' : 'Not Ready'}`);
    this.ui.setTooltip(`WASM analyzer is ${ready ? 'ready' : 'not ready'}`);
  }
}
