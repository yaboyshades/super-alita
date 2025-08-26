import { Connection, RAL } from '@vscode/wasm-component-model';
// Placeholder import path; actual generated bindings would live under ./wasm/calculator after build tooling integration.
// import { calculator } from './wasm/calculator';

async function main(): Promise<void> {
  // In a full implementation we'd pass the generated namespace root (e.g., calculator._)
  // For now this is a scaffold; integration happens once wit->ts generation is added.
  const connection = await Connection.createWorker<any>({});
  connection.listen();
}

main().catch(err => {
  RAL().console.error("Worker initialization failed", err);
});
