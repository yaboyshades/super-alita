# Alita Language Extension

Provides basic language support for the Alita language, including:

- Syntax highlighting via TextMate grammar.
- Snippets for common patterns.
- Language Server Protocol features through a bundled server.
- Semantic tokens, task provider, and debug adapter wiring.
- MCP search and a skillset command with telemetry.

## Telemetry

The extension records anonymous activation events to help improve the
extension. See `telemetry.json` for details and disable collection via
VS Code settings if desired.

## WASM Worker & WIT Bindings

The extension includes a lightweight Web Worker scaffold at `src/worker.ts` to
host components via `@vscode/wasm-component-model`. It currently uses a
placeholder world shape. Once a wit -> TypeScript codegen pipeline is in place
(e.g., using jco/wasm-tools or componentize-js), replace the placeholder with
generated bindings, for example:

- Import the generated world: `import { world } from './bindings/alita-world.generated'`.
- Create the connection: `await Connection.createWorker(world)`.

Codegen: Use `npm run codegen:wit` to generate TypeScript bindings.

- Default input: `wasm/code_radar/radar.wit` (override with env `WIT_INPUT=...`).
- Tooling: Prefers local `jco` (`jco transpile <wit> --out src/generated`).
- Optional: Set `COMPONENTIZE=1` to integrate a `wasm-tools` component step if configured.
- Skipping unchanged: The script writes `src/generated/.codegen.hash` and skips regeneration when inputs are unchanged (override with `FORCE_REGEN=1`).
- Symbol adapter: If the generator does not export `world`, an adapter `alita-world.generated.ts` is created to re-export the first generated const as `world`.

Worker usage: `src/worker.ts` imports `world` from `./generated/alita-world.generated` and connects via `Connection.createWorker`. If your bindings require host imports, populate the typed `hostServices` in `worker.ts` and they’ll be passed to the worker factory.

Status indicator: Enable `alita.codegen.showBindingStatus` to show a status bar item (stub vs generated). Command palette: `Alita: Show WIT Binding Status` displays current mode and metrics.

CI check: The GitHub workflow runs `npm run codegen:wit && npm run test:codegen` in this extension. To require real bindings in CI, set env `REQUIRE_REAL_WIT=1`; the check fails when in stub mode.

Multi-world registry: The codegen step aggregates all discovered `*.wit` files in `wasm/**` and emits `src/generated/registry.ts` that re-exports each generated module under a registry map. Use this to reference multiple components as needed.

WASM componentization: When `COMPONENTIZE=1` and `wasm-tools` is available, and you provide `COMPONENT_INPUT_WASM=/path/to/module.wasm` (and optionally `ADAPTER_WASM=/path/to/adapter.wasm`), the codegen step runs `wasm-tools component new` to produce a componentized `.wasm` under `src/generated/components/`. Integrate this artifact with your runtime if needed.

Sample componentization:
- Build the sample module: `npm run --prefix extensions/alita-language-tools codegen:componentize:sample`
- This compiles `sample-wasm/add.wat` → `sample-wasm/add.wasm`, then componentizes it to `src/generated/components/add.component.wasm` (if `wasm-tools` is installed).
- Validate artifact: `npm run --prefix extensions/alita-language-tools test:component` (runs `wasm-tools validate` when available).

Experimental analyze prefetch: Enable `alita.predictive.wasmAnalysisEnabled` to have the extension invoke a generated `analyze` export (if present) with a sample snippet and record telemetry (`predictive/wasm/analyze`). Diagnostics produced here are currently placeholders for future predictive refactor seeding.

Host-call telemetry: Toggle `alita.codegen.hostTelemetry` to collect per-host function latency metrics emitted from the worker via a `postMessage` bridge. Events are reported as `alita/hostCall` (name, duration, ok flag).

Linting: ESLint is configured to error on explicit `any` in `src/worker.ts`. Replace placeholders with generated types to satisfy the rule.

### Code Generation Stub

A minimal stub generator (`scripts/generate-wit-stubs.cjs`) emits
`src/generated/alita-world.generated.ts` with a placeholder `world` object.
It runs automatically as part of `npm run compile` (via `npm run codegen:wit`).
Replace this script with real WIT tooling when ready.

### Toolchain Installation

Install optional toolchain utilities globally (or add to devDependencies):

```bash
npm install -g @bytecodealliance/jco
# (optional) wasm-tools via rustup
rustup component add rust-src
cargo install wasm-tools
```

### Development Watch Mode

Use continuous generation + type checking:

```bash
npm run watch:dev
```

### Codegen Telemetry

On activation the extension emits a `alita/codegen/meta` telemetry event with:

- mode: `generated | stub | skip`
- jco: whether `jco` was detected
- wasmTools: whether `wasm-tools` was detected

This helps track adoption of real bindings versus the fallback stub.

### Testing the Generated Export

Run the lightweight test to assert the `world` export exists:

```bash
npm run codegen:wit
npm run test:codegen
```

CI can enforce real bindings by setting `REQUIRE_REAL_WIT=1` (future enhancement) and failing if `mode === 'stub'`.
