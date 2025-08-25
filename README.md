# Super Alita

A production-ready autonomous agent system with streaming orchestration and rich tooling.

## Quick Start

1. Create an environment file:
   ```bash
   cp .env.example .env  # then add one provider API key
   ```
2. Install dependencies:
   ```bash
   make deps
   ```
3. Run the development server:
   ```bash
   make run
   ```
4. Run the runtime test suite:
   ```bash
   make test
   ```

See [docs/runtime.md](docs/runtime.md) for detailed configuration and Docker instructions.
