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


## Quick start

1. **Create environment**

   Copy the sample environment file and add at least one API key:

   ```bash
   cp .env.example .env
   ```

2. **Install dependencies and lint**

   Use the provided Make targets to set up and validate the project:

   ```bash
   make deps
   make lint
   ```

3. **Run the development server**

   ```bash
   make run
   ```

For additional runtime details, see [docs/runtime.md](docs/runtime.md).

Debug utilities (`debug_fixed.py`, `debug_matching.py`, and `utility_debug.py`) now live under `scripts/`.

Use these scripts for exploring decision policy behavior and testing policy tweaks.



