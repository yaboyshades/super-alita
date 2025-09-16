# quickstart.md - Constitutional Mastery Architect v5.3

## Prerequisites

- Python 3.11 installed and available on PATH
- `jq` installed and available on PATH
- A virtual environment (recommended)

## Quickstart Steps

1. Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

1. Install development requirements (example):

```powershell
pip install -r requirements.txt
```

1. Run the specify command using the CLI entrypoint (example):

```powershell
python -m src.sdd.cli specify "Constitutional Mastery Architect v5.3"
```

1. Inspect the generated spec:

```powershell
Get-Content specs\020-constitutional-mastery-architect\feature-spec.md -Raw
```

1. Continue to `/plan` and `/tasks` once spec is approved.
