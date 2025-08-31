# Mangle Integration Setup Guide

This guide provides detailed steps to set up and use the Mangle integration with Super Alita.

## Prerequisites

1. Python 3.11+ with pip installed
2. Super Alita codebase cloned to your local machine

## Installation Options

### Option 1: Using the Auto-Setup Script (Recommended)

1. **Run the setup script**:

   ```bash
   # Windows
   python start_mangle.py

   # Linux/Mac
   python start_mangle.py
   ```

   This script will:

   - Create a mock Mangle binary for testing
   - Set up the necessary environment variables
   - Create required directories
   - Start the Super Alita server with Mangle integration enabled

2. **Verify the integration**:

   Visit: http://127.0.0.1:8080/tools/catalog

   You should see the Mangle tools listed in the API catalog.

### Option 2: Manual Setup

1. **Create a mock Mangle binary**:

   On Windows:

   ```powershell
   # Create a mock mangle.bat file
   $mockContent = '@echo off
   echo [{"Name": "log4j", "Version": "2.14.0"}, {"Name": "junit", "Version": "4.13.1"}]'
   Set-Content -Path "$env:TEMP\mangle.bat" -Value $mockContent
   ```

   On Linux/Mac:

   ```bash
   # Create a mock mangle script
   echo '#!/bin/sh
   echo '\''[{"Name": "log4j", "Version": "2.14.0"}, {"Name": "junit", "Version": "4.13.1"}]'\''' > /tmp/mangle
   chmod +x /tmp/mangle
   ```

2. **Set environment variables**:

   On Windows:

   ```powershell
   $env:MANGLE_BIN_PATH = "$env:TEMP\mangle.bat"
   $env:ALITA_AUTO_DISCOVER_ABILITIES = "on"
   ```

   On Linux/Mac:

   ```bash
   export MANGLE_BIN_PATH="/tmp/mangle"
   export ALITA_AUTO_DISCOVER_ABILITIES="on"
   ```

3. **Create the data directory**:

   ```bash
   mkdir -p ./data/mangle
   ```

4. **Start the Super Alita server**:

   ```bash
   python -m uvicorn app:app --reload --port 8080
   ```

## Production Setup with Actual Mangle

For a production setup, you should install the actual Mangle binary:

1. **Install Go**:

   Download and install Go from [golang.org](https://golang.org/doc/install)

2. **Install Mangle**:

   ```bash
   go install github.com/google/mangle/cmd/mangle@latest
   ```

3. **Set the path to the actual Mangle binary**:

   On Windows:

   ```powershell
   $env:MANGLE_BIN_PATH = "$HOME\go\bin\mangle.exe"
   ```

   On Linux/Mac:

   ```bash
   export MANGLE_BIN_PATH="$HOME/go/bin/mangle"
   ```

4. **Start the server as usual**:

   ```bash
   python -m uvicorn app:app --reload --port 8080
   ```

## Testing the Integration

### Using the API

You can test the Mangle integration directly through the API:

```python
import requests

# Add a fact
response = requests.post(
    'http://127.0.0.1:8080/ability/execute/mangle_add_fact',
    json={'fact': 'vulnerable("log4j", "2.14.0")'}
)
print(response.json())

# Add a rule
response = requests.post(
    'http://127.0.0.1:8080/ability/execute/mangle_add_rule',
    json={
        'name': 'transitive_deps',
        'rule': 'transitive_depends_on(X, Z) :- depends_on(X, Y), depends_on(Y, Z).'
    }
)
print(response.json())

# Execute a query
response = requests.post(
    'http://127.0.0.1:8080/ability/execute/mangle_query',
    json={'query': 'vulnerable(Name, Version)'}
)
print(response.json())

# Analyze dependencies
response = requests.post(
    'http://127.0.0.1:8080/ability/execute/mangle_analyze_dependencies',
    json={
        'dependencies': [
            {'name': 'log4j', 'version': '2.14.0'},
            {'name': 'spring-core', 'version': '5.3.20'}
        ]
    }
)
print(response.json())
```

### Using the Demo Scripts

Run the demo scripts to see Mangle in action:

```bash
# Run the simple demo
python mangle_simple_demo.py

# Run the full integration demo
python run_mangle_demo.py
```

## Troubleshooting

1. **Mangle tools not showing up in catalog**:

   - Make sure `ALITA_AUTO_DISCOVER_ABILITIES` is set to "on"
   - Verify that `MANGLE_BIN_PATH` points to a valid executable
   - Restart the server and check the logs for any errors

2. **Query execution fails**:

   - Check if the mock Mangle binary is executable
   - Ensure the syntax of your queries follows Mangle's Datalog syntax
   - Look for error messages in the server logs

3. **Slow performance**:
   - Consider reducing the timeout in `MangleAbility` configuration
   - Check if your knowledge base has grown too large
   - Use more specific queries to reduce processing time

## Next Steps

Now that you have Mangle integration working, you can:

1. Build more sophisticated reasoning rules
2. Create custom security policies
3. Integrate with external vulnerability databases
4. Develop knowledge graphs for your specific domain
5. Connect with dependency management systems
