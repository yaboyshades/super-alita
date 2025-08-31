#!/usr/bin/env python3
"""
Wrapper script for validate_deployment.py that handles encoding issues
"""
import os
import subprocess
import sys

# Force ASCII encoding for all child processes
os.environ["PYTHONIOENCODING"] = "ascii:replace"

# Run the validate_deployment.py script
result = subprocess.run(
    [sys.executable, "validate_deployment.py"],
    env=os.environ,
    text=True,
    encoding="ascii",
    errors="replace"
)

# Return the same exit code
sys.exit(result.returncode)
