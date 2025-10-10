#!/usr/bin/env python3
"""
DEPRECATED LAUNCHER STUB
========================
This launcher has been deprecated and replaced by the unified launcher system.

Original file: test_consensus_debug.py
Migration date: 2025-09-06 06:54:06
Reason: Consolidated into unified launcher

Usage Migration:
    OLD: python test_consensus_debug.py [args]
    NEW: python start.py --mode=test-consensus-debug [args]

For more information, see LAUNCHER_MIGRATION_GUIDE.md
"""

import subprocess
import sys
from pathlib import Path


def main():
    print(f"DEPRECATION WARNING: {Path(__file__).name} is deprecated")
    print("Use instead: python start.py --mode=test-consensus-debug")
    print("📖 See LAUNCHER_MIGRATION_GUIDE.md for details")

    # Attempt automatic migration
    try:
        cmd = ["python", "start.py", "--mode=test-consensus-debug"] + sys.argv[
            1:
        ]
        print(f"🔄 Auto-migrating: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
    except Exception as e:
        print(f"❌ Auto-migration failed: {e}")
        print("💡 Please run manually: python start.py --help")
        sys.exit(1)


if __name__ == "__main__":
    main()
