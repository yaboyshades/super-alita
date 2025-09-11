#!/usr/bin/env python3
"""
Super-Alita Launcher Migration Script
====================================
Migrates legacy launcher scripts to deprecation stubs.
Part of the unified launcher consolidation effort.
"""

import shutil
import sys
from datetime import datetime
from pathlib import Path

# Migration configuration
WORKSPACE_ROOT = Path(".")
UNIFIED_LAUNCHER = "start.py"
MIGRATION_LOG = "launcher_migration.log"

# Legacy launchers to migrate (based on current directory scan)
LEGACY_LAUNCHERS = [
    "complete_agent_demo.py",
    "cortex_development_demo.py",
    "debug_autogen_pipeline.py",
    "debug_consensus_500_error.py",
    "debug_reug_streaming.py",
    "demo_consensus_chat.py",
    "demo_enhanced_consensus.py",
    "demo_github_integration.py",
    "demo_jest_for_python.py",
    "demo_leanrag.py",
    "demo_ollama_working.py",
    "demo_perplexica_integration.py",
    "ladder_cortex_integration_demo.py",
    "ladder_demo_clean.py",
    "ladder_demo.py",
    "live_agent_demo.py",
    "mangle_simple_demo.py",
    "run_mangle_demo.py",
    "test_consensus_debug.py",
    "test_paper2code_debug.py",
]

# Special cases to preserve (keep functional)
PRESERVE_FUNCTIONAL = {
    "app.py": "FastAPI application entry point",
    "mcp_server_entrypoint.py": "MCP server standalone entry",
    "start.py": "Unified launcher (target)",
}


def create_deprecation_stub(
    original_path: Path, reason: str = "Consolidated into unified launcher"
) -> str:
    """Create a deprecation stub for a legacy launcher."""
    stem = original_path.stem.replace("_", "-")
    return f'''#!/usr/bin/env python3
"""
DEPRECATED LAUNCHER STUB
========================
This launcher has been deprecated and replaced by the unified launcher system.

Original file: {original_path.name}
Migration date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Reason: {reason}

Usage Migration:
    OLD: python {original_path.name} [args]
    NEW: python start.py --mode={stem} [args]

For more information, see LAUNCHER_MIGRATION_GUIDE.md
"""

import sys
import subprocess
from pathlib import Path

def main():
    print(f"⚠️  DEPRECATION WARNING: {{Path(__file__).name}} is deprecated")
    print(f"✅ Use instead: python start.py --mode={stem}")
    print(f"📖 See LAUNCHER_MIGRATION_GUIDE.md for details")

    # Attempt automatic migration
    try:
        cmd = ["python", "start.py", "--mode={stem}"] + sys.argv[1:]
        print(f"🔄 Auto-migrating: {{' '.join(cmd)}}")
        subprocess.run(cmd, check=True)
    except Exception as e:
        print(f"❌ Auto-migration failed: {{e}}")
        print(f"💡 Please run manually: python start.py --help")
        sys.exit(1)

if __name__ == "__main__":
    main()
'''


def migrate_launcher(launcher_path: Path) -> bool:
    """Migrate a single launcher to deprecation stub."""
    try:
        # Create backup
        backup_path = launcher_path.with_suffix(
            f".backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        shutil.copy2(launcher_path, backup_path)

        # Check if this should be preserved
        if launcher_path.name in PRESERVE_FUNCTIONAL:
            reason = PRESERVE_FUNCTIONAL[launcher_path.name]
            print(f"🔒 PRESERVING: {launcher_path.name} ({reason})")
            return True

        # Create deprecation stub
        stub_content = create_deprecation_stub(launcher_path)
        launcher_path.write_text(stub_content, encoding="utf-8")

        print(f"✅ MIGRATED: {launcher_path.name} -> {backup_path.name}")
        return True

    except Exception as e:
        print(f"❌ FAILED: {launcher_path.name} - {e}")
        return False


def main():
    """Execute the launcher migration."""
    print("🚀 Starting Super-Alita Launcher Migration")
    print("=" * 50)

    # Verify unified launcher exists
    unified_path = WORKSPACE_ROOT / UNIFIED_LAUNCHER
    if not unified_path.exists():
        print(f"❌ ERROR: Unified launcher not found: {unified_path}")
        return False

    print(f"✅ Unified launcher confirmed: {unified_path}")

    # Migration tracking
    migrated = []
    failed = []
    preserved = []

    # Process each legacy launcher
    for launcher_name in LEGACY_LAUNCHERS:
        launcher_path = WORKSPACE_ROOT / launcher_name

        if not launcher_path.exists():
            print(f"⏭️  SKIP: {launcher_name} (not found)")
            continue

        if launcher_name in PRESERVE_FUNCTIONAL:
            preserved.append(launcher_name)
            print(f"🔒 PRESERVE: {launcher_name}")
            continue

        if migrate_launcher(launcher_path):
            migrated.append(launcher_name)
        else:
            failed.append(launcher_name)

    # Migration summary
    print("\n" + "=" * 50)
    print("📊 MIGRATION SUMMARY")
    print("=" * 50)
    print(f"✅ Migrated: {len(migrated)} launchers")
    print(f"🔒 Preserved: {len(preserved)} launchers")
    print(f"❌ Failed: {len(failed)} launchers")

    if migrated:
        print("\n📦 Migrated Launchers:")
        for name in migrated:
            print(f"  • {name}")

    if preserved:
        print("\n🔒 Preserved Launchers:")
        for name in preserved:
            print(f"  • {name} ({PRESERVE_FUNCTIONAL.get(name, 'N/A')})")

    if failed:
        print("\n❌ Failed Migrations:")
        for name in failed:
            print(f"  • {name}")

    # Create migration log
    log_content = f"""Super-Alita Launcher Migration Log
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

Migrated ({len(migrated)}):
{chr(10).join(f"  • {name}" for name in migrated)}

Preserved ({len(preserved)}):
{chr(10).join(f"  • {name} - {PRESERVE_FUNCTIONAL.get(name, 'N/A')}" for name in preserved)}

Failed ({len(failed)}):
{chr(10).join(f"  • {name}" for name in failed)}
"""

    (WORKSPACE_ROOT / MIGRATION_LOG).write_text(log_content, encoding="utf-8")
    print(f"\n📝 Migration log saved: {MIGRATION_LOG}")

    # Final status
    success = len(failed) == 0
    if success:
        print("\n🎉 MIGRATION COMPLETED SUCCESSFULLY!")
        print("💡 Next steps:")
        print("  1. Test unified launcher: python start.py --help")
        print("  2. Update CI/CD configurations")
        print("  3.
              backup files and clean up when satisfied")
    else:
        print(f"\n⚠️  MIGRATION COMPLETED WITH {len(failed)} ERRORS")
        print("💡 Please review failed migrations and fix manually")

    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
