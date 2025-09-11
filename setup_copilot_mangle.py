#!/usr/bin/env python3
"""
Auto-initialization script for GitHub Copilot Mangle Integration.    required_modules = [
        "src.sdd.mangle_reasoner",
        "src.sdd.enhanced_sdd_framework",
        "src.constitutional.scorer",
        "src.abilities.mangle_reasoning_ability"
    ]s script automatically enables Mangle reasoning for all GitHub Copilot
interactions by setting up the necessary hooks and middleware.
"""

import os
import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))


def setup_copilot_mangle_integration():
    """Set up native Mangle integration for GitHub Copilot."""
    print("🚀 Setting up GitHub Copilot Mangle Integration...")

    # Set environment variables
    os.environ["COPILOT_MANGLE_MODE"] = "true"
    os.environ["COPILOT_AUTO_ENHANCE"] = "true"
    os.environ["SDD_CONSTITUTIONAL_MODE"] = "true"

    # Import and initialize the middleware
    try:
        from src.copilot.mangle_middleware import enhance_copilot_with_mangle

        success = enhance_copilot_with_mangle()

        if success:
            print("✅ GitHub Copilot Mangle Integration activated!")
            print("   Now every Copilot interaction includes:")
            print("   • Automatic code knowledge graph analysis")
            print("   • Constitutional compliance checking")
            print("   • Natural language code querying")
            print("   • Specification-to-code traceability")
            print("   • Quality analysis and recommendations")

            print("\n🎯 Try asking GitHub Copilot:")
            print("   • 'What functions are untested?'")
            print("   • 'What violates the constitution?'")
            print("   • 'Show me quality issues'")
            print("   • 'Trace this function to its specification'")

            return True
        else:
            print("⚠️  Setup completed but some features may be limited")
            return False

    except ImportError as e:
        print(f"❌ Setup failed: {e}")
        print("   Make sure all dependencies are installed")
        return False


def install_vs_code_extension():
    """Install the VS Code extension for enhanced integration."""
    extension_path = Path(__file__).parent / "extensions" / "copilot-mangle"

    if extension_path.exists():
        print(f"\n📦 VS Code extension available at: {extension_path}")
        print("   To install:")
        print("   1. Open VS Code")
        print("   2. Go to Extensions (Ctrl+Shift+X)")
        print(f"   3. Install from VSIX: {extension_path}")
        print("   4. Enable 'GitHub Copilot Mangle Integration'")

        # Try to auto-install if possible
        try:
            import subprocess

            result = subprocess.run(
                ["code", "--install-extension", str(extension_path)],
                capture_output=True,
                text=True,
            )

            if result.returncode == 0:
                print("✅ VS Code extension installed automatically!")
            else:
                print("ℹ️  Manual installation required")

        except Exception:
            print("ℹ️  Manual installation required")
    else:
        print("⚠️  VS Code extension not found")


def check_dependencies():
    """Check that all required dependencies are available."""
    print("🔍 Checking dependencies...")

    required_modules = [
        "sdd.mangle_reasoner",
        "sdd.enhanced_sdd_framework",
        "constitutional.scorer",
        "abilities.mangle_reasoning_ability",
    ]

    missing = []
    for module in required_modules:
        try:
            __import__(module)
            print(f"   ✅ {module}")
        except ImportError as e:
            print(f"   ❌ {module} - {e}")
            missing.append(module)

    if missing:
        print(f"\n⚠️  Missing dependencies: {', '.join(missing)}")
        print("   Run the main installation first")
        return False

    print("✅ All dependencies available")
    return True


def main():
    """Main setup function."""
    print("=" * 60)
    print("🧠 GITHUB COPILOT MANGLE INTEGRATION SETUP")
    print("=" * 60)

    # Check dependencies
    if not check_dependencies():
        sys.exit(1)

    # Set up the integration
    success = setup_copilot_mangle_integration()

    # Install VS Code extension
    install_vs_code_extension()

    print("\n" + "=" * 60)
    if success:
        print(
            "🎉 Setup complete! GitHub Copilot is now enhanced with Mangle reasoning."
        )
        print(
            "   Every interaction will include automatic code knowledge graph analysis."
        )
        print(
            "   Just start using GitHub Copilot normally - the enhancement is automatic!"
        )
    else:
        print("⚠️  Setup completed with some limitations.")
        print(
            "   Basic functionality should work, but some features may be unavailable."
        )

    print("=" * 60)


if __name__ == "__main__":
    main()
