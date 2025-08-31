#!/usr/bin/env python3
"""
Create Desktop Shortcut for Super Alita

This script creates a desktop shortcut for easy one-click access to Super Alita.
"""

import sys
from pathlib import Path


def create_windows_shortcut():
    """Create a Windows desktop shortcut."""
    try:
        import winshell
        from win32com.client import Dispatch
    except ImportError:
        print("❌ Windows shortcut creation requires 'pywin32' and 'winshell'")
        print("Install with: pip install pywin32 winshell")
        return False
    
    desktop = winshell.desktop()
    shortcut_path = Path(desktop) / "Super Alita.lnk"
    
    target = sys.executable
    arguments = str(Path.cwd() / "start_super_alita.py")
    working_dir = str(Path.cwd())
    icon_path = str(Path.cwd() / "static" / "favicon.ico")
    
    # Create shortcut
    shell = Dispatch('WScript.Shell')
    shortcut = shell.CreateShortCut(str(shortcut_path))
    shortcut.Targetpath = target
    shortcut.Arguments = f'"{arguments}"'
    shortcut.WorkingDirectory = working_dir
    shortcut.Description = "Super Alita AI Agent - One-click startup"
    
    # Use favicon if available, otherwise use Python icon
    if Path(icon_path).exists():
        shortcut.IconLocation = icon_path
    
    shortcut.save()
    
    print(f"✅ Windows shortcut created: {shortcut_path}")
    return True

def create_linux_desktop_file():
    """Create a Linux desktop file."""
    desktop_dir = Path.home() / "Desktop"
    applications_dir = Path.home() / ".local" / "share" / "applications"
    
    # Ensure directories exist
    desktop_dir.mkdir(exist_ok=True)
    applications_dir.mkdir(parents=True, exist_ok=True)
    
    desktop_content = f"""[Desktop Entry]
Name=Super Alita
Comment=Super Alita AI Agent - One-click startup
Exec={sys.executable} "{Path.cwd() / 'start_super_alita.py'}"
Icon={Path.cwd() / 'static' / 'favicon.ico'}
Terminal=true
Type=Application
Categories=Development;AI;
StartupWMClass=Super Alita
"""
    
    # Create desktop file
    desktop_file = desktop_dir / "super-alita.desktop"
    with open(desktop_file, 'w') as f:
        f.write(desktop_content)
    
    # Make executable
    desktop_file.chmod(0o755)
    
    # Also create in applications directory
    app_file = applications_dir / "super-alita.desktop"
    with open(app_file, 'w') as f:
        f.write(desktop_content)
    
    app_file.chmod(0o755)
    
    print(f"✅ Linux desktop file created: {desktop_file}")
    print(f"✅ Application entry created: {app_file}")
    return True

def create_macos_app():
    """Create a macOS app bundle."""
    app_dir = Path.cwd() / "Super Alita.app"
    contents_dir = app_dir / "Contents"
    macos_dir = contents_dir / "MacOS"
    resources_dir = contents_dir / "Resources"
    
    # Create directory structure
    macos_dir.mkdir(parents=True, exist_ok=True)
    resources_dir.mkdir(parents=True, exist_ok=True)
    
    # Create Info.plist
    info_plist = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleDisplayName</key>
    <string>Super Alita</string>
    <key>CFBundleExecutable</key>
    <string>super_alita</string>
    <key>CFBundleIdentifier</key>
    <string>com.superalita.agent</string>
    <key>CFBundleName</key>
    <string>Super Alita</string>
    <key>CFBundleVersion</key>
    <string>1.0</string>
    <key>CFBundleShortVersionString</key>
    <string>1.0</string>
</dict>
</plist>"""
    
    with open(contents_dir / "Info.plist", 'w') as f:
        f.write(info_plist)
    
    # Create executable script
    executable_script = f"""#!/bin/bash
cd "{Path.cwd()}"
{sys.executable} start_super_alita.py
"""
    
    executable_path = macos_dir / "super_alita"
    with open(executable_path, 'w') as f:
        f.write(executable_script)
    
    executable_path.chmod(0o755)
    
    print(f"✅ macOS app bundle created: {app_dir}")
    return True

def main():
    """Main entry point."""
    print("🚀 Creating Super Alita Desktop Shortcut...")
    print()
    
    # Check if startup script exists
    startup_script = Path.cwd() / "start_super_alita.py"
    if not startup_script.exists():
        print("❌ start_super_alita.py not found in current directory")
        print("Please run this script from the Super Alita root directory")
        return 1
    
    system = sys.platform.lower()
    success = False
    
    if system.startswith('win'):
        print("🪟 Detected Windows, creating .lnk shortcut...")
        success = create_windows_shortcut()
    elif system.startswith('linux'):
        print("🐧 Detected Linux, creating .desktop file...")
        success = create_linux_desktop_file()
    elif system.startswith('darwin'):
        print("🍎 Detected macOS, creating .app bundle...")
        success = create_macos_app()
    else:
        print(f"❌ Unsupported platform: {system}")
        return 1
    
    if success:
        print()
        print("✅ Desktop shortcut created successfully!")
        print("You can now double-click the shortcut to start Super Alita")
        print()
        print("📝 What the shortcut does:")
        print("   • Starts the Super Alita server")
        print("   • Starts the MCP server")
        print("   • Opens the chat interface in your browser")
        print("   • Displays service status")
    else:
        print("❌ Failed to create desktop shortcut")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())