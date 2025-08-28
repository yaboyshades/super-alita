#!/usr/bin/env python3
"""
Demo script for the Automatic Merge Conflict Resolution workflow.

This script demonstrates how the merge conflict resolution system works
by creating a mock conflict scenario and showing the resolution strategies.
"""

import os
import sys

# Add the project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import builtins
import contextlib

from cortex.automation.git_workflow import GitAutomation


def create_demo_conflict_file(file_path: str, conflict_type: str = "simple"):
    """Create a demo file with different types of conflicts."""

    if conflict_type == "simple":
        content = """# Demo file with simple conflict
import os
import sys
<<<<<<< HEAD
# Current branch: added logging
import logging
logging.basicConfig(level=logging.INFO)
=======
# Incoming branch: added json
import json
print("Starting application...")
>>>>>>> feature-branch

def main():
    print("Hello, World!")
"""

    elif conflict_type == "imports":
        content = """# Demo file with import conflicts
<<<<<<< HEAD
import os
import sys
import logging
from pathlib import Path
=======
import os
import json
import requests
from datetime import datetime
>>>>>>> feature-branch

def main():
    pass
"""

    elif conflict_type == "additive":
        content = """# Demo file with additive conflicts
def existing_function():
    return "existing"

<<<<<<< HEAD
def new_function_current():
    return "from current branch"
=======
def new_function_incoming():
    return "from incoming branch"
>>>>>>> feature-branch

def another_existing_function():
    return "also existing"
"""

    elif conflict_type == "complex":
        content = """# Demo file with complex conflicts
class DataProcessor:
<<<<<<< HEAD
    def __init__(self, config_file: str):
        self.config = self.load_config(config_file)
        self.logger = logging.getLogger(__name__)
        
    def process(self, data: list) -> dict:
        self.logger.info(f"Processing {len(data)} items")
        return {"processed": len(data), "status": "success"}
=======
    def __init__(self, database_url: str):
        self.db = self.connect_database(database_url)
        self.cache = {}
        
    def process(self, data: list) -> dict:
        result = {"items": []}
        for item in data:
            processed = self.transform_item(item)
            result["items"].append(processed)
        return result
>>>>>>> feature-branch
"""

    with open(file_path, "w") as f:
        f.write(content)


def demo_conflict_resolution():
    """Demonstrate the conflict resolution capabilities."""
    print("🚀 Super Alita Merge Conflict Resolution Demo")
    print("=" * 50)

    git_auto = GitAutomation()

    # Create demo files with different conflict types
    demo_files = {
        "simple_conflict.py": "simple",
        "import_conflict.py": "imports",
        "additive_conflict.py": "additive",
        "complex_conflict.py": "complex",
    }

    for file_name, conflict_type in demo_files.items():
        print(f"\n📄 Analyzing {file_name} ({conflict_type} conflict):")
        print("-" * 40)

        # Create the demo file
        create_demo_conflict_file(file_name, conflict_type)

        # Analyze the conflict
        conflict_info = git_auto._analyze_conflict_file(file_name)

        if conflict_info:
            print(f"✅ Found {len(conflict_info.conflict_sections)} conflict(s)")

            for i, section in enumerate(conflict_info.conflict_sections):
                print(f"\n🔍 Conflict {i+1}:")
                print(
                    f"   Current branch: {len(section['current_branch'].split())} words"
                )
                print(
                    f"   Incoming branch: {len(section['incoming_branch'].split())} words"
                )

                # Test resolution strategies
                print("\n🛠️ Resolution strategies:")

                # Check if it's an import section
                if git_auto._is_import_section(
                    section["current_branch"], section["incoming_branch"]
                ):
                    print("   📦 Import conflict detected - will merge imports")
                    merged = git_auto._merge_imports(
                        section["current_branch"], section["incoming_branch"]
                    )
                    print(f"   ✅ Merged result: {len(merged.split())} unique imports")

                # Check if it's additive
                elif git_auto._is_additive_change(
                    section["current_branch"], section["incoming_branch"]
                ):
                    print("   ➕ Additive conflict detected - will combine both sides")

                # Check for empty sides
                elif not section["current_branch"].strip():
                    print("   ⬅️ Current branch empty - will take incoming changes")
                elif not section["incoming_branch"].strip():
                    print("   ➡️ Incoming branch empty - will take current changes")
                else:
                    print("   ⚠️ Complex conflict - requires manual review")
        else:
            print("❌ No conflicts found in file")

        # Clean up demo file
        with contextlib.suppress(builtins.BaseException):
            os.remove(file_name)

    print("\n🎯 Workflow Integration:")
    print("-" * 40)
    print("In GitHub, this workflow will:")
    print("1. 🔍 Automatically detect conflicts in PRs")
    print("2. 🤖 Attempt intelligent resolution using multiple strategies")
    print("3. 📝 Create a new PR with resolved conflicts")
    print("4. 💬 Comment on the original PR with results")
    print("5. 🎛️ Accept commands via comments for manual triggering")

    print("\n📋 Available Commands:")
    print("-" * 40)
    print("• @github-actions resolve conflicts auto      # Smart resolution")
    print("• @github-actions resolve conflicts current   # Take current branch")
    print("• @github-actions resolve conflicts incoming  # Take incoming branch")

    print("\n✨ Features:")
    print("-" * 40)
    print("• 🧠 Intelligent conflict analysis")
    print("• 📦 Smart import merging")
    print("• ➕ Additive change combination")
    print("• 🔄 Multiple resolution strategies")
    print("• 🤖 Fully automated workflow")
    print("• 💬 Comment-based manual triggering")
    print("• 📊 Comprehensive conflict reporting")


if __name__ == "__main__":
    demo_conflict_resolution()
