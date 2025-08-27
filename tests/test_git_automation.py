"""
Test the git automation merge conflict resolution functionality.
"""
import unittest
import tempfile
import os
import shutil
from pathlib import Path
import subprocess
import sys

# Add the project root to the path so we can import the module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cortex.automation.git_workflow import GitAutomation, MergeConflictInfo


class TestGitAutomation(unittest.TestCase):
    def setUp(self):
        """Set up a temporary git repository for testing."""
        self.test_dir = tempfile.mkdtemp()
        self.original_dir = os.getcwd()
        os.chdir(self.test_dir)
        
        # Initialize git repo
        subprocess.run(["git", "init"], capture_output=True)
        subprocess.run(["git", "config", "user.name", "Test User"], capture_output=True)
        subprocess.run(["git", "config", "user.email", "test@example.com"], capture_output=True)
        
        self.git_auto = GitAutomation()

    def tearDown(self):
        """Clean up the temporary directory."""
        os.chdir(self.original_dir)
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_detect_conflicts_no_merge(self):
        """Test conflict detection when not in a merge state."""
        result = self.git_auto.detect_merge_conflicts()
        self.assertFalse(result["has_conflicts"])
        self.assertEqual(result["conflicts"], [])

    def test_merge_conflict_info(self):
        """Test MergeConflictInfo class."""
        conflicts = [{"test": "data"}]
        info = MergeConflictInfo("test.py", conflicts)
        self.assertEqual(info.file_path, "test.py")
        self.assertEqual(info.conflict_sections, conflicts)

    def test_analyze_conflict_file(self):
        """Test analyzing a file with conflict markers."""
        # Create a test file with conflict markers
        test_content = """some code
<<<<<<< HEAD
current branch code
=======
incoming branch code
>>>>>>> branch
more code"""
        
        test_file = Path("test_conflict.py")
        test_file.write_text(test_content)
        
        result = self.git_auto._analyze_conflict_file("test_conflict.py")
        self.assertIsNotNone(result)
        self.assertEqual(result.file_path, "test_conflict.py")
        self.assertEqual(len(result.conflict_sections), 1)
        self.assertEqual(result.conflict_sections[0]["current_branch"], "current branch code")
        self.assertEqual(result.conflict_sections[0]["incoming_branch"], "incoming branch code")

    def test_analyze_conflict_file_no_conflicts(self):
        """Test analyzing a file without conflict markers."""
        test_content = "normal code without conflicts"
        test_file = Path("normal.py")
        test_file.write_text(test_content)
        
        result = self.git_auto._analyze_conflict_file("normal.py")
        self.assertIsNone(result)

    def test_is_import_section(self):
        """Test import section detection."""
        self.assertTrue(self.git_auto._is_import_section("import os", "import sys"))
        self.assertTrue(self.git_auto._is_import_section("from pathlib import Path", "import json"))
        self.assertFalse(self.git_auto._is_import_section("def function():", "return value"))

    def test_is_additive_change(self):
        """Test additive change detection."""
        current = "line1\nline2"
        incoming = "line3\nline4"
        self.assertTrue(self.git_auto._is_additive_change(current, incoming))
        
        # Overlapping changes should not be additive
        current = "line1\nline2"
        incoming = "line1\nline3"
        self.assertFalse(self.git_auto._is_additive_change(current, incoming))

    def test_merge_imports(self):
        """Test import merging."""
        current = "import os\nimport sys"
        incoming = "import json\nimport pathlib"
        result = self.git_auto._merge_imports(current, incoming)
        
        # Should contain all imports
        lines = result.split('\n')
        self.assertIn("import os", lines)
        self.assertIn("import sys", lines)
        self.assertIn("import json", lines)
        self.assertIn("import pathlib", lines)

    def test_merge_additive_changes(self):
        """Test merging additive changes."""
        current = "line1\nline2"
        incoming = "line3\nline4"
        result = self.git_auto._merge_additive_changes(current, incoming)
        
        expected = "line1\nline2\nline3\nline4"
        self.assertEqual(result, expected)


if __name__ == "__main__":
    unittest.main()