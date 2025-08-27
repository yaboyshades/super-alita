#!/usr/bin/env python3
"""Generate requirements.txt and requirements-test.txt from source dependencies.

This script scans the source code to find import statements and generates
requirements files based on discovered dependencies.
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path
from typing import Set


# Known standard library modules (Python 3.11+)
STDLIB_MODULES = {
    'abc', 'argparse', 'ast', 'asyncio', 'base64', 'collections', 'contextlib',
    'copy', 'datetime', 'decimal', 'enum', 'functools', 'hashlib', 'http',
    'inspect', 'itertools', 'json', 'logging', 'math', 'operator', 'os',
    'pathlib', 'pickle', 'random', 're', 'socket', 'string', 'sys', 'tempfile',
    'time', 'typing', 'urllib', 'uuid', 'warnings', 'weakref', 'xml'
}

# Map import names to pip package names
PACKAGE_MAPPING = {
    'aiohttp': 'aiohttp>=3.9',
    'aioredis': 'aioredis>=2.0.0',
    'chromadb': 'chromadb>=0.4.0',
    'fastapi': 'fastapi>=0.104.0',
    'google': 'google-generativeai>=0.3.0',
    'hypothesis': 'hypothesis>=6.0.0',
    'networkx': 'networkx>=3.0',
    'numpy': 'numpy>=1.24.0',
    'openai': 'openai>=1.0',
    'orjson': 'orjson>=3.10.7',
    'pydantic': 'pydantic>=2.11',
    'pytest': 'pytest>=8.0.0',
    'redis': 'redis>=5.0.0',
    'sentence_transformers': 'sentence-transformers>=2.2.0',
    'starlette': 'starlette',
    'uvicorn': 'uvicorn>=0.18.0',
    'yaml': 'pyyaml',
    'dotenv': 'python-dotenv>=1.0.0',
    'protobuf': 'protobuf>=4.0',
}

# Test-only dependencies
TEST_ONLY_DEPS = {
    'pytest', 'hypothesis', 'fakeredis', 'pytest_asyncio'
}


class ImportVisitor(ast.NodeVisitor):
    """AST visitor to extract import statements."""
    
    def __init__(self):
        self.imports: Set[str] = set()
    
    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self.imports.add(alias.name.split('.')[0])
    
    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module:
            self.imports.add(node.module.split('.')[0])


def extract_imports_from_file(file_path: Path) -> Set[str]:
    """Extract import statements from a Python file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        tree = ast.parse(content)
        visitor = ImportVisitor()
        visitor.visit(tree)
        return visitor.imports
    except (SyntaxError, UnicodeDecodeError):
        # Skip files with syntax errors or encoding issues
        return set()


def scan_directory(directory: Path) -> Set[str]:
    """Scan a directory recursively for Python files and extract imports."""
    imports = set()
    
    for py_file in directory.rglob('*.py'):
        if '__pycache__' in str(py_file):
            continue
        imports.update(extract_imports_from_file(py_file))
    
    return imports


def filter_external_imports(imports: Set[str]) -> Set[str]:
    """Filter out standard library and relative imports."""
    external = set()
    
    for imp in imports:
        # Skip standard library modules
        if imp in STDLIB_MODULES:
            continue
        
        # Skip relative imports (starting with .)
        if imp.startswith('.'):
            continue
        
        # Skip local modules (common patterns)
        if imp in {'src', 'tests', 'tools', 'scripts', 'config'}:
            continue
        
        external.add(imp)
    
    return external


def generate_requirements(imports: Set[str], is_test: bool = False) -> list[str]:
    """Generate requirements list from imports."""
    requirements = []
    
    for imp in sorted(imports):
        if imp in PACKAGE_MAPPING:
            req = PACKAGE_MAPPING[imp]
            
            # Check if this is a test-only dependency
            if is_test or imp not in TEST_ONLY_DEPS:
                requirements.append(req)
    
    # Add core dependencies that might not be explicitly imported
    if not is_test:
        core_deps = [
            'pydantic>=2.11',
            'protobuf>=4.0',
            'pyyaml',
            'numpy>=1.24.0',
            'redis>=5.0.0',
            'python-dotenv>=1.0.0',
            'orjson>=3.10.7',
            'hiredis>=2.3.2',
            'faiss-cpu>=1.7.4',
            'chromadb>=0.4.0',
            'sentence-transformers>=2.2.0',
            'openai>=1.0',
            'openai-agents>=0.1'
        ]
        for dep in core_deps:
            if dep not in requirements:
                requirements.append(dep)
    else:
        test_deps = [
            'pytest>=8.0.0',
            'pytest-asyncio>=0.21.0',
            'redis>=5.0',
            'aiohttp>=3.9',
            'pydantic>=2.0.0',
            'aioredis>=2.0.0',
            'chromadb>=0.4.0',
            'google-generativeai>=0.3.0',
            'numpy>=1.24.0',
            'fastapi>=0.104.0',
            'networkx>=3.0',
            'hypothesis>=6.0.0',
            'fakeredis>=2.0.0'
        ]
        for dep in test_deps:
            if dep not in requirements:
                requirements.append(dep)
    
    return sorted(set(requirements))


def main():
    parser = argparse.ArgumentParser(description='Generate requirements files from source code')
    parser.add_argument('--src', nargs='*', default=['src'], help='Source directories to scan')
    parser.add_argument('--tests', default='tests', help='Test directory to scan')
    parser.add_argument('--write', action='store_true', help='Write to requirements files')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    # Scan source directories
    src_imports = set()
    for src_dir in args.src:
        src_path = Path(src_dir)
        if src_path.exists():
            if args.verbose:
                print(f"Scanning {src_path}")
            src_imports.update(scan_directory(src_path))
    
    # Scan test directory
    test_imports = set()
    test_path = Path(args.tests)
    if test_path.exists():
        if args.verbose:
            print(f"Scanning {test_path}")
        test_imports.update(scan_directory(test_path))
    
    # Filter external imports
    src_external = filter_external_imports(src_imports)
    test_external = filter_external_imports(test_imports)
    
    if args.verbose:
        print(f"Found {len(src_external)} external imports in source")
        print(f"Found {len(test_external)} external imports in tests")
    
    # Generate requirements
    main_reqs = generate_requirements(src_external, is_test=False)
    test_reqs = generate_requirements(test_external | src_external, is_test=True)
    
    # Output or write files
    if args.write:
        with open('requirements.txt', 'w') as f:
            f.write('# Core dependencies\n')
            for req in main_reqs:
                f.write(f'{req}\n')
        
        with open('requirements-test.txt', 'w') as f:
            f.write('# Test requirements\n')
            f.write('# Keep pytest aligned with the minversion specified in pyproject.toml\n')
            for req in test_reqs:
                f.write(f'{req}\n')
        
        print("Updated requirements.txt and requirements-test.txt")
    else:
        print("Main requirements:")
        for req in main_reqs:
            print(f"  {req}")
        
        print("\nTest requirements:")
        for req in test_reqs:
            print(f"  {req}")


if __name__ == '__main__':
    main()