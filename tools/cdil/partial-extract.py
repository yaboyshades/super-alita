#!/usr/bin/env python3
"""
Partial extraction cache for Spec Kit + CDIL
Caches symbol graphs per-file for faster local runs
"""
import hashlib
import json
import os
import sys
from typing import Dict, Any, Optional


def get_file_hash(file_path: str) -> str:
    """
    Calculate SHA256 hash of a file.
    
    Args:
        file_path: Path to the file
        
    Returns:
        SHA256 hash as hex string
    """
    sha256 = hashlib.sha256()
    try:
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                sha256.update(chunk)
        return sha256.hexdigest()
    except Exception as e:
        print(f"Error hashing file {file_path}: {e}")
        return ""


def load_cache_index() -> Dict[str, Any]:
    """
    Load the cache index from disk.
    
    Returns:
        Cache index dictionary
    """
    cache_dir = ".contracts/cache"
    index_file = os.path.join(cache_dir, "index.json")
    
    # Create cache directory if it doesn't exist
    os.makedirs(cache_dir, exist_ok=True)
    
    if os.path.exists(index_file):
        try:
            with open(index_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading cache index: {e}")
            return {}
    else:
        return {}


def save_cache_index(index: Dict[str, Any]) -> None:
    """
    Save the cache index to disk.
    
    Args:
        index: Cache index dictionary
    """
    cache_dir = ".contracts/cache"
    index_file = os.path.join(cache_dir, "index.json")
    
    try:
        with open(index_file, 'w') as f:
            json.dump(index, f, indent=2)
    except Exception as e:
        print(f"Error saving cache index: {e}")


def get_cached_symbol_graph(file_path: str) -> Optional[Dict[str, Any]]:
    """
    Get cached symbol graph for a file if it exists and is valid.
    
    Args:
        file_path: Path to the source file
        
    Returns:
        Cached symbol graph or None if not available
    """
    # Get file hash
    file_hash = get_file_hash(file_path)
    if not file_hash:
        return None
    
    # Load cache index
    index = load_cache_index()
    
    # Check if file is cached
    cache_key = f"{file_path}:{file_hash}"
    if cache_key in index:
        cache_entry = index[cache_key]
        cache_file = cache_entry.get("cache_file")
        
        if cache_file and os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading cached symbol graph: {e}")
                # Remove invalid cache entry
                del index[cache_key]
                save_cache_index(index)
    
    return None


def cache_symbol_graph(
    file_path: str, 
    file_hash: str, 
    symbol_graph: Dict[str, Any]
) -> None:
    """
    Cache a symbol graph for a file.
    
    Args:
        file_path: Path to the source file
        file_hash: Hash of the file content
        symbol_graph: Symbol graph to cache
    """
    # Load cache index
    index = load_cache_index()
    
    # Create cache entry
    cache_dir = ".contracts/cache"
    cache_filename = f"{hashlib.md5(file_path.encode()).hexdigest()}.json"
    cache_file = os.path.join(cache_dir, cache_filename)
    
    cache_key = f"{file_path}:{file_hash}"
    index[cache_key] = {
        "file_path": file_path,
        "file_hash": file_hash,
        "cache_file": cache_file,
        "timestamp": os.path.getmtime(file_path)
    }
    
    # Save symbol graph
    try:
        with open(cache_file, 'w') as f:
            json.dump(symbol_graph, f, indent=2)
    except Exception as e:
        print(f"Error saving cached symbol graph: {e}")
        return
    
    # Save updated index
    save_cache_index(index)
    print(f"Cached symbol graph for {file_path}")


def extract_symbol_graph(file_path: str) -> Dict[str, Any]:
    """
    Extract symbol graph from a source file.
    In a real implementation, this would parse the AST and extract symbols.
    
    Args:
        file_path: Path to the source file
        
    Returns:
        Symbol graph dictionary
    """
    # This is a placeholder implementation
    # In a real implementation, you would:
    # 1. Parse the file with AST
    # 2. Extract function/class definitions
    # 3. Extract type information
    # 4. Build symbol graph
    
    print(f"Extracting symbol graph from {file_path}")
    
    # For demonstration, return a simple structure
    return {
        "file": file_path,
        "functions": [],
        "classes": [],
        "exports": [],
        "hash": get_file_hash(file_path)
    }


def main():
    """
    Main entry point for the partial extraction tool.
    """
    if len(sys.argv) < 2:
        print("Usage: python partial-extract.py <file-path>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        sys.exit(1)
    
    # Check if we have a cached version
    cached_graph = get_cached_symbol_graph(file_path)
    if cached_graph:
        print(f"Using cached symbol graph for {file_path}")
        print(json.dumps(cached_graph, indent=2))
        return cached_graph
    
    # Extract symbol graph
    symbol_graph = extract_symbol_graph(file_path)
    
    # Cache the result
    file_hash = get_file_hash(file_path)
    if file_hash:
        cache_symbol_graph(file_path, file_hash, symbol_graph)
    
    # Output the result
    print(json.dumps(symbol_graph, indent=2))
    return symbol_graph


if __name__ == "__main__":
    main()