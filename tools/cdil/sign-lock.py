#!/usr/bin/env python3
"""
Sign lock files with HMAC for tamper evidence
"""
import hashlib
import hmac
import json
import os
import sys


def sign_lock_file(lock_file_path: str, signing_key: str) -> None:
    """
    Sign a lock file with HMAC.
    
    Args:
        lock_file_path: Path to the lock file to sign
        signing_key: HMAC signing key
    """
    # Read the lock file
    with open(lock_file_path, encoding="utf-8") as f:
        data = json.load(f)
    
    # Canonicalize the data
    raw = json.dumps(
        data, 
        ensure_ascii=False, 
        sort_keys=True, 
        separators=(",", ":")
    ).encode()
    
    # Generate HMAC signature
    sig = hmac.new(
        signing_key.encode(), 
        raw, 
        hashlib.sha256
    ).hexdigest()
    
    # Create signed payload
    signed_payload = {
        "payload": data,
        "signature": f"hmac-sha256:{sig}"
    }
    
    # Write signed lock file
    with open(lock_file_path, "w", encoding="utf-8") as f:
        json.dump(signed_payload, f, ensure_ascii=False, separators=(",", ":"))
    
    print(f"Signed lock file: {lock_file_path}")


def main() -> None:
    """
    Main entry point for the lock signing tool.
    """
    if len(sys.argv) != 2:
        print("Usage: python sign-lock.py <lock-file-path>")
        sys.exit(1)
    
    lock_file_path = sys.argv[1]
    
    # Get signing key from environment
    signing_key = os.environ.get("LOCK_SIGNING_KEY")
    if not signing_key:
        print("Error: LOCK_SIGNING_KEY environment variable not set")
        sys.exit(1)
    
    # Sign the lock file
    try:
        sign_lock_file(lock_file_path, signing_key)
    except Exception as e:
        print(f"Error signing lock file: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()