#!/usr/bin/env python3
"""
Verify signed lock files with HMAC
"""
import hmac
import hashlib
import json
import os
import sys


def verify_lock_file(lock_file_path: str, signing_key: str) -> bool:
    """
    Verify a signed lock file with HMAC.
    
    Args:
        lock_file_path: Path to the signed lock file
        signing_key: HMAC signing key
        
    Returns:
        True if verification passes, False otherwise
    """
    # Read the signed lock file
    with open(lock_file_path, "r", encoding="utf-8") as f:
        blob = json.load(f)
    
    # Extract payload and signature
    payload = blob.get("payload")
    signature = blob.get("signature")
    
    if not payload or not signature:
        print(f"Error: Invalid lock file format in {lock_file_path}")
        return False
    
    # Canonicalize the payload
    raw = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":")
    ).encode()
    
    # Generate expected signature
    expected_sig = "hmac-sha256:" + hmac.new(
        signing_key.encode(),
        raw,
        hashlib.sha256
    ).hexdigest()
    
    # Compare signatures using constant-time comparison
    if hmac.compare_digest(signature, expected_sig):
        print(f"Lock file verification passed: {lock_file_path}")
        return True
    else:
        print(f"Lock file verification failed: {lock_file_path}")
        print(f"Expected: {expected_sig}")
        print(f"Actual: {signature}")
        return False


def main() -> None:
    """
    Main entry point for the lock verification tool.
    """
    if len(sys.argv) != 2:
        print("Usage: python verify-lock.py <lock-file-path>")
        sys.exit(1)
    
    lock_file_path = sys.argv[1]
    
    # Get signing key from environment
    signing_key = os.environ.get("LOCK_SIGNING_KEY")
    if not signing_key:
        print("Error: LOCK_SIGNING_KEY environment variable not set")
        sys.exit(1)
    
    # Verify the lock file
    try:
        is_valid = verify_lock_file(lock_file_path, signing_key)
        sys.exit(0 if is_valid else 2)
    except Exception as e:
        print(f"Error verifying lock file: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()