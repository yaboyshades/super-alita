"""Plugin loading and discovery utilities."""

import logging
from typing import Dict, List, Any
from pathlib import Path
import json

def load_plugin_manifest() -> Dict[str, Any]:
    """Load plugin manifest if it exists."""
    manifest_path = Path("plugins/manifest.json")
    
    if manifest_path.exists():
        with open(manifest_path) as f:
            return json.load(f)
    
    # Return default manifest
    return {
        "version": "1.0",
        "plugins": []
    }

def discover_plugins(manifest: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Discover available plugins from manifest."""
    logger = logging.getLogger(__name__)
    
    plugins = manifest.get("plugins", [])
    logger.info(f"Discovered {len(plugins)} plugins from manifest")
    
    return plugins