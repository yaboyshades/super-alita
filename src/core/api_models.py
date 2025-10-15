"""API response models and utilities."""

from typing import Dict, Any

def api_error(error_type: str, category: str, details: Dict[str, Any] = None) -> Dict[str, Any]:
    """Create standardized API error response."""
    return {
        "error": {
            "type": error_type,
            "category": category,
            "details": details or {},
            "timestamp": __import__('time').time()
        }
    }

def api_success(data: Any = None, message: str = None) -> Dict[str, Any]:
    """Create standardized API success response."""
    response = {
        "success": True,
        "timestamp": __import__('time').time()
    }
    
    if data is not None:
        response["data"] = data
        
    if message:
        response["message"] = message
        
    return response