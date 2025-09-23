"""
User creation module implementing the user.create specification.
"""

import uuid
from datetime import datetime
from typing import TypedDict


class UserIn(TypedDict):
    email: str
    name: str


class UserOut(TypedDict):
    id: str
    email: str
    name: str
    created_at: str  # ISO8601Z


def create_user(data: UserIn, referrer: str | None = None) -> UserOut:
    """
    Create a new user account.

    Args:
        data: User input data containing email and name
        referrer: Optional referrer information

    Returns:
        UserOut: Created user information

    Raises:
        ValueError: If email already exists
    """
    # In a real implementation, this would check for duplicate emails
    # and save to a database

    # Validate preconditions (normally done by generated validators)
    if not data["email"] or "@" not in data["email"]:
        raise ValueError("Invalid email")

    if not data["name"]:
        raise ValueError("Name is required")

    # Create user
    user_id = f"u_{uuid.uuid4().hex[:8]}"
    created_at = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

    return {
        "id": user_id,
        "email": data["email"],
        "name": data["name"],
        "created_at": created_at,
    }
