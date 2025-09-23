"""
Tests for the user.create module.
"""

import pytest

from src.user.create import UserIn, create_user


def test_create_user_success():
    """Test successful user creation."""
    data: UserIn = {"email": "test@example.com", "name": "Test User"}

    result = create_user(data)

    assert result["email"] == "test@example.com"
    assert result["name"] == "Test User"
    assert result["id"].startswith("u_")
    assert "T" in result["created_at"]
    assert result["created_at"].endswith("Z")


def test_create_user_with_referrer():
    """Test user creation with referrer."""
    data: UserIn = {"email": "test@example.com", "name": "Test User"}

    result = create_user(data, "referrer123")

    assert result["email"] == "test@example.com"
    assert result["name"] == "Test User"


def test_create_user_invalid_email():
    """Test user creation with invalid email."""
    data: UserIn = {"email": "invalid-email", "name": "Test User"}

    with pytest.raises(ValueError, match="Invalid email"):
        create_user(data)


def test_create_user_empty_name():
    """Test user creation with empty name."""
    data: UserIn = {"email": "test@example.com", "name": ""}

    with pytest.raises(ValueError, match="Name is required"):
        create_user(data)
