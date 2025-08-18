"""
Tests for Core Utils functionality.
"""

import pytest

from src.tools.core_utils import CoreUtils


@pytest.mark.parametrize("expr,expected", [
    ("1+2*3", 7.0),
    ("(1+2)*3", 9.0),
    ("-3 + 5", 2.0),
    ("4 / 2", 2.0),
    ("2 + (3 * (4 - 1))", 11.0),
    ("10 + 5 * 2", 20.0),
    ("(2+3)*4", 20.0),
    ("-3.5 + 2", -1.5),
    ("+5", 5.0),
    ("--4", 4.0),
    ("2.5 * 4", 10.0),
])
def test_calculate(expr, expected):
    """Test arithmetic calculation with various expressions."""
    assert CoreUtils.calculate(expr) == pytest.approx(expected)


def test_calculate_empty_expression():
    """Test that empty expressions raise ValueError."""
    with pytest.raises(ValueError, match="empty expression"):
        CoreUtils.calculate("")


def test_calculate_invalid_expression():
    """Test that invalid expressions raise ValueError."""
    with pytest.raises(ValueError, match="invalid expression"):
        CoreUtils.calculate("1 + + 2")


def test_calculate_unsupported_operation():
    """Test that unsupported operations raise ValueError."""
    with pytest.raises(ValueError, match="unsupported expression"):
        CoreUtils.calculate("1 ** 2")  # Power operator not supported


def test_div_by_zero():
    """Test that division by zero raises ZeroDivisionError."""
    with pytest.raises(ZeroDivisionError, match="division by zero"):
        CoreUtils.calculate("1/0")


def test_div_by_zero_complex():
    """Test division by zero in complex expressions."""
    with pytest.raises(ZeroDivisionError, match="division by zero"):
        CoreUtils.calculate("5 + 1/(2-2)")


@pytest.mark.parametrize("text,expected", [
    ("hello", "olleh"),
    ("Super Alita", "atilA repuS"),
    ("12345", "54321"),
    ("", ""),
    ("a", "a"),
    ("race car", "rac ecar"),
])
def test_reverse_string(text, expected):
    """Test string reversal with various inputs."""
    assert CoreUtils.reverse_string(text) == expected


def test_reverse_non_string():
    """Test that non-string inputs are converted to strings."""
    assert CoreUtils.reverse_string(123) == "321"
    assert CoreUtils.reverse_string(45.67) == "76.54"
    assert CoreUtils.reverse_string(True) == "eurT"


def test_reverse_none():
    """Test that None is handled gracefully."""
    assert CoreUtils.reverse_string(None) == "enoN"


class TestCoreUtilsIntegration:
    """Integration tests for CoreUtils functionality."""

    def test_calculate_and_reverse_chain(self):
        """Test chaining calculator and string operations."""
        # Calculate a result
        result = CoreUtils.calculate("2 + 3 * 4")
        assert result == 14.0

        # Convert to string and reverse it
        reversed_result = CoreUtils.reverse_string(str(int(result)))
        assert reversed_result == "41"

    def test_complex_mathematical_expressions(self):
        """Test complex mathematical expressions."""
        # Test operator precedence
        assert CoreUtils.calculate("2 + 3 * 4 - 1") == 13.0
        assert CoreUtils.calculate("(2 + 3) * (4 - 1)") == 15.0

        # Test unary operators
        assert CoreUtils.calculate("-(-5)") == 5.0
        assert CoreUtils.calculate("+(+3)") == 3.0

        # Test nested parentheses
        assert CoreUtils.calculate("((2 + 3) * (4 + 1)) / 5") == 5.0
