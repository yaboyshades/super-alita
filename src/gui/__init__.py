"""Lightweight GUI component library for Super Alita.

Provides:
- Registry for server-rendered components.
- FastAPI router exposing /gui index and dynamic component endpoints.
- Minimal static assets (CSS/JS) for progressive enhancement.

Goal: allow rapid assembly of small dashboards and interactive panels
without pulling heavy frontend frameworks.
"""

from .registry import gui_registry, register_component  # noqa: F401
