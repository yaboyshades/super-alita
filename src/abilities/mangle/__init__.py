"""
Mangle Integration Package

This package provides integration with Google's Mangle deductive database
programming language for Super Alita, enabling advanced logical reasoning,
security analysis, and knowledge graph capabilities.
"""

from src.abilities.mangle.mangle_ability import MangleAbility, ManglePluginInterface
from src.abilities.mangle.register import register_mangle_abilities, register_mangle_plugin

__all__ = [
    "MangleAbility",
    "ManglePluginInterface",
    "register_mangle_abilities",
    "register_mangle_plugin"
]
