"""
Declarative plugin loader for Super Alita.

Enables manifest-driven plugin discovery, priority ordering, and dependency validation.
Replaces hard-coded plugin lists with extensible YAML configuration.
"""

import importlib
import inspect
import logging
from pathlib import Path
from typing import Any

import yaml

from .plugin_interface import PluginInterface

logger = logging.getLogger(__name__)


class PluginManifestError(Exception):
    """Raised when plugin manifest is invalid or cannot be loaded."""


class PluginLoadError(Exception):
    """Raised when a plugin cannot be loaded from its module specification."""


def load_plugin_manifest(path: str | None = None) -> list[dict[str, Any]]:
    """
    Load plugin manifest from YAML file.

    Args:
        path: Path to manifest file. Defaults to 'plugins.yaml' in current directory.

    Returns:
        List of plugin configuration dictionaries

    Raises:
        PluginManifestError: If manifest cannot be loaded or is invalid
    """
    if path is None:
        # Look for plugins.yaml in current directory and super-alita/
        candidates = ["plugins.yaml", "super-alita/plugins.yaml"]
        for candidate in candidates:
            if Path(candidate).exists():
                path = candidate
                break
        else:
            logger.warning("No plugins.yaml found, falling back to static plugin list")
            return []

    try:
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if not isinstance(data, dict) or "plugins" not in data:
            raise PluginManifestError(
                f"Invalid manifest format in {path}: missing 'plugins' key"
            )

        plugins = data["plugins"]
        if not isinstance(plugins, list):
            raise PluginManifestError(
                f"Invalid manifest format in {path}: 'plugins' must be a list"
            )

        # Validate required fields
        for i, plugin in enumerate(plugins):
            if not isinstance(plugin, dict):
                raise PluginManifestError(f"Plugin {i} is not a dictionary")

            required_fields = ["name", "module"]
            for field in required_fields:
                if field not in plugin:
                    raise PluginManifestError(
                        f"Plugin {i} missing required field: {field}"
                    )

        logger.info(f"Loaded {len(plugins)} plugin definitions from {path}")
        return plugins

    except yaml.YAMLError as e:
        raise PluginManifestError(f"Failed to parse YAML from {path}: {e}") from e
    except FileNotFoundError:
        raise PluginManifestError(f"Plugin manifest not found: {path}") from None
    except Exception as e:
        raise PluginManifestError(
            f"Unexpected error loading manifest from {path}: {e}"
        ) from e


def validate_dependencies(plugins: list[dict[str, Any]]) -> set[str]:
    """
    Validate plugin dependencies and return set of missing dependencies.

    Args:
        plugins: List of plugin configurations

    Returns:
        Set of plugin names that are referenced as dependencies but not defined
    """
    plugin_names = {plugin["name"] for plugin in plugins}
    missing_deps = set()

    for plugin in plugins:
        deps = plugin.get("depends_on", [])
        if not isinstance(deps, list):
            continue

        for dep in deps:
            if dep not in plugin_names:
                missing_deps.add(dep)

    return missing_deps


def discover_plugins(
    plugins: list[dict[str, Any]],
) -> list[tuple[str, type[PluginInterface]]]:
    """
    Discover and load plugin classes from manifest, ordered by priority.

    Args:
        plugins: List of plugin configurations from manifest

    Returns:
        List of (plugin_name, plugin_class) tuples, ordered by priority

    Raises:
        PluginLoadError: If a plugin cannot be loaded
    """
    result = []

    # Filter enabled plugins and sort by priority
    enabled_plugins = [p for p in plugins if p.get("enabled", True)]
    sorted_plugins = sorted(enabled_plugins, key=lambda p: p.get("priority", 100))

    for plugin_config in sorted_plugins:
        name = plugin_config["name"]
        module_spec = plugin_config["module"]

        try:
            # Parse module:class specification
            if ":" not in module_spec:
                raise PluginLoadError(
                    f"Invalid module specification '{module_spec}': must be 'module:class'"
                )

            module_path, class_name = module_spec.split(":", 1)

            # Import module and get class
            try:
                module = importlib.import_module(module_path)
            except ImportError as e:
                raise PluginLoadError(
                    f"Cannot import module '{module_path}' for plugin '{name}': {e}"
                ) from e

            if not hasattr(module, class_name):
                raise PluginLoadError(
                    f"Class '{class_name}' not found in module '{module_path}' for plugin '{name}'"
                )

            plugin_class = getattr(module, class_name)

            # Validate that it's a class and implements PluginInterface
            if not inspect.isclass(plugin_class):
                raise PluginLoadError(
                    f"'{class_name}' is not a class in plugin '{name}'"
                )

            if not issubclass(plugin_class, PluginInterface):
                logger.warning(
                    f"Plugin class '{class_name}' does not inherit from PluginInterface"
                )

            result.append((name, plugin_class))
            logger.debug(f"Loaded plugin '{name}' from {module_spec}")

        except Exception as e:
            if isinstance(e, PluginLoadError):
                raise
            raise PluginLoadError(
                f"Unexpected error loading plugin '{name}': {e}"
            ) from e

    logger.info(f"Successfully loaded {len(result)} plugins")
    return result


def get_plugin_info(plugins: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """
    Extract plugin metadata for introspection.

    Args:
        plugins: List of plugin configurations

    Returns:
        Dictionary mapping plugin names to their metadata
    """
    info = {}

    for plugin in plugins:
        name = plugin["name"]
        info[name] = {
            "module": plugin["module"],
            "enabled": plugin.get("enabled", True),
            "priority": plugin.get("priority", 100),
            "depends_on": plugin.get("depends_on", []),
            "description": plugin.get("description", ""),
            "category": plugin.get("category", "uncategorized"),
        }

    return info


def validate_plugin_order(plugins: list[dict[str, Any]]) -> list[str]:
    """
    Validate that plugin dependencies can be satisfied in priority order.

    Args:
        plugins: List of plugin configurations

    Returns:
        List of warnings about dependency ordering issues
    """
    warnings = []
    enabled_plugins = [p for p in plugins if p.get("enabled", True)]
    sorted_plugins = sorted(enabled_plugins, key=lambda p: p.get("priority", 100))

    loaded_plugins = set()

    for plugin in sorted_plugins:
        name = plugin["name"]
        deps = plugin.get("depends_on", [])

        for dep in deps:
            if dep not in loaded_plugins:
                warnings.append(
                    f"Plugin '{name}' depends on '{dep}' which loads later (or is disabled)"
                )

        loaded_plugins.add(name)

    return warnings
