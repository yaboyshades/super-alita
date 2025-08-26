# Architectural Overview

## Overview
Super Alita is a sophisticated self-evolving AI agent system built on an event-driven neural architecture. This guide provides a high-level view of the system's major components and design principles.

## Core Architecture Principles
1. **Event-Driven Neural Architecture**: All components communicate through a Redis/Memurai-backed event bus.
2. **MCP Integration**: Tools and VS Code integration rely on the Model Context Protocol for standardized communication.
3. **Atoms/Bonds Cognitive Fabric**: All outputs are structured as atoms with deterministic UUIDs to maintain lineage.
4. **Plugin-Based Modularity**: Components implement `PluginInterface` for hot-swappable functionality.
5. **Sandboxed Execution**: Dynamic code execution flows through a secure sandbox.
6. **Multi-Modal LLM Support**: The system supports multiple LLM providers with automatic fallback.

## Further Reading
- [Refactoring Guide](./02_refactoring_guide.md)
- [Agentic Workflows](./03_agentic_workflows.md)
- [Advanced Development Patterns](./04_advanced_patterns.md)
