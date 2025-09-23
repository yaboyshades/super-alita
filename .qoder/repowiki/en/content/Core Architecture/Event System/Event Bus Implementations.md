
# Event Bus Implementations

<cite>
**Referenced Files in This Document**   
- [in_memory_event_bus.py](file://src/core/in_memory_event_bus.py)
- [event_bus.py](file://src/core/event_bus.py)
- [reliable_event_bus.py](file://src/core/reliable_event_bus.py)
- [events.py](file://src/core/events.py)
- [redis_event_bus.py](file://src/adapters/redis_event_bus.py)
</cite>

## Table of Contents
1. [Introduction](#introduction)
2. [In-Memory Event Bus](#in-memory-event-bus)
3. [Redis-Backed Event Bus](#redis-backed-event-bus)
4. [Reliable Event Bus](#reliable-event-bus)
5. [Interface Contracts](#interface-contracts)
6. [Performance Characteristics](#performance-characteristics)
7. [Failure Modes and Resilience](#failure-modes-and-resilience)
8. [Usage Patterns and Configuration](#usage-patterns-and-configuration)
9. [Trade-offs Analysis](#trade-offs-analysis)
10. [Troubleshooting Common Issues](#troubleshooting-common-issues)

## Introduction
The Super Alita system implements a sophisticated event-driven architecture through three distinct event bus implementations, each designed for specific operational requirements. This document provides comprehensive documentation of the in-memory, Redis-backed, and reliable event bus implementations, detailing their interface contracts, performance characteristics, and failure modes. The event bus serves as the central nervous system for inter-component communication, enabling decoupled, asynchronous interactions between system components. Each implementation offers different trade-offs between latency, durability, and complexity, allowing developers to select the appropriate solution based on their specific use case requirements.

## In-Memory Event Bus
The In-Memory Event Bus provides a lightweight, high-performance implementation suitable for testing, development, and local execution scenarios. This implementation stores event handlers in a dictionary structure and processes events entirely within the current process memory space, eliminating network overhead and external dependencies. The bus maintains a simple event routing mechanism that maps event types to their corresponding handler functions, enabling efficient message dispatch. When an event is emitted, the bus creates a BaseEvent object from the provided parameters and concurrently invokes all registered handlers through asyncio.gather, ensuring non-blocking execution. The implementation includes lifecycle management through start and stop methods that control the operational state of the bus, with appropriate logging to track its status. This implementation is particularly valuable for unit testing and rapid prototyping, as it allows for predictable, isolated event processing without requiring external infrastructure.

```mermaid
classDiagram