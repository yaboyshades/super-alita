"""
State type definitions for REUG v9.0

Separated to avoid circular imports between session.py and states.py
"""

from enum import Enum, auto


class StateType(Enum):
    """REUG v9.0 State Types"""

    READY = auto()  # Initial state, waiting for input
    ENGAGE = auto()  # Processing user request, intent detection
    UNDERSTAND = auto()  # Context building, memory loading, tool routing
    GENERATE = auto()  # Tool execution, response generation
    CREATE_DYNAMIC_TOOL = auto()  # Dynamic tool creation from natural language
    ERROR_RECOVERY = auto()  # Graceful error handling and recovery
    COMPLETE = auto()  # Completion state, ready for next turn
    SHUTDOWN = auto()  # System shutdown state
