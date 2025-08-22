"""
Session management for Super Alita with concurrency safety.

Provides input mailbox, operation tracking, and concurrency controls
to handle re-entrant input and stale completions safely.
"""

import asyncio
import uuid
from datetime import UTC, datetime
from typing import Any

from src.core.state_types import StateType


class Session:
    """
    Session object with input mailbox and concurrency controls.

    Handles:
    - Input queueing when FSM is not accepting input
    - Operation ID tracking for idempotent completions
    - Per-session locks for transition serialization
    """

    def __init__(self, session_id: str | None = None):
        self.session_id = session_id or f"session_{uuid.uuid4().hex[:8]}"

        # Input mailbox for re-entrant input handling
        self.input_mailbox: asyncio.Queue[str] = asyncio.Queue(maxsize=50)

        # States that accept user input
        self.accepting_input_states = {
            StateType.READY,
            StateType.ENGAGE,
            StateType.COMPLETE,
        }

        # Operation tracking for idempotent completions
        self.inflight_op_id: str | None = None

        # Concurrency control
        self.transition_lock = asyncio.Lock()

        # Metrics
        self.input_queued_count = 0
        self.input_dropped_count = 0
        self.stale_completions_count = 0
        self.ignored_triggers_count = 0

        # Creation timestamp
        self.created_at = datetime.now(UTC)

    def start_operation(self) -> str:
        """Start a new operation and return its ID"""
        self.inflight_op_id = uuid.uuid4().hex
        return self.inflight_op_id

    def register_operation(self, operation_name: str) -> str:
        """Register a new operation (alias for start_operation with name)"""
        op_id = f"{operation_name}_{uuid.uuid4().hex[:8]}"
        self.inflight_op_id = op_id
        return op_id

    def is_operation_current(self, op_id: str) -> bool:
        """Check if operation ID matches current inflight operation"""
        return op_id == self.inflight_op_id

    def complete_operation(self, op_id: str) -> bool:
        """Complete operation if ID matches current inflight"""
        if self.is_operation_current(op_id):
            self.inflight_op_id = None
            return True
        else:
            self.stale_completions_count += 1
            return False

    def can_accept_input(self, current_state: StateType) -> bool:
        """Check if current state accepts user input"""
        return current_state in self.accepting_input_states

    def queue_input(self, user_input: str) -> bool:
        """
        Queue user input for later processing.

        Returns:
            bool: True if queued successfully, False if dropped
        """
        try:
            self.input_mailbox.put_nowait(user_input)
            self.input_queued_count += 1
            return True
        except asyncio.QueueFull:
            # Drop oldest to keep system responsive
            try:
                _ = self.input_mailbox.get_nowait()
                self.input_mailbox.put_nowait(user_input)
                self.input_dropped_count += 1
                self.input_queued_count += 1
                return True
            except asyncio.QueueEmpty:
                # Race condition, queue is empty now
                self.input_mailbox.put_nowait(user_input)
                self.input_queued_count += 1
                return True

    async def drain_one_input(self) -> str | None:
        """
        Drain exactly one input from mailbox.

        Returns:
            Optional[str]: Next queued input, or None if empty
        """
        try:
            return self.input_mailbox.get_nowait()
        except asyncio.QueueEmpty:
            return None

    def get_mailbox_size(self) -> int:
        """Get current mailbox size"""
        return self.input_mailbox.qsize()

    def increment_ignored_triggers(self):
        """Increment ignored triggers counter"""
        self.ignored_triggers_count += 1

    def get_metrics(self) -> dict[str, Any]:
        """Get session metrics"""
        return {
            "session_id": self.session_id,
            "created_at": self.created_at.isoformat(),
            "inflight_op_id": self.inflight_op_id,
            "mailbox_size": self.get_mailbox_size(),
            "input_queued_count": self.input_queued_count,
            "input_dropped_count": self.input_dropped_count,
            "stale_completions_count": self.stale_completions_count,
            "ignored_triggers_count": self.ignored_triggers_count,
        }


# Simple session registry for tests and lightweight usage
_SESSIONS: dict[str, Session] = {}


def get_session(session_id: str | None = None) -> Session:
    """Return a session by id, creating it if necessary."""
    if session_id is None:
        return Session()
    if session_id not in _SESSIONS:
        _SESSIONS[session_id] = Session(session_id)
    return _SESSIONS[session_id]


__all__ = ["Session", "get_session"]
