"""Code-Execute-Observe loop for the CodeAct system."""

from __future__ import annotations

import logging
from typing import Callable

from .actions import AgentFinish, IPythonRunCell
from .observation import Observation
from .sandbox import PythonSandbox

logger = logging.getLogger(__name__)


class CodeActRunner:
    """Simple loop executing actions and feeding observations to a policy."""

    def __init__(self, sandbox: PythonSandbox, policy: Callable[[Observation], AgentFinish | IPythonRunCell]):
        self.sandbox = sandbox
        self.policy = policy

    async def step(self, action: IPythonRunCell) -> Observation:
        result = await self.sandbox.run(action.code)
        obs = Observation(stdout=result.stdout, stderr=result.stderr, error=result.error)
        self._track_resources(obs)
        return obs

    async def run(self, initial: IPythonRunCell, max_steps: int = 10) -> Observation:
        action: AgentFinish | IPythonRunCell = initial
        observation = Observation()
        for _ in range(max_steps):
            if isinstance(action, AgentFinish):
                logger.debug("Agent finished early")
                break
            observation = await self.step(action)
            action = self.policy(observation)
            if isinstance(action, AgentFinish):
                break
        return observation

    def _track_resources(self, observation: Observation) -> None:
        """Placeholder for memory/energy accounting."""
        logger.debug("Resource hook - stdout %d chars", len(observation.stdout))
