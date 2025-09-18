"""Lightweight continual learning utilities for HAIRF."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Iterable, Tuple

from .types import Query, ReasoningResult


@dataclass
class Experience:
    query: Query
    result: ReasoningResult
    reward: float


class ExperienceReplayLearner:
    """Maintains a replay buffer to tune router preferences online."""

    def __init__(self, capacity: int = 256) -> None:
        self.capacity = capacity
        self._buffer: Deque[Experience] = deque(maxlen=capacity)

    def record(self, experience: Experience) -> None:
        self._buffer.append(experience)

    def batch(self, size: int = 16) -> Iterable[Experience]:
        return list(self._buffer)[-size:]

    def summarize(self) -> Tuple[float, float]:
        if not self._buffer:
            return (0.0, 0.0)
        rewards = [exp.reward for exp in self._buffer]
        return (float(sum(rewards) / len(rewards)), max(rewards))
