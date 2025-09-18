"""Quantum-Inspired Reasoning Alignment (QIRA) implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, List, Sequence

import math

from .base import ReasoningModule
from .types import Query, ReasoningState

StateGenerator = Callable[[Query, Sequence[ReasoningState]], Iterable[ReasoningState]]
CollapseStrategy = Callable[[Sequence[ReasoningState]], ReasoningState]


def default_generator(query: Query, context: Sequence[ReasoningState]) -> Iterable[ReasoningState]:
    """Generate naive hypotheses by chunking the query text.

    The goal is to provide a deterministic, dependency-free baseline that can be
    replaced with calls into a real language model by end users.
    """

    tokens = query.text.split()
    if not tokens:
        yield ReasoningState(content="", confidence=0.0, cost=1.0)
        return

    stride = max(1, len(tokens) // 3)
    for i in range(0, len(tokens), stride):
        chunk = " ".join(tokens[i : i + stride])
        confidence = 0.5 + 0.5 * math.tanh(len(chunk) / 20)
        metadata = {"source": "default_generator", "position": i}
        yield ReasoningState(content=f"Hypothesis: {chunk}", confidence=confidence, cost=1.0, metadata=metadata)


def default_collapse(states: Sequence[ReasoningState]) -> ReasoningState:
    """Select the highest-confidence state."""

    return max(states, key=lambda state: state.confidence)


@dataclass
class QIRAConfig:
    """Configuration for the QIRA reasoning module."""

    generator: StateGenerator = default_generator
    collapse: CollapseStrategy = default_collapse
    superposition_temperature: float = 0.7


class QIRAReasoner(ReasoningModule):
    """Quantum-inspired reasoning alignment module."""

    name = "qira"

    def __init__(self, config: QIRAConfig | None = None) -> None:
        super().__init__()
        self.config = config or QIRAConfig()

    def execute(
        self, query: Query, *, context: Sequence[ReasoningState] | None = None
    ) -> Iterable[ReasoningState]:
        context = context or []
        candidates = list(self.config.generator(query, context))
        if not candidates:
            return []

        superposed = self._superpose(candidates)
        collapsed = self.config.collapse(superposed)

        return [collapsed]

    def _superpose(self, candidates: Sequence[ReasoningState]) -> List[ReasoningState]:
        weights = self._softmax([state.confidence for state in candidates])
        superposed: List[ReasoningState] = []
        aggregate_cost = sum(state.cost for state in candidates) / max(len(candidates), 1)

        for weight, state in zip(weights, candidates):
            superposed.append(state.scaled(weight))

        merged_content = "\n".join(state.content for state in superposed)
        merged_confidence = sum(state.confidence for state in superposed)
        merged_state = ReasoningState(
            content=f"Superposed Summary:\n{merged_content}",
            confidence=min(1.0, merged_confidence),
            cost=aggregate_cost + len(superposed) * 0.1,
            metadata={"superposed": True, "components": len(superposed)},
        )
        return [*superposed, merged_state]

    def _softmax(self, confidences: Sequence[float]) -> List[float]:
        temperature = max(1e-6, self.config.superposition_temperature)
        exps = [math.exp(c / temperature) for c in confidences]
        total = sum(exps)
        if total == 0:
            return [1.0 / len(confidences)] * len(confidences)
        return [value / total for value in exps]
