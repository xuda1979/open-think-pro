"""Reference implementations of classical reasoning strategies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from .base import ReasoningModule
from .types import Query, ReasoningState


@dataclass
class Heuristic:
    """Simple heuristic that scores a string."""

    name: str
    weight: float

    def score(self, text: str) -> float:
        occurrences = text.lower().count(self.name.lower())
        return min(1.0, occurrences * self.weight)


class ChainOfThoughtReasoner(ReasoningModule):
    """Produces sequential reasoning states by chunking the query."""

    name = "cot"

    def __init__(self, *, step_size: int = 32) -> None:
        super().__init__()
        self.step_size = step_size

    def execute(
        self, query: Query, *, context: Sequence[ReasoningState] | None = None
    ) -> Iterable[ReasoningState]:
        tokens = query.text.split()
        if not tokens:
            return []
        for i in range(0, len(tokens), self.step_size):
            chunk = " ".join(tokens[i : i + self.step_size])
            yield ReasoningState(
                content=f"Step {i // self.step_size + 1}: {chunk}",
                confidence=0.55,
                cost=0.5,
                metadata={"chunk_index": i},
            )


class SelfConsistencyReasoner(ReasoningModule):
    """Aggregates context states and boosts consensus confidence."""

    name = "self_consistency"

    def __init__(self, heuristics: Sequence[Heuristic] | None = None) -> None:
        super().__init__()
        self.heuristics = tuple(heuristics or (Heuristic(name="thus", weight=0.1),))

    def supports(self, query: Query) -> bool:
        return bool(query.text)

    def execute(
        self, query: Query, *, context: Sequence[ReasoningState] | None = None
    ) -> Iterable[ReasoningState]:
        context = context or []
        combined_text = "\n".join(state.content for state in context) or query.text
        score = 0.3 + sum(heuristic.score(combined_text) for heuristic in self.heuristics)
        yield ReasoningState(
            content=f"Consensus synthesis for: {query.text[:80]}",
            confidence=min(1.0, score),
            cost=0.4,
            metadata={"heuristics": [h.name for h in self.heuristics]},
        )


class ReflectiveCritic(ReasoningModule):
    """Critiques previous reasoning states and suggests refinements."""

    name = "critic"

    def execute(
        self, query: Query, *, context: Sequence[ReasoningState] | None = None
    ) -> Iterable[ReasoningState]:
        context = context or []
        if not context:
            return []
        best = max(context, key=lambda state: state.confidence)
        yield ReasoningState(
            content=f"Critique: Validate -> {best.content}",
            confidence=min(1.0, best.confidence + 0.1),
            cost=0.3,
            metadata={"target": best.metadata},
        )
