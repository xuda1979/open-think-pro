"""Aggregation utilities for combining module outputs."""

from __future__ import annotations

from statistics import mean
from typing import Iterable, Sequence

from .types import ModuleTrace, ReasoningResult, ReasoningState


class OutputAggregator:
    """Fuse reasoning states into a final :class:`ReasoningResult`."""

    def __init__(self, *, token_budget: int = 2048) -> None:
        self.token_budget = token_budget

    def synthesize(
        self,
        states: Sequence[ReasoningState],
        *,
        rationale: str | None = None,
    ) -> ReasoningResult:
        if not states:
            raise ValueError("Cannot synthesize from an empty set of states")

        answer = self._compose_answer(states)
        confidence = mean(state.confidence for state in states)
        used_tokens = min(self.token_budget, int(sum(state.cost for state in states)))
        result = ReasoningResult(
            answer=answer,
            confidence=confidence,
            used_tokens=used_tokens,
            states=tuple(states),
        )
        if rationale:
            trace = ModuleTrace(
                module="aggregator",
                states=list(states),
                info={"rationale": rationale},
            )
            result = result.with_trace(trace)
        return result

    def _compose_answer(self, states: Iterable[ReasoningState]) -> str:
        summaries = [f"[{s.confidence:.2f}] {s.content}" for s in states]
        return "\n".join(summaries)
