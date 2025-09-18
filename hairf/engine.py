"""Execution engine for orchestrating reasoning modules."""

from __future__ import annotations

from typing import Iterable, List, Sequence

from .aggregator import OutputAggregator
from .base import ReasoningModule
from .types import ModuleTrace, Query, ReasoningResult, ReasoningState


class ModularExecutionEngine:
    """Run a collection of modules and aggregate their states."""

    def __init__(self, aggregator: OutputAggregator | None = None) -> None:
        self.aggregator = aggregator or OutputAggregator()

    def execute(
        self,
        query: Query,
        modules: Sequence[ReasoningModule],
        *,
        context: Sequence[ReasoningState] | None = None,
    ) -> tuple[list[ModuleTrace], List[ReasoningState]]:
        traces: list[ModuleTrace] = []
        collected: List[ReasoningState] = []

        for module in modules:
            trace = module.run(query, context=context or collected)
            traces.append(trace)
            collected.extend(trace.states)
            context = collected  # progressive context sharing

        return traces, collected

    def finalize(
        self,
        query: Query,
        traces: Iterable[ModuleTrace],
        states: Sequence[ReasoningState],
    ) -> ReasoningResult:
        rationale = f"Synthesized from {len(states)} states for query: {query.text[:80]}"
        result = self.aggregator.synthesize(states, rationale=rationale)
        for trace in traces:
            result = result.with_trace(trace)
        return result
