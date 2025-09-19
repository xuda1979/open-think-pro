"""Common datatypes used across HAIRF modules."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional


@dataclass
class Query:
    """Represents a user query and optional task metadata."""

    text: str
    task_type: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def enrich(self, **kwargs: Any) -> "Query":
        """Return a copy of the query with additional metadata."""

        merged = {**self.metadata, **kwargs}
        return Query(text=self.text, task_type=self.task_type, metadata=merged)


@dataclass
class ReasoningState:
    """A single reasoning hypothesis within the framework."""

    content: str
    confidence: float
    cost: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def scaled(self, weight: float) -> "ReasoningState":
        """Return a new state whose confidence is scaled by ``weight``."""

        return ReasoningState(
            content=self.content,
            confidence=self.confidence * weight,
            cost=self.cost,
            metadata=dict(self.metadata),
        )

    def merge(self, other: "ReasoningState", *, strategy: str = "average") -> "ReasoningState":
        """Merge two states into a single state using the given ``strategy``."""

        if strategy == "average":
            confidence = (self.confidence + other.confidence) / 2.0
        elif strategy == "max":
            confidence = max(self.confidence, other.confidence)
        else:
            raise ValueError(f"Unsupported merge strategy: {strategy}")

        return ReasoningState(
            content=f"{self.content}\n---\n{other.content}",
            confidence=confidence,
            cost=(self.cost + other.cost) / 2.0,
            metadata={**self.metadata, **other.metadata},
        )


@dataclass
class ModuleTrace:
    """Diagnostic trace produced by a reasoning module."""

    module: str
    states: List[ReasoningState]
    info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReasoningResult:
    """Final output produced by the framework."""

    answer: str
    confidence: float
    used_tokens: int
    states: Iterable[ReasoningState]
    traces: List[ModuleTrace] = field(default_factory=list)

    def with_trace(self, trace: ModuleTrace) -> "ReasoningResult":
        """Return a new result with ``trace`` appended to the trace list."""

        return ReasoningResult(
            answer=self.answer,
            confidence=self.confidence,
            used_tokens=self.used_tokens,
            states=self.states,
            traces=[*self.traces, trace],
        )


@dataclass
class RoutingDecision:
    """Describes how the router configured the reasoning stack."""

    selected_modules: List[str]
    rationale: str
    estimated_cost: float
    budget: int = 0
    difficulty: float = 0.0
    allocation: Dict[str, float] = field(default_factory=dict)
