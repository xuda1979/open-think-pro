"""Compute-optimal test-time scheduling inspired by Snell et al. (2024)."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import exp
from typing import Dict, Iterable, Mapping, Sequence

from .types import Query


@dataclass
class ComputePlan:
    """Describes how much compute to allocate for a given query."""

    budget: int
    difficulty: float
    allocation: Dict[str, float] = field(default_factory=dict)
    rationale: str = ""


class ComputeOptimalBudgeter:
    """Implements a lightweight analogue of compute-optimal scaling."""

    def __init__(
        self,
        *,
        min_budget: int = 1,
        max_budget: int = 4,
        thresholds: Sequence[tuple[float, int]] | None = None,
        smoothing: float = 0.3,
    ) -> None:
        if min_budget < 1:
            raise ValueError("min_budget must be positive")
        if max_budget < min_budget:
            raise ValueError("max_budget must be >= min_budget")
        self.min_budget = min_budget
        self.max_budget = max_budget
        self.thresholds = tuple(thresholds or ((0.25, min_budget), (0.55, min_budget + 1), (0.75, max_budget - 1)))
        self.smoothing = smoothing

    # Public API ---------------------------------------------------------
    def plan(
        self,
        query: Query,
        *,
        performance: Mapping[str, float],
        rules: Iterable[object],
    ) -> ComputePlan:
        difficulty = self._estimate_difficulty(query)
        budget = self._resolve_budget(difficulty)
        allocation = self._score_modules(performance, rules, difficulty)
        rationale = (
            f"compute-optimal budget={budget} diff={difficulty:.2f}"
        )
        return ComputePlan(budget=budget, difficulty=difficulty, allocation=allocation, rationale=rationale)

    # Internal helpers ---------------------------------------------------
    def _estimate_difficulty(self, query: Query) -> float:
        text = query.text.strip()
        if not text:
            return 0.0
        token_count = len(text.split())
        span_score = min(1.0, token_count / 48.0)
        punctuation_bonus = 0.1 if any(ch in text for ch in "?;:") else 0.0
        numeric_bonus = 0.1 if any(ch.isdigit() for ch in text) else 0.0
        task_bonus = 0.15 if query.task_type in {"math", "coding", "planning"} else 0.0
        raw = span_score + punctuation_bonus + numeric_bonus + task_bonus
        return max(0.0, min(1.0, raw))

    def _resolve_budget(self, difficulty: float) -> int:
        budget = self.min_budget
        for threshold, value in self.thresholds:
            if difficulty >= threshold:
                budget = max(budget, value)
        budget = min(budget, self.max_budget)
        return budget

    def _score_modules(
        self,
        performance: Mapping[str, float],
        rules: Iterable[object],
        difficulty: float,
    ) -> Dict[str, float]:
        # Higher difficulty -> concentrate on strong modules, lower -> uniform.
        temperature = max(0.05, self.smoothing * (1.0 - difficulty) + 0.1)
        scores: Dict[str, float] = {}
        raw_scores: Dict[str, float] = {}
        for rule in rules:
            name = getattr(rule, "module", getattr(rule, "name", None))
            module_name = getattr(name, "name", None) if hasattr(name, "name") else getattr(rule, "module_name", None)
            if module_name is None and hasattr(rule, "module"):
                module = getattr(rule, "module")
                module_name = getattr(module, "name", None)
            if module_name is None:
                continue
            perf = float(performance.get(module_name, 0.5))
            priority = float(getattr(rule, "priority", 1))
            raw = perf + 0.1 * (priority / 10.0)
            raw_scores[module_name] = raw
        if not raw_scores:
            return {}
        weights = self._softmax(raw_scores.values(), temperature)
        for (module_name, raw), weight in zip(raw_scores.items(), weights):
            scores[module_name] = weight
        return scores

    def _softmax(self, values: Sequence[float], temperature: float) -> list[float]:
        scaled = [v / temperature for v in values]
        max_val = max(scaled)
        exps = [exp(v - max_val) for v in scaled]
        total = sum(exps)
        if total == 0:
            uniform = 1.0 / len(values)
            return [uniform for _ in values]
        return [val / total for val in exps]


__all__ = ["ComputeOptimalBudgeter", "ComputePlan"]
