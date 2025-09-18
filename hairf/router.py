"""Dynamic routing of reasoning modules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

from .base import ReasoningModule
from .types import Query, RoutingDecision


@dataclass
class RouterRule:
    """Simple rule describing when a module should be preferred."""

    module: ReasoningModule
    priority: int
    keywords: Tuple[str, ...] = ()
    task_types: Tuple[str, ...] = ()

    def matches(self, query: Query) -> bool:
        if self.task_types and (query.task_type not in self.task_types):
            return False
        if self.keywords:
            lowered = query.text.lower()
            return any(keyword in lowered for keyword in self.keywords)
        return True


class AdaptiveRouter:
    """Select reasoning modules based on query semantics and past performance."""

    def __init__(self) -> None:
        self._rules: List[RouterRule] = []
        self._performance: Dict[str, float] = {}

    def register(self, rule: RouterRule) -> None:
        self._rules.append(rule)
        self._rules.sort(key=lambda r: r.priority, reverse=True)

    def update_performance(self, module_name: str, reward: float) -> None:
        prev = self._performance.get(module_name, 0.0)
        self._performance[module_name] = 0.8 * prev + 0.2 * reward

    def route(self, query: Query, *, budget: int = 3) -> RoutingDecision:
        selected: List[str] = []
        rationale: List[str] = []
        estimated_cost = 0.0

        for rule in self._rules:
            if len(selected) >= budget:
                break
            if rule.matches(query) and rule.module.supports(query):
                score = self._performance.get(rule.module.name, 0.5)
                rationale.append(
                    f"selected {rule.module.name} (priority={rule.priority}, score={score:.2f})"
                )
                selected.append(rule.module.name)
                estimated_cost += score

        return RoutingDecision(
            selected_modules=selected,
            rationale="; ".join(rationale) or "default route",
            estimated_cost=estimated_cost,
        )

    def modules_for_decision(
        self, decision: RoutingDecision, modules: Sequence[ReasoningModule]
    ) -> List[ReasoningModule]:
        lookup = {module.name: module for module in modules}
        return [lookup[name] for name in decision.selected_modules if name in lookup]
