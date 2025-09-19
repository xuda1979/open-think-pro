"""Dynamic routing of reasoning modules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

from .base import ReasoningModule
from .compute_optimal import ComputeOptimalBudgeter, ComputePlan
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

    def __init__(self, *, scheduler: ComputeOptimalBudgeter | None = None) -> None:
        self._rules: List[RouterRule] = []
        self._performance: Dict[str, float] = {}
        self._scheduler = scheduler or ComputeOptimalBudgeter()

    def register(self, rule: RouterRule) -> None:
        self._rules.append(rule)
        self._rules.sort(key=lambda r: r.priority, reverse=True)

    def update_performance(self, module_name: str, reward: float) -> None:
        prev = self._performance.get(module_name, 0.0)
        self._performance[module_name] = 0.8 * prev + 0.2 * reward

    def route(self, query: Query, *, budget: int | None = None) -> RoutingDecision:
        manual_budget = budget
        plan: ComputePlan | None = None
        if budget is None and self._scheduler is not None:
            plan = self._scheduler.plan(
                query,
                performance=self._performance,
                rules=self._rules,
            )
            budget = plan.budget
        if budget is None:
            budget = 3

        candidates: List[tuple[float, RouterRule, float, float]] = []
        for rule in self._rules:
            if not rule.matches(query):
                continue
            if not rule.module.supports(query):
                continue
            perf_score = self._performance.get(rule.module.name, 0.5)
            allocation = plan.allocation.get(rule.module.name, 0.0) if plan else 1.0
            combined = perf_score * (1.0 + allocation) + rule.priority / 10.0
            candidates.append((combined, rule, perf_score, allocation))

        candidates.sort(key=lambda item: item[0], reverse=True)

        selected: List[str] = []
        rationale: List[str] = []
        estimated_cost = 0.0

        if plan is not None:
            rationale.append(plan.rationale)
        elif manual_budget is not None:
            rationale.append(f"fixed budget={budget}")

        for combined, rule, perf_score, allocation in candidates:
            if len(selected) >= budget:
                break
            selected.append(rule.module.name)
            estimated_cost += perf_score + allocation
            rationale.append(
                (
                    f"selected {rule.module.name} "
                    f"(score={perf_score:.2f}, weight={allocation:.2f}, priority={rule.priority})"
                )
            )

        allocation = {name: (plan.allocation.get(name, 0.0) if plan else 0.0) for name in selected}
        if estimated_cost == 0.0:
            baseline = plan.difficulty if plan else 0.5
            estimated_cost = max(0.1, float(budget) * max(0.1, baseline))

        return RoutingDecision(
            selected_modules=selected,
            rationale="; ".join(rationale) or "default route",
            estimated_cost=estimated_cost,
            budget=budget,
            difficulty=plan.difficulty if plan else 0.0,
            allocation=allocation,
        )

    def modules_for_decision(
        self, decision: RoutingDecision, modules: Sequence[ReasoningModule]
    ) -> List[ReasoningModule]:
        lookup = {module.name: module for module in modules}
        return [lookup[name] for name in decision.selected_modules if name in lookup]
