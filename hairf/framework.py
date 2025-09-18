"""High-level Hybrid Adaptive Inference Reasoning Framework orchestration."""

from __future__ import annotations

from typing import Sequence

from .engine import ModularExecutionEngine
from .inference import LLMConfig, ensure_llm_config
from .learning import Experience, ExperienceReplayLearner
from .modules import ChainOfThoughtReasoner, ReflectiveCritic, SelfConsistencyReasoner
from .qira import QIRAReasoner
from .router import AdaptiveRouter, RouterRule
from .types import Query, ReasoningResult
from .dcmn import DynamicContextualMemoryNetwork


class HAIRF:
    """Reference implementation that wires together the HAIRF components."""

    def __init__(self, *, default_llm: LLMConfig | str | dict[str, object] | None = None) -> None:
        self.memory = DynamicContextualMemoryNetwork()
        self.router = AdaptiveRouter()
        self.engine = ModularExecutionEngine()
        self.learner = ExperienceReplayLearner()
        self.default_llm = ensure_llm_config(default_llm)
        self._register_default_modules()

    def _register_default_modules(self) -> None:
        self.modules = [
            ChainOfThoughtReasoner(),
            QIRAReasoner(),
            SelfConsistencyReasoner(),
            ReflectiveCritic(),
        ]
        keyword_map = {
            "cot": ("step", "calculate", "derive"),
            "qira": ("optimize", "explore", "design"),
            "self_consistency": ("verify", "proof", "explain"),
            "critic": ("check", "validate", "review"),
        }
        for module in self.modules:
            self.router.register(
                RouterRule(
                    module=module,
                    priority=10 if module.name == "qira" else 5,
                    keywords=keyword_map.get(module.name, ()),
                )
            )

    def process(
        self, query: Query, *, llm: LLMConfig | str | dict[str, object] | None = None
    ) -> ReasoningResult:
        llm_config = self._resolve_llm_config(query, llm)
        if llm_config is not None:
            query = query.enrich(llm_config=llm_config.as_dict())

        self.memory.ingest("latest_query", query.text, boost=0.1)
        decision = self.router.route(query)
        modules = self.router.modules_for_decision(decision, self.modules)
        memory_states = list(self.memory.contextualize(query))
        traces, states = self.engine.execute(query, modules, context=memory_states)
        states = memory_states + states
        result = self.engine.finalize(query, traces, states)
        self._update_learning(query, result, modules)
        return result

    def _update_learning(
        self, query: Query, result: ReasoningResult, modules: Sequence
    ) -> None:
        reward = result.confidence - result.used_tokens * 0.001
        for module in modules:
            self.router.update_performance(module.name, reward)
        self.learner.record(Experience(query=query, result=result, reward=reward))

    def summary(self) -> str:
        mean_reward, max_reward = self.learner.summarize()
        return (
            "HAIRF Summary:\n"
            f"  Modules: {[module.name for module in self.modules]}\n"
            f"  Mean reward: {mean_reward:.3f}\n"
            f"  Max reward: {max_reward:.3f}"
        )

    def _resolve_llm_config(
        self, query: Query, override: LLMConfig | str | dict[str, object] | None
    ) -> LLMConfig | None:
        if override is not None:
            return ensure_llm_config(override)
        existing = query.metadata.get("llm_config") if query.metadata else None
        if existing is not None:
            try:
                return ensure_llm_config(existing)
            except (TypeError, ValueError):
                return self.default_llm
        return self.default_llm
