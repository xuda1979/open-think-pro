"""Dynamic Contextual Memory Network implementation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Sequence

from .types import Query, ReasoningState


@dataclass
class MemoryNode:
    """Represents a unit of contextual memory."""

    key: str
    content: str
    salience: float = 0.5
    metadata: Dict[str, float] = field(default_factory=dict)

    def update_salience(self, amount: float) -> None:
        self.salience = max(0.0, min(1.0, self.salience + amount))


class DynamicContextualMemoryNetwork:
    """Adaptive memory graph for retrieving task-relevant context."""

    def __init__(self) -> None:
        self._nodes: Dict[str, MemoryNode] = {}

    def ingest(self, key: str, content: str, *, boost: float = 0.0) -> MemoryNode:
        node = self._nodes.get(key)
        if node:
            node.content = content
            node.update_salience(boost)
        else:
            node = MemoryNode(key=key, content=content, salience=0.5 + boost)
            self._nodes[key] = node
        return node

    def decay(self, amount: float = 0.05) -> None:
        for node in self._nodes.values():
            node.update_salience(-amount)

    def retrieve(self, query: Query, *, top_k: int = 3) -> List[MemoryNode]:
        scored: List[tuple[float, MemoryNode]] = []
        lowered = query.text.lower()
        for node in self._nodes.values():
            overlap = self._keyword_overlap(lowered, node.content.lower())
            score = 0.7 * node.salience + 0.3 * overlap
            scored.append((score, node))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [node for _, node in scored[:top_k]]

    def contextualize(
        self, query: Query, *, top_k: int = 3
    ) -> Iterable[ReasoningState]:
        for node in self.retrieve(query, top_k=top_k):
            yield ReasoningState(
                content=f"Memory[{node.key}]: {node.content}",
                confidence=node.salience,
                cost=0.2,
                metadata={"memory_key": node.key},
            )

    def _keyword_overlap(self, query: str, content: str) -> float:
        q_tokens = set(query.split())
        c_tokens = set(content.split())
        if not q_tokens or not c_tokens:
            return 0.0
        return len(q_tokens & c_tokens) / len(q_tokens | c_tokens)
