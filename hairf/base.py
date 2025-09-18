"""Base abstractions for reasoning modules."""

from __future__ import annotations

import abc
from typing import Iterable, Sequence

from .types import ModuleTrace, Query, ReasoningState


class ReasoningModule(abc.ABC):
    """Interface for all reasoning modules in HAIRF."""

    name: str = "base"

    def __init__(self) -> None:
        self._invocation_count = 0

    @property
    def invocation_count(self) -> int:
        """Return how many times the module has been executed."""

        return self._invocation_count

    def supports(self, query: Query) -> bool:
        """Return whether the module is applicable to ``query``.

        Subclasses can override to implement custom routing guards.
        """

        return True

    def run(self, query: Query, *, context: Sequence[ReasoningState] | None = None) -> ModuleTrace:
        """Execute the module and return a diagnostic trace.

        Subclasses must override :meth:`execute` to return states.  The default
        implementation wraps the results in a :class:`ModuleTrace`.
        """

        self._invocation_count += 1
        states = tuple(self.execute(query, context=context))
        info = {"invocation": self._invocation_count, "module": self.name}
        return ModuleTrace(module=self.name, states=list(states), info=info)

    @abc.abstractmethod
    def execute(
        self, query: Query, *, context: Sequence[ReasoningState] | None = None
    ) -> Iterable[ReasoningState]:
        """Yield :class:`ReasoningState` objects produced by the module."""

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"{self.__class__.__name__}(name={self.name!r})"
