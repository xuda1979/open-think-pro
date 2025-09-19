"""High-level convenience functions for interacting with HAIRF."""

from __future__ import annotations

from typing import Any

from .framework import HAIRF
from .types import Query, ReasoningResult

_DEFAULT_FRAMEWORK: HAIRF | None = None


def get_default_framework(*, default_llm: Any | None = None, reset: bool = False) -> HAIRF:
    """Return a process-wide :class:`HAIRF` instance.

    Parameters
    ----------
    default_llm:
        Optional configuration used when instantiating the framework.  When
        provided alongside ``reset=True`` a new :class:`HAIRF` instance is
        created using the configuration.
    reset:
        If ``True`` the existing cached framework is discarded and a new one is
        created.  This is primarily useful for tests that need isolation.
    """

    global _DEFAULT_FRAMEWORK

    if reset or _DEFAULT_FRAMEWORK is None:
        _DEFAULT_FRAMEWORK = HAIRF(default_llm=default_llm)
    return _DEFAULT_FRAMEWORK


def answer_question(
    question: str,
    *,
    llm: Any | None = None,
    framework: HAIRF | None = None,
) -> ReasoningResult:
    """Run the HAIRF pipeline for ``question`` and return the result.

    Parameters
    ----------
    question:
        Natural-language question to be processed by the framework.  Leading
        and trailing whitespace is ignored and the question must be non-empty.
    llm:
        Optional LLM configuration passed through to :meth:`HAIRF.process`.
        Strings are interpreted as model identifiers, while dictionaries can
        provide richer configuration.
    framework:
        Optional pre-configured :class:`HAIRF` instance.  When omitted a
        singleton instance managed by :func:`get_default_framework` is used.
    """

    if not question or not question.strip():
        raise ValueError("question must be a non-empty string")

    hairf = framework if framework is not None else get_default_framework()
    query = Query(text=question.strip())
    return hairf.process(query, llm=llm)
