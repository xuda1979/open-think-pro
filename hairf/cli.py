"""Command line interface for the HAIRF framework."""

from __future__ import annotations

import argparse
import json
from typing import Any, Iterable

from .api import answer_question, get_default_framework
from .types import ModuleTrace, ReasoningState


def _format_states(states: Iterable[ReasoningState]) -> str:
    lines = []
    for state in states:
        lines.append(f"- {state.content} (confidence={state.confidence:.2f}, cost={state.cost:.2f})")
    return "\n".join(lines)


def _format_traces(traces: Iterable[ModuleTrace]) -> str:
    rendered = []
    for trace in traces:
        header = f"Module: {trace.module}"
        rendered.append(header)
        if trace.states:
            rendered.append(_format_states(trace.states))
        if trace.info:
            rendered.append(f"  info: {json.dumps(trace.info, sort_keys=True)}")
    return "\n".join(rendered)


def run_cli(argv: list[str] | None = None) -> int:
    """Entry point used by ``python -m hairf`` and the console script."""

    parser = argparse.ArgumentParser(
        description=(
            "Run the Hybrid Adaptive Inference Reasoning Framework (HAIRF) for a "
            "single natural-language question."
        )
    )
    parser.add_argument(
        "--open-think-pro",
        "-q",
        dest="question",
        metavar="QUESTION",
        help="Question or prompt to send through the HAIRF pipeline.",
    )
    parser.add_argument(
        "--llm",
        dest="llm",
        metavar="LLM",
        help=(
            "Optional LLM configuration. Provide a model name or a JSON object "
            "with configuration settings."
        ),
    )
    parser.add_argument(
        "--show-traces",
        action="store_true",
        help="Include intermediate module traces in the output.",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print a performance summary gathered from the default framework.",
    )

    args = parser.parse_args(argv)

    if not args.question:
        parser.error("the following arguments are required: --open-think-pro")

    llm_config: Any | None = None
    if args.llm:
        llm_config = _parse_llm_argument(args.llm)

    result = answer_question(args.question, llm=llm_config)

    print("Answer:")
    print(result.answer)
    print()
    print(f"Confidence: {result.confidence:.2%}")
    print(f"Estimated tokens used: {result.used_tokens}")

    if args.show_traces and result.traces:
        print()
        print("Traces:")
        print(_format_traces(result.traces))

    if args.summary:
        framework = get_default_framework()
        print()
        print(framework.summary())

    return 0


def _parse_llm_argument(raw: str) -> Any:
    """Interpret ``raw`` as JSON when possible, otherwise return it verbatim."""

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return raw
    if isinstance(parsed, dict):
        if "model" in parsed and "provider" not in parsed:
            parsed = {**parsed, "provider": "custom"}
        return parsed
    return parsed


__all__ = ["run_cli"]
