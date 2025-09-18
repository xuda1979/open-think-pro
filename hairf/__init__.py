"""Hybrid Adaptive Inference Reasoning Framework (HAIRF).

This package implements the core abstractions from the HAIRF paper, including
adaptive routing, quantum-inspired reasoning, and dynamic contextual memory
management.  The modules are designed to be composable so practitioners can
assemble complex inference stacks that blend symbolic, statistical, and
retrieval-based reasoning.
"""

from .types import Query, ReasoningResult, ReasoningState
from .base import ReasoningModule
from .qira import QIRAReasoner
from .dcmn import DynamicContextualMemoryNetwork
from .router import AdaptiveRouter, RoutingDecision
from .engine import ModularExecutionEngine
from .aggregator import OutputAggregator
from .framework import HAIRF
from .learning import ExperienceReplayLearner

__all__ = [
    "AdaptiveRouter",
    "DynamicContextualMemoryNetwork",
    "ExperienceReplayLearner",
    "HAIRF",
    "ModularExecutionEngine",
    "OutputAggregator",
    "QIRAReasoner",
    "Query",
    "ReasoningModule",
    "ReasoningResult",
    "ReasoningState",
    "RoutingDecision",
]
