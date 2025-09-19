# Hybrid Adaptive Inference Reasoning Framework (HAIRF)

HAIRF is a modular research stack for building composite reasoning pipelines around large language models (LLMs). It blends classical prompting heuristics with novel controllers such as Quantum-Inspired Reasoning Alignment (QIRA), Dynamic Contextual Memory Networks (DCMN), and an adaptive router that follows compute-optimal scheduling insights from *Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters* (Snell et al., 2024).

The repository accompanies the reference paper in [`paper.tex`](paper.tex) and exposes a minimal Python implementation that can be wired into custom LLM backends.

## Core Algorithms

### Adaptive Routing with Compute-Optimal Budgets
- **Difficulty estimation.** `hairf.compute_optimal.ComputeOptimalBudgeter` scores each query using token-span, punctuation, numeric, and task-type signals to approximate the hardness heuristics proposed by Snell et al. Harder queries receive scores close to 1.0, while trivial requests collapse toward 0.0.
- **Budget selection.** Difficulty is mapped to an execution budget between a configurable minimum and maximum. Budgets represent the number of reasoning modules the router should activate for the query.
- **Allocation weights.** Module performance priors and rule priorities are transformed through a tempered softmax to emphasize strong specialists when compute is scarce. The resulting `ComputePlan` drives downstream selection.
- **Routing decision.** `hairf.router.AdaptiveRouter` consumes the plan, assembles the highest-value modules, and records the rationale, estimated cost, and realized budget inside a `RoutingDecision`. Manual overrides can still supply fixed budgets when needed.

### Quantum-Inspired Reasoning Alignment (QIRA)
- Maintains a superposition of hypotheses, allowing constructive interference before collapse.
- Uses configurable generators and collapse policies so teams can experiment with different exploration strategies.
- Particularly effective for multi-step synthesis and planning tasks that benefit from retaining multiple intermediate lines of thought.

### Dynamic Contextual Memory Network (DCMN)
- Tracks recent context in a salience-weighted memory graph.
- Boosts or decays edges based on query features, surfacing the most relevant snippets for subsequent modules.
- Provides retrieved context as `ReasoningState` objects that can be fed directly into QIRA or other reasoning modules.

### Modular Execution and Aggregation
- `hairf.engine.ModularExecutionEngine` executes the chosen modules sequentially, sharing context and metadata.
- `hairf.aggregator.OutputAggregator` synthesizes module traces into a final answer while retaining diagnostic information.
- The framework exposes hooks for plugging in proprietary LLMs through `hairf.inference` helpers.

### Continual Learning Loop
- `hairf.learning.ExperienceReplayLearner` records experiences (query, result, reward) and updates the router’s performance priors.
- Rewards combine confidence and compute cost, encouraging routes that stay accurate while respecting the compute-optimal budgets.

## Putting It Together

`hairf.framework.HAIRF` orchestrates the components:
1. Normalize or enrich the incoming `Query` with LLM configuration metadata.
2. Use DCMN to retrieve contextual memories.
3. Ask the adaptive router for a `RoutingDecision`; the compute-optimal scheduler determines the budget when not manually provided.
4. Execute the selected modules via the execution engine, aggregate traces, and produce a `ReasoningResult`.
5. Update router priors and the experience replay buffer based on observed reward.

The routing state (including budget, difficulty, allocation weights, and textual rationale) is surfaced as a `ReasoningState` so downstream consumers can audit how compute was spent per query.

## Repository Structure

```
hairf/                Core implementation modules
├── compute_optimal.py  – Compute-optimal scheduler utilities
├── router.py           – Adaptive router that consumes compute plans
├── modules.py          – Reference reasoning modules (CoT, self-consistency, critic)
├── qira.py             – Quantum-inspired reasoning alignment module
├── dcmn.py             – Dynamic contextual memory network
├── framework.py        – High-level HAIRF orchestrator
└── ...                 – Additional engine, inference, and learning helpers
tests/               Unit tests (router regression currently verifies budget scaling)
paper.tex            LaTeX source of the accompanying paper
```

## Getting Started

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt  # (not required for the standard-library-only demo)
pytest
```

### Command-line usage

The project exposes a lightweight CLI so you can run the framework without
writing Python code:

```bash
python -m hairf --open-think-pro "How does adaptive routing allocate compute?"
```

Use `--show-traces` to include per-module traces or `--summary` to print the
running performance summary collected by the default framework instance.

### Python API convenience helpers

When importing the package you can call `hairf.answer_question` to route a
question through a cached :class:`hairf.framework.HAIRF` instance:

```python
from hairf import answer_question

result = answer_question("Explain compute-optimal scheduling.")
print(result.answer)
```

The unit tests validate that the router increases compute allocation for harder prompts, ensuring that the compute-optimal scheduler is wired into the decision flow.

## References

- Snell, C., Lee, J., Xu, K., & Kumar, A. (2024). *Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters*. arXiv:2408.03314.
- Additional related work citations are collected in [`paper.tex`](paper.tex).
