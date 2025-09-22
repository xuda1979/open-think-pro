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

### 1. Install dependencies

The reference implementation only relies on the Python standard library for its core logic, so you can get started with any modern Python (>=3.10) interpreter. We still recommend working inside a virtual environment to keep experiments isolated:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt  # optional: supply your own requirements file for additional tools
pip install pytest              # install locally if you want to run the test suite
```

### 2. Configure API credentials

HAIRF can call different hosted LLM providers via the thin wrappers in [`hairf/inference.py`](hairf/inference.py). Each client expects an API key to be supplied through environment variables. Set the variables for the providers you plan to use before running any scripts:

| Provider | Required environment variable | Optional base URL override | Notes |
|----------|-------------------------------|-----------------------------|-------|
| OpenAI   | `OPENAI_API_KEY`              | `OPENAI_API_BASE`           | Uses the `/chat/completions` endpoint with bearer authentication. |
| Gemini   | `GEMINI_API_KEY` (falls back to `GOOGLE_API_KEY`) | `GEMINI_API_BASE`           | Key is passed as a query parameter; bearer token is not required. Pass either a bare model name (e.g. `gemini-2.0-flash`) or a full resource path such as `tunedModels/your-model-id`. |
| DeepSeek | `DEEPSEEK_API_KEY`            | `DEEPSEEK_API_BASE`         | Default base URL is `https://api.deepseek.com/v1`. |
| Qwen     | `QWEN_API_KEY`                | `QWEN_API_BASE`             | Targets the DashScope text-generation endpoint. |

For example, on macOS/Linux you can export variables in your shell session:

```bash
export OPENAI_API_KEY="sk-your-key"
export DEEPSEEK_API_KEY="ds-your-key"
```

When testing locally without network access, set the `offline` option in an `LLMConfig` to bypass the HTTP call and return a stub response.

### 3. Run the test suite

```bash
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

## Usage Example

Below is a minimal snippet that wires the components together, providing a concrete example of how to prepare a query and request a completion from OpenAI (the same pattern applies to Gemini, DeepSeek, or Qwen by changing the provider/model fields):

```python
from hairf.framework import HAIRF
from hairf.inference import LLMConfig

framework = HAIRF()

query = {
    "prompt": "Summarise the main differences between transformers and RNNs.",
    "llm": LLMConfig(provider="openai", model="gpt-4o", options={"temperature": 0.2}),
}

result = framework.run(query)
print(result.answer)
```

`HAIRF.run` will automatically invoke DCMN for context retrieval, compute the appropriate budget, route across the configured reasoning modules, and call the specified LLM client using the credentials provided through environment variables.

## References

- Snell, C., Lee, J., Xu, K., & Kumar, A. (2024). *Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters*. arXiv:2408.03314.
- Additional related work citations are collected in [`paper.tex`](paper.tex).
