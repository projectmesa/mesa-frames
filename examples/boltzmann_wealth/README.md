# Boltzmann Wealth Exchange Model

## Overview

This example implements a simple wealth exchange ("Boltzmann money") model in two
backends:

- `backend_frames.py` (Mesa Frames / vectorised `AgentSet`)
- `backend_mesa.py` (classic Mesa / object-per-agent)

Both expose a Typer CLI with symmetric options so you can compare correctness
and performance directly.

## Concept

Each agent starts with 1 unit of wealth. At every step:

1. Frames backend: all agents with strictly positive wealth become potential donors.
   Each donor gives 1 unit of wealth, and a recipient is drawn (with replacement)
   for every donating agent. A single vectorised update applies donor losses and
   recipient gains.
2. Mesa backend: agents are shuffled and iterate sequentially; each agent with
   positive wealth transfers 1 unit to a randomly selected peer.

The stochastic exchange process leads to an emergent, increasingly unequal
wealth distribution and rising Gini coefficient, typically approaching a stable
level below 1 (due to conservation and continued mixing).

## Reported Metrics

The model records per-step population Gini (`gini`). You can extend reporters by
adding lambdas to `model_reporters` in either backend's constructor.

Notes on interpretation:

- Early steps: Gini ~ 0 (uniform initial wealth).
- Mid phase: Increasing Gini as random exchanges concentrate wealth.
- Late phase: Fluctuating plateau (a stochastic steady state) — exact level
  varies with agent count and RNG seed.

## Running

Always run examples from the project root using `uv`:

```bash
uv run examples/boltzmann_wealth/backend_frames.py --agents 5000 --steps 200 --seed 123 --plot --save-results
uv run examples/boltzmann_wealth/backend_mesa.py --agents 5000 --steps 200 --seed 123 --plot --save-results
```

## CLI options

- `--agents` Number of agents (default 5000)
- `--steps` Simulation steps (default 100)
- `--seed` Optional RNG seed for reproducibility
- `--plot / --no-plot` Generate line plot(s) of Gini
- `--save-results / --no-save-results` Persist CSV metrics
- `--results-dir` Override the auto-timestamped directory under `results/`

Frames backend additionally warns if runtime type checking is enabled because it
slows vectorised operations: set `MESA_FRAMES_RUNTIME_TYPECHECKING=0` for fair
performance comparisons.

## Outputs

Each run creates (or uses) a results directory like:

```text
examples/boltzmann_wealth/results/20251016_173702/
  model.csv            # step,gini
  gini_<timestamp>_dark.png (and possibly other theme variants)
```

Tail metrics are printed to console for quick inspection:

```text
Metrics in the final 5 steps: shape: (5, 2)
┌──────┬───────┐
│ step ┆ gini  │
│ ---  ┆ ---   │
│ i64  ┆ f64   │
├──────┼───────┤
│ ...  ┆ ...   │
└──────┴───────┘
```

## Performance & Benchmarking

Use the shared benchmarking CLI to compare scaling, checkout `benchmarks/README.md`.

## Programmatic Use

```python
from examples.boltzmann_wealth import backend_frames as bw_frames
result = bw_frames.simulate(agents=10000, steps=250, seed=42)
metrics = result.datacollector.data["model"]  # Polars DataFrame
```
