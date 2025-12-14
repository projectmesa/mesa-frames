# Sugarscape IG (Instant Growback)

## Overview

This directory contains a minimal Instant Growback Sugarscape implementation in
both backends:

- `backend_frames/` parallel (vectorised) movement variant using Mesa Frames
- `backend_mesa/` sequential (asynchronous) movement variant using classic Mesa

The Instant Growback (IG) rule sequence is: move -> eat -> regrow -> collect.
Agents harvest sugar, pay metabolism costs, possibly die (starve), and empty
cells instantly regrow to their `max_sugar` value.

## Concept

Each agent has integer traits:

- `sugar` (current stores)
- `metabolism` (per-step consumption)
- `vision` (how far the agent can see in cardinal directions)

Movement policy (both backends conceptually):

1. Sense visible cells along N/E/S/W up to `vision` steps (including origin).
2. Rank candidate cells by: (a) sugar (desc), (b) distance (asc), (c) coordinates
   as deterministic tie-breaker.
3. Choose highest-ranked empty cell; fall back to origin if none available.

The Frames parallel variant resolves conflicts by iterative lottery rounds using
rank promotion; the sequential Mesa variant inherently orders moves by shuffled
agent iteration.

After moving, agents harvest sugar on their cell, pay metabolism, and starved
agents are removed. Empty cells regrow to their `max_sugar` value immediately.

## Reported Metrics

Both backends record population-level reporters each step:

- `mean_sugar` Average sugar per surviving agent.
- `total_sugar` Aggregate sugar held by living agents.
- `agents_alive` Population size (declines as agents starve).
- `gini` Inequality in sugar holdings (0 = equal, higher = more unequal).
- `corr_sugar_metabolism` Pearson correlation (do high-metabolism agents retain sugar?).
- `corr_sugar_vision` Pearson correlation (does greater vision correlate with sugar?).

Notes on interpretation:

- `agents_alive` typically decreases until a quasi-steady state (metabolism vs regrowth) or total collapse.
- `mean_sugar` and `total_sugar` may stabilise if regrowth balances metabolism.
- Rising `gini` indicates emerging inequality; sustained high values suggest strong positional advantages.
- Correlations near 0 imply weak linear relationships; positive `corr_sugar_vision` suggests high-vision agents aid resource gathering. Negative `corr_sugar_metabolism` can emerge if high-metabolism agents accelerate starvation.

## Running

From project root using `uv`:

```bash
uv run examples/sugarscape_ig/backend_frames/model.py --agents 400 --width 40 --height 40 --steps 60 --seed 123 --plot --save-results
uv run examples/sugarscape_ig/backend_mesa/model.py --agents 400 --width 40 --height 40 --steps 60 --seed 123 --plot --save-results
```

## CLI options

- `--agents` Number of agents (default 400)
- `--width`, `--height` Grid dimensions (default 40x40)
- `--steps` Max steps (default 60)
- `--max-sugar` Initial/regrowth max sugar per cell (default 4)
- `--seed` Optional RNG seed
- `--plot / --no-plot` Generate per-metric plots
- `--save-results / --no-save-results` Persist CSV outputs
- `--results-dir` Override auto timestamped directory under `results/`

Frames backend warns if `MESA_FRAMES_RUNTIME_TYPECHECKING` is enabled (disable for benchmarks).

## Outputs

Example output directory (frames):

```text
examples/sugarscape_ig/backend_frames/results/20251016_173702/
  model.csv
  plots/
    gini_<timestamp>_dark.png
    agents_alive_<timestamp>_dark.png
    mean_sugar_<timestamp>_dark.png
    ...
```

`model.csv` columns include: `step`, `mean_sugar`, `total_sugar`, `agents_alive`,
`gini`, `corr_sugar_metabolism`, `corr_sugar_vision`, plus backend-specific bookkeeping.
Mesa backend normalises to the same layout (excluding internal columns).

## Performance & Benchmarking

Use the shared benchmarking CLI to compare scaling, checkout `benchmarks/README.md`.

## Programmatic Use

```python
from examples.sugarscape_ig.backend_frames import model as sg_frames
res = sg_frames.simulate(agents=500, steps=80, width=50, height=50, seed=42)
metrics = res.datacollector.data["model"]  # Polars DataFrame
```
