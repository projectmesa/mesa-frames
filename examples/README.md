# Examples

This directory contains runnable example models and shared plotting/utilities
used in the tutorials and benchmarks. Each example provides **two backends**:

- `mesa` (classic Mesa, object-per-agent)
- `frames` (Mesa Frames, vectorised agent sets / dataframe-centric)

They expose a consistent Typer CLI so you can compare outputs and timings.

## Contents

```
examples/
  boltzmann_wealth/
    backend_mesa.py       # Mesa implementation + CLI (simulate() + run)
    backend_frames.py     # Frames implementation + CLI (simulate() + run)
  sugarscape_ig/
    backend_mesa/         # Mesa Sugarscape (agents + model + CLI)
    backend_frames/       # Frames Sugarscape (agents + model + CLI)
  plotting.py             # Shared plotting helpers (Seaborn + dark theme)
  utils.py                # Small dataclasses for simulation results
```

## Quick start

Always run via `uv` from the project root. The simplest way to run an example
backend is to execute the module:

```
uv run examples/boltzmann_wealth/backend_frames.py
```

Each command will:

1. Print a short banner with configuration.
2. Run the simulation and show elapsed time.
3. Emit a tail of the collected metrics (e.g. last 5 Gini values).
4. Save CSV metrics and optional plots in a timestamped directory under that
   example's `results/` folder (unless overridden by `--results-dir`).

## CLI symmetry

Both backends accept similar options:

- `--agents` (population size)
- `--steps` (number of simulated steps)
- `--seed` (optional RNG seed; Mesa backend resets model RNG)
- `--plot / --no-plot` (toggle plot generation)
- `--save-results / --no-save-results` (persist CSV outputs)
- `--results-dir` (override auto-created timestamped folder)

The Frames Boltzmann backend stores model metrics in a Polars DataFrame via
`mesa_frames.DataCollector`; the Mesa backend uses the standard `mesa.DataCollector`
returning pandas DataFrames, then converts to Polars only for plotting so plots
look identical.

## Data and metrics

The saved CSV layout (Frames) places `model.csv` in the results directory with
columns like: `step, gini, <other reporters...>`.
The Mesa implementations write
compatible CSVs.

## Plotting helpers

`examples/plotting.py` provides:

- `plot_model_metrics(df, output_dir, stem, title, subtitle, agents, steps)`
  Produces dark theme line plots of model-level metrics (currently Gini) and
  stores PNG files under `output_dir` with names like `gini_<timestamp>_dark.png`.
- `plot_performance(df, output_dir, stem, title)` used by `benchmarks/cli.py` to
  generate runtime scaling plots.

The dark theme matches the styling used in the documentation for visual
consistency.

## Interacting programmatically

Instead of using the CLIs you can import the simulation entry points directly:

```python
from examples.boltzmann_wealth import backend_frames as bw_frames
result = bw_frames.simulate(agents=2000, steps=100, seed=123)
polars_df = result.datacollector.data["model"]  # Polars DataFrame of metrics
```

Each `simulate()` returns a small dataclass (`FramesSimulationResult` or
`MesaSimulationResult`) holding the respective `DataCollector` instance so you
can further analyse the collected data.

## Tips

- To compare backends fairly, disable runtime type checking when measuring performance:
  set environment variable `MESA_FRAMES_RUNTIME_TYPECHECKING=0`.
- Use the same `--seed` across runs for reproducible trajectories (given the
  stochastic nature of agent interactions).
- Larger Sugarscape grids (width/height) increase memory and runtime; choose
  sizes proportional to the square root of agent count for balanced density.

## Adding Examples

You can adapt these scripts to prototype new models: copy a backend pair,
rename the module, and implement your agent rules while keeping the API
surface (`simulate`, `run`) consistent so tooling and documentation patterns
continue to apply.
