# Benchmarks

Performance benchmarks compare Mesa Frames backends ("frames") with classic Mesa ("mesa")
implementations for a small set of representative models. They help track runtime scaling
and regressions.

Currently included models:

- **boltzmann**: Simple wealth exchange ("Boltzmann wealth") model.
- **sugarscape**: Sugarscape Immediate Growback variant (square grid sized relative to agent count).

## Quick start

```
uv run benchmarks/cli.py
```

That command (with defaults) will:

- Benchmark both models (`boltzmann`, `sugarscape`).
- Use agent counts 1000, 2000, 3000, 4000, 5000.
- Run 100 steps per simulation.
- Repeat each configuration once.
- Save CSV results and generate plots.

## CLI options

Invoke `uv run benchmarks/cli.py --help` to see full help. Key options:

| Option | Default | Description |
| ------ | ------- | ----------- |
| `--models` | `all` | Comma list or `all`; accepted: `boltzmann`, `sugarscape`. |
| `--agents` | `1000:5000:1000` | Single int or range `start:stop:step`. |
| `--steps` | `100` | Steps per simulation run. |
| `--repeats` | `1` | How many repeats per (model, backend, agents) config. Seed increments per repeat. |
| `--seed` | `42` | Base RNG seed. Incremented by repeat index. |
| `--save / --no-save` | `--save` | Persist per‑model CSVs. |
| `--plot / --no-plot` | `--plot` | Generate scaling plots (PNG + possibly other formats). |
| `--results-dir` | `benchmarks/results` | Root directory that will receive a timestamped subdirectory. |

Range parsing: `A:B:S` includes `A, A+S, ... <= B`. Final value > B is dropped.

## Output layout

Each invocation uses a single UTC timestamp, e.g. `20251016_173702`:

```
benchmarks/
  results/
    20251016_173702/
      boltzmann_perf_20251016_173702.csv
      sugarscape_perf_20251016_173702.csv
      plots/
        boltzmann_runtime_20251016_173702_dark.png
        sugarscape_runtime_20251016_173702_dark.png
        ... (other themed variants if enabled)
```

CSV schema (one row per completed run):

| Column | Meaning |
| ------ | ------- |
| `model` | Model key (`boltzmann`, `sugarscape`). |
| `backend` | `mesa` or `frames`. |
| `agents` | Agent count for that run. |
| `steps` | Steps simulated. |
| `seed` | Seed used (base seed + repeat index). |
| `repeat_idx` | Repeat counter starting at 0. |
| `runtime_seconds` | Wall-clock runtime for that run. |
| `timestamp` | Shared timestamp identifier for the benchmark batch. |

## Performance tips

- Ensure the environment variable `MESA_FRAMES_RUNTIME_TYPECHECKING` is **unset** or set to `0` / `false` when collecting performance numbers. Enabling it adds runtime type validation overhead and the CLI will warn you.
- Run multiple repeats (`--repeats 5`) to smooth variance.

## Extending benchmarks

To benchmark an additional model:

1. Add or import both a Mesa implementation and a Frames implementation exposing a `simulate(agents:int, steps:int, seed:int|None, ...)` function.
2. Register it in `benchmarks/cli.py` inside the `MODELS` dict with two backends (names must be `mesa` and `frames`).
3. Ensure any extra spatial parameters are derived from `agents` inside the runner lambda (see sugarscape example).
4. Run the CLI to verify new CSV columns still align.

## Related documentation

See `docs/user-guide/5_benchmarks.md` (user-facing narrative) and the main project `README.md` for overall context.