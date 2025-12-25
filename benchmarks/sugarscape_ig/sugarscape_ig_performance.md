# Sugarscape IG performance work (performance-improvement-sugarscape)

This document explains, in detail, what changed on the `performance-improvement-sugarscape` branch to accelerate the Sugarscape IG example, and what parts are (a) local to the example, (b) generally useful mesa-frames improvements that can be upstreamed, and (c) a standalone “benchmark engine” (NumPy/Numba) that bypasses mesa-frames entirely.

It also includes measured timings and a component-by-component attribution based on controlled flag toggles.

## Scope and goals

Sugarscape IG is a **capacity=1**, **synchronous parallel move** model with a “ranked candidate moves + conflict resolution” step.

The original bottleneck in the mesa-frames implementation was dominated by repeated high-level DataFrame work inside the step loop:

- Building neighborhood/candidate tables (many rows per agent).
- Joining candidate tables to cell sugar.
- Grouping/sorting by agent + by candidate cell.
- Conflict resolution using repeated rebuild/filter/join passes.
- Per-step agent shuffles, and per-step Polars writes for sugar/cell updates.

The branch work attacked those costs in two ways:

1. **Make the mesa-frames path as fast as possible** while preserving semantics.
2. Provide a separate **ultra-fast benchmark engine** that removes Polars/mesa-frames overhead entirely (for pure performance benchmarking).

## Repository separation: what lives where

### Standalone ultra-fast engine (bypasses mesa-frames)

- [benchmarks/sugarscape_ig/numpy_engine.py](benchmarks/sugarscape_ig/numpy_engine.py)

This is “the super fast version”. It does *not* attempt to be a reusable mesa-frames backend; it’s a specialized benchmark kernel.

### Sugarscape example optimizations (mesa-frames still used)

- [examples/sugarscape_ig/backend_frames/agents.py](examples/sugarscape_ig/backend_frames/agents.py)
- [examples/sugarscape_ig/backend_frames/model.py](examples/sugarscape_ig/backend_frames/model.py)

These implement the high-performance conflict resolver(s), cached NumPy buffers, and eat/regrow fast paths.

### mesa-frames core improvements (generalizable)

- [mesa_frames/abstract/space/grid.py](mesa_frames/abstract/space/grid.py)

This adds **Grid full-move fast paths** that avoid expensive Polars operations when moving all agents every step.

> Note: this branch also includes a larger refactor where the space implementation lives under `mesa_frames/abstract/space/…` rather than a single module. The Sugarscape performance work described here is compatible with either layout; upstreaming to `main` would apply the same logic in whatever file defines `Grid._place_or_move_agents` on `main`.

## How to run each variant

### Baseline mesa-frames / Polars engine (optimized code paths available)

Use the standard simulation entrypoint:

- [examples/sugarscape_ig/backend_frames/model.py](examples/sugarscape_ig/backend_frames/model.py)

Key environment variables:

- `MESA_FRAMES_SUGARSCAPE_ENGINE`
  - `frames` = run through mesa-frames/Polars
  - `numpy` = run the standalone ultra-fast engine
- `MESA_FRAMES_SUGARSCAPE_DISABLE_DATACOLLECTOR`
  - set to `1` to disable data collection for benchmark-mode. Several fast paths require this.

Conflict resolver selection (mesa-frames path):

- `MESA_FRAMES_SUGARSCAPE_CONFLICT_RESOLVER`
  - `rounds` = original round-based DataFrame approach
  - `kernel` = NumPy/Numba conflict resolution kernel but still DataFrame candidate generation
  - `full_kernel` = full-kernel path (NumPy/Numba candidate generation + conflict resolution)

Grid move safety/fast path:

- `MESA_FRAMES_GRID_TRUST_FULL_MOVE=1`
  - enables a fast path that skips a per-step membership check when moving *all agents*.
  - safe for Sugarscape because the model moves all placed agents each step.

Sugar/regrow fast paths (mesa-frames path):

- `MESA_FRAMES_SUGARSCAPE_ULTRA_FAST=1`
  - enables cached flat sugar buffer and bulk writeback/regrow fast paths.
- `MESA_FRAMES_SUGARSCAPE_SUGAR_MODEL` (currently used by the example)
  - `buffer` (default) = maintain a `sugar_flat` buffer aligned to cell ids
  - `stamps` = experimental stamp-based model (measured slower in the current configuration)
- `MESA_FRAMES_SUGARSCAPE_REGROW_MODEL` (currently used by the example)
  - `bulk` (default) = refill empty cells in bulk
  - `delta` = refill only cells that became empty this step

### Ultra-fast NumPy engine

- `MESA_FRAMES_SUGARSCAPE_ENGINE=numpy`
- `MESA_FRAMES_SUGARSCAPE_DISABLE_DATACOLLECTOR=1` (required; the engine intentionally does not produce datacollector output)

The engine is intended to be used directly from the benchmarks folder (it is not imported by the example code).

For performance runs, prefer `python -O` so `__debug__` checks are skipped.

## Benchmarks: headline numbers (50k agents, 100 steps)

All measurements below were run with:

- `agents=50000`, `steps=100`, `width=448`, `height=448`, `seed=11`
- `POLARS_MAX_THREADS=1`, `NUMBA_NUM_THREADS=1` (to reduce noise and keep comparisons stable)
- A short warmup run was executed before timing to avoid first-call JIT overhead.

### Cross-implementation comparison

- Mesa (default branch `main`, Mesa backend): **24.42s**
- mesa-frames (default branch `main`, frames backend): **31.07s**
- mesa-frames (this branch, frames backend, optimized flags): **19.75s** (see attribution table)
- Ultra-fast NumPy engine (this branch): **0.090s**

> The NumPy engine time is after warmup/JIT. It is an intentionally specialized benchmark kernel.

## Attribution: which part improved what (measured)

These are measured on this branch with `MESA_FRAMES_SUGARSCAPE_ENGINE=frames` and `MESA_FRAMES_SUGARSCAPE_DISABLE_DATACOLLECTOR=1`.

Each row changes *one major layer* (resolver / grid move path / sugar-regrow path), so you can see what each feature buys.

| Configuration | Seconds | Speedup vs rounds (trust=1, ultra=0) |
| --- | ---: | ---: |
| `resolver=rounds`, `trust_full_move=0`, `ultra_fast=0` | 19.84 | 0.996× |
| `resolver=rounds`, `trust_full_move=1`, `ultra_fast=0` | 19.75 | 1.000× |
| `resolver=kernel`, `trust_full_move=1`, `ultra_fast=0` | 17.90 | 1.10× |
| `resolver=full_kernel`, `trust_full_move=1`, `ultra_fast=0` | 5.55 | 3.56× |
| `resolver=full_kernel`, `trust_full_move=1`, `ultra_fast=1` | 1.84 | 10.71× |
| `resolver=full_kernel`, `trust_full_move=0`, `ultra_fast=1` | 1.91 | 10.33× |

Interpretation:

- Moving from DataFrame-based conflict resolution (`rounds`) to the full Numba “allocation kernel” (`full_kernel`) is the dominant win.
- `ultra_fast=1` then removes the remaining Polars-heavy sugar/cell updates, cutting ~5.55s → ~1.84s.
- `MESA_FRAMES_GRID_TRUST_FULL_MOVE=1` is a smaller but real win at the “already very fast” level (~1.91s → ~1.84s).

### Sugar/regrow model variants under `full_kernel + ultra_fast`

Measured with `resolver=full_kernel`, `ultra_fast=1`, `trust_full_move=1`:

| Sugar model | Regrow model | Seconds |
| --- | --- | ---: |
| `buffer` | `bulk` | 1.845s |
| `buffer` | `delta` | 1.808s |
| `stamps` | `bulk` | 2.229s |
| `stamps` | `delta` | 2.128s |

Interpretation:

- `delta` regrow is a small win vs `bulk` in this configuration.
- `stamps` is slower than the `buffer` approach here (likely because the current step still needs cell ids and array indexing, so the stamp model doesn’t reduce enough work to offset its extra bookkeeping).

## Detailed explanation: mesa-frames path improvements

This section explains *exactly* what changed in the mesa-frames execution path, why it was slow before, and what the branch does instead.

### 1) Stop shuffling every step for “parallel semantics” resolvers

Location: [examples/sugarscape_ig/backend_frames/agents.py](examples/sugarscape_ig/backend_frames/agents.py)

In synchronous parallel movement, agent ordering should not affect correctness.

- For `resolver` in `{kernel, full_kernel}`, the code **skips `self.shuffle()`**.
- For `resolver=rounds`, shuffle is preserved to match the original implementation’s assumptions.

Why this matters:

- `shuffle(inplace=True)` is a substantial Polars operation.
- Skipping shuffle also keeps cached NumPy buffers aligned with the AgentSet row ordering across steps.

### 2) Replace join-heavy “eat” implementation with position-join + cached arrays

Locations:

- [examples/sugarscape_ig/backend_frames/agents.py](examples/sugarscape_ig/backend_frames/agents.py)

Key changes:

- The base `eat()` path was made more direct by joining **agent positions** to `cells(include="properties")` to fetch sugar, instead of scanning/filtering the full cells table.
- For `full_kernel`, `eat()` becomes mostly NumPy work:
  - positions are cached as `pos_dim0/pos_dim1` aligned with the AgentSet
  - sugar is read from `sugar_flat[cell_id]`
  - metabolism/sugar stocks are cached NumPy arrays
  - if the datacollector is disabled, the code **does not write the new sugar values back to Polars every step**

Why this matters:

- The original approach paid repeated DataFrame costs: `cells.filter(is_in(...))`, joins, and `with_columns`.
- Under `full_kernel`, the “per-agent update” is reduced to a few vectorized NumPy ops.

### 3) Conflict resolution: from DataFrame rounds → Numba kernel → full-kernel

Location: [examples/sugarscape_ig/backend_frames/agents.py](examples/sugarscape_ig/backend_frames/agents.py)

There are three resolver families:

#### `rounds` (DataFrame-heavy)

- Builds neighborhood/candidate tables in Polars.
- Resolves conflicts with per-round DataFrame joins/filters.

This is conceptually clean but expensive.

#### `kernel` (mixed approach)

- Still uses Polars to build the candidate table.
- Converts candidate information into compact arrays.
- Runs a Numba conflict kernel to decide winners/losers in O(proposals) time.

Measured improvement: **~19.75s → ~17.90s**.

#### `full_kernel` (full-kernel approach)

- Avoids the global neighborhood/candidate DataFrame construction.
- Generates each agent’s ranked candidate list directly into arrays.
- Resolves conflicts using a Numba kernel.
- Produces destination positions as NumPy arrays aligned with the AgentSet.

Measured improvement (vs `rounds`): **~19.75s → ~5.55s**.

### 4) Cache aligned NumPy buffers to avoid repeated conversions

Location: [examples/sugarscape_ig/backend_frames/agents.py](examples/sugarscape_ig/backend_frames/agents.py)

The full-kernel path keeps a `self._full_kernel_cache` dict with:

- `agent_ids_np`, `pos_dim0`, `pos_dim1` (aligned to the AgentSet)
- `metabolism_np`, `sugar_np`
- `origin_cell_id`, `dest_cell_id` (cell ids for origin/destination)

Why this matters:

- Converting Polars Series to NumPy arrays is not free.
- Re-building `self.pos` is particularly expensive (it often triggers joins/collect).
- Once the model is in “benchmark mode” (no datacollector, no shuffle), the ordering is stable enough to reuse these buffers.

### 5) Ultra-fast sugar/cell updates: keep a flat sugar buffer and bulk-write once

Locations:

- [examples/sugarscape_ig/backend_frames/agents.py](examples/sugarscape_ig/backend_frames/agents.py)
- [examples/sugarscape_ig/backend_frames/model.py](examples/sugarscape_ig/backend_frames/model.py)

When `MESA_FRAMES_SUGARSCAPE_ULTRA_FAST=1`:

- `eat()` updates `sugar_flat[cell_id]=0` for consumed cells.
- The model’s regrow step can then:
  - update `sugar_flat` for empty cells
  - and do a **single Polars column writeback** (or skip it entirely depending on mode)

Why this matters:

- Per-step `space.cells.update(...)` calls are expensive at large scale.
- A single `with_columns(pl.Series("sugar", ...))` is much cheaper than many row-based updates.

Measured improvement (under `full_kernel`): **~5.55s → ~1.84s**.

### 6) Optional delta regrow

Location: [examples/sugarscape_ig/backend_frames/agents.py](examples/sugarscape_ig/backend_frames/agents.py)

When `MESA_FRAMES_SUGARSCAPE_REGROW_MODEL=delta` and `ultra_fast=1`:

- Only cells that became empty due to movement are refilled.

Measured effect in this configuration: modest improvement (1.845s → 1.808s).

### 7) Starvation removal fast path

Location: [examples/sugarscape_ig/backend_frames/agents.py](examples/sugarscape_ig/backend_frames/agents.py)

- If `eat()` already computed `sugar_np`, `_remove_starved()` checks `sugar_np.min()`.
- If no one starved, it returns immediately (avoids a Polars filter scan).
- If starvation occurs, it discards by id.

Why this matters:

- In many parameter regimes, starvation is rare, so the early-exit avoids repeated DataFrame work.

## Detailed explanation: mesa-frames core improvement (Grid full-move fast path)

Location: [mesa_frames/abstract/space/grid.py](mesa_frames/abstract/space/grid.py)

Sugarscape moves **all agents** every step.

The default `Grid._place_or_move_agents` path is generic and correct, but it does a lot of DataFrame construction and may call `combine_first`.

### New behavior

When moving all agents:

1) If `pos` is a stacked NumPy array shaped `(n_agents, n_dims)`, the grid:

- optionally checks membership (`agents.is_in(self._agents["agent_id"])`) unless `MESA_FRAMES_GRID_TRUST_FULL_MOVE=1`
- wraps/validates coordinates
- recomputes capacity via `np.bincount(cell_id)`
- rebuilds the agents table in one shot

1) If `pos` is provided as a tuple/list of per-dimension arrays `(dim0, dim1, ...)`, the grid performs the same operations without allocating a stacked `(n, d)` array.

Why this matters:

- It removes repeated Polars-heavy “partial update” logic from the hot path.
- It allows high-performance kernels to pass raw NumPy arrays directly into the grid.

Measured contribution in this benchmark setup:

- With full-kernel + ultra-fast, turning trust off → on is ~1.91s → ~1.84s.

## Detailed explanation: the ultra-fast NumPy/Numba engine

Location: [examples/sugarscape_ig/backend_frames/numpy_engine.py](examples/sugarscape_ig/backend_frames/numpy_engine.py)

This engine is designed for one purpose: **absolute maximum step throughput** for Sugarscape IG-like rules.

### Key design choices

- No Polars DataFrames in the step loop.
- No mesa-frames Space/Cells APIs in the step loop.
- Flat arrays for agent state:
  - position (`dim0`, `dim1`)
  - sugar stock, metabolism, vision
- Flat arrays for grid state:
  - `sugar_flat`
  - occupancy arrays
- A deterministic, fast RNG (SplitMix64) used inside Numba for conflict lotteries.

### Conflict semantics

The conflict resolution kernel follows the same conceptual rules as the Polars rounds implementation:

- each round, each unresolved agent proposes its best available candidate
- each cell selects one proposer uniformly at random
- winners claim the cell; losers advance rank and try again
- when a winner moves away, its origin becomes available for later rounds

### Why it is so fast

It removes almost all overhead that remains even in the optimized mesa-frames path:

- No DataFrame allocations
- No joins
- No group-bys
- No object-model overhead
- Pure O(n_agents * candidates) array work

Measured time: **~0.090s for 50k×100** after warmup.

### Limitations (intentional)

- Requires `MESA_FRAMES_SUGARSCAPE_DISABLE_DATACOLLECTOR=1`.
- Intended for benchmarking; not a general mesa-frames engine.
- Any additional features (agent heterogeneity, richer datacollection, extra agent attributes) will reduce the speed advantage.

## What we can bring back to `main` (recommended upstream candidates)

### High-confidence upstream candidates (general and reusable)

1) Grid full-move fast path + trust toggle

- The logic in [mesa_frames/abstract/space/grid.py](mesa_frames/abstract/space/grid.py) is generally useful for any model that moves all agents each step.
- Keep `MESA_FRAMES_GRID_TRUST_FULL_MOVE` default-off to preserve safety.

1) Accept per-dimension NumPy arrays for full-move

- Avoids allocating stacked `(n, d)` position arrays.
- Useful whenever a model computes destination coordinates as separate arrays.

### Example-only candidates (keep in Sugarscape example)

1) `full_kernel` resolver and caching

- It’s highly specific to Sugarscape IG rules and candidate structure.
- Can remain an example “performance mode” without complicating core APIs.

1) Ultra-fast sugar buffer + bulk writeback

- This depends on specific `cells` layouts and on being able to treat the grid as dense row-major.
- It is safe as an example optimization but not yet a general core feature.

### Not recommended for upstream as-is

- The standalone NumPy engine: it’s great for benchmarking, but it bypasses the library abstractions and doesn’t integrate with general mesa-frames features.

## Practical recommendation for upstreaming

If the goal is to make `main` mesa-frames materially faster for *real* models:

1) Upstream Grid full-move fast paths (core).
2) Add optional “array-first” hooks in Spaces/Cells only if needed (careful: complexity).
3) Keep the NumPy engine as a benchmark reference implementation.

## Appendix: exact benchmark scripts used

The timings in this document were collected by running `simulate(...)` with environment flags set (see tables above). If you want a reproducible script, the same runs can be executed by importing the models and timing them with `time.perf_counter()`.
