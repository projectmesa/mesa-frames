# Fast Update + Lookup Plan (No-Join Polars) + Buffer Branch Split

## Goals

- Keep the public API **simple**: `update(...)` for both `GridCells` and `AgentSet`.
- Implement **no-join** updates where possible (dense, row-major, mask alignment).
- Add a **lookup(...)** method for both `GridCells` and `AgentSet` to fetch rows by key without joins.
- Move current **NumPy buffer implementation** to a secondary branch; **no buffers in main** until lazy/deferred sync exists.
- Benchmark before/after to quantify impact.

## High-level strategy

1) **Main branch**: no buffers. Implement `update` + `lookup` with a shared mixin for *masked update application* (not mask normalization).
2) **Secondary branch**: keep buffer implementation for future lazyframes/deferred sync work.
3) Prioritize **no-join Polars paths**:
   - Use boolean masks aligned with row-major order.
   - Avoid joins by mapping coords → `cell_id` → boolean mask.

---

## Part A — Branch split (buffers)

### A1) Create a secondary branch

- Branch name suggestion: `feature/buffers-deferred-sync`.
- Move all buffer-related code to that branch:
  - `mesa_frames/concrete/_buffers.py`
  - `_BufferedColumnsMixin` usage in `AgentSet` and `GridCells`
  - Any calls to `_ensure_buffer`, `_sync_buffer`, `_invalidate_buffers`
- Main branch should not import `_buffers` or depend on it.

### A2) Remove buffer logic from main

- Replace buffer-based fast paths with Polars no-join path logic (see Part C).
- Ensure all tests still pass.

---

## Part B — Common API additions

### B1) Add `update(...)` to AgentSet + GridCells

- Public method name: `update`.
- Remove `set` method completely (breaking change accepted).
- `update` signature (both classes):

```python
update(
    self,
    target: ... | dict[str, object] | None = None,
    updates: dict[str, object] | None = None,
    *,
    mask: str | DataFrame | Series | np.ndarray | None = None,
    backend: Literal["auto", "polars"] = "auto",
    mask_col: str | None = None,
) -> None
```

- Accepted update values:
  - Scalars
  - Array-like (list/np.ndarray/pl.Series)
  - Column-name string (copy from other column)
  - `pl.Expr` (forces polars backend)
- Explicitly **reject callables**.

### B2) Add `lookup(...)` to AgentSet + GridCells

- Purpose: fetch values by key **without join**.

Suggested signature:

```python
lookup(
    self,
    target: ids | coords | cell_id | DataFrame,
    columns: list[str] | None = None,
    *,
    as_df: bool = True,
) -> DataFrame | dict[str, np.ndarray] | np.ndarray
```

- `AgentSet.lookup`: target = ids or DataFrame with `unique_id` column.
- `GridCells.lookup`: target = coords DataFrame or `cell_id` array.
- Use `cell_id` mapping for GridCells to avoid joins.

---

## Part C — No-Join Polars update implementation

### C1) Common mixin for masked updates

Create a small mixin (shared helper) that **applies** updates given a boolean mask aligned to the target DataFrame.

File suggestion: `mesa_frames/concrete/_update_masked.py`

```python
class _MaskedUpdateMixin:
    def _apply_masked_updates(
        self,
        df: pl.DataFrame,
        mask: np.ndarray | pl.Series,
        updates: dict[str, object],
    ) -> pl.DataFrame:
        # Build a single with_columns(...) with when(mask)
```

Notes:

- This helper **does not** compute the mask; it only applies it.
- For each column in updates:
  - if `pl.Expr`: `rhs = expr`
  - if column name: `rhs = pl.col(name)`
  - if scalar: `rhs = pl.lit(value)`
  - if Series/array: must be row-aligned (len == df height)
- Use `pl.when(mask).then(rhs).otherwise(pl.col(col))`.

### C2) GridCells: mask normalization to boolean (row-major)

Add `_mask_to_bool(...)` method on `GridCells`:

Inputs:

- `mask`: str | DataFrame | Series | np.ndarray | None
- `mask_col`: optional name for bool column in DataFrame mask

Outputs:

- `mask_bool`: np.ndarray[bool] aligned to row-major cell_id

Rules:

- **Dense row-major required** (enforce and/or rebuild once):
  - ensure `self._cells.height == width*height`
  - call `_ensure_dense_row_major_cells()` once
- Mask handling:
  - `None` or `"all"` → full True mask
  - `"empty"` → `remaining == capacity`
  - `"full"` → `remaining == 0`
  - `"available"` → `remaining > 0`
  - `np.ndarray`:
    - if 2D: ravel row-major
    - if 1D: must equal n_cells
  - `pl.Series` bool: must be length n_cells
  - `DataFrame mask`:
    - supports either:
      - coords only (treat all as True)
      - coords + bool column (use bool values)
    - compute `cell_id = dim_0 * height + dim_1`
    - create boolean array with those indices

### C3) GridCells: update implementation

- In `GridCells.update`, do:
  1) normalize updates
  2) if `pl.Expr` present → polars path
  3) call `_mask_to_bool` to get boolean mask
  4) use `_apply_masked_updates` to update `self._cells`

### C4) AgentSet: mask normalization

Add `_mask_to_bool(...)` for AgentSet:

- Accept:
  - `None` / `"all"` / `"active"`
  - ids (Series, np.ndarray, list)
  - boolean Series
  - DataFrame with `unique_id`
- Convert ids → boolean mask using `is_in` on `unique_id` or index mapping.
- For `"active"`, use `self.active_agents` if available.

### C5) AgentSet: update implementation

- Same pattern as GridCells:
  1) normalize updates
  2) if `pl.Expr` present → polars path
  3) mask → bool
  4) `_apply_masked_updates`

---

## Part D — Sugarscape refactor

### D1) Update all `set` calls to `update`

- Files:
  - `examples/sugarscape_ig/backend_frames/model.py`
  - `examples/sugarscape_ig/backend_frames/agents.py`
  - tests and docs referencing `set`

### D2) Replace join in `eat`

Current:

```python
positions.join(cells).select("sugar")
```

New:

```python
sugar = self.space.cells.lookup(positions, columns=["sugar"])
```

This should use no-join path (coords → cell_id → take).

### D3) Regrow with update + named masks

```python
self.space.cells.update({"sugar": "max_sugar"}, mask="empty")
self.space.cells.update({"sugar": 0}, mask="full")
```

No joins.

---

## Part E — Benchmarks

### E1) Sugarscape performance

Command:

```bash
POLARS_MAX_THREADS=1 uv run python -O examples/sugarscape_ig/backend_frames/model.py \
  --agents 50000 --steps 100 --width 448 --height 448 --seed 11 --no-plot --no-save-results
```

Record:

- baseline (current main)
- new no-join update/lookup path

### E2) Microbench

Add a small benchmark script if needed:

- compare `lookup` vs join for sugar fetch
- compare `update` no-join vs join for mask updates

---

## Part F — Tests

### F1) Update tests for `set` → `update`

- `tests/space/test_cells.py`
- `tests/space/test_grid.py`
- `tests/agentset/test_agentset.py`

### F2) Add tests for `lookup`

- GridCells lookup by coords
- GridCells lookup by cell_id
- AgentSet lookup by ids

### F3) Mask coverage

- `mask="empty"`/`"full"`
- boolean masks (array and Series)
- DataFrame mask with bool column

---

## Part G — Documentation

- Update docs/tutorials/examples to use `update` instead of `set`.
- Document `lookup` and masks.
- Mention **dense row-major requirement** for fast path; fallback if not dense.

---

## Success criteria

- Sugarscape benchmark improves significantly vs join path.
- API remains simple: users call `update` and `lookup`, no array exposure.
- No buffer code in main branch.
- Tests updated + passing.
