"""Microbenchmark for Grid.move_to_best fast path vs DF fallback.

This is intentionally a lightweight script (not part of the CLI) to make it
simple to run locally while iterating on NumPy/Numba kernels.

Usage:
    uv run python benchmarks/move_to_best_fastpath.py

You can force the path used by setting:
    MESA_FRAMES_GRID_MOVE_TO_BEST_FORCE_PATH=fast|df
"""

from __future__ import annotations

from time import perf_counter

import numpy as np

from mesa_frames import AgentSet, Grid, Model


def _setup(
    *, agents: int, width: int, height: int, seed: int
) -> tuple[Grid, np.ndarray]:
    model = Model(seed=seed)
    aset = AgentSet(model)
    aset.add({"x": np.zeros(agents, dtype=np.int64)})
    model.sets.add(aset)

    grid = Grid(
        model, dimensions=[width, height], capacity=1, neighborhood_type="moore"
    )
    model.space = grid

    coords = np.array(
        [(i, j) for i in range(width) for j in range(height)], dtype=np.int64
    )
    sugar = np.random.default_rng(seed + 1).integers(
        0, 100, size=coords.shape[0], dtype=np.int64
    )
    grid.cells.set(coords, properties={"sugar": sugar})

    # Place agents without overlap.
    rng = np.random.default_rng(seed + 2)
    chosen = rng.choice(coords.shape[0], size=agents, replace=False)
    grid.place_agents(aset["unique_id"], coords[chosen])

    return grid, aset["unique_id"].to_numpy()


def main() -> None:
    """Run a small set of move_to_best timing configurations."""
    configs = [
        (10_000, 3),
        (50_000, 3),
        (100_000, 3),
    ]

    width = 800
    height = 800

    for n_agents, radius in configs:
        grid, ids = _setup(agents=n_agents, width=width, height=height, seed=42)
        t0 = perf_counter()
        grid.move_to_best(ids, radius=radius, property="sugar", include_center=True)
        dt = perf_counter() - t0
        print(f"agents={n_agents:>7} radius={radius} runtime_s={dt:.4f}")


if __name__ == "__main__":
    main()
