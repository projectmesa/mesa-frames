"""Ultra-fast Sugarscape engine (NumPy/Numba).

This module provides an opt-in execution path intended purely for maximum
performance benchmarking. It bypasses Polars and most of mesa-frames' core
abstractions during the step loop.

Enable via:
- `MESA_FRAMES_SUGARSCAPE_ENGINE=numpy`

Notes
-----
- This engine is designed for benchmark runs where
  `MESA_FRAMES_SUGARSCAPE_DISABLE_DATACOLLECTOR=1`.
- The semantics match the tutorial model ordering: move -> eat -> regrow.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any

import numpy as np

from numba import njit


_VISION_MAX = 5
_CAND_MAX = 1 + 4 * _VISION_MAX  # center + von Neumann rays


@dataclass
class _NullDataCollector:
    data: dict[str, Any]

    def collect(self) -> None:  # pragma: no cover
        return

    def flush(self) -> None:  # pragma: no cover
        return


def null_datacollector() -> _NullDataCollector:
    return _NullDataCollector(data={"model": [], "agent": []})


# --- RNG helpers (deterministic, fast) ---

def _splitmix64_next(state: np.uint64) -> tuple[np.uint64, np.uint64]:
    z = state + np.uint64(0x9E3779B97F4A7C15)
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    z = z ^ (z >> np.uint64(31))
    return state + np.uint64(0x9E3779B97F4A7C15), z


def _randbelow_u64(state: np.uint64, n: int) -> tuple[np.uint64, int]:
    """Uniform integer in [0, n)."""
    if n <= 1:
        return state, 0
    x = np.uint64(n)
    limit = (np.uint64(0xFFFFFFFFFFFFFFFF) // x) * x
    while True:
        state, r = _splitmix64_next(state)
        if r < limit:
            return state, int(r % x)


def _resolve_conflicts_kernel(
    cand_cells: np.ndarray,
    cand_offsets: np.ndarray,
    origin_cell: np.ndarray,
    max_rank: np.ndarray,
    n_cells: int,
    seed: np.uint64,
) -> np.ndarray:
    """Resolve conflicts with proposal rounds in O(total proposals).

    Semantics match the Polars rounds implementation:
    - each round, each unresolved agent proposes its best available candidate
      (first available at rank >= current_rank), where availability is
      `owner[cell] == -1` OR cell is the agent's origin.
    - each cell picks one proposer uniformly at random.
    - losers advance to proposed_rank+1 (clamped) and try again.
    - when a winner moves away, its origin becomes available next rounds.
    """
    n_agents = origin_cell.size
    owner = np.full(n_cells, -1, dtype=np.int64)
    for i in range(n_agents):
        owner[int(origin_cell[i])] = i

    dest = origin_cell.copy()
    current_rank = np.zeros(n_agents, dtype=np.int64)
    unresolved_a = np.arange(n_agents, dtype=np.int64)
    unresolved_b = np.empty(n_agents, dtype=np.int64)
    unresolved_n = n_agents

    head = np.full(n_cells, -1, dtype=np.int64)
    next_prop = np.full(n_agents, -1, dtype=np.int64)
    proposed_rank = np.zeros(n_agents, dtype=np.int64)
    # At most one proposal per agent per round, so <= n_agents touched cells.
    touched = np.empty(n_agents, dtype=np.int64)

    state = seed
    touched_n = 0

    while unresolved_n:
        # reset only touched buckets
        for t in range(touched_n):
            head[int(touched[t])] = -1
        touched_n = 0

        # proposals
        for idx in range(unresolved_n):
            a = int(unresolved_a[idx])
            start = int(cand_offsets[a])
            end = int(cand_offsets[a + 1])
            r = int(current_rank[a])
            if r < 0:
                r = 0
            if r >= end - start:
                r = end - start - 1
                if r < 0:
                    r = 0

            chosen = int(origin_cell[a])
            chosen_rank = r

            # scan forward for available
            j = start + r
            while j < end:
                cell = int(cand_cells[j])
                if owner[cell] == -1 or cell == int(origin_cell[a]):
                    chosen = cell
                    chosen_rank = j - start
                    break
                j += 1

            proposed_rank[a] = chosen_rank
            # bucket by destination
            if head[chosen] == -1:
                touched[touched_n] = chosen
                touched_n += 1
            next_prop[a] = head[chosen]
            head[chosen] = a

        # resolve per touched cell
        next_n = 0

        for t in range(touched_n):
            cell = int(touched[t])
            a = head[cell]
            if a == -1:
                continue

            # reservoir sample uniform winner
            winner = a
            count = 1
            a = next_prop[a]
            while a != -1:
                count += 1
                state, k = _randbelow_u64(state, count)
                if k == 0:
                    winner = a
                a = next_prop[a]

            # apply winner
            w = int(winner)
            dest[w] = cell
            if cell != int(origin_cell[w]):
                owner[int(origin_cell[w])] = -1
                owner[cell] = w

            # losers advance
            a = head[cell]
            while a != -1:
                if a != w:
                    pr = int(proposed_rank[a]) + 1
                    mr = int(max_rank[a])
                    if pr > mr:
                        pr = mr
                    current_rank[a] = pr
                    unresolved_b[next_n] = a
                    next_n += 1
                a = next_prop[a]

        unresolved_a, unresolved_b = unresolved_b, unresolved_a
        unresolved_n = next_n

    return dest


def _build_ranked_candidates_von_neumann(
    origin_cell: np.ndarray,
    vision: np.ndarray,
    sugar_flat: np.ndarray,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build CSR candidate lists ranked by (sugar desc, radius asc, dim0, dim1).

    This implementation avoids per-agent temporary allocations by filling the CSR
    slice directly and sorting it in-place (insertion sort on small K).
    """
    n_agents = origin_cell.size

    # offsets
    offsets = np.empty(n_agents + 1, dtype=np.int64)
    total = 0
    offsets[0] = 0
    for i in range(n_agents):
        total += 1 + 4 * int(vision[i])
        offsets[i + 1] = total

    cand_cells = np.empty(total, dtype=np.int64)
    cand_radius = np.empty(total, dtype=np.int64)
    max_rank = np.empty(n_agents, dtype=np.int64)

    for i in range(n_agents):
        start = int(offsets[i])
        v = int(vision[i])
        k = 0

        oc = int(origin_cell[i])
        o0 = oc // height
        o1 = oc - o0 * height

        cand_cells[start + k] = oc
        cand_radius[start + k] = 0
        k += 1

        for r in range(1, v + 1):
            if o0 + r < width:
                cand_cells[start + k] = (o0 + r) * height + o1
                cand_radius[start + k] = r
                k += 1
            if o0 - r >= 0:
                cand_cells[start + k] = (o0 - r) * height + o1
                cand_radius[start + k] = r
                k += 1
            if o1 + r < height:
                cand_cells[start + k] = o0 * height + (o1 + r)
                cand_radius[start + k] = r
                k += 1
            if o1 - r >= 0:
                cand_cells[start + k] = o0 * height + (o1 - r)
                cand_radius[start + k] = r
                k += 1

        # insertion sort by: sugar desc, radius asc, dim0 asc, dim1 asc
        for a in range(1, k):
            ca = int(cand_cells[start + a])
            ra = int(cand_radius[start + a])
            sa = int(sugar_flat[ca])
            d0a = ca // height
            d1a = ca - d0a * height

            j = a - 1
            while j >= 0:
                cj = int(cand_cells[start + j])
                sj = int(sugar_flat[cj])
                if sj > sa:
                    break
                if sj == sa:
                    rj = int(cand_radius[start + j])
                    if rj < ra:
                        break
                    if rj == ra:
                        d0j = cj // height
                        d1j = cj - d0j * height
                        if d0j < d0a:
                            break
                        if d0j == d0a and d1j <= d1a:
                            break

                cand_cells[start + j + 1] = cand_cells[start + j]
                cand_radius[start + j + 1] = cand_radius[start + j]
                j -= 1

            cand_cells[start + j + 1] = ca
            cand_radius[start + j + 1] = ra

        max_rank[i] = k - 1

    return offsets, cand_cells, max_rank


def _step_once_inplace(
    origin_cell: np.ndarray,
    sugar_stock: np.ndarray,
    metabolism: np.ndarray,
    vision: np.ndarray,
    max_sugar_flat: np.ndarray,
    occ0_gen: np.ndarray,
    occ_gen: np.ndarray,
    head_first: np.ndarray,
    head_gen: np.ndarray,
    next_prop: np.ndarray,
    proposed_rank: np.ndarray,
    touched: np.ndarray,
    unresolved_a: np.ndarray,
    unresolved_b: np.ndarray,
    current_rank: np.ndarray,
    dest_cell: np.ndarray,
    cand_cell: np.ndarray,
    cand_rad: np.ndarray,
    cand_sugar: np.ndarray,
    cand_len: np.ndarray,
    width: int,
    height: int,
    n_agents: int,
    gen: np.uint32,
    seed: np.uint64,
) -> tuple[int, np.uint64]:
    """One step (move+eat+regrow) with zero per-step allocations.

    Key idea for instant growback:
    - Sugar available during move/eat is 0 on cells occupied at *start of step*,
      else it is max_sugar.
    - Cells vacated during conflict rounds become available to move into, but
      their sugar remains 0 because they were occupied at start of step.

    We therefore maintain:
    - `occ0_gen`: start-of-step occupancy stamps (for sugar lookup)
    - `occ_gen`: dynamic occupancy stamps during conflict rounds (for availability)
    Both use `gen` as the active stamp.
    """

    # Build start occupancy (sugar) and dynamic occupancy (availability).
    for i in range(n_agents):
        c = int(origin_cell[i])
        occ0_gen[c] = gen
        occ_gen[c] = gen

    # Build ranked candidates per agent (fixed width, small K, insertion sort).
    for i in range(n_agents):
        base = i * _CAND_MAX
        oc = int(origin_cell[i])
        o0 = oc // height
        o1 = oc - o0 * height
        v = int(vision[i])
        if v > _VISION_MAX:
            v = _VISION_MAX

        k = 0
        cand_cell[base + k] = oc
        cand_rad[base + k] = 0
        cand_sugar[base + k] = 0
        k += 1

        for r in range(1, v + 1):
            if o0 + r < width:
                c = (o0 + r) * height + o1
                cand_cell[base + k] = c
                cand_rad[base + k] = r
                cand_sugar[base + k] = 0 if occ0_gen[c] == gen else max_sugar_flat[c]
                k += 1
            if o0 - r >= 0:
                c = (o0 - r) * height + o1
                cand_cell[base + k] = c
                cand_rad[base + k] = r
                cand_sugar[base + k] = 0 if occ0_gen[c] == gen else max_sugar_flat[c]
                k += 1
            if o1 + r < height:
                c = o0 * height + (o1 + r)
                cand_cell[base + k] = c
                cand_rad[base + k] = r
                cand_sugar[base + k] = 0 if occ0_gen[c] == gen else max_sugar_flat[c]
                k += 1
            if o1 - r >= 0:
                c = o0 * height + (o1 - r)
                cand_cell[base + k] = c
                cand_rad[base + k] = r
                cand_sugar[base + k] = 0 if occ0_gen[c] == gen else max_sugar_flat[c]
                k += 1

        # insertion sort by: sugar desc, radius asc, cell_id asc
        for a in range(1, k):
            ca = int(cand_cell[base + a])
            ra = int(cand_rad[base + a])
            sa = int(cand_sugar[base + a])
            j = a - 1
            while j >= 0:
                sj = int(cand_sugar[base + j])
                if sj > sa:
                    break
                if sj == sa:
                    rj = int(cand_rad[base + j])
                    if rj < ra:
                        break
                    if rj == ra:
                        cj = int(cand_cell[base + j])
                        if cj <= ca:
                            break
                cand_cell[base + j + 1] = cand_cell[base + j]
                cand_rad[base + j + 1] = cand_rad[base + j]
                cand_sugar[base + j + 1] = cand_sugar[base + j]
                j -= 1
            cand_cell[base + j + 1] = ca
            cand_rad[base + j + 1] = ra
            cand_sugar[base + j + 1] = sa

        cand_len[i] = k

    # Conflict resolution (proposal rounds). Buckets use generation stamps.
    for i in range(n_agents):
        current_rank[i] = 0
        dest_cell[i] = origin_cell[i]
        unresolved_a[i] = i
    unresolved_n = n_agents
    state = seed
    round_gen = np.uint32(1)

    while unresolved_n:
        touched_n = 0

        # proposals
        for idx in range(unresolved_n):
            a = int(unresolved_a[idx])
            k = int(cand_len[a])
            r0 = int(current_rank[a])
            if r0 < 0:
                r0 = 0
            if r0 >= k:
                r0 = k - 1
                if r0 < 0:
                    r0 = 0

            oc = int(origin_cell[a])
            chosen = oc
            chosen_rank = r0

            j = r0
            while j < k:
                c = int(cand_cell[a * _CAND_MAX + j])
                # available if empty now, OR is the agent's own origin
                if occ_gen[c] != gen or c == oc:
                    chosen = c
                    chosen_rank = j
                    break
                j += 1

            proposed_rank[a] = chosen_rank

            if head_gen[chosen] != round_gen:
                head_gen[chosen] = round_gen
                head_first[chosen] = a
                touched[touched_n] = chosen
                touched_n += 1
                next_prop[a] = -1
            else:
                next_prop[a] = head_first[chosen]
                head_first[chosen] = a

        # resolve
        next_n = 0

        for ti in range(touched_n):
            cell = int(touched[ti])
            a = int(head_first[cell])
            if a < 0:
                continue

            # reservoir sample uniform winner
            winner = a
            count = 1
            a2 = int(next_prop[a])
            while a2 >= 0:
                count += 1
                state, ksel = _randbelow_u64(state, count)
                if ksel == 0:
                    winner = a2
                a2 = int(next_prop[a2])

            w = int(winner)
            oc = int(origin_cell[w])
            dest_cell[w] = cell
            if cell != oc:
                # free origin for later rounds, occupy destination
                occ_gen[oc] = np.uint32(0)
                occ_gen[cell] = gen

            # losers advance
            a = int(head_first[cell])
            while a >= 0:
                if a != w:
                    pr = int(proposed_rank[a]) + 1
                    mr = int(cand_len[a]) - 1
                    if pr > mr:
                        pr = mr
                    current_rank[a] = pr
                    unresolved_b[next_n] = a
                    next_n += 1
                a = int(next_prop[a])

        # swap unresolved buffers
        for i in range(next_n):
            unresolved_a[i] = unresolved_b[i]
        unresolved_n = next_n
        round_gen = round_gen + np.uint32(1)

    # Eat + in-place compaction.
    write = 0
    for i in range(n_agents):
        d = int(dest_cell[i])
        gain = 0
        if occ0_gen[d] != gen:
            gain = int(max_sugar_flat[d])
        ns = int(sugar_stock[i]) + gain - int(metabolism[i])
        if ns > 0:
            origin_cell[write] = d
            sugar_stock[write] = ns
            metabolism[write] = metabolism[i]
            vision[write] = vision[i]
            write += 1

    # Note: we intentionally do NOT clear occ0_gen/occ_gen; stamps advance each step.
    return write, state


def _step_once(
    origin_cell: np.ndarray,
    sugar_stock: np.ndarray,
    metabolism: np.ndarray,
    vision: np.ndarray,
    sugar_flat: np.ndarray,
    max_sugar_flat: np.ndarray,
    width: int,
    height: int,
    seed: np.uint64,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_cells = width * height

    offsets, cand_cells, max_rank = _build_ranked_candidates_von_neumann(
        origin_cell=origin_cell,
        vision=vision,
        sugar_flat=sugar_flat,
        width=width,
        height=height,
    )
    dest_cell = _resolve_conflicts_kernel(
        cand_cells=cand_cells,
        cand_offsets=offsets,
        origin_cell=origin_cell,
        max_rank=max_rank,
        n_cells=n_cells,
        seed=seed,
    )

    sugar = sugar_flat[dest_cell]
    new_sugar = sugar_stock + sugar - metabolism
    sugar_flat[dest_cell] = 0

    alive = new_sugar > 0

    # instant growback: refill all cells, then keep occupied at 0
    sugar_flat[:] = max_sugar_flat
    sugar_flat[dest_cell[alive]] = 0

    return dest_cell, new_sugar, alive


if njit is not None:  # pragma: no cover
    _splitmix64_next = njit(cache=True)(_splitmix64_next)
    _randbelow_u64 = njit(cache=True)(_randbelow_u64)
    _resolve_conflicts_kernel = njit(cache=True)(_resolve_conflicts_kernel)
    _build_ranked_candidates_von_neumann = njit(cache=True)(_build_ranked_candidates_von_neumann)
    _step_once_inplace = njit(cache=True)(_step_once_inplace)
    _step_once = njit(cache=True)(_step_once)


def simulate_numpy(
    *,
    agents: int,
    steps: int,
    width: int,
    height: int,
    max_sugar: int = 4,
    seed: int | None = None,
) -> None:
    """Run Sugarscape with a fully NumPy/Numba step loop (max performance)."""
    if seed is None:
        seed = 1

    rng = np.random.default_rng(seed)
    n_cells = width * height

    # Instant growback: only the per-cell max sugar matters.
    max_sugar_flat = rng.integers(0, max_sugar + 1, size=n_cells, dtype=np.uint8)

    # Fixed-capacity agent arrays (compact in-place).
    sugar_stock = rng.integers(6, 25, size=agents, dtype=np.int32)
    metabolism = rng.integers(2, 5, size=agents, dtype=np.uint8)
    vision = rng.integers(1, 6, size=agents, dtype=np.uint8)
    origin_cell = rng.choice(n_cells, size=agents, replace=False).astype(np.uint32)

    # Occupancy stamps: 0 means empty; gen increments per step.
    occ0_gen = np.zeros(n_cells, dtype=np.uint32)
    occ_gen = np.zeros(n_cells, dtype=np.uint32)

    # Conflict-resolution scratch.
    head_first = np.empty(n_cells, dtype=np.int32)
    head_gen = np.zeros(n_cells, dtype=np.uint32)
    next_prop = np.empty(agents, dtype=np.int32)
    proposed_rank = np.empty(agents, dtype=np.uint8)
    touched = np.empty(agents, dtype=np.uint32)
    unresolved_a = np.empty(agents, dtype=np.int32)
    unresolved_b = np.empty(agents, dtype=np.int32)
    current_rank = np.empty(agents, dtype=np.uint8)
    dest_cell = np.empty(agents, dtype=np.uint32)

    # Candidate buffers (fixed width, flat).
    cand_cell = np.empty(agents * _CAND_MAX, dtype=np.uint32)
    cand_rad = np.empty(agents * _CAND_MAX, dtype=np.uint8)
    cand_sugar = np.empty(agents * _CAND_MAX, dtype=np.uint8)
    cand_len = np.empty(agents, dtype=np.uint8)

    state = np.uint64(seed) ^ np.uint64(0xD1B54A32D192ED03)
    mask_u64 = (1 << 64) - 1
    state = np.uint64(int(state) & mask_u64)
    n_agents = agents
    gen = np.uint32(1)

    for _ in range(steps):
        if n_agents <= 0:
            break
        state = np.uint64(int(state) & mask_u64)
        n_agents, state = _step_once_inplace(
            origin_cell=origin_cell,
            sugar_stock=sugar_stock,
            metabolism=metabolism,
            vision=vision,
            max_sugar_flat=max_sugar_flat,
            occ0_gen=occ0_gen,
            occ_gen=occ_gen,
            head_first=head_first,
            head_gen=head_gen,
            next_prop=next_prop,
            proposed_rank=proposed_rank,
            touched=touched,
            unresolved_a=unresolved_a,
            unresolved_b=unresolved_b,
            current_rank=current_rank,
            dest_cell=dest_cell,
            cand_cell=cand_cell,
            cand_rad=cand_rad,
            cand_sugar=cand_sugar,
            cand_len=cand_len,
            width=width,
            height=height,
            n_agents=n_agents,
            gen=gen,
            seed=state,
        )

        state = np.uint64(int(state) & mask_u64)

        gen = gen + np.uint32(1)


def engine_enabled() -> bool:
    engine = os.environ.get("MESA_FRAMES_SUGARSCAPE_ENGINE", "").strip().lower()
    return engine == "numpy"
