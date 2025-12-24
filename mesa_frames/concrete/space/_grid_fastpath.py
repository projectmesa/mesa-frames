"""Internal NumPy/Numba fast paths for grid operations.

This module is internal (not part of the public API).

Design: Grid composes this helper via the private `_GridFastPath` class.
"""

from __future__ import annotations

from dataclasses import dataclass
import os

import numpy as np

try:  # pragma: no cover
    import numba as _numba  # type: ignore

    _NUMBA_AVAILABLE = True
except ImportError:  # pragma: no cover
    _numba = None
    _NUMBA_AVAILABLE = False


def _numba_enabled() -> bool:
    """Return True if numba is importable and not explicitly disabled."""
    if not _NUMBA_AVAILABLE:
        return False
    flag = os.environ.get("MESA_FRAMES_GRID_MOVE_TO_BEST_DISABLE_NUMBA", "")
    return flag.strip().lower() not in {"1", "true", "yes", "on"}


# Imported to keep beartype forward-ref resolution happy without introducing a
# circular import at `grid.py` import time (Grid imports this module only from
# `Grid.__init__`).
from mesa_frames.concrete.space.grid import Grid  # noqa: E402


if _NUMBA_AVAILABLE:  # pragma: no cover
    # NOTE: cache=False to avoid "no locator available" errors under wrapped
    # functions (e.g., runtime type-checking / beartype instrumentation).
    _njit = _numba.njit(cache=False)


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


if _NUMBA_AVAILABLE:  # pragma: no cover

    @_njit
    def _neighbors_count_kernel_2d(
        centers: np.ndarray,
        radius_per_agent: np.ndarray,
        radius_scalar: int,
        use_per_agent: bool,
        base_dirs: np.ndarray,
        width: int,
        height: int,
        torus: bool,
        include_center: bool,
    ) -> np.ndarray:
        n_agents = int(centers.shape[0])
        n_dirs = int(base_dirs.shape[0])
        counts = np.zeros(n_agents, dtype=np.int64)

        for i in range(n_agents):
            cx = int(centers[i, 0])
            cy = int(centers[i, 1])
            rmax = radius_scalar
            if use_per_agent:
                rmax = int(radius_per_agent[i])

            # Upper bound for this agent (including center). Used only for the
            # local de-dupe buffer.
            seg_upper = (1 if include_center else 0) + n_dirs * rmax
            seen = np.empty(seg_upper if seg_upper > 0 else 1, dtype=np.int64)
            seen_n = 0

            if include_center:
                x0 = cx
                y0 = cy
                if torus:
                    x0 %= width
                    y0 %= height
                else:
                    if x0 < 0 or x0 >= width or y0 < 0 or y0 >= height:
                        counts[i] = 0
                        continue
                cid0 = x0 * height + y0
                seen[seen_n] = cid0
                seen_n += 1

            for d in range(n_dirs):
                dx0 = int(base_dirs[d, 0])
                dy0 = int(base_dirs[d, 1])
                for r in range(1, rmax + 1):
                    x = cx + dx0 * r
                    y = cy + dy0 * r
                    if torus:
                        x %= width
                        y %= height
                    else:
                        if x < 0 or x >= width or y < 0 or y >= height:
                            continue
                    cid = x * height + y
                    if torus:
                        dup = False
                        for k in range(seen_n):
                            if int(seen[k]) == cid:
                                dup = True
                                break
                        if dup:
                            continue
                    seen[seen_n] = cid
                    seen_n += 1

            counts[i] = seen_n

        return counts


if _NUMBA_AVAILABLE:  # pragma: no cover

    @_njit
    def _neighbors_fill_kernel_2d(
        centers: np.ndarray,
        radius_per_agent: np.ndarray,
        radius_scalar: int,
        use_per_agent: bool,
        base_dirs: np.ndarray,
        width: int,
        height: int,
        torus: bool,
        include_center: bool,
        offsets: np.ndarray,
        out_cell_id: np.ndarray,
        out_rad: np.ndarray,
        out_d0: np.ndarray,
        out_d1: np.ndarray,
    ) -> None:
        n_agents = int(centers.shape[0])
        n_dirs = int(base_dirs.shape[0])

        for i in range(n_agents):
            start = int(offsets[i])
            cx = int(centers[i, 0])
            cy = int(centers[i, 1])
            rmax = radius_scalar
            if use_per_agent:
                rmax = int(radius_per_agent[i])

            seg_upper = (1 if include_center else 0) + n_dirs * rmax
            seen = np.empty(seg_upper if seg_upper > 0 else 1, dtype=np.int64)
            seen_n = 0

            out = start

            if include_center:
                x0 = cx
                y0 = cy
                if torus:
                    x0 %= width
                    y0 %= height
                else:
                    if x0 < 0 or x0 >= width or y0 < 0 or y0 >= height:
                        continue
                cid0 = x0 * height + y0
                seen[seen_n] = cid0
                seen_n += 1

                out_d0[out] = x0
                out_d1[out] = y0
                out_rad[out] = 0
                out_cell_id[out] = cid0
                out += 1

            for d in range(n_dirs):
                dx0 = int(base_dirs[d, 0])
                dy0 = int(base_dirs[d, 1])
                for r in range(1, rmax + 1):
                    x = cx + dx0 * r
                    y = cy + dy0 * r
                    if torus:
                        x %= width
                        y %= height
                    else:
                        if x < 0 or x >= width or y < 0 or y >= height:
                            continue
                    cid = x * height + y
                    if torus:
                        dup = False
                        for k in range(seen_n):
                            if int(seen[k]) == cid:
                                dup = True
                                break
                        if dup:
                            continue
                    seen[seen_n] = cid
                    seen_n += 1

                    out_d0[out] = x
                    out_d1[out] = y
                    out_rad[out] = r
                    out_cell_id[out] = cid
                    out += 1


if _NUMBA_AVAILABLE:  # pragma: no cover

    @_njit
    def _rank_candidates_kernel(
        offsets: np.ndarray,
        cell_id: np.ndarray,
        radius: np.ndarray,
        dim0: np.ndarray,
        dim1: np.ndarray,
        score_flat: np.ndarray,
    ) -> None:
        n_agents = int(offsets.shape[0] - 1)
        for i in range(n_agents):
            start = int(offsets[i])
            stop = int(offsets[i + 1])
            # insertion sort on small K
            for j in range(start + 1, stop):
                key_cell = int(cell_id[j])
                key_rad = int(radius[j])
                key_d0 = int(dim0[j])
                key_d1 = int(dim1[j])
                key_score = float(score_flat[key_cell])

                k = j - 1
                while k >= start:
                    cur_cell = int(cell_id[k])
                    cur_score = float(score_flat[cur_cell])
                    # comparator: score desc, radius asc, dim0 asc, dim1 asc
                    better = False
                    if cur_score > key_score:
                        better = True
                    elif cur_score < key_score:
                        better = False
                    else:
                        cur_rad = int(radius[k])
                        if cur_rad < key_rad:
                            better = True
                        elif cur_rad > key_rad:
                            better = False
                        else:
                            cur_d0 = int(dim0[k])
                            if cur_d0 < key_d0:
                                better = True
                            elif cur_d0 > key_d0:
                                better = False
                            else:
                                cur_d1 = int(dim1[k])
                                better = cur_d1 <= key_d1

                    if better:
                        break

                    cell_id[k + 1] = cell_id[k]
                    radius[k + 1] = radius[k]
                    dim0[k + 1] = dim0[k]
                    dim1[k + 1] = dim1[k]
                    k -= 1

                cell_id[k + 1] = key_cell
                radius[k + 1] = key_rad
                dim0[k + 1] = key_d0
                dim1[k + 1] = key_d1


if _NUMBA_AVAILABLE:  # pragma: no cover

    @_njit
    def _rank_candidates_by_score_kernel(
        offsets: np.ndarray,
        cell_id: np.ndarray,
        radius: np.ndarray,
        dim0: np.ndarray,
        dim1: np.ndarray,
        cand_score: np.ndarray,
    ) -> None:
        n_agents = int(offsets.shape[0] - 1)
        for i in range(n_agents):
            start = int(offsets[i])
            stop = int(offsets[i + 1])
            for j in range(start + 1, stop):
                key_cell = int(cell_id[j])
                key_rad = int(radius[j])
                key_d0 = int(dim0[j])
                key_d1 = int(dim1[j])
                key_score = float(cand_score[j])
                if key_score != key_score:
                    key_score = -1e300

                k = j - 1
                while k >= start:
                    cur_score = float(cand_score[k])
                    if cur_score != cur_score:
                        cur_score = -1e300

                    better = False
                    if cur_score > key_score:
                        better = True
                    elif cur_score < key_score:
                        better = False
                    else:
                        cur_rad = int(radius[k])
                        if cur_rad < key_rad:
                            better = True
                        elif cur_rad > key_rad:
                            better = False
                        else:
                            cur_d0 = int(dim0[k])
                            if cur_d0 < key_d0:
                                better = True
                            elif cur_d0 > key_d0:
                                better = False
                            else:
                                cur_d1 = int(dim1[k])
                                better = cur_d1 <= key_d1

                    if better:
                        break

                    cell_id[k + 1] = cell_id[k]
                    radius[k + 1] = radius[k]
                    dim0[k + 1] = dim0[k]
                    dim1[k + 1] = dim1[k]
                    cand_score[k + 1] = cand_score[k]
                    k -= 1

                cell_id[k + 1] = key_cell
                radius[k + 1] = key_rad
                dim0[k + 1] = key_d0
                dim1[k + 1] = key_d1
                cand_score[k + 1] = key_score


if _NUMBA_AVAILABLE:  # pragma: no cover

    @_njit
    def _resolve_conflicts_lottery_kernel(
        offsets: np.ndarray,
        cand_cell: np.ndarray,
        origin: np.ndarray,
        capacity_flat: np.ndarray,
        seed: np.uint64,
    ) -> tuple[np.ndarray, int]:
        n_agents = int(offsets.shape[0] - 1)
        dest = np.full(n_agents, -1, dtype=np.int64)
        if n_agents == 0:
            return dest, seed

        cap = capacity_flat.copy()

        ptr = np.zeros(n_agents, dtype=np.int64)
        seg_len = (offsets[1:] - offsets[:-1]).astype(np.int64)
        unassigned = np.ones(n_agents, dtype=np.uint8)  # 1 if unassigned

        max_candidates = 0
        for i in range(n_agents):
            if int(seg_len[i]) > max_candidates:
                max_candidates = int(seg_len[i])
        if max_candidates == 0:
            for i in range(n_agents):
                dest[i] = int(origin[i])
            return dest, seed

        state = seed

        # Scratch for proposals (<= n_agents each round).
        prop_agents = np.empty(n_agents, dtype=np.int64)
        prop_cells = np.empty(n_agents, dtype=np.int64)
        prop_wait = np.empty(n_agents, dtype=np.uint8)

        # Upper bound similar to Python implementation.
        max_iters = int(n_agents * max_candidates + n_agents + 1)

        for _ in range(max_iters):
            any_unassigned = False
            for i in range(n_agents):
                if unassigned[i] == 1:
                    any_unassigned = True
                    break
            if not any_unassigned:
                break

            made_progress = False

            freeable = np.zeros(cap.shape[0], dtype=np.uint8)
            for i in range(n_agents):
                if unassigned[i] == 1:
                    freeable[int(origin[i])] = 1

            prop_n = 0

            for i in range(n_agents):
                if unassigned[i] != 1:
                    continue
                base = int(offsets[i])
                nseg = int(seg_len[i])
                if nseg <= 0:
                    dest[i] = int(origin[i])
                    unassigned[i] = 0
                    made_progress = True
                    continue

                # Walk candidates until we find a proposal or exhaust.
                while int(ptr[i]) < nseg:
                    j = base + int(ptr[i])
                    cell = int(cand_cell[j])

                    if cell == int(origin[i]):
                        dest[i] = cell
                        unassigned[i] = 0
                        made_progress = True
                        break

                    if int(cap[cell]) > 0:
                        prop_agents[prop_n] = i
                        prop_cells[prop_n] = cell
                        prop_wait[prop_n] = 0
                        prop_n += 1
                        break

                    if freeable[cell] == 1:
                        prop_agents[prop_n] = i
                        prop_cells[prop_n] = cell
                        prop_wait[prop_n] = 1
                        prop_n += 1
                        break

                    ptr[i] += 1
                    made_progress = True

                if unassigned[i] == 1 and int(ptr[i]) >= nseg:
                    dest[i] = int(origin[i])
                    unassigned[i] = 0
                    made_progress = True

            if prop_n == 0:
                if not made_progress:
                    break
                continue

            order = np.argsort(prop_cells[:prop_n])
            # Walk proposal runs by cell.
            run_start = 0
            while run_start < prop_n:
                idx0 = int(order[run_start])
                cell = int(prop_cells[idx0])
                run_end = run_start + 1
                while run_end < prop_n:
                    idxn = int(order[run_end])
                    if int(prop_cells[idxn]) != cell:
                        break
                    run_end += 1

                remaining = int(cap[cell])
                run_len = run_end - run_start

                if remaining <= 0:
                    # Full: non-waiters advance.
                    for t in range(run_start, run_end):
                        ii = int(order[t])
                        a = int(prop_agents[ii])
                        if prop_wait[ii] == 0:
                            ptr[a] += 1
                            made_progress = True
                else:
                    # Pick winners without replacement (simple in-place shuffle).
                    k = remaining
                    if k > run_len:
                        k = run_len

                    tmp = np.empty(run_len, dtype=np.int64)
                    for t in range(run_len):
                        ii = int(order[run_start + t])
                        tmp[t] = int(prop_agents[ii])

                    # Partial Fisher-Yates for first k.
                    for t in range(k):
                        # Inline splitmix64 + randbelow to avoid calling external
                        # helpers that may be runtime-wrapped.
                        n = run_len - t
                        if n <= 1:
                            j = t
                        else:
                            x = np.uint64(n)
                            limit = (np.uint64(0xFFFFFFFFFFFFFFFF) // x) * x
                            while True:
                                z = state + np.uint64(0x9E3779B97F4A7C15)
                                z = (z ^ (z >> np.uint64(30))) * np.uint64(
                                    0xBF58476D1CE4E5B9
                                )
                                z = (z ^ (z >> np.uint64(27))) * np.uint64(
                                    0x94D049BB133111EB
                                )
                                z = z ^ (z >> np.uint64(31))
                                state = state + np.uint64(0x9E3779B97F4A7C15)
                                if z < limit:
                                    j = t + int(z % x)
                                    break
                        tmp[t], tmp[j] = tmp[j], tmp[t]

                    for t in range(k):
                        w = int(tmp[t])
                        dest[w] = cell
                        unassigned[w] = 0
                    cap[cell] -= k
                    made_progress = True

                    # Free origins for winners that moved.
                    for t in range(k):
                        w = int(tmp[t])
                        if int(origin[w]) != cell:
                            cap[int(origin[w])] += 1

                    # Losers advance.
                    # Build a small winner marker (linear scan: run_len small).
                    for t in range(run_start, run_end):
                        ii = int(order[t])
                        a = int(prop_agents[ii])
                        won = False
                        for u in range(k):
                            if int(tmp[u]) == a:
                                won = True
                                break
                        if not won:
                            ptr[a] += 1

                run_start = run_end

            if not made_progress:
                break

        for i in range(n_agents):
            if unassigned[i] == 1:
                dest[i] = int(origin[i])

        return dest, state


@dataclass(frozen=True, slots=True)
class _CSR:
    offsets: np.ndarray  # shape (n_agents + 1,)
    cell_id: np.ndarray  # shape (n_candidates,)
    radius: np.ndarray  # shape (n_candidates,)
    dim0: np.ndarray  # shape (n_candidates,)
    dim1: np.ndarray  # shape (n_candidates,)


def _cell_id_from_coords_2d(coords: np.ndarray, height: int) -> np.ndarray:
    return coords[:, 0].astype(np.int64, copy=False) * int(height) + coords[
        :, 1
    ].astype(np.int64, copy=False)


def _coords_from_cell_id_2d(cell_id: np.ndarray, height: int) -> np.ndarray:
    cell_id = cell_id.astype(np.int64, copy=False)
    dim0 = cell_id // int(height)
    dim1 = cell_id % int(height)
    return np.stack([dim0, dim1], axis=1)


class _GridFastPath:
    def __init__(self, grid: Grid) -> None:
        self._grid = grid

    @property
    def numba_enabled(self) -> bool:
        return _numba_enabled()

    def csr(
        self,
        *,
        offsets: np.ndarray,
        cell_id: np.ndarray,
        radius: np.ndarray,
        dim0: np.ndarray,
        dim1: np.ndarray,
    ) -> _CSR:
        return _CSR(
            offsets=np.asarray(offsets),
            cell_id=np.asarray(cell_id),
            radius=np.asarray(radius),
            dim0=np.asarray(dim0),
            dim1=np.asarray(dim1),
        )

    def neighbors_for_agents_array(
        self,
        *,
        centers: np.ndarray,
        radius: int | np.ndarray,
        include_center: bool,
    ) -> _CSR:
        """Build neighborhood candidates matching GridNeighborhood semantics.

        Important: This replicates mesa-frames' current neighborhood generation:
        for each r in 1..radius, candidates are center + (base_offset * r),
        where base_offset comes from grid._offsets (radius-1 directions).
        """
        if centers.ndim != 2 or centers.shape[1] != 2:
            raise ValueError("centers must be (n, 2)")

        n_agents = int(centers.shape[0])
        if n_agents == 0:
            return _CSR(
                offsets=np.zeros(1, dtype=np.int64),
                cell_id=np.empty(0, dtype=np.int64),
                radius=np.empty(0, dtype=np.int64),
                dim0=np.empty(0, dtype=np.int64),
                dim1=np.empty(0, dtype=np.int64),
            )

        radius_scalar: int | None = None
        radius_per_agent: np.ndarray | None = None

        if isinstance(radius, np.ndarray):
            radius_per_agent = np.asarray(radius)
            if radius_per_agent.ndim != 1:
                raise ValueError("radius sequence must be 1-D")
            if int(radius_per_agent.shape[0]) != n_agents:
                raise ValueError("radius sequence length must match number of centers")
            if not np.issubdtype(radius_per_agent.dtype, np.integer):
                raise TypeError("radius sequence must have integer dtype")
            if np.any(radius_per_agent < 0):
                raise ValueError("radius values must be >= 0")
            radius_per_agent = radius_per_agent.astype(np.int64, copy=False)
        else:
            radius_scalar = int(radius)
            if radius_scalar < 0:
                raise ValueError("radius must be >= 0")

        grid = self._grid

        base_dirs = (
            grid._offsets.select(grid._pos_col_names)
            .to_numpy()
            .astype(np.int64, copy=False)
        )
        n_dirs = int(base_dirs.shape[0])

        if _numba_enabled():
            width = int(grid._dimensions[0])
            height = int(grid._dimensions[1])
            torus = bool(grid._torus)
            use_per_agent = radius_per_agent is not None
            rad_arr = radius_per_agent if radius_per_agent is not None else np.empty(0)
            rad_scalar = int(radius_scalar) if radius_scalar is not None else 0

            counts = _neighbors_count_kernel_2d(
                centers.astype(np.int64, copy=False),
                rad_arr.astype(np.int64, copy=False),
                rad_scalar,
                use_per_agent,
                base_dirs,
                width,
                height,
                torus,
                include_center,
            )
            offsets = np.empty(n_agents + 1, dtype=np.int64)
            offsets[0] = 0
            np.cumsum(counts, out=offsets[1:])
            total = int(offsets[-1])
            cell_id_all = np.empty(total, dtype=np.int64)
            rad_all = np.empty(total, dtype=np.int64)
            dim0_all = np.empty(total, dtype=np.int64)
            dim1_all = np.empty(total, dtype=np.int64)
            _neighbors_fill_kernel_2d(
                centers.astype(np.int64, copy=False),
                rad_arr.astype(np.int64, copy=False),
                rad_scalar,
                use_per_agent,
                base_dirs,
                width,
                height,
                torus,
                include_center,
                offsets,
                cell_id_all,
                rad_all,
                dim0_all,
                dim1_all,
            )
            return _CSR(
                offsets=offsets,
                cell_id=cell_id_all,
                radius=rad_all,
                dim0=dim0_all,
                dim1=dim1_all,
            )

        # Upper bound (no bounds filtering, no torus de-dupe). When radii vary,
        # allocate using the sum of per-agent bounds instead of n_agents * max_radius.
        if radius_per_agent is None:
            max_per_agent = (1 if include_center else 0) + (int(radius_scalar) * n_dirs)
            total_upper = n_agents * max_per_agent
        else:
            total_upper = (n_agents if include_center else 0) + int(
                n_dirs * int(radius_per_agent.sum())
            )

        cell_id_all = np.empty(total_upper, dtype=np.int64)
        rad_all = np.empty_like(cell_id_all)
        dim0_all = np.empty_like(cell_id_all)
        dim1_all = np.empty_like(cell_id_all)
        offsets = np.zeros(n_agents + 1, dtype=np.int64)

        width = int(grid._dimensions[0])
        height = int(grid._dimensions[1])

        out = 0
        for i in range(n_agents):
            offsets[i] = out
            cx, cy = int(centers[i, 0]), int(centers[i, 1])

            rmax = (
                int(radius_scalar)
                if radius_per_agent is None
                else int(radius_per_agent[i])
            )

            # Include center first (matches GridNeighborhood._finalize_neighbors)
            if include_center:
                x0, y0 = cx, cy
                if grid._torus:
                    x0 %= width
                    y0 %= height
                else:
                    if x0 < 0 or x0 >= width or y0 < 0 or y0 >= height:
                        # centers should be valid already; keep defensive
                        continue
                dim0_all[out] = x0
                dim1_all[out] = y0
                rad_all[out] = 0
                cell_id_all[out] = x0 * height + y0
                out += 1

            # Match GridNeighborhood._expanded_offsets ordering: direction-major, radius-minor.
            for d in range(n_dirs):
                dx0 = int(base_dirs[d, 0])
                dy0 = int(base_dirs[d, 1])
                for r in range(1, rmax + 1):
                    dx = dx0 * r
                    dy = dy0 * r
                    x = cx + dx
                    y = cy + dy
                    if grid._torus:
                        x %= width
                        y %= height
                    else:
                        if x < 0 or x >= width or y < 0 or y >= height:
                            continue
                    dim0_all[out] = x
                    dim1_all[out] = y
                    rad_all[out] = r
                    cell_id_all[out] = x * height + y
                    out += 1

            # Torus mode de-dupes (agent_id, coords) pairs with keep="first".
            if grid._torus and out > offsets[i] + 1:
                start = offsets[i]
                stop = out
                seen: set[int] = set()
                write = start
                for k in range(start, stop):
                    cid = int(cell_id_all[k])
                    if cid in seen:
                        continue
                    seen.add(cid)
                    if write != k:
                        cell_id_all[write] = cell_id_all[k]
                        rad_all[write] = rad_all[k]
                        dim0_all[write] = dim0_all[k]
                        dim1_all[write] = dim1_all[k]
                    write += 1
                out = write

        offsets[n_agents] = out

        return _CSR(
            offsets=offsets,
            cell_id=cell_id_all[:out],
            radius=rad_all[:out],
            dim0=dim0_all[:out],
            dim1=dim1_all[:out],
        )

    @staticmethod
    def rank_candidates_array_by_score(csr: _CSR, cand_score: np.ndarray) -> _CSR:
        """Sort candidates within each agent segment using per-candidate scores."""
        offsets = csr.offsets
        out_cell = csr.cell_id.copy()
        out_rad = csr.radius.copy()
        out_d0 = csr.dim0.copy()
        out_d1 = csr.dim1.copy()
        cand_score = np.asarray(cand_score)
        if cand_score.dtype.kind == "f":
            cand_score = np.nan_to_num(cand_score, nan=-np.inf)

        if _numba_enabled() and out_cell.size:
            score_buf = cand_score.astype(np.float64, copy=True)
            _rank_candidates_by_score_kernel(
                offsets,
                out_cell,
                out_rad,
                out_d0,
                out_d1,
                score_buf,
            )
            # score_buf is updated in-place to preserve stability; not returned.
        else:
            n_agents = int(offsets.shape[0] - 1)
            for i in range(n_agents):
                start = int(offsets[i])
                stop = int(offsets[i + 1])
                if stop - start <= 1:
                    continue
                seg = slice(start, stop)
                seg_score = cand_score[seg]
                order = np.lexsort((out_d1[seg], out_d0[seg], out_rad[seg], -seg_score))
                out_cell[seg] = out_cell[seg][order]
                out_rad[seg] = out_rad[seg][order]
                out_d0[seg] = out_d0[seg][order]
                out_d1[seg] = out_d1[seg][order]

        return _CSR(
            offsets=offsets,
            cell_id=out_cell,
            radius=out_rad,
            dim0=out_d0,
            dim1=out_d1,
        )

    @staticmethod
    def rank_candidates_array(csr: _CSR, score_flat: np.ndarray) -> _CSR:
        """Sort candidates within each agent segment.

        Sort key (descending/ascending):
        - score desc
        - radius asc
        - dim0 asc
        - dim1 asc
        """
        offsets = csr.offsets
        cell_id = csr.cell_id
        radius = csr.radius
        dim0 = csr.dim0
        dim1 = csr.dim1

        out_cell = cell_id.copy()
        out_rad = radius.copy()
        out_d0 = dim0.copy()
        out_d1 = dim1.copy()

        # Normalize NaNs to -inf so they always lose.
        score_flat = np.asarray(score_flat)
        if score_flat.dtype.kind in {"f"}:
            score = np.nan_to_num(score_flat, nan=-np.inf)
        else:
            score = score_flat

        if _numba_enabled() and out_cell.size:
            score_buf = score.astype(np.float64, copy=False)
            _rank_candidates_kernel(
                offsets,
                out_cell,
                out_rad,
                out_d0,
                out_d1,
                score_buf,
            )
        else:
            n_agents = int(offsets.shape[0] - 1)
            for i in range(n_agents):
                start = int(offsets[i])
                stop = int(offsets[i + 1])
                if stop - start <= 1:
                    continue
                seg = slice(start, stop)
                seg_cell = out_cell[seg]
                seg_score = score[seg_cell]
                # lexsort uses last key as primary.
                order = np.lexsort((out_d1[seg], out_d0[seg], out_rad[seg], -seg_score))
                out_cell[seg] = seg_cell[order]
                out_rad[seg] = out_rad[seg][order]
                out_d0[seg] = out_d0[seg][order]
                out_d1[seg] = out_d1[seg][order]

        return _CSR(
            offsets=offsets,
            cell_id=out_cell,
            radius=out_rad,
            dim0=out_d0,
            dim1=out_d1,
        )

    @staticmethod
    def resolve_conflicts_lottery(
        *,
        rng: np.random.Generator,
        csr: _CSR,
        origin_cell_id: np.ndarray,
        capacity_flat: np.ndarray,
    ) -> np.ndarray:
        """Resolve destination cells with round-based lottery.

        Algorithm:
        - Each unassigned agent proposes its best remaining candidate.
        - For each cell, if proposals exceed remaining capacity, pick winners uniformly.
        - Losers advance to next candidate and retry in the next round.
        - Agents with no candidates (or only over-capacity candidates) stay put.

        Parameters
        ----------
        rng : np.random.Generator
            NumPy Generator from the model.
        csr : _CSR
            Ranked candidate CSR.
        origin_cell_id : np.ndarray
            Origin cell id per agent (shape (n_agents,)). Used as fallback.
        capacity_flat : np.ndarray
            Remaining capacity per cell id (shape (n_cells,)). This should already
            account for agents leaving their origin cells.

        Returns
        -------
        np.ndarray
            Destination cell id per agent (shape (n_agents,)). Dtype is int64.
        """
        offsets = csr.offsets
        cand_cell = csr.cell_id
        origin = origin_cell_id.astype(np.int64, copy=False)

        if _numba_enabled():
            seed = np.uint64(rng.integers(0, 2**63 - 1, dtype=np.uint64))
            dest, _ = _resolve_conflicts_lottery_kernel(
                offsets.astype(np.int64, copy=False),
                cand_cell.astype(np.int64, copy=False),
                origin.astype(np.int64, copy=False),
                capacity_flat.astype(np.int64, copy=False),
                seed,
            )
            return dest

        n_agents = int(offsets.shape[0] - 1)
        dest = np.full(n_agents, -1, dtype=np.int64)

        if n_agents == 0:
            return origin.astype(np.int64, copy=True)

        cap = capacity_flat.astype(np.int64, copy=True)

        # Pointer into each agent's candidate list.
        ptr = np.zeros(n_agents, dtype=np.int64)
        end = (offsets[1:] - offsets[:-1]).astype(np.int64, copy=False)
        unassigned = np.ones(n_agents, dtype=bool)

        max_candidates = int(end.max(initial=0))
        if max_candidates == 0:
            return origin.astype(np.int64, copy=True)

        # Upper bound: in the worst case we advance a pointer at most
        # `n_agents * max_candidates` times, plus some extra iterations for waiting
        # on origins to be freed.
        max_iters = int(n_agents * max_candidates + n_agents + 1)

        for _ in range(max_iters):
            if not unassigned.any():
                break

            made_progress = False

            # Cells that could become available if an unassigned agent leaves.
            freeable = np.zeros(cap.shape[0], dtype=bool)
            freeable[origin[unassigned]] = True

            active_idx = np.flatnonzero(unassigned)
            if active_idx.size == 0:
                break

            # Build proposals.
            prop_agents: list[int] = []
            prop_cells: list[int] = []
            prop_is_wait: list[bool] = []

            for i in active_idx.tolist():
                base = int(offsets[i])
                seg_len = int(end[i])
                if seg_len <= 0:
                    dest[i] = int(origin[i])
                    unassigned[i] = False
                    made_progress = True
                    continue

                stop = base + seg_len
                _ = stop  # keep local var for parity with old code

                # Advance past permanently impossible candidates.
                while int(ptr[i]) < seg_len:
                    j = base + int(ptr[i])
                    cell = int(cand_cell[j])

                    if cell == int(origin[i]):
                        # Staying never consumes free capacity.
                        dest[i] = cell
                        unassigned[i] = False
                        made_progress = True
                        break

                    if cap[cell] > 0:
                        prop_agents.append(i)
                        prop_cells.append(cell)
                        prop_is_wait.append(False)
                        break

                    # Cell is currently full: if it might be freed later (because the
                    # current occupant is an unassigned mover), wait on it; otherwise
                    # skip it permanently.
                    if freeable[cell]:
                        prop_agents.append(i)
                        prop_cells.append(cell)
                        prop_is_wait.append(True)
                        break

                    ptr[i] += 1
                    made_progress = True

                # No candidates left; stay at origin.
                if unassigned[i] and int(ptr[i]) >= seg_len:
                    dest[i] = int(origin[i])
                    unassigned[i] = False
                    made_progress = True

            if not prop_agents:
                if not made_progress:
                    break
                continue

            prop_agents_arr = np.asarray(prop_agents, dtype=np.int64)
            prop_cells_arr = np.asarray(prop_cells, dtype=np.int64)
            prop_wait_arr = np.asarray(prop_is_wait, dtype=bool)

            # Process proposals grouped by cell id.
            order = np.argsort(prop_cells_arr, kind="stable")
            prop_cells_arr = prop_cells_arr[order]
            prop_agents_arr = prop_agents_arr[order]
            prop_wait_arr = prop_wait_arr[order]

            run_start = 0
            n_props = int(prop_cells_arr.shape[0])
            while run_start < n_props:
                cell = int(prop_cells_arr[run_start])
                run_end = run_start + 1
                while run_end < n_props and int(prop_cells_arr[run_end]) == cell:
                    run_end += 1

                proposers = prop_agents_arr[run_start:run_end]
                waiters = prop_wait_arr[run_start:run_end]
                remaining = int(cap[cell])

                if remaining <= 0:
                    # Cell still full. Waiters keep their pointer; non-waiters advance.
                    non_wait = proposers[~waiters]
                    if non_wait.size:
                        ptr[non_wait] += 1
                        made_progress = True
                else:
                    if proposers.shape[0] <= remaining:
                        winners = proposers
                    else:
                        winners = rng.choice(proposers, size=remaining, replace=False)

                    winners = np.asarray(winners, dtype=np.int64)
                    dest[winners] = cell
                    unassigned[winners] = False
                    cap[cell] -= int(winners.shape[0])
                    made_progress = True

                    # Free origin capacity for winners that moved away.
                    moved_mask = origin[winners] != cell
                    if moved_mask.any():
                        moved_winners = winners[moved_mask]
                        np.add.at(cap, origin[moved_winners], 1)

                    # Losers advance to next candidate.
                    if proposers.shape[0] > remaining:
                        winner_set = {int(x) for x in winners.tolist()}
                        losers = np.array(
                            [
                                int(a)
                                for a in proposers.tolist()
                                if int(a) not in winner_set
                            ],
                            dtype=np.int64,
                        )
                        if losers.size:
                            ptr[losers] += 1

                run_start = run_end

            if not made_progress:
                break

        # Any remaining unassigned agents stay put.
        if unassigned.any():
            dest[unassigned] = origin[unassigned]

        return dest
