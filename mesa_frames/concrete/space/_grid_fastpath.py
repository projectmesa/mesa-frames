"""Internal NumPy/Numba fast paths for grid operations.

This module is internal (not part of the public API).

Design: Grid composes this helper via the private `_GridFastPath` class.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any

import numpy as np


def _prange(n: int) -> range:
    return range(n)


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


def _numba_cache_enabled() -> bool:
    """Return True if Numba on-disk caching should be enabled.

    Controlled via the `MESA_FRAMES_NUMBA_CACHE` environment variable.

    - Unset: defaults to enabled ("1").
    - "0"/"false"/"no"/"off": disabled.
    - "1"/"true"/"yes"/"on": enabled.

    Notes
    -----
    When enabled, Numba will attempt to write cache files into the module's
    `__pycache__` directory. On read-only installs this may warn and/or fall
    back to no cache.
    """
    raw = os.environ.get("MESA_FRAMES_NUMBA_CACHE")
    if raw is None:
        return True
    value = raw.strip().lower()
    if value in {"0", "false", "no", "off"}:
        return False
    if value in {"1", "true", "yes", "on"}:
        return True
    # Defensive default: treat unknown values as enabled.
    return True


# Imported to keep beartype forward-ref resolution happy without introducing a
# circular import at `grid.py` import time (Grid imports this module only from
# `Grid.__init__`).
from mesa_frames.concrete.space.grid import Grid  # noqa: E402


if _NUMBA_AVAILABLE:  # pragma: no cover
    _njit = _numba.njit(cache=_numba_cache_enabled())


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
        dirs_scaled: np.ndarray,
        max_per_agent: int,
        width: int,
        height: int,
        torus: bool,
        dedup_torus: bool,
        include_center: bool,
    ) -> np.ndarray:
        n_agents = int(centers.shape[0])
        n_dirs = int(base_dirs.shape[0])
        counts = np.zeros(n_agents, dtype=np.int64)

        # Local scratch buffer reused across agents (only used for torus de-dupe).
        seen = np.empty(max_per_agent if max_per_agent > 0 else 1, dtype=np.int64)

        for i in range(n_agents):
            cx = int(centers[i, 0])
            cy = int(centers[i, 1])
            rmax = radius_scalar
            if use_per_agent:
                rmax = int(radius_per_agent[i])

            if not torus:
                c = 0
                if include_center:
                    x0 = cx
                    y0 = cy
                    if x0 < 0 or x0 >= width or y0 < 0 or y0 >= height:
                        counts[i] = 0
                        continue
                    c += 1
                for d in range(n_dirs):
                    for r in range(1, rmax + 1):
                        x = cx + int(dirs_scaled[r, d, 0])
                        y = cy + int(dirs_scaled[r, d, 1])
                        if x < 0 or x >= width or y < 0 or y >= height:
                            continue
                        c += 1
                counts[i] = c
                continue

            if torus and not dedup_torus:
                # No in-bounds filtering and no duplicates possible.
                counts[i] = (1 if include_center else 0) + (n_dirs * rmax)
                continue

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
                for r in range(1, rmax + 1):
                    x = cx + int(dirs_scaled[r, d, 0])
                    y = cy + int(dirs_scaled[r, d, 1])
                    if torus:
                        x %= width
                        y %= height
                    else:
                        if x < 0 or x >= width or y < 0 or y >= height:
                            continue
                    cid = x * height + y
                    if dedup_torus:
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
    def _neighbors_table_counts_kernel(
        n_cells: int,
        height: int,
        base_dirs: np.ndarray,
        dirs_scaled: np.ndarray,
        rmax: int,
        torus: bool,
        dedup_torus: bool,
        include_center: bool,
    ) -> np.ndarray:
        """Count candidates per origin cell_id for the precomputed neighbor table."""
        width = int(n_cells) // int(height)
        n_dirs = int(base_dirs.shape[0])

        max_per_cell = (1 if include_center else 0) + (n_dirs * int(rmax))
        counts = np.empty(int(n_cells), dtype=np.int64)

        if torus and not dedup_torus:
            for i in range(int(n_cells)):
                counts[i] = max_per_cell
            return counts

        seen = np.empty(max_per_cell if max_per_cell > 0 else 1, dtype=np.int64)

        for cell in range(int(n_cells)):
            cx = int(cell) // int(height)
            cy = int(cell) - cx * int(height)

            if not torus:
                c = 0
                if include_center:
                    c += 1
                for d in range(n_dirs):
                    for r in range(1, int(rmax) + 1):
                        x = cx + int(dirs_scaled[r, d, 0])
                        y = cy + int(dirs_scaled[r, d, 1])
                        if x < 0 or x >= width or y < 0 or y >= int(height):
                            continue
                        c += 1
                counts[cell] = c
                continue

            # torus + possible de-dupe
            seen_n = 0
            if include_center:
                seen[seen_n] = int(cell)
                seen_n += 1

            for d in range(n_dirs):
                for r in range(1, int(rmax) + 1):
                    x = (cx + int(dirs_scaled[r, d, 0])) % width
                    y = (cy + int(dirs_scaled[r, d, 1])) % int(height)
                    cid = x * int(height) + y
                    if dedup_torus:
                        dup = False
                        for k in range(seen_n):
                            if int(seen[k]) == cid:
                                dup = True
                                break
                        if dup:
                            continue
                    seen[seen_n] = cid
                    seen_n += 1

            counts[cell] = seen_n

        return counts


if _NUMBA_AVAILABLE:  # pragma: no cover

    @_njit
    def _neighbors_table_fill_kernel(
        n_cells: int,
        height: int,
        base_dirs: np.ndarray,
        dirs_scaled: np.ndarray,
        rmax: int,
        torus: bool,
        dedup_torus: bool,
        include_center: bool,
        offsets: np.ndarray,
        cand_cell: np.ndarray,
        cand_rad: np.ndarray,
    ) -> None:
        """Fill cand_cell/cand_rad for the precomputed neighbor table."""
        width = int(n_cells) // int(height)
        n_dirs = int(base_dirs.shape[0])
        max_per_cell = (1 if include_center else 0) + (n_dirs * int(rmax))
        seen = np.empty(max_per_cell if max_per_cell > 0 else 1, dtype=np.int64)

        if torus and not dedup_torus:
            for cell in range(int(n_cells)):
                out = int(offsets[cell])
                cx = int(cell) // int(height)
                cy = int(cell) - cx * int(height)

                if include_center:
                    cand_cell[out] = int(cell)
                    cand_rad[out] = 0
                    out += 1

                for d in range(n_dirs):
                    for r in range(1, int(rmax) + 1):
                        x = (cx + int(dirs_scaled[r, d, 0])) % width
                        y = (cy + int(dirs_scaled[r, d, 1])) % int(height)
                        cand_cell[out] = x * int(height) + y
                        cand_rad[out] = r
                        out += 1
            return

        for cell in range(int(n_cells)):
            out = int(offsets[cell])
            cx = int(cell) // int(height)
            cy = int(cell) - cx * int(height)

            if not torus:
                if include_center:
                    cand_cell[out] = int(cell)
                    cand_rad[out] = 0
                    out += 1
                for d in range(n_dirs):
                    for r in range(1, int(rmax) + 1):
                        x = cx + int(dirs_scaled[r, d, 0])
                        y = cy + int(dirs_scaled[r, d, 1])
                        if x < 0 or x >= width or y < 0 or y >= int(height):
                            continue
                        cand_cell[out] = x * int(height) + y
                        cand_rad[out] = r
                        out += 1
                continue

            # torus + de-dupe
            seen_n = 0
            if include_center:
                seen[seen_n] = int(cell)
                seen_n += 1
                cand_cell[out] = int(cell)
                cand_rad[out] = 0
                out += 1

            for d in range(n_dirs):
                for r in range(1, int(rmax) + 1):
                    x = (cx + int(dirs_scaled[r, d, 0])) % width
                    y = (cy + int(dirs_scaled[r, d, 1])) % int(height)
                    cid = x * int(height) + y
                    if dedup_torus:
                        dup = False
                        for k in range(seen_n):
                            if int(seen[k]) == cid:
                                dup = True
                                break
                        if dup:
                            continue
                    seen[seen_n] = cid
                    seen_n += 1
                    cand_cell[out] = cid
                    cand_rad[out] = r
                    out += 1


if _NUMBA_AVAILABLE:  # pragma: no cover

    @_njit
    def _neighbors_fill_kernel_2d(
        centers: np.ndarray,
        radius_per_agent: np.ndarray,
        radius_scalar: int,
        use_per_agent: bool,
        base_dirs: np.ndarray,
        dirs_scaled: np.ndarray,
        max_per_agent: int,
        width: int,
        height: int,
        torus: bool,
        dedup_torus: bool,
        include_center: bool,
        offsets: np.ndarray,
        out_cell_id: np.ndarray,
        out_rad: np.ndarray,
        out_d0: np.ndarray,
        out_d1: np.ndarray,
    ) -> None:
        n_agents = int(centers.shape[0])
        n_dirs = int(base_dirs.shape[0])

        # Local scratch buffer reused across agents (only used for torus de-dupe).
        seen = np.empty(max_per_agent if max_per_agent > 0 else 1, dtype=np.int64)

        for i in range(n_agents):
            start = int(offsets[i])
            cx = int(centers[i, 0])
            cy = int(centers[i, 1])
            rmax = radius_scalar
            if use_per_agent:
                rmax = int(radius_per_agent[i])

            out = start

            if not torus:
                if include_center:
                    x0 = cx
                    y0 = cy
                    if x0 < 0 or x0 >= width or y0 < 0 or y0 >= height:
                        continue
                    cid0 = x0 * height + y0
                    out_d0[out] = x0
                    out_d1[out] = y0
                    out_rad[out] = 0
                    out_cell_id[out] = cid0
                    out += 1

                for d in range(n_dirs):
                    for r in range(1, rmax + 1):
                        x = cx + int(dirs_scaled[r, d, 0])
                        y = cy + int(dirs_scaled[r, d, 1])
                        if x < 0 or x >= width or y < 0 or y >= height:
                            continue
                        cid = x * height + y
                        out_d0[out] = x
                        out_d1[out] = y
                        out_rad[out] = r
                        out_cell_id[out] = cid
                        out += 1
                continue

            if torus and not dedup_torus:
                if include_center:
                    x0 = cx % width
                    y0 = cy % height
                    cid0 = x0 * height + y0
                    out_d0[out] = x0
                    out_d1[out] = y0
                    out_rad[out] = 0
                    out_cell_id[out] = cid0
                    out += 1

                for d in range(n_dirs):
                    for r in range(1, rmax + 1):
                        x = (cx + int(dirs_scaled[r, d, 0])) % width
                        y = (cy + int(dirs_scaled[r, d, 1])) % height
                        cid = x * height + y
                        out_d0[out] = x
                        out_d1[out] = y
                        out_rad[out] = r
                        out_cell_id[out] = cid
                        out += 1
                continue

            seen_n = 0

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
                for r in range(1, rmax + 1):
                    x = cx + int(dirs_scaled[r, d, 0])
                    y = cy + int(dirs_scaled[r, d, 1])
                    if torus:
                        x %= width
                        y %= height
                    else:
                        if x < 0 or x >= width or y < 0 or y >= height:
                            continue
                    cid = x * height + y
                    if dedup_torus:
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
    def _gather_precomputed_neighbors_kernel(
        origin_cell_id: np.ndarray,
        cell_offsets: np.ndarray,
        cell_cand_cell: np.ndarray,
        cell_cand_rad: np.ndarray,
        out_offsets: np.ndarray,
        out_cell_id: np.ndarray,
        out_rad: np.ndarray,
    ) -> None:
        n_agents = int(origin_cell_id.shape[0])
        for i in range(n_agents):
            cid = int(origin_cell_id[i])
            src_start = int(cell_offsets[cid])
            src_stop = int(cell_offsets[cid + 1])
            dst = int(out_offsets[i])
            for j in range(src_start, src_stop):
                out_cell_id[dst] = cell_cand_cell[j]
                out_rad[dst] = cell_cand_rad[j]
                dst += 1


if _NUMBA_AVAILABLE:  # pragma: no cover

    @_njit
    def _gather_precomputed_neighbors_by_radius_counts_kernel(
        origin_cell_id: np.ndarray,
        radius_per_agent: np.ndarray,
        cell_offsets_by_radius: np.ndarray,
    ) -> np.ndarray:
        """Count candidates for each agent given per-agent radius and stacked tables.

        cell_offsets_by_radius is shape (max_radius + 1, n_cells + 1).
        """
        n_agents = int(origin_cell_id.shape[0])
        counts = np.empty(n_agents, dtype=np.int64)
        for i in range(n_agents):
            cid = int(origin_cell_id[i])
            r = int(radius_per_agent[i])
            start = int(cell_offsets_by_radius[r, cid])
            stop = int(cell_offsets_by_radius[r, cid + 1])
            counts[i] = stop - start
        return counts


if _NUMBA_AVAILABLE:  # pragma: no cover

    @_njit
    def _gather_precomputed_neighbors_by_radius_fill_kernel(
        origin_cell_id: np.ndarray,
        radius_per_agent: np.ndarray,
        cell_offsets_by_radius: np.ndarray,
        cand_starts_by_radius: np.ndarray,
        cand_cell_all: np.ndarray,
        cand_rad_all: np.ndarray,
        out_offsets: np.ndarray,
        out_cell_id: np.ndarray,
        out_rad: np.ndarray,
    ) -> None:
        """Fill candidates for each agent given per-agent radius and stacked tables."""
        n_agents = int(origin_cell_id.shape[0])
        for i in range(n_agents):
            cid = int(origin_cell_id[i])
            r = int(radius_per_agent[i])
            base = int(cand_starts_by_radius[r])
            src_start = base + int(cell_offsets_by_radius[r, cid])
            src_stop = base + int(cell_offsets_by_radius[r, cid + 1])
            dst = int(out_offsets[i])
            for j in range(src_start, src_stop):
                out_cell_id[dst] = cand_cell_all[j]
                out_rad[dst] = cand_rad_all[j]
                dst += 1


if _NUMBA_AVAILABLE:  # pragma: no cover

    @_njit
    def _better_by_score_triplet(
        cell_id: np.ndarray,
        radius: np.ndarray,
        cand_score: np.ndarray,
        i: int,
        j: int,
    ) -> bool:
        si = float(cand_score[i])
        if si != si:
            si = -1e300
        sj = float(cand_score[j])
        if sj != sj:
            sj = -1e300
        if si > sj:
            return True
        if si < sj:
            return False
        ri = int(radius[i])
        rj = int(radius[j])
        if ri < rj:
            return True
        if ri > rj:
            return False
        return int(cell_id[i]) < int(cell_id[j])


if _NUMBA_AVAILABLE:  # pragma: no cover

    @_njit
    def _swap_score_triplet(
        cell_id: np.ndarray,
        radius: np.ndarray,
        cand_score: np.ndarray,
        a: int,
        b: int,
    ) -> None:
        cell_id[a], cell_id[b] = cell_id[b], cell_id[a]
        radius[a], radius[b] = radius[b], radius[a]
        cand_score[a], cand_score[b] = cand_score[b], cand_score[a]


if _NUMBA_AVAILABLE:  # pragma: no cover

    @_njit
    def _sift_down_score_triplet(
        cell_id: np.ndarray,
        radius: np.ndarray,
        cand_score: np.ndarray,
        base: int,
        size: int,
        root: int,
    ) -> None:
        while True:
            left = 2 * root + 1
            if left >= size:
                return
            right = left + 1
            best = left
            if right < size and _better_by_score_triplet(
                cell_id, radius, cand_score, base + right, base + left
            ):
                best = right
            if _better_by_score_triplet(
                cell_id, radius, cand_score, base + best, base + root
            ):
                _swap_score_triplet(
                    cell_id, radius, cand_score, base + root, base + best
                )
                root = best
            else:
                return


if _NUMBA_AVAILABLE:  # pragma: no cover

    @_njit
    def _heapsort_score_triplet(
        cell_id: np.ndarray,
        radius: np.ndarray,
        cand_score: np.ndarray,
        start: int,
        stop: int,
    ) -> None:
        seg_len = stop - start
        if seg_len <= 1:
            return

        # Heapify (max-heap: best at root)
        for root in range((seg_len // 2) - 1, -1, -1):
            _sift_down_score_triplet(cell_id, radius, cand_score, start, seg_len, root)

        # Extract
        for end in range(seg_len - 1, 0, -1):
            _swap_score_triplet(cell_id, radius, cand_score, start, start + end)
            _sift_down_score_triplet(cell_id, radius, cand_score, start, end, 0)

        # Reverse worst..best -> best..worst
        lo = start
        hi = stop - 1
        while lo < hi:
            _swap_score_triplet(cell_id, radius, cand_score, lo, hi)
            lo += 1
            hi -= 1


if _NUMBA_AVAILABLE:  # pragma: no cover

    @_njit
    def _better_by_flat_triplet(
        cell_id: np.ndarray,
        radius: np.ndarray,
        score_flat: np.ndarray,
        i: int,
        j: int,
    ) -> bool:
        si = float(score_flat[int(cell_id[i])])
        sj = float(score_flat[int(cell_id[j])])
        if si > sj:
            return True
        if si < sj:
            return False
        ri = int(radius[i])
        rj = int(radius[j])
        if ri < rj:
            return True
        if ri > rj:
            return False
        return int(cell_id[i]) < int(cell_id[j])


if _NUMBA_AVAILABLE:  # pragma: no cover

    @_njit
    def _swap_flat_triplet(
        cell_id: np.ndarray,
        radius: np.ndarray,
        a: int,
        b: int,
    ) -> None:
        cell_id[a], cell_id[b] = cell_id[b], cell_id[a]
        radius[a], radius[b] = radius[b], radius[a]


if _NUMBA_AVAILABLE:  # pragma: no cover

    @_njit
    def _sift_down_flat_triplet(
        cell_id: np.ndarray,
        radius: np.ndarray,
        score_flat: np.ndarray,
        base: int,
        size: int,
        root: int,
    ) -> None:
        while True:
            left = 2 * root + 1
            if left >= size:
                return
            right = left + 1
            best = left
            if right < size and _better_by_flat_triplet(
                cell_id, radius, score_flat, base + right, base + left
            ):
                best = right
            if _better_by_flat_triplet(
                cell_id, radius, score_flat, base + best, base + root
            ):
                _swap_flat_triplet(cell_id, radius, base + root, base + best)
                root = best
            else:
                return


if _NUMBA_AVAILABLE:  # pragma: no cover

    @_njit
    def _heapsort_flat_triplet(
        cell_id: np.ndarray,
        radius: np.ndarray,
        score_flat: np.ndarray,
        start: int,
        stop: int,
    ) -> None:
        seg_len = stop - start
        if seg_len <= 1:
            return

        for root in range((seg_len // 2) - 1, -1, -1):
            _sift_down_flat_triplet(cell_id, radius, score_flat, start, seg_len, root)
        for end in range(seg_len - 1, 0, -1):
            _swap_flat_triplet(cell_id, radius, start, start + end)
            _sift_down_flat_triplet(cell_id, radius, score_flat, start, end, 0)
        lo = start
        hi = stop - 1
        while lo < hi:
            _swap_flat_triplet(cell_id, radius, lo, hi)
            lo += 1
            hi -= 1


if _NUMBA_AVAILABLE:  # pragma: no cover

    @_njit
    def _rank_candidates_kernel(
        offsets: np.ndarray,
        cell_id: np.ndarray,
        radius: np.ndarray,
        score_flat: np.ndarray,
        k_thresh: int = 24,
    ) -> None:
        """Sort candidates in-place within each CSR segment.

        Comparator (desc/asc/asc): score(cell_id), radius, cell_id.
        """
        n_agents = int(offsets.shape[0] - 1)
        for i in range(n_agents):
            start = int(offsets[i])
            stop = int(offsets[i + 1])
            seg_len = stop - start
            if seg_len <= 1:
                continue

            if seg_len <= k_thresh:
                for j in range(start + 1, stop):
                    key_cell = int(cell_id[j])
                    key_rad = int(radius[j])
                    key_score = float(score_flat[key_cell])

                    k = j - 1
                    while k >= start:
                        cur_cell = int(cell_id[k])
                        cur_score = float(score_flat[cur_cell])
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
                                better = int(cell_id[k]) <= key_cell

                        if better:
                            break

                        cell_id[k + 1] = cell_id[k]
                        radius[k + 1] = radius[k]
                        k -= 1

                    cell_id[k + 1] = key_cell
                    radius[k + 1] = key_rad
                continue

            # Large-K: heapsort (O(K log K)) without calling global helpers.
            size = seg_len

            # Heapify (max-heap: best at root)
            for root in range((size // 2) - 1, -1, -1):
                r = root
                while True:
                    left = 2 * r + 1
                    if left >= size:
                        break
                    best = left
                    right = left + 1
                    if right < size:
                        ia = start + right
                        ib = start + left
                        sa = float(score_flat[int(cell_id[ia])])
                        sb = float(score_flat[int(cell_id[ib])])
                        better = False
                        if sa > sb:
                            better = True
                        elif sa < sb:
                            better = False
                        else:
                            ra = int(radius[ia])
                            rb = int(radius[ib])
                            if ra < rb:
                                better = True
                            elif ra > rb:
                                better = False
                            else:
                                better = int(cell_id[ia]) < int(cell_id[ib])
                        if better:
                            best = right

                    ia = start + best
                    ib = start + r
                    sa = float(score_flat[int(cell_id[ia])])
                    sb = float(score_flat[int(cell_id[ib])])
                    better = False
                    if sa > sb:
                        better = True
                    elif sa < sb:
                        better = False
                    else:
                        ra = int(radius[ia])
                        rb = int(radius[ib])
                        if ra < rb:
                            better = True
                        elif ra > rb:
                            better = False
                        else:
                            better = int(cell_id[ia]) < int(cell_id[ib])

                    if better:
                        a = start + r
                        b = start + best
                        cell_id[a], cell_id[b] = cell_id[b], cell_id[a]
                        radius[a], radius[b] = radius[b], radius[a]
                        r = best
                    else:
                        break

            # Extract max repeatedly to end.
            for end in range(size - 1, 0, -1):
                a = start
                b = start + end
                cell_id[a], cell_id[b] = cell_id[b], cell_id[a]
                radius[a], radius[b] = radius[b], radius[a]

                r = 0
                while True:
                    left = 2 * r + 1
                    if left >= end:
                        break
                    best = left
                    right = left + 1
                    if right < end:
                        ia = start + right
                        ib = start + left
                        sa = float(score_flat[int(cell_id[ia])])
                        sb = float(score_flat[int(cell_id[ib])])
                        better = False
                        if sa > sb:
                            better = True
                        elif sa < sb:
                            better = False
                        else:
                            ra = int(radius[ia])
                            rb = int(radius[ib])
                            if ra < rb:
                                better = True
                            elif ra > rb:
                                better = False
                            else:
                                better = int(cell_id[ia]) < int(cell_id[ib])
                        if better:
                            best = right

                    ia = start + best
                    ib = start + r
                    sa = float(score_flat[int(cell_id[ia])])
                    sb = float(score_flat[int(cell_id[ib])])
                    better = False
                    if sa > sb:
                        better = True
                    elif sa < sb:
                        better = False
                    else:
                        ra = int(radius[ia])
                        rb = int(radius[ib])
                        if ra < rb:
                            better = True
                        elif ra > rb:
                            better = False
                        else:
                            better = int(cell_id[ia]) < int(cell_id[ib])

                    if better:
                        a = start + r
                        b = start + best
                        cell_id[a], cell_id[b] = cell_id[b], cell_id[a]
                        radius[a], radius[b] = radius[b], radius[a]
                        r = best
                    else:
                        break

            # Reverse worst..best -> best..worst.
            lo = start
            hi = stop - 1
            while lo < hi:
                cell_id[lo], cell_id[hi] = cell_id[hi], cell_id[lo]
                radius[lo], radius[hi] = radius[hi], radius[lo]
                lo += 1
                hi -= 1


if _NUMBA_AVAILABLE:  # pragma: no cover

    @_njit
    def _rank_candidates_by_score_kernel(
        offsets: np.ndarray,
        cell_id: np.ndarray,
        radius: np.ndarray,
        cand_score: np.ndarray,
        k_thresh: int = 24,
    ) -> None:
        """Sort candidates in-place within each CSR segment.

        Comparator (desc/asc/asc): score, radius, cell_id.
        Note: cell_id ordering is equivalent to (dim0, dim1) ordering because
        cell_id = dim0 * height + dim1 with constant height.
        """
        n_agents = int(offsets.shape[0] - 1)
        for i in range(n_agents):
            start = int(offsets[i])
            stop = int(offsets[i + 1])
            seg_len = stop - start
            if seg_len <= 1:
                continue

            # Small-K: insertion sort (lower constant factors).
            if seg_len <= k_thresh:
                for j in range(start + 1, stop):
                    key_cell = int(cell_id[j])
                    key_rad = int(radius[j])
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
                                better = int(cell_id[k]) <= key_cell

                        if better:
                            break

                        cell_id[k + 1] = cell_id[k]
                        radius[k + 1] = radius[k]
                        cand_score[k + 1] = cand_score[k]
                        k -= 1

                    cell_id[k + 1] = key_cell
                    radius[k + 1] = key_rad
                    cand_score[k + 1] = key_score
                continue

            # Large-K: heapsort (O(K log K)) without calling global helpers.
            size = seg_len

            # Heapify (max-heap: best at root)
            for root in range((size // 2) - 1, -1, -1):
                r = root
                while True:
                    left = 2 * r + 1
                    if left >= size:
                        break
                    best = left
                    right = left + 1
                    if right < size:
                        ia = start + right
                        ib = start + left
                        sa = float(cand_score[ia])
                        if sa != sa:
                            sa = -1e300
                        sb = float(cand_score[ib])
                        if sb != sb:
                            sb = -1e300
                        better = False
                        if sa > sb:
                            better = True
                        elif sa < sb:
                            better = False
                        else:
                            ra = int(radius[ia])
                            rb = int(radius[ib])
                            if ra < rb:
                                better = True
                            elif ra > rb:
                                better = False
                            else:
                                better = int(cell_id[ia]) < int(cell_id[ib])
                        if better:
                            best = right

                    ia = start + best
                    ib = start + r
                    sa = float(cand_score[ia])
                    if sa != sa:
                        sa = -1e300
                    sb = float(cand_score[ib])
                    if sb != sb:
                        sb = -1e300
                    better = False
                    if sa > sb:
                        better = True
                    elif sa < sb:
                        better = False
                    else:
                        ra = int(radius[ia])
                        rb = int(radius[ib])
                        if ra < rb:
                            better = True
                        elif ra > rb:
                            better = False
                        else:
                            better = int(cell_id[ia]) < int(cell_id[ib])

                    if better:
                        a = start + r
                        b = start + best
                        cell_id[a], cell_id[b] = cell_id[b], cell_id[a]
                        radius[a], radius[b] = radius[b], radius[a]
                        cand_score[a], cand_score[b] = cand_score[b], cand_score[a]
                        r = best
                    else:
                        break

            # Extract max repeatedly to end.
            for end in range(size - 1, 0, -1):
                a = start
                b = start + end
                cell_id[a], cell_id[b] = cell_id[b], cell_id[a]
                radius[a], radius[b] = radius[b], radius[a]
                cand_score[a], cand_score[b] = cand_score[b], cand_score[a]

                r = 0
                while True:
                    left = 2 * r + 1
                    if left >= end:
                        break
                    best = left
                    right = left + 1
                    if right < end:
                        ia = start + right
                        ib = start + left
                        sa = float(cand_score[ia])
                        if sa != sa:
                            sa = -1e300
                        sb = float(cand_score[ib])
                        if sb != sb:
                            sb = -1e300
                        better = False
                        if sa > sb:
                            better = True
                        elif sa < sb:
                            better = False
                        else:
                            ra = int(radius[ia])
                            rb = int(radius[ib])
                            if ra < rb:
                                better = True
                            elif ra > rb:
                                better = False
                            else:
                                better = int(cell_id[ia]) < int(cell_id[ib])
                        if better:
                            best = right

                    ia = start + best
                    ib = start + r
                    sa = float(cand_score[ia])
                    if sa != sa:
                        sa = -1e300
                    sb = float(cand_score[ib])
                    if sb != sb:
                        sb = -1e300
                    better = False
                    if sa > sb:
                        better = True
                    elif sa < sb:
                        better = False
                    else:
                        ra = int(radius[ia])
                        rb = int(radius[ib])
                        if ra < rb:
                            better = True
                        elif ra > rb:
                            better = False
                        else:
                            better = int(cell_id[ia]) < int(cell_id[ib])

                    if better:
                        a = start + r
                        b = start + best
                        cell_id[a], cell_id[b] = cell_id[b], cell_id[a]
                        radius[a], radius[b] = radius[b], radius[a]
                        cand_score[a], cand_score[b] = cand_score[b], cand_score[a]
                        r = best
                    else:
                        break

            # Reverse worst..best -> best..worst.
            lo = start
            hi = stop - 1
            while lo < hi:
                cell_id[lo], cell_id[hi] = cell_id[hi], cell_id[lo]
                radius[lo], radius[hi] = radius[hi], radius[lo]
                cand_score[lo], cand_score[hi] = cand_score[hi], cand_score[lo]
                lo += 1
                hi -= 1


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

        n_cells = int(cap.shape[0])

        # Reusable stamp array for "freeable" cells (avoid allocating/zeroing
        # a full n_cells array each round).
        freeable_stamp = np.zeros(cap.shape[0], dtype=np.int32)
        gen = np.int32(1)

        # Scratch for proposals (<= n_agents each round).
        prop_agents = np.empty(n_agents, dtype=np.int64)
        prop_cells = np.empty(n_agents, dtype=np.int64)
        prop_wait = np.empty(n_agents, dtype=np.uint8)

        # Proposal bucketing by cell_id (avoid per-round argsort allocations).
        # We keep deterministic order by:
        # - iterating cells in ascending cell_id
        # - filling each cell bucket in proposal index order
        bucket_stamp = np.zeros(n_cells, dtype=np.int32)
        bucket_counts = np.zeros(n_cells, dtype=np.int32)
        bucket_start = np.empty(n_cells, dtype=np.int64)
        bucket_next = np.empty(n_cells, dtype=np.int64)
        bucket_prop = np.empty(n_agents, dtype=np.int64)
        bucket_gen = np.int32(1)

        # Reusable scratch for winner selection.
        tmp_agents = np.empty(n_agents, dtype=np.int64)
        winner_stamp = np.zeros(n_agents, dtype=np.int32)
        winner_gen = np.int32(1)

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

            # Bump generation; reset on overflow.
            gen = np.int32(gen + 1)
            if gen == np.int32(0):
                freeable_stamp[:] = 0
                gen = np.int32(1)

            for i in range(n_agents):
                if unassigned[i] == 1:
                    freeable_stamp[int(origin[i])] = gen

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

                    if freeable_stamp[cell] == gen:
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

            # Bucket proposals by destination cell (stable by proposal index).
            bucket_gen = np.int32(bucket_gen + 1)
            if bucket_gen == np.int32(0):
                bucket_stamp[:] = 0
                bucket_gen = np.int32(1)

            for p in range(prop_n):
                cell = int(prop_cells[p])
                if bucket_stamp[cell] != bucket_gen:
                    bucket_stamp[cell] = bucket_gen
                    bucket_counts[cell] = np.int32(1)
                else:
                    bucket_counts[cell] = np.int32(bucket_counts[cell] + 1)

            # Prefix sum over cells to compute bucket slices.
            pos = np.int64(0)
            for cell in range(n_cells):
                if bucket_stamp[cell] == bucket_gen:
                    bucket_start[cell] = pos
                    bucket_next[cell] = pos
                    pos += np.int64(bucket_counts[cell])

            # Fill buckets in proposal index order.
            for p in range(prop_n):
                cell = int(prop_cells[p])
                idx = int(bucket_next[cell])
                bucket_prop[idx] = p
                bucket_next[cell] = np.int64(idx + 1)

            # Walk cells in ascending cell_id (matches sorted-by-cell behavior).
            for cell in range(n_cells):
                if bucket_stamp[cell] != bucket_gen:
                    continue

                start = int(bucket_start[cell])
                end = int(bucket_next[cell])
                run_len = end - start
                if run_len <= 0:
                    continue

                remaining = int(cap[cell])

                if remaining <= 0:
                    # Full: non-waiters advance.
                    for t in range(start, end):
                        ii = int(bucket_prop[t])
                        a = int(prop_agents[ii])
                        if prop_wait[ii] == 0:
                            ptr[a] += 1
                            made_progress = True
                    continue

                # Pick winners without replacement (simple in-place shuffle).
                k = remaining
                if k > run_len:
                    k = run_len

                for t in range(run_len):
                    ii = int(bucket_prop[start + t])
                    tmp_agents[t] = int(prop_agents[ii])

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
                    tmp_agents[t], tmp_agents[j] = tmp_agents[j], tmp_agents[t]

                # Stamp winners for loser detection.
                winner_gen = np.int32(winner_gen + 1)
                if winner_gen == np.int32(0):
                    winner_stamp[:] = 0
                    winner_gen = np.int32(1)

                for t in range(k):
                    w = int(tmp_agents[t])
                    dest[w] = cell
                    unassigned[w] = 0
                    winner_stamp[w] = winner_gen

                cap[cell] -= k
                made_progress = True

                # Free origins for winners that moved.
                for t in range(k):
                    w = int(tmp_agents[t])
                    if int(origin[w]) != cell:
                        cap[int(origin[w])] += 1

                # Losers advance.
                for t in range(start, end):
                    ii = int(bucket_prop[t])
                    a = int(prop_agents[ii])
                    if winner_stamp[a] != winner_gen:
                        ptr[a] += 1

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
        # Cache of precomputed neighbor tables keyed by:
        # (width, height, torus, include_center, radius, base_dirs_hash, dedup_torus)
        # Value: dict with cell_offsets, cand_cell, cand_rad.
        self._neighbors_by_cell_cache: dict[tuple[Any, ...], dict[str, np.ndarray]] = {}

        # Cache of stacked neighbor tables for per-agent varying radii.
        # Key: (width, height, torus, include_center, max_radius, base_dirs_hash)
        # Value: dict with cell_offsets_by_radius, cand_starts_by_radius,
        #        cand_cell_all, cand_rad_all.
        self._neighbors_by_cell_stacked_cache: dict[
            tuple[Any, ...], dict[str, np.ndarray]
        ] = {}

    def _neighbors_tables_stacked_by_radius(
        self,
        *,
        max_radius: int,
        include_center: bool,
    ) -> dict[str, np.ndarray]:
        grid = self._grid
        width = int(grid._dimensions[0])
        height = int(grid._dimensions[1])
        torus = bool(grid._torus)

        base_dirs = (
            grid._offsets.select(grid._pos_col_names)
            .to_numpy()
            .astype(np.int64, copy=False)
        )
        base_dirs_hash = hash(base_dirs.tobytes())

        key = (
            width,
            height,
            torus,
            bool(include_center),
            int(max_radius),
            base_dirs_hash,
        )
        cached = self._neighbors_by_cell_stacked_cache.get(key)
        if cached is not None:
            return cached

        n_cells = width * height
        rmax = int(max_radius)
        n_r = rmax + 1

        cell_offsets_by_radius = np.empty((n_r, n_cells + 1), dtype=np.int64)
        cand_starts_by_radius = np.zeros(n_r, dtype=np.int64)

        totals = np.empty(n_r, dtype=np.int64)
        for r in range(n_r):
            tbl = self._neighbors_table_by_cell_id(
                radius=r, include_center=include_center
            )
            cell_offsets_by_radius[r, :] = tbl["cell_offsets"].astype(
                np.int64, copy=False
            )
            totals[r] = int(cell_offsets_by_radius[r, -1])

        # Prefix sum over per-radius totals to compute global starts.
        for r in range(1, n_r):
            cand_starts_by_radius[r] = cand_starts_by_radius[r - 1] + totals[r - 1]

        grand_total = int(cand_starts_by_radius[-1] + totals[-1])
        cand_cell_all = np.empty(grand_total, dtype=np.int64)
        cand_rad_all = np.empty(grand_total, dtype=np.int64)

        for r in range(n_r):
            tbl = self._neighbors_table_by_cell_id(
                radius=r, include_center=include_center
            )
            start = int(cand_starts_by_radius[r])
            stop = start + int(totals[r])
            cand_cell_all[start:stop] = tbl["cand_cell"].astype(np.int64, copy=False)
            cand_rad_all[start:stop] = tbl["cand_rad"].astype(np.int64, copy=False)

        stacked = {
            "cell_offsets_by_radius": cell_offsets_by_radius,
            "cand_starts_by_radius": cand_starts_by_radius,
            "cand_cell_all": cand_cell_all,
            "cand_rad_all": cand_rad_all,
        }
        self._neighbors_by_cell_stacked_cache[key] = stacked
        return stacked

    @property
    def numba_enabled(self) -> bool:
        return _numba_enabled()

    def csr(
        self,
        *,
        offsets: np.ndarray,
        cell_id: np.ndarray,
        radius: np.ndarray,
    ) -> _CSR:
        return _CSR(
            offsets=np.asarray(offsets),
            cell_id=np.asarray(cell_id),
            radius=np.asarray(radius),
        )

    def _neighbors_table_by_cell_id(
        self,
        *,
        radius: int,
        include_center: bool,
    ) -> dict[str, np.ndarray]:
        grid = self._grid
        width = int(grid._dimensions[0])
        height = int(grid._dimensions[1])
        torus = bool(grid._torus)
        rmax = int(radius)

        base_dirs = (
            grid._offsets.select(grid._pos_col_names)
            .to_numpy()
            .astype(np.int64, copy=False)
        )
        base_dirs_hash = hash(base_dirs.tobytes())

        # If neighborhood is strictly contained, duplicates are impossible.
        dedup_torus = bool(torus and not ((2 * rmax) < width and (2 * rmax) < height))

        key = (
            width,
            height,
            torus,
            bool(include_center),
            rmax,
            base_dirs_hash,
            dedup_torus,
        )
        cached = self._neighbors_by_cell_cache.get(key)
        if cached is not None:
            return cached

        use_numba = _numba_enabled()

        n_cells = width * height
        n_dirs = int(base_dirs.shape[0])
        # Upper bound per cell.
        max_per_cell = (1 if include_center else 0) + (n_dirs * rmax)

        if use_numba:
            dirs_scaled = (
                base_dirs[np.newaxis, :, :]
                * np.arange(rmax + 1, dtype=np.int64)[:, np.newaxis, np.newaxis]
            )
            counts = _neighbors_table_counts_kernel(
                int(n_cells),
                int(height),
                base_dirs,
                dirs_scaled,
                int(rmax),
                bool(torus),
                bool(dedup_torus),
                bool(include_center),
            )
        else:
            # Pass 1: counts per cell.
            counts = np.empty(n_cells, dtype=np.int64)
            for cell in range(n_cells):
                x0 = int(cell // height)
                y0 = int(cell % height)
                if torus and not dedup_torus:
                    counts[cell] = max_per_cell
                    continue

                c = 0
                if include_center:
                    c += 1
                if not torus:
                    for d in range(n_dirs):
                        dx0 = int(base_dirs[d, 0])
                        dy0 = int(base_dirs[d, 1])
                        for r in range(1, rmax + 1):
                            x = x0 + dx0 * r
                            y = y0 + dy0 * r
                            if x < 0 or x >= width or y < 0 or y >= height:
                                continue
                            c += 1
                    counts[cell] = c
                    continue

                # torus + possible de-dupe
                seen_n = 0
                seen = np.empty(max_per_cell if max_per_cell > 0 else 1, dtype=np.int64)
                if include_center:
                    seen[seen_n] = cell
                    seen_n += 1
                for d in range(n_dirs):
                    dx0 = int(base_dirs[d, 0])
                    dy0 = int(base_dirs[d, 1])
                    for r in range(1, rmax + 1):
                        x = (x0 + dx0 * r) % width
                        y = (y0 + dy0 * r) % height
                        cid = x * height + y
                        if dedup_torus:
                            dup = False
                            for k in range(seen_n):
                                if int(seen[k]) == cid:
                                    dup = True
                                    break
                            if dup:
                                continue
                        seen[seen_n] = cid
                        seen_n += 1
                counts[cell] = seen_n

        cell_offsets = np.empty(n_cells + 1, dtype=np.int64)
        cell_offsets[0] = 0
        np.cumsum(counts, out=cell_offsets[1:])
        total = int(cell_offsets[-1])
        cand_cell = np.empty(
            total, dtype=np.int32 if n_cells <= np.iinfo(np.int32).max else np.int64
        )
        cand_rad = np.empty(
            total, dtype=np.uint8 if rmax <= np.iinfo(np.uint8).max else np.int32
        )

        if use_numba:
            _neighbors_table_fill_kernel(
                int(n_cells),
                int(height),
                base_dirs,
                dirs_scaled,
                int(rmax),
                bool(torus),
                bool(dedup_torus),
                bool(include_center),
                cell_offsets,
                cand_cell,
                cand_rad,
            )
        else:
            # Pass 2: fill arrays.
            out = 0
            for cell in range(n_cells):
                x0 = int(cell // height)
                y0 = int(cell % height)

                if torus and not dedup_torus:
                    if include_center:
                        cand_cell[out] = cell
                        cand_rad[out] = 0
                        out += 1
                    for d in range(n_dirs):
                        dx0 = int(base_dirs[d, 0])
                        dy0 = int(base_dirs[d, 1])
                        for r in range(1, rmax + 1):
                            x = (x0 + dx0 * r) % width
                            y = (y0 + dy0 * r) % height
                            cand_cell[out] = x * height + y
                            cand_rad[out] = r
                            out += 1
                    continue

                if not torus:
                    if include_center:
                        cand_cell[out] = cell
                        cand_rad[out] = 0
                        out += 1
                    for d in range(n_dirs):
                        dx0 = int(base_dirs[d, 0])
                        dy0 = int(base_dirs[d, 1])
                        for r in range(1, rmax + 1):
                            x = x0 + dx0 * r
                            y = y0 + dy0 * r
                            if x < 0 or x >= width or y < 0 or y >= height:
                                continue
                            cand_cell[out] = x * height + y
                            cand_rad[out] = r
                            out += 1
                    continue

                # torus + de-dupe
                seen_n = 0
                seen = np.empty(max_per_cell if max_per_cell > 0 else 1, dtype=np.int64)
                if include_center:
                    seen[seen_n] = cell
                    seen_n += 1
                    cand_cell[out] = cell
                    cand_rad[out] = 0
                    out += 1
                for d in range(n_dirs):
                    dx0 = int(base_dirs[d, 0])
                    dy0 = int(base_dirs[d, 1])
                    for r in range(1, rmax + 1):
                        x = (x0 + dx0 * r) % width
                        y = (y0 + dy0 * r) % height
                        cid = x * height + y
                        if dedup_torus:
                            dup = False
                            for k in range(seen_n):
                                if int(seen[k]) == cid:
                                    dup = True
                                    break
                            if dup:
                                continue
                        seen[seen_n] = cid
                        seen_n += 1
                        cand_cell[out] = cid
                        cand_rad[out] = r
                        out += 1

        tbl = {
            "cell_offsets": cell_offsets,
            "cand_cell": cand_cell,
            "cand_rad": cand_rad,
            "height": np.array([height], dtype=np.int64),
        }
        self._neighbors_by_cell_cache[key] = tbl
        return tbl

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

            max_radius = rad_scalar
            if use_per_agent:
                if int(rad_arr.shape[0]) == 0:
                    max_radius = 0
                else:
                    max_radius = int(np.max(rad_arr))

            # Compute origin cell_id (centers are already within bounds).
            origin_cell_id = centers[:, 0].astype(
                np.int64, copy=False
            ) * height + centers[:, 1].astype(np.int64, copy=False)
            if torus:
                origin_cell_id = (
                    centers[:, 0].astype(np.int64, copy=False) % width
                ) * height + (centers[:, 1].astype(np.int64, copy=False) % height)

            # Fast path: use precomputed neighbor tables by origin cell_id.
            n_cells = width * height
            cell_id_dtype = np.int32 if n_cells <= np.iinfo(np.int32).max else np.int64
            radius_dtype = (
                np.uint8 if max_radius <= np.iinfo(np.uint8).max else np.int32
            )

            if use_per_agent:
                stacked = self._neighbors_tables_stacked_by_radius(
                    max_radius=max_radius,
                    include_center=include_center,
                )
                cell_offsets_by_radius = stacked["cell_offsets_by_radius"]
                counts = _gather_precomputed_neighbors_by_radius_counts_kernel(
                    origin_cell_id.astype(np.int64, copy=False),
                    rad_arr,
                    cell_offsets_by_radius,
                )

                offsets = np.empty(n_agents + 1, dtype=np.int64)
                offsets[0] = 0
                np.cumsum(counts, out=offsets[1:])
                total = int(offsets[-1])
                cell_id_all = np.empty(total, dtype=cell_id_dtype)
                rad_all = np.empty(total, dtype=radius_dtype)

                _gather_precomputed_neighbors_by_radius_fill_kernel(
                    origin_cell_id.astype(np.int64, copy=False),
                    rad_arr,
                    cell_offsets_by_radius,
                    stacked["cand_starts_by_radius"],
                    stacked["cand_cell_all"],
                    stacked["cand_rad_all"],
                    offsets.astype(np.int64, copy=False),
                    cell_id_all,
                    rad_all,
                )
            else:
                tbl = self._neighbors_table_by_cell_id(
                    radius=max_radius, include_center=include_center
                )
                cell_offsets = tbl["cell_offsets"].astype(np.int64, copy=False)
                cell_counts = cell_offsets[1:] - cell_offsets[:-1]
                counts = cell_counts[origin_cell_id]

                offsets = np.empty(n_agents + 1, dtype=np.int64)
                offsets[0] = 0
                np.cumsum(counts, out=offsets[1:])
                total = int(offsets[-1])
                cell_id_all = np.empty(total, dtype=cell_id_dtype)
                rad_all = np.empty(total, dtype=radius_dtype)

                _gather_precomputed_neighbors_kernel(
                    origin_cell_id.astype(np.int64, copy=False),
                    cell_offsets,
                    tbl["cand_cell"].astype(np.int64, copy=False),
                    tbl["cand_rad"].astype(np.int64, copy=False),
                    offsets.astype(np.int64, copy=False),
                    cell_id_all,
                    rad_all,
                )

            return _CSR(offsets=offsets, cell_id=cell_id_all, radius=rad_all)

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
        offsets = np.zeros(n_agents + 1, dtype=np.int64)

        width = int(grid._dimensions[0])
        height = int(grid._dimensions[1])

        dedup_torus = bool(
            grid._torus
            and (
                2
                * (
                    int(radius_scalar)
                    if radius_per_agent is None
                    else int(np.max(radius_per_agent, initial=0))
                )
                >= width
                or 2
                * (
                    int(radius_scalar)
                    if radius_per_agent is None
                    else int(np.max(radius_per_agent, initial=0))
                )
                >= height
            )
        )

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
                    rad_all[out] = r
                    cell_id_all[out] = x * height + y
                    out += 1

            # Torus mode de-dupes (agent_id, coords) pairs with keep="first".
            if grid._torus and dedup_torus and out > offsets[i] + 1:
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
                    write += 1
                out = write

        offsets[n_agents] = out

        return _CSR(
            offsets=offsets,
            cell_id=cell_id_all[:out],
            radius=rad_all[:out],
        )

    @staticmethod
    def rank_candidates_array_by_score(
        csr: _CSR,
        cand_score: np.ndarray,
        *,
        height: int,
    ) -> _CSR:
        """Sort candidates within each agent segment using per-candidate scores."""
        offsets = csr.offsets
        out_cell = csr.cell_id.copy()
        out_rad = csr.radius.copy()
        cand_score = np.asarray(cand_score)
        if cand_score.dtype.kind == "f":
            cand_score = np.nan_to_num(cand_score, nan=-np.inf)

        if _numba_enabled() and out_cell.size:
            score_buf = cand_score.astype(np.float64, copy=True)
            _rank_candidates_by_score_kernel(
                offsets,
                out_cell,
                out_rad,
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
                seg_cell = out_cell[seg]
                h = int(height)
                seg_d0 = seg_cell // h
                seg_d1 = seg_cell % h
                order = np.lexsort((seg_d1, seg_d0, out_rad[seg], -seg_score))
                out_cell[seg] = out_cell[seg][order]
                out_rad[seg] = out_rad[seg][order]

        return _CSR(
            offsets=offsets,
            cell_id=out_cell,
            radius=out_rad,
        )

    @staticmethod
    def rank_candidates_array(
        csr: _CSR, score_flat: np.ndarray, *, height: int
    ) -> _CSR:
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

        out_cell = cell_id.copy()
        out_rad = radius.copy()

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
                h = int(height)
                seg_d0 = seg_cell // h
                seg_d1 = seg_cell % h
                # lexsort uses last key as primary.
                order = np.lexsort((seg_d1, seg_d0, out_rad[seg], -seg_score))
                out_cell[seg] = seg_cell[order]
                out_rad[seg] = out_rad[seg][order]

        return _CSR(
            offsets=offsets,
            cell_id=out_cell,
            radius=out_rad,
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
