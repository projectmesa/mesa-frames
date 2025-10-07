"""Mesa agents for the Sugarscape IG example (sequential/asynchronous update).

Implements the movement rule (sense along cardinal axes up to `vision`, choose
highest-sugar cell with tie-breakers by distance and coordinates). Eating,
starvation, and regrowth are orchestrated by the model to preserve the order
move -> eat -> regrow -> collect, mirroring the tutorial schedule.
"""

from __future__ import annotations

from typing import Tuple

import mesa


class AntAgent(mesa.Agent):
    """Sugarscape ant with sugar/metabolism/vision traits and movement."""

    def __init__(
        self,
        model: "Sugarscape",
        *,
        sugar: int,
        metabolism: int,
        vision: int,
    ) -> None:
        super().__init__(model)
        self.sugar = int(sugar)
        self.metabolism = int(metabolism)
        self.vision = int(vision)

    # --- Movement helpers (sequential/asynchronous) ---

    def _visible_cells(self, origin: Tuple[int, int]) -> list[Tuple[int, int]]:
        x0, y0 = origin
        width, height = self.model.width, self.model.height
        cells: list[Tuple[int, int]] = [origin]
        for step in range(1, self.vision + 1):
            if x0 + step < width:
                cells.append((x0 + step, y0))
            if x0 - step >= 0:
                cells.append((x0 - step, y0))
            if y0 + step < height:
                cells.append((x0, y0 + step))
            if y0 - step >= 0:
                cells.append((x0, y0 - step))
        return cells

    def _choose_best_cell(self, origin: Tuple[int, int]) -> Tuple[int, int]:
        # Highest sugar; tie-break by Manhattan distance from origin; then coords.
        best_cell = origin
        best_sugar = int(self.model.sugar_current[origin[0], origin[1]])
        best_distance = 0
        ox, oy = origin
        for cx, cy in self._visible_cells(origin):
            # Block occupied cells except the origin (own cell allowed as fallback).
            if (cx, cy) != origin and not self.model.grid.is_cell_empty((cx, cy)):
                continue
            sugar_here = int(self.model.sugar_current[cx, cy])
            distance = abs(cx - ox) + abs(cy - oy)
            better = False
            if sugar_here > best_sugar:
                better = True
            elif sugar_here == best_sugar:
                if distance < best_distance:
                    better = True
                elif distance == best_distance and (cx, cy) < best_cell:
                    better = True
            if better:
                best_cell = (cx, cy)
                best_sugar = sugar_here
                best_distance = distance
        return best_cell

    def move(self) -> None:
        best = self._choose_best_cell(self.pos)
        if best != self.pos:
            self.model.grid.move_agent(self, best)


__all__ = ["AntAgent"]

