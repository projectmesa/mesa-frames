import numpy as np
import polars as pl

from mesa_frames import GridPolars, ModelDF

from .agents import AntPolars


class SugarscapePolars(ModelDF):
    def __init__(
        self,
        n_agents: int,
        sugar_grid: np.ndarray | None = None,
        initial_sugar: np.ndarray | None = None,
        metabolism: np.ndarray | None = None,
        vision: np.ndarray | None = None,
        width: int | None = None,
        height: int | None = None,
    ):
        super().__init__()
        if sugar_grid is None:
            sugar_grid = self.random.integers(0, 4, (width, height))
        grid_dimensions = sugar_grid.shape
        self.space = GridPolars(
            self, grid_dimensions, neighborhood_type="von_neumann", capacity=1
        )
        dim_0 = pl.Series("dim_0", pl.arange(grid_dimensions[0], eager=True)).to_frame()
        dim_1 = pl.Series("dim_1", pl.arange(grid_dimensions[1], eager=True)).to_frame()
        sugar_grid = dim_0.join(dim_1, how="cross").with_columns(
            sugar=sugar_grid.flatten(), max_sugar=sugar_grid.flatten()
        )
        self.space.set_cells(sugar_grid)
        self.agents += AntPolars(self, n_agents, initial_sugar, metabolism, vision)
        self.space.place_to_empty(self.agents)

    def run_model(self, steps: int) -> list[int]:
        for _ in range(steps):
            if len(self.agents) == 0:
                return
            self.step()
            empty_cells = self.space.empty_cells
            full_cells = self.space.full_cells

            max_sugar = self.space.cells.join(
                empty_cells, on=["dim_0", "dim_1"]
            ).select(pl.col("max_sugar"))

            self.space.set_cells(full_cells, {"sugar": 0})
            self.space.set_cells(empty_cells, {"sugar": max_sugar})
