import numpy as np
import pandas as pd

from mesa_frames import GridPandas, ModelDF

from .agents import AntPandas


class SugarscapePandas(ModelDF):
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

        self.space = GridPandas(
            self, grid_dimensions, neighborhood_type="von_neumann", capacity=1
        )

        # NOTE: set_cells should automatically broadcast the property if the dimensions of DF
        # are same as the grid so there is no need to pass the dimensions with pd.MultiIndex
        sugar_grid = pd.DataFrame(
            {
                "sugar": sugar_grid.flatten(),
            },
            index=pd.MultiIndex.from_product(
                [np.arange(grid_dimensions[0]), np.arange(grid_dimensions[1])],
                names=["dim_0", "dim_1"],
            ),
        )
        self.space.set_cells(sugar_grid)
        self.agents += AntPandas(self, n_agents, initial_sugar, metabolism, vision)
        self.space.place_to_empty(self.agents)

    def run_model(self, steps: int) -> list[int]:
        for _ in range(steps):
            self.step()
