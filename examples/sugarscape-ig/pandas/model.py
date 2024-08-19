import numpy as np
from agents import AntPandas
from mesa_frames import GridPandas, ModelDF

import pandas as pd


class SugarscapePandas(ModelDF):
    def __init__(self, grid_dimensions: list[int], n_agents: int):
        super().__init__()
        self.space = GridPandas(
            self, grid_dimensions, neighborhood_type="von_neumann", capacity=1
        )
        # NOTE: set_cells should automatically broadcast the property if the dimensions of DF
        # are same as the grid so there is no need to pass the dimensions with pd.MultiIndex
        sugar_grid = pd.DataFrame(
            {
                "sugar": self.random.integers(0, 4, grid_dimensions).flatten(),
            },
            index=pd.MultiIndex.from_product(
                [np.arange(grid_dimensions[0]), np.arange(grid_dimensions[1])],
                names=["dim_0", "dim_1"],
            ),
        )
        self.space.set_cells(sugar_grid)
        self.agents += AntPandas(self, n_agents)
        self.space.place_to_empty(self.agents)

    def run_model(self, steps: int) -> list[int]:
        agents_count = []
        for i in range(steps):
            print(f"Step {i}")
            self.step()
            agents_count.append(len(self.agents))
        return agents_count
