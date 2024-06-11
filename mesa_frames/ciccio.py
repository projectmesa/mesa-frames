from copy import deepcopy

import pandas as pd

from mesa_frames import AgentSetPandas, ModelDF


class CiccioAgentSet(AgentSetPandas):
    def __init__(self, model: ModelDF):
        self.starting_wealth = pd.Series([1, 2, 3, 4], name="wealth")

    def add_wealth(self, amount: int) -> None:
        self.agents["wealth"] += amount


example_model = ModelDF()
example = CiccioAgentSet(example_model)

example._agents = pd.DataFrame({"unique_id": [0, 1, 2, 3, 4]}).set_index("unique_id")
example.add({"unique_id": [0, 1, 2, 3]})



deepcopy(example)
print(example.model)
print(dir(example))
