from hmac import new
from random import seed
from typing import TYPE_CHECKING

import mesa
import numpy as np
import pandas as pd
import perfplot

from mesa_frames import AgentSetDF, ModelDF


# Mesa implementation
def mesa_implementation(n_agents: int) -> None:
    model = MoneyModel(n_agents)
    model.run_model(100)


class MoneyAgent(mesa.Agent):
    """An agent with fixed initial wealth."""

    def __init__(self, unique_id, model):
        # Pass the parameters to the parent class.
        super().__init__(unique_id, model)

        # Create the agent's variable and set the initial values.
        self.wealth = 1

    def step(self):
        # Verify agent has some wealth
        if self.wealth > 0:
            other_agent = self.random.choice(self.model.schedule.agents)
            if other_agent is not None:
                other_agent.wealth += 1
                self.wealth -= 1


class MoneyModel(mesa.Model):
    """A model with some number of agents."""

    def __init__(self, N):
        super().__init__
        self.num_agents = N
        # Create scheduler and assign it to the model
        self.schedule = mesa.time.RandomActivation(self)

        # Create agents
        for i in range(self.num_agents):
            a = MoneyAgent(i, self)
            # Add the agent to the scheduler
            self.schedule.add(a)

    def step(self):
        """Advance the model by one step."""

        # The model's step will go here for now this will call the step method of each agent and print the agent's unique_id
        self.schedule.step()

    def run_model(self, n_steps) -> None:
        for _ in range(n_steps):
            self.step()


"""def compute_gini(model):
    agent_wealths = model.agents.get("wealth")
    x = sorted(agent_wealths)
    N = model.num_agents
    B = sum(xi * (N - i) for i, xi in enumerate(x)) / (N * sum(x))
    return 1 + (1 / N) - 2 * B"""


# Mesa Frames implementation
def mesa_frames_implementation(n_agents: int) -> None:
    model = MoneyModelDF(n_agents)
    model.run_model(100)


class MoneyModelDF(ModelDF):
    def __init__(self, N):
        super().__init__()
        self.num_agents = N
        self.agents = self.agents.add(MoneyAgentsDF(N, model=self))

    def step(self):
        self.agents = self.agents.do("step")

    def run_model(self, n):
        for _ in range(n):
            self.step()


class MoneyAgentsDF(AgentSetDF):
    def __init__(self, n: int, model: MoneyModelDF):
        super().__init__(model=model)
        self.add(n, data={"wealth": np.ones(n)})

    def step(self):
        wealthy_agents = self.agents["wealth"] > 0
        self.select(wealthy_agents).do("give_money")

    def give_money(self):
        other_agents = self.agents.sample(len(self.active_agents), replace=True)
        new_wealth = (
            other_agents.index.value_counts()
            .reindex(self.active_agents.index)
            .fillna(-1)
        )
        self.set_attribute("wealth", self.get_attribute("wealth") + new_wealth)


def main():
    mesa_frames_implementation(100)
    out = perfplot.bench(
        setup=lambda n: n,
        kernels=[mesa_implementation, mesa_frames_implementation],
        labels=["mesa", "mesa-frames"],
        n_range=[k for k in range(100, 1000, 100)],
        xlabel="Number of agents",
        equality_check=None,
        title="100 steps of the Boltzmann Wealth model",
    )
    out.show()
    # out.save("docs/images/readme_plot.png")


if __name__ == "__main__":
    main()
