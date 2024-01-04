import mesa
import numpy as np
import perfplot

from mesa_frames.agent import AgentDF
from mesa_frames.model import ModelDF


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


# Mesa Frames implementation
def mesa_frames_implementation(n_agents: int) -> None:
    model = MoneyModelDF(n_agents)
    model.run_model(100)


class MoneyModelDF(ModelDF):
    def __init__(self, N):
        super().__init__()
        self.num_agents = N
        self.create_agents(N, {MoneyAgentDF: 1})

    def step(self, merged_mro=True):
        self.agents = self.agents.sample(frac=1)
        self.update_agents_masks()
        super().step(merged_mro)


class MoneyAgentDF(AgentDF):
    dtypes: dict[str, str] = {"wealth": "int64"}

    @classmethod
    def __init__(cls):
        super().__init__()
        cls.model.agents.loc[cls.mask, "wealth"] = 1

    @classmethod
    def step(cls):
        wealthy_agents = cls.model.agents.loc[cls.mask, "wealth"] > 0
        if wealthy_agents.any():
            other_agents = cls.model.agents.index.isin(
                cls.model.agents.sample(n=wealthy_agents.sum()).index
            )
            cls.model.agents.loc[wealthy_agents, "wealth"] -= 1
            cls.model.agents.loc[other_agents, "wealth"] += 1


def main():
    out = perfplot.bench(
        setup=lambda n: n,
        kernels=[mesa_implementation, mesa_frames_implementation],
        labels=["mesa", "mesa-frames"],
        n_range=[k for k in range(10, 10000, 100)],
        xlabel="Number of agents",
        equality_check=None,
        title="100 steps of the Boltzmann Wealth model",
    )
    out.show()
    out.save("docs/images/readme_plot.png")


if __name__ == "__main__":
    main()
