from __future__ import annotations

# %% [markdown]
"""[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/projectmesa/mesa-frames/blob/main/docs/general/user-guide/2_introductory-tutorial.ipynb)"""

# %% [markdown]
"""## Installation (if running in Colab)

Run the following cell to install `mesa-frames` if you are using Google Colab."""

# %%
# #!pip install git+https://github.com/projectmesa/mesa-frames mesa

# %% [markdown]
""" # Introductory Tutorial: Boltzmann Wealth Model with mesa-frames 💰🚀

In this tutorial, we'll implement the Boltzmann Wealth Model using mesa-frames. This model simulates the distribution of wealth among agents, where agents randomly give money to each other.

## Setting Up the Model 🏗️

First, let's import the necessary modules and set up our model class:"""

# %%
from mesa_frames import Model, AgentSet, DataCollector


class MoneyModel(Model):
    def __init__(self, N: int, agents_cls):
        super().__init__()
        self.n_agents = N
        self.sets += agents_cls(N, self)
        self.datacollector = DataCollector(
            model=self,
            model_reporters={
                "total_wealth": lambda m: m.sets["MoneyAgents"].df["wealth"].sum()
            },
            agent_reporters={"wealth": "wealth"},
            storage="csv",
            storage_uri="./data",
            trigger=lambda m: m.steps % 2 == 0,
        )

    def step(self):
        # Executes the step method for every agentset in self.sets
        self.sets.do("step")

    def run_model(self, n):
        for _ in range(n):
            self.step()
            self.datacollector.conditional_collect
        self.datacollector.flush()


# %% [markdown]
"""## Implementing the AgentSet 👥

Now, let's implement our `MoneyAgents` using polars backends."""

# %%
import polars as pl


class MoneyAgents(AgentSet):
    def __init__(self, n: int, model: Model):
        super().__init__(model)
        self += pl.DataFrame({"wealth": pl.ones(n, eager=True)})

    def step(self) -> None:
        self.do("give_money")

    def give_money(self):
        self.select(self.wealth > 0)
        other_agents = self.df.sample(n=len(self.active_agents), with_replacement=True)
        self["active", "wealth"] -= 1
        new_wealth = other_agents.group_by("unique_id").len()
        self[new_wealth["unique_id"], "wealth"] += new_wealth["len"]


# %% [markdown]
"""
## Running the Model ▶️

Now that we have our model and agent set defined, let's run a simulation:"""

# %%
# Create and run the model
model = MoneyModel(1000, MoneyAgents)
model.run_model(100)

wealth_dist = list(model.sets.df.values())[0]

# Print the final wealth distribution
print(wealth_dist.select(pl.col("wealth")).describe())

# %% [markdown]
"""
This output shows the statistical summary of the wealth distribution after 100 steps of the simulation with 1000 agents.

## Performance Comparison 🏎️💨

One of the key advantages of mesa-frames is its performance with large numbers of agents. Let's compare the performance of mesa and polars:"""


# %%
class MoneyAgentsConcise(AgentSet):
    def __init__(self, n: int, model: Model):
        super().__init__(model)
        ## Adding the agents to the agent set
        # 1. Changing the df attribute directly (not recommended, if other agents were added before, they will be lost)
        """self.df = pl.DataFrame(
            {"wealth": pl.ones(n, eager=True)}
        )"""
        # 2. Adding the dataframe with add
        """self.add(
            pl.DataFrame(
                {
                    "wealth": pl.ones(n, eager=True),
                }
            )
        )"""
        # 3. Adding the dataframe with __iadd__
        self += pl.DataFrame({"wealth": pl.ones(n, eager=True)})

    def step(self) -> None:
        # The give_money method is called
        # self.give_money()
        self.do("give_money")

    def give_money(self):
        ## Active agents are changed to wealthy agents
        # 1. Using the __getitem__ method
        # self.select(self["wealth"] > 0)
        # 2. Using the fallback __getattr__ method
        self.select(self.wealth > 0)

        # Receiving agents are sampled (only native expressions currently supported)
        other_agents = self.df.sample(n=len(self.active_agents), with_replacement=True)

        # Wealth of wealthy is decreased by 1
        # 1. Using the __setitem__ method with self.active_agents mask
        # self[self.active_agents, "wealth"] -= 1
        # 2. Using the __setitem__ method with "active" mask
        self["active", "wealth"] -= 1

        # Compute the income of the other agents (only native expressions currently supported)
        new_wealth = other_agents.group_by("unique_id").len()

        # Add the income to the other agents
        # 1. Using the set method
        """self.set(
            attr_names="wealth",
            values=pl.col("wealth") + new_wealth["len"],
            mask=new_wealth,
        )"""

        # 2. Using the __setitem__ method
        self[new_wealth, "wealth"] += new_wealth["len"]


class MoneyAgentsNative(AgentSet):
    def __init__(self, n: int, model: Model):
        super().__init__(model)
        self += pl.DataFrame({"wealth": pl.ones(n, eager=True)})

    def step(self) -> None:
        self.do("give_money")

    def give_money(self):
        ## Active agents are changed to wealthy agents
        self.select(pl.col("wealth") > 0)

        other_agents = self.df.sample(n=len(self.active_agents), with_replacement=True)

        # Wealth of wealthy is decreased by 1
        self.df = self.df.with_columns(
            wealth=pl.when(
                pl.col("unique_id").is_in(self.active_agents["unique_id"].implode())
            )
            .then(pl.col("wealth") - 1)
            .otherwise(pl.col("wealth"))
        )

        new_wealth = other_agents.group_by("unique_id").len()

        # Add the income to the other agents
        self.df = (
            self.df.join(new_wealth, on="unique_id", how="left")
            .fill_null(0)
            .with_columns(wealth=pl.col("wealth") + pl.col("len"))
            .drop("len")
        )


# %% [markdown]
"""Add Mesa implementation of MoneyAgent and MoneyModel classes to test Mesa performance"""

# %%
import mesa


class MesaMoneyAgent(mesa.Agent):
    """An agent with fixed initial wealth."""

    def __init__(self, model):
        # Pass the parameters to the parent class.
        super().__init__(model)

        # Create the agent's variable and set the initial values.
        self.wealth = 1

    def step(self):
        # Verify agent has some wealth
        if self.wealth > 0:
            other_agent: MesaMoneyAgent = self.model.random.choice(self.model.agents)
            if other_agent is not None:
                other_agent.wealth += 1
                self.wealth -= 1


class MesaMoneyModel(mesa.Model):
    """A model with some number of agents."""

    def __init__(self, N: int):
        super().__init__()
        self.num_agents = N
        for _ in range(N):
            self.agents.add(MesaMoneyAgent(self))

    def step(self):
        """Advance the model by one step."""
        self.agents.shuffle_do("step")

    def run_model(self, n_steps) -> None:
        for _ in range(n_steps):
            self.step()


# %%
import time


def run_simulation(model: MesaMoneyModel | MoneyModel, n_steps: int):
    start_time = time.time()
    model.run_model(n_steps)
    end_time = time.time()
    return end_time - start_time


# Compare mesa and mesa-frames implementations
n_agents_list = [10**2, 10**3 + 1, 2 * 10**3]
n_steps = 100
print("Execution times:")
for implementation in [
    "mesa",
    "mesa-frames (pl concise)",
    "mesa-frames (pl native)",
]:
    print(f"---------------\n{implementation}:")
    for n_agents in n_agents_list:
        if implementation == "mesa":
            ntime = run_simulation(MesaMoneyModel(n_agents), n_steps)
        elif implementation == "mesa-frames (pl concise)":
            ntime = run_simulation(MoneyModel(n_agents, MoneyAgentsConcise), n_steps)
        elif implementation == "mesa-frames (pl native)":
            ntime = run_simulation(MoneyModel(n_agents, MoneyAgentsNative), n_steps)

        print(f"  Number of agents: {n_agents}, Time: {ntime:.2f} seconds")
    print("---------------")

# %% [markdown]
"""
## Conclusion 🎉

- All mesa-frames implementations significantly outperform the original mesa implementation. 🏆
- The native implementation for Polars shows better performance than their concise counterparts. 💪
- The Polars native implementation shows the most impressive speed-up, ranging from 10.86x to 17.60x faster than mesa! 🚀🚀🚀
- The performance advantage of mesa-frames becomes more pronounced as the number of agents increases. 📈"""
