import matplotlib.pyplot as plt
import mesa
import numpy as np
import pandas as pd
import perfplot
import polars as pl
import seaborn as sns

from mesa_frames import AgentSetPandas, AgentSetPolars, ModelDF


### ---------- Mesa implementation ---------- ###
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
        super().__init__()
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


### ---------- Mesa-frames implementation ---------- ###


class MoneyAgentPolars(AgentSetPolars):
    def __init__(self, n: int, model: ModelDF):
        super().__init__(model)
        ## Adding the agents to the agent set
        # 1. Changing the agents attribute directly (not recommended, if other agents were added before, they will be lost)
        """self.agents = pl.DataFrame(
            {"unique_id": pl.arange(n, eager=True), "wealth": pl.ones(n, eager=True)}
        )"""
        # 2. Adding the dataframe with add
        """self.add(
            pl.DataFrame(
                {
                    "unique_id": pl.arange(n, eager=True),
                    "wealth": pl.ones(n, eager=True),
                }
            )
        )"""
        # 3. Adding the dataframe with __iadd__
        self += pl.DataFrame(
            {"unique_id": pl.arange(n, eager=True), "wealth": pl.ones(n, eager=True)}
        )

    def step(self) -> None:
        # The give_money method is called
        # self.give_money()
        self.do("give_money")

    def give_money(self):
        ## Active agents are changed to wealthy agents
        # 1. Using a native expression
        # self.select(pl.col("wealth") > 0)
        # 2. Using the __getitem__ method
        # self.select(self["wealth"] > 0)
        # 3. Using the fallback __getattr__ method
        self.select(self.wealth > 0)

        # Receiving agents are sampled (only native expressions currently supported)
        other_agents = self.agents.sample(
            n=len(self.active_agents), with_replacement=True
        )

        # Wealth of wealthy is decreased by 1
        # 1. Using a native expression
        """self.agents = self.agents.with_columns(
            wealth=pl.when(pl.col("unique_id").is_in(self.active_agents["unique_id"]))
            .then(pl.col("wealth") - 1)
            .otherwise(pl.col("wealth"))
        )"""
        # 2. Using the __setitem__ method with self.active_agents mask
        # self[self.active_agents, "wealth"] -= 1
        # 3. Using the __setitem__ method with "active" mask
        self["active", "wealth"] -= 1

        # Compute the income of the other agents (only native expressions currently supported)
        new_wealth = other_agents.group_by("unique_id").len()

        # Add the income to the other agents
        # 1. Using native expressions
        """self.agents = self.agents.with_columns(
            pl.when(pl.col("unique_id").is_in(new_wealth["unique_id"]))
            .then(pl.col("wealth") + new_wealth["wealth"])
            .otherwise(pl.col("wealth"))
        )"""

        # 2. Using the set method
        """self.set(
            attr_names="wealth",
            values=pl.col("wealth") + new_wealth["len"],
            mask=new_wealth,
        )"""

        # 3. Using the __setitem__ method
        self[new_wealth, "wealth"] += new_wealth["len"]


class MoneyAgentPandas(AgentSetPandas):
    def __init__(self, n: int, model: ModelDF) -> None:
        super().__init__(model)
        ## Adding the agents to the agent set
        # 1. Changing the agents attribute directly (not recommended, if other agents were added before, they will be lost)
        # self.agents = pd.DataFrame({"unique_id": np.arange(n), "wealth": np.ones(n)})
        # 2. Adding the dataframe with add
        # self.add(pd.DataFrame({"unique_id": np.arange(n), "wealth": np.ones(n)}))
        # 3. Adding the dataframe with __iadd__
        self += pd.DataFrame({"unique_id": np.arange(n), "wealth": np.ones(n)})

    def step(self) -> None:
        # The give_money method is called
        self.do("give_money")

    def give_money(self):
        ## Active agents are changed to wealthy agents
        # 1. Using a native expression
        # self.select(self.agents['wealth'] > 0)
        # 2. Using the __getitem__ method
        # self.select(self["wealth"] > 0)
        # 3. Using the fallback __getattr__ method
        self.select(self.wealth > 0)

        # Receiving agents are sampled (only native expressions currently supported)
        other_agents = self.agents.sample(n=len(self.active_agents), replace=True)

        # Wealth of wealthy is decreased by 1
        # 1. Using a native expression
        """b_mask = self.active_agents.index.isin(self.agents)
        self.agents.loc[b_mask, "wealth"] -= 1"""
        # 2. Using the __setitem__ method with self.active_agents mask
        # self[self.active_agents, "wealth"] -= 1
        # 3. Using the __setitem__ method with "active" mask
        self["active", "wealth"] -= 1

        # Compute the income of the other agents (only native expressions currently supported)
        new_wealth = other_agents.groupby("unique_id").count()

        # Add the income to the other agents
        # 1. Using native expressions
        """merged = pd.merge(
            self.agents, new_wealth, on="unique_id", how="left", suffixes=("", "_new")
        )
        merged["wealth"] = merged["wealth"] + merged["wealth_new"].fillna(0)
        self.agents = merged.drop(columns=["wealth_new"])"""

        # 2. Using the set method
        # self.set(attr_names="wealth", values=self["wealth"] + new_wealth["wealth"], mask=new_wealth)

        # 3. Using the __setitem__ method
        self[new_wealth, "wealth"] += new_wealth["wealth"]


class MoneyModelDF(ModelDF):
    def __init__(self, N: int, agents_cls):
        super().__init__()
        self.n_agents = N
        self.agents += agents_cls(N, self)

    def step(self):
        # Executes the step method for every agentset in self.agents
        self.agents.do("step")

    def run_model(self, n):
        for _ in range(n):
            self.step()


def mesa_frames_polars(n_agents: int) -> None:
    model = MoneyModelDF(n_agents, MoneyAgentPolars)
    model.run_model(100)


def mesa_frames_pandas(n_agents: int) -> None:
    model = MoneyModelDF(n_agents, MoneyAgentPandas)
    model.run_model(100)


def main():
    sns.set_theme(style="whitegrid")

    out = perfplot.bench(
        setup=lambda n: n,
        kernels=[mesa_implementation, mesa_frames_polars, mesa_frames_pandas],
        labels=["mesa", "mesa-frames (polars)", "mesa-frames (pandas)"],
        n_range=[k for k in range(100, 10000, 1000)],
        xlabel="Number of agents",
        equality_check=None,
        title="100 steps of the Boltzmann Wealth model: mesa vs mesa-frames (polars) vs mesa-frames (pandas)",
    )
    plt.ylabel("Execution time (s)")
    out.save("docs/images/readme_plot_1.png")


if __name__ == "__main__":
    main()
