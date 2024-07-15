from random import choice, shuffle

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
            other_agent = choice(self.model.agents)
            if other_agent is not None:
                other_agent.wealth += 1
                self.wealth -= 1


class MoneyModel(mesa.Model):
    """A model with some number of agents."""

    def __init__(self, N):
        super().__init__()
        self.num_agents = N
        # Create scheduler and assign it to the model
        self.agents = [MoneyAgent(i, self) for i in range(self.num_agents)]

    def step(self):
        """Advance the model by one step."""
        shuffle(self.agents)
        for agent in self.agents:
            agent.step()

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


class MoneyAgentPolarsConcise(AgentSetPolars):
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
        # 1. Using the __getitem__ method
        # self.select(self["wealth"] > 0)
        # 2. Using the fallback __getattr__ method
        self.select(self.wealth > 0)

        # Receiving agents are sampled (only native expressions currently supported)
        other_agents = self.agents.sample(
            n=len(self.active_agents), with_replacement=True
        )

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


class MoneyAgentPolarsNative(AgentSetPolars):
    def __init__(self, n: int, model: ModelDF):
        super().__init__(model)
        self += pl.DataFrame(
            {"unique_id": pl.arange(n, eager=True), "wealth": pl.ones(n, eager=True)}
        )

    def step(self) -> None:
        self.do("give_money")

    def give_money(self):
        ## Active agents are changed to wealthy agents
        self.select(pl.col("wealth") > 0)

        other_agents = self.agents.sample(
            n=len(self.active_agents), with_replacement=True
        )

        # Wealth of wealthy is decreased by 1
        self.agents = self.agents.with_columns(
            wealth=pl.when(pl.col("unique_id").is_in(self.active_agents["unique_id"]))
            .then(pl.col("wealth") - 1)
            .otherwise(pl.col("wealth"))
        )

        new_wealth = other_agents.group_by("unique_id").len()

        # Add the income to the other agents
        self.agents = (
            self.agents.join(new_wealth, on="unique_id", how="left")
            .fill_null(0)
            .with_columns(wealth=pl.col("wealth") + pl.col("len"))
            .drop("len")
        )


class MoneyAgentPandasConcise(AgentSetPandas):
    def __init__(self, n: int, model: ModelDF) -> None:
        super().__init__(model)
        ## Adding the agents to the agent set
        # 1. Changing the agents attribute directly (not recommended, if other agents were added before, they will be lost)
        # self.agents = pd.DataFrame({"unique_id": np.arange(n), "wealth": np.ones(n)})
        # 2. Adding the dataframe with add
        # self.add(pd.DataFrame({"unique_id": np.arange(n), "wealth": np.ones(n)}))
        # 3. Adding the dataframe with __iadd__
        self += pd.DataFrame(
            {"unique_id": np.arange(n, dtype="int64"), "wealth": np.ones(n)}
        )

    def step(self) -> None:
        # The give_money method is called
        self.do("give_money")

    def give_money(self):
        ## Active agents are changed to wealthy agents
        # 1. Using the __getitem__ method
        # self.select(self["wealth"] > 0)
        # 2. Using the fallback __getattr__ method
        self.select(self.wealth > 0)

        # Receiving agents are sampled (only native expressions currently supported)
        other_agents = self.agents.sample(n=len(self.active_agents), replace=True)

        # Wealth of wealthy is decreased by 1
        # 1. Using the __setitem__ method with self.active_agents mask
        # self[self.active_agents, "wealth"] -= 1
        # 2. Using the __setitem__ method with "active" mask
        self["active", "wealth"] -= 1

        # Compute the income of the other agents (only native expressions currently supported)
        new_wealth = other_agents.groupby("unique_id").count()

        # Add the income to the other agents
        # 1. Using the set method
        # self.set(attr_names="wealth", values=self["wealth"] + new_wealth["wealth"], mask=new_wealth)
        # 2. Using the __setitem__ method
        self[new_wealth, "wealth"] += new_wealth["wealth"]


class MoneyAgentPandasNative(AgentSetPandas):
    def __init__(self, n: int, model: ModelDF) -> None:
        super().__init__(model)
        ## Adding the agents to the agent set
        self += pd.DataFrame(
            {"unique_id": np.arange(n, dtype="int64"), "wealth": np.ones(n)}
        )

    def step(self) -> None:
        # The give_money method is called
        self.do("give_money")

    def give_money(self):
        self.select(self.agents["wealth"] > 0)

        # Receiving agents are sampled (only native expressions currently supported)
        other_agents = self.agents.sample(n=len(self.active_agents), replace=True)

        # Wealth of wealthy is decreased by 1
        b_mask = self.active_agents.index.isin(self.agents)
        self.agents.loc[b_mask, "wealth"] -= 1

        # Compute the income of the other agents (only native expressions currently supported)
        new_wealth = other_agents.groupby("unique_id").count()

        # Add the income to the other agents
        merged = pd.merge(
            self.agents, new_wealth, on="unique_id", how="left", suffixes=("", "_new")
        )
        merged["wealth"] = merged["wealth"] + merged["wealth_new"].fillna(0)
        self.agents = merged.drop(columns=["wealth_new"])


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


def mesa_frames_polars_concise(n_agents: int) -> None:
    model = MoneyModelDF(n_agents, MoneyAgentPolarsConcise)
    model.run_model(100)


def mesa_frames_polars_native(n_agents: int) -> None:
    model = MoneyModelDF(n_agents, MoneyAgentPolarsNative)
    model.run_model(100)


def mesa_frames_pandas_concise(n_agents: int) -> None:
    model = MoneyModelDF(n_agents, MoneyAgentPandasConcise)
    model.run_model(100)


def mesa_frames_pandas_native(n_agents: int) -> None:
    model = MoneyModelDF(n_agents, MoneyAgentPandasNative)
    model.run_model(100)


def plot_and_print_benchmark(labels, kernels, n_range, title, image_path):
    out = perfplot.bench(
        setup=lambda n: n,
        kernels=kernels,
        labels=labels,
        n_range=n_range,
        xlabel="Number of agents",
        equality_check=None,
        title=title,
    )

    plt.ylabel("Execution time (s)")
    out.save(image_path)

    print("\nExecution times:")
    for i, label in enumerate(labels):
        print(f"---------------\n{label}:")
        for n, t in zip(out.n_range, out.timings_s[i]):
            print(f"  Number of agents: {n}, Time: {t:.2f} seconds")
        print("---------------")


def main():
    sns.set_theme(style="whitegrid")

    labels_0 = [
        "mesa",
        "mesa-frames (pl concise)",
        "mesa-frames (pl native)",
        "mesa-frames (pd concise)",
        "mesa-frames (pd native)",
    ]
    kernels_0 = [
        mesa_implementation,
        mesa_frames_polars_concise,
        mesa_frames_polars_native,
        mesa_frames_pandas_concise,
        mesa_frames_pandas_native,
    ]
    n_range_0 = [k for k in range(0, 100001, 10000)]
    title_0 = "100 steps of the Boltzmann Wealth model:\n" + " vs ".join(labels_0)
    image_path_0 = "docs/images/readme_plot_0.png"

    plot_and_print_benchmark(labels_0, kernels_0, n_range_0, title_0, image_path_0)

    labels_1 = [
        "mesa-frames (pl concise)",
        "mesa-frames (pl native)",
        "mesa-frames (pd concise)",
        "mesa-frames (pd native)",
    ]
    kernels_1 = [
        mesa_frames_polars_concise,
        mesa_frames_polars_native,
        mesa_frames_pandas_concise,
        mesa_frames_pandas_native,
    ]
    n_range_1 = [k for k in range(100000, 1000001, 100000)]
    title_1 = "100 steps of the Boltzmann Wealth model:\n" + " vs ".join(labels_1)
    image_path_1 = "docs/images/readme_plot_1.png"

    plot_and_print_benchmark(labels_1, kernels_1, n_range_1, title_1, image_path_1)


if __name__ == "__main__":
    main()
