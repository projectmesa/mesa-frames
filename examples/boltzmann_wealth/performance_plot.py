import importlib.metadata

import matplotlib.pyplot as plt
import mesa
import numpy as np
import perfplot
import polars as pl
import seaborn as sns
from packaging import version

from mesa_frames import AgentSet, Model


### ---------- Mesa implementation ---------- ###
def mesa_implementation(n_agents: int) -> None:
    model = MoneyModel(n_agents)
    model.run_model(100)


class MoneyAgent(mesa.Agent):
    """An agent with fixed initial wealth."""

    def __init__(self, model):
        # Pass the parameters to the parent class.
        super().__init__(model)

        # Create the agent's variable and set the initial values.
        self.wealth = 1

    def step(self):
        # Verify agent has some wealth
        if self.wealth > 0:
            other_agent = self.random.choice(self.model.sets)
            if other_agent is not None:
                other_agent.wealth += 1
                self.wealth -= 1


class MoneyModel(mesa.Model):
    """A model with some number of agents."""

    def __init__(self, N):
        super().__init__()
        self.num_agents = N
        for _ in range(self.num_agents):
            self.sets.add(MoneyAgent(self))

    def step(self):
        """Advance the model by one step."""
        self.sets.shuffle_do("step")

    def run_model(self, n_steps) -> None:
        for _ in range(n_steps):
            self.step()


"""def compute_gini(model):
    agent_wealths = model.sets.get("wealth")
    x = sorted(agent_wealths)
    N = model.num_agents
    B = sum(xi * (N - i) for i, xi in enumerate(x)) / (N * sum(x))
    return 1 + (1 / N) - 2 * B"""


### ---------- Mesa-frames implementation ---------- ###


class MoneyAgentConcise(AgentSet):
    def __init__(self, n: int, model: Model):
        super().__init__(model)
        ## Adding the agents to the agent set
        # 1. Changing the agents attribute directly (not recommended, if other agents were added before, they will be lost)
        """self.sets = pl.DataFrame(
            "wealth": pl.ones(n, eager=True)}
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


class MoneyAgentNative(AgentSet):
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


class MoneyModel(Model):
    def __init__(self, N: int, agents_cls):
        super().__init__()
        self.n_agents = N
        self.sets += agents_cls(N, self)

    def step(self):
        # Executes the step method for every agentset in self.sets
        self.sets.do("step")

    def run_model(self, n):
        for _ in range(n):
            self.step()


def mesa_frames_polars_concise(n_agents: int) -> None:
    model = MoneyModel(n_agents, MoneyAgentConcise)
    model.run_model(100)


def mesa_frames_polars_native(n_agents: int) -> None:
    model = MoneyModel(n_agents, MoneyAgentNative)
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
    out.save(image_path, transparent=False)

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
    ]
    kernels_0 = [
        mesa_implementation,
        mesa_frames_polars_concise,
        mesa_frames_polars_native,
    ]
    n_range_0 = [k for k in range(0, 100001, 10000)]
    title_0 = "100 steps of the Boltzmann Wealth model:\n" + " vs ".join(labels_0)
    image_path_0 = "boltzmann_with_mesa.png"

    plot_and_print_benchmark(labels_0, kernels_0, n_range_0, title_0, image_path_0)

    labels_1 = [
        "mesa-frames (pl concise)",
        "mesa-frames (pl native)",
    ]
    kernels_1 = [
        mesa_frames_polars_concise,
        mesa_frames_polars_native,
    ]
    n_range_1 = [k for k in range(100000, 1000001, 100000)]
    title_1 = "100 steps of the Boltzmann Wealth model:\n" + " vs ".join(labels_1)
    image_path_1 = "boltzmann_no_mesa.png"

    plot_and_print_benchmark(labels_1, kernels_1, n_range_1, title_1, image_path_1)


if __name__ == "__main__":
    main()
