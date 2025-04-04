# Introductory Tutorial: Boltzmann Wealth Model with mesa-frames 💰🚀

In this tutorial, we'll implement the Boltzmann Wealth Model using mesa-frames. This model simulates the distribution of wealth among agents, where agents randomly give money to each other.

## Setting Up the Model 🏗️

First, let's import the necessary modules and set up our model class:

```python
from mesa_frames import ModelDF, AgentSetPolars

class MoneyModelDF(ModelDF):
    def __init__(self, N: int, agents_cls):
        super().__init__()
        self.n_agents = N
        self.agents += agents_cls(N, self)

    def step(self):
        self.agents.do("step")

    def run_model(self, n):
        for _ in range(n):
            self.step()
```

This `MoneyModelDF` class will work for Polars implementations.

## Implementing the AgentSet 👥

Now, let's implement our `MoneyAgentSet` using Polars backends. You can switch between the two implementations:

=== "Polars 🐻‍❄️"

    ```python
        import polars as pl

        class MoneyAgentPolars(AgentSetPolars):
            def __init__(self, n: int, model: ModelDF):
                super().__init__(model)
                self += pl.DataFrame(
                    {"unique_id": pl.arange(n, eager=True), "wealth": pl.ones(n, eager=True)}
                )

            def step(self) -> None:
                self.do("give_money")

            def give_money(self):
                self.select(self.wealth > 0)
                other_agents = self.agents.sample(n=len(self.active_agents), with_replacement=True)
                self["active", "wealth"] -= 1
                new_wealth = other_agents.group_by("unique_id").len()
                self[new_wealth["unique_id"], "wealth"] += new_wealth["len"]
    ```

## Running the Model ▶️

Now that we have our model and agent set defined, let's run a simulation:

```python

agent_class = MoneyAgentPolars

# Create and run the model
model = MoneyModelDF(1000, agent_class)
model.run_model(100)

# Print the final wealth distribution
print(model.agents["wealth"].describe())
```

Output:

```python
count    1000.000000
mean        1.000000
std         1.414214
min         0.000000
25%         0.000000
50%         1.000000
75%         1.000000
max        13.000000
Name: wealth, dtype: float64
```

This output shows the statistical summary of the wealth distribution after 100 steps of the simulation with 1000 agents.

## Performance Comparison 🏎️💨

One of the key advantages of mesa-frames is its performance with large numbers of agents. Let's compare the performance of our pandas and Polars implementations:

```python
import time

def run_simulation(model_class, n_agents, n_steps):
    start_time = time.time()
    model = model_class(n_agents)
    model.run_model(n_steps)
    end_time = time.time()
    return end_time - start_time

# Compare mesa and mesa-frames implementations
n_agents_list = [100000, 300000, 500000, 700000]
n_steps = 100

print("Execution times:")
for implementation in ["mesa", "mesa-frames (pl concise)", "mesa-frames (pl native)", "mesa-frames (pd concise)", "mesa-frames (pd native)"]:
    print(f"---------------\n{implementation}:")
    for n_agents in n_agents_list:
        if implementation == "mesa":
            time = run_simulation(MoneyModel, n_agents, n_steps)
        elif implementation == "mesa-frames (pl concise)":
            time = run_simulation(lambda n: MoneyModelDF(n, MoneyAgentPolarsConcise), n_agents, n_steps)
        elif implementation == "mesa-frames (pl native)":
            time = run_simulation(lambda n: MoneyModelDF(n, MoneyAgentPolarsNative), n_agents, n_steps)

        print(f"  Number of agents: {n_agents}, Time: {time:.2f} seconds")
    print("---------------")
```

Example output:

```python
---------------
mesa:
  Number of agents: 100000, Time: 3.80 seconds
  Number of agents: 300000, Time: 14.96 seconds
  Number of agents: 500000, Time: 26.88 seconds
  Number of agents: 700000, Time: 40.34 seconds
---------------
---------------
mesa-frames (pl concise):
  Number of agents: 100000, Time: 0.76 seconds
  Number of agents: 300000, Time: 2.01 seconds
  Number of agents: 500000, Time: 4.77 seconds
  Number of agents: 700000, Time: 7.26 seconds
---------------
---------------
mesa-frames (pl native):
  Number of agents: 100000, Time: 0.35 seconds
  Number of agents: 300000, Time: 0.85 seconds
  Number of agents: 500000, Time: 1.55 seconds
  Number of agents: 700000, Time: 2.61 seconds
---------------
```

Speed-up over mesa: 🚀

```python
mesa-frames (pl concise):
  Number of agents: 100000, Speed-up: 5.00x 💨
  Number of agents: 300000, Speed-up: 7.44x 💨
  Number of agents: 500000, Speed-up: 5.63x 💨
  Number of agents: 700000, Speed-up: 5.56x 💨
---------------
mesa-frames (pl native):
  Number of agents: 100000, Speed-up: 10.86x 💨
  Number of agents: 300000, Speed-up: 17.60x 💨
  Number of agents: 500000, Speed-up: 17.34x 💨
  Number of agents: 700000, Speed-up: 15.46x 💨
```

## Conclusion 🎉

- All mesa-frames implementations significantly outperform the original mesa implementation. 🏆
- The native implementation for Polars shows better performance than their concise counterparts. 💪
- The Polars native implementation shows the most impressive speed-up, ranging from 10.86x to 17.60x faster than mesa! 🚀🚀🚀
- The performance advantage of mesa-frames becomes more pronounced as the number of agents increases. 📈
