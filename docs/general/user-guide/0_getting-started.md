# Getting Started 🚀

## Main Concepts 🧠

### DataFrame-Based Object-Oriented Framework 📊

Unlike traditional mesa models where each agent is an individual Python object, mesa-frames stores all agents of a particular type in a single DataFrame. We operate only at the AgentSet level.

This approach allows for:

- Efficient memory usage
- Improved performance through vectorized operations on agent attributes (This is what makes `mesa-frames` fast)

Objects can be easily subclassed to respect mesa's object-oriented philosophy.

### Vectorized Operations ⚡

`mesa-frames` leverages **Polars** to replace Python loops with **column-wise expressions** executed in native Rust.
This allows you to update all agents simultaneously, the main source of `mesa-frames`' performance advantage.

Unlike traditional `mesa` models, where the **activation order** of agents can affect results (see [Comer, 2014](http://mars.gmu.edu/bitstream/handle/1920/9070/Comer_gmu_0883E_10539.pdf)),
`mesa-frames` processes all agents **in parallel by default**.
This removes order-dependent effects, though you should handle conflicts explicitly when sequential logic is required.

!!! tip "Best practice"
    Always start by expressing agent logic in a vectorized form.
    Fall back to loops only when ordering or conflict resolution is essential.

For a deeper understanding of vectorization and why it accelerates computation, see:

- [How vectorization speeds up your Python code — PythonSpeed](https://pythonspeed.com/articles/vectorization-python)

Here's a comparison between mesa-frames and mesa:

=== "mesa-frames"

    ```python
    class MoneyAgents(AgentSet):
        # initialization...
        def give_money(self):
            # Active agents are changed to wealthy agents
            self.select(self.wealth > 0)

            # Receiving agents are sampled (only native expressions currently supported)
            other_agents = self.df.sample(
                n=len(self.active_agents), with_replacement=True
            )

            # Wealth of wealthy is decreased by 1
            self["active", "wealth"] -= 1

            # Compute the income of the other agents (only native expressions currently supported)
            new_wealth = other_agents.group_by("unique_id").len()

            # Add the income to the other agents
            self[new_wealth, "wealth"] += new_wealth["len"]
    ```

=== "mesa"

    ```python
    class MoneyAgent(mesa.Agent):
        # initialization...
        def give_money(self):
            # Verify agent has some wealth
            if self.wealth > 0:
                other_agent = self.random.choice(self.model.sets)
                if other_agent is not None:
                    other_agent.wealth += 1
                    self.wealth -= 1
    ```

As you can see, while in mesa you should iterate through all the agents' steps in the model class, here you execute the method once for all agents.

## Coming from mesa 🔀

If you're familiar with mesa, this guide will help you understand the key differences in code structure between mesa and mesa-frames.

### Agent Representation 👥

- mesa: Each agent is an individual object instance. Methods are defined for individual agents and called on each agent.
- mesa-frames: Agents are rows in a DataFrame, grouped into AgentSets. Methods are defined for AgentSets and operate on all agents simultaneously.

=== "mesa-frames"

    ```python
    class MoneyAgents(AgentSet):
        def __init__(self, n, model):
            super().__init__(model)
            self += pl.DataFrame({
                "wealth": pl.ones(n)
                })
        def step(self):
            givers = self.wealth > 0
            receivers = self.df.sample(n=len(self.active_agents), with_replacement=True)
            self[givers, "wealth"] -= 1
            new_wealth = receivers.group_by("unique_id").len()
            self[new_wealth["unique_id"], "wealth"] += new_wealth["len"]
    ```

=== "mesa"

    ```python
    class MoneyAgent(Agent):
        def __init__(self, unique_id, model):
            super().__init__(unique_id, model)
            self.wealth = 1

        def step(self):
            if self.wealth > 0:
                other_agent = self.random.choice(self.model.schedule.agents)
                other_agent.wealth += 1
                self.wealth -= 1
    ```

### Model Structure 🏗️

- mesa: Models manage individual agents and use a scheduler.
- mesa-frames: Models manage AgentSets and directly control the simulation flow.

=== "mesa-frames"

    ```python
    class MoneyModel(Model):
        def __init__(self, N):
            super().__init__()
            self.sets += MoneyAgents(N, self)

        def step(self):
            self.sets.do("step")

    ```

=== "mesa"

    ```python
    class MoneyModel(Model):
        def __init__(self, N):
            self.num_agents = N
            self.schedule = RandomActivation(self)
            for i in range(self.num_agents):
                a = MoneyAgent(i, self)
                self.schedule.add(a)

        def step(self):
            self.schedule.step()
    ```

### Transition Tips 💡

1. **Think in Sets 🎭**: Instead of individual agents, think about operations on groups of agents.
2. **Leverage DataFrame Operations 🛠️**: Familiarize yourself with Polars operations for efficient agent manipulation.
3. **Vectorize Logic 🚅**: Convert loops and conditionals to vectorized operations where possible.
4. **Use AgentSets 📦**: Group similar agents into AgentSets instead of creating many individual agent classes.

### Handling Race Conditions 🏁

When simultaneous activation is not possible, you need to handle race conditions carefully. There are two main approaches:

1. **Custom UDF with Numba 🔧**: Use a custom User Defined Function (UDF) with Numba for efficient sequential processing.

   - [Polars UDF Guide](https://docs.pola.rs/user-guide/expressions/user-defined-functions/)

2. **Looping Mechanism 🔁**: Implement a looping mechanism on vectorized operations.

For a more detailed implementation of handling race conditions, see the [Advanced Tutorial](../tutorials/3_advanced_tutorial.ipynb). It walks through the Sugarscape model with instantaneous growback and shows practical patterns for staged vectorization and conflict resolution.
