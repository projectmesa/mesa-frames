# Getting Started 🚀

## Main Concepts 🧠

### DataFrame-Based Object-Oriented Framework 📊

Unlike traditional mesa models where each agent is an individual Python object, mesa-frames stores all agents of a particular type in a single DataFrame. We operate only at the AgentSet level.

This approach allows for:

- Efficient memory usage
- Improved performance through vectorized operations on agent attributes (This is what makes `mesa-frames` fast)

Objects can be easily subclassed to respect mesa's object-oriented philosophy.

### Vectorized Operations ⚡

mesa-frames leverages the power of vectorized operations provided by DataFrame libraries:

- Operations are performed on entire columns of data at once
- This approach is significantly faster than iterating over individual agents
- Complex behaviors can be expressed in fewer lines of code

You should never use loops to iterate through your agents. Instead, use vectorized operations and implemented methods. If you need to loop, loop through vectorized operations (see the advanced tutorial SugarScape IG for more information).

It's important to note that in traditional `mesa` models, the order in which agents are activated can significantly impact the results of the model (see [Comer, 2014](http://mars.gmu.edu/bitstream/handle/1920/9070/Comer_gmu_0883E_10539.pdf)). `mesa-frames`, by default, doesn't have this issue as all agents are processed simultaneously. However, this comes with the trade-off of needing to carefully implement conflict resolution mechanisms when sequential processing is required. We'll discuss how to handle these situations later in this guide.

Check out these resources to understand vectorization and why it speeds up the code:

- [What is vectorization?](https://stackoverflow.com/a/1422181)
- [Vectorization Explained, Step by Step](https://machinelearningcompass.com/machine_learning_math/vectorization/)

Here's a comparison between mesa-frames and mesa:

=== "mesa-frames"
    ```python
    class MoneyAgentPolarsConcise(AgentSetPolars):
        # initialization...

        def give_money(self):
            # Active agents are changed to wealthy agents
            self.select(self.wealth > 0)

            # Receiving agents are sampled (only native expressions currently supported)
            other_agents = self.agents.sample(
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
                other_agent = self.random.choice(self.model.agents)
                if other_agent is not None:
                    other_agent.wealth += 1
                    self.wealth -= 1
    ```

As you can see, while in mesa you should iterate through all the agents' steps in the model class, here you execute the method once for all agents.

### Backend Flexibility 🔄

mesa-frames aims to support multiple DataFrame backends:
The supported backends right now are

- **pandas**: A widely-used data manipulation library
- **Polars**: A high-performance DataFrame library written in Rust

Users can choose the backend that best suits their needs:

    ```python
    from mesa_frames import AgentSetPandas  # or AgentSetPolars
    ```

Currently, there are two implementations of AgentSetDF and GridDF, one for each backend implementation: AgentSetPandas and AgentSetPolars, and GridPandas and GridPolars.
We encourage you to use the Polars implementation for increased performance. We are working on creating a unique interface [here](https://github.com/projectmesa/mesa-frames/discussions/12). Let us know what you think!

Soon we will also have multiple other backends like Dask, cuDF, and Dask-cuDF!

## Coming from mesa 🔀

If you're familiar with mesa, this guide will help you understand the key differences in code structure between mesa and mesa-frames.

### Agent Representation 👥

- mesa: Each agent is an individual object instance. Methods are defined for individual agents and called on each agent.
- mesa-frames: Agents are rows in a DataFrame, grouped into AgentSets. Methods are defined for AgentSets and operate on all agents simultaneously.

=== "mesa-frames"
    ```python
    class MoneyAgentSet(AgentSetPolars):
        def **init**(self, n, model):
            super().**init**(model)
            self += pl.DataFrame({
                "unique_id": pl.arange(n),
                "wealth": pl.ones(n)
            })

        def step(self):
            givers = self.wealth > 0
            receivers = self.agents.sample(n=len(self.active_agents))
            self[givers, "wealth"] -= 1
            new_wealth = receivers.groupby("unique_id").count()
            self[new_wealth["unique_id"], "wealth"] += new_wealth["count"]
    ```

=== "mesa"
    ```python
    class MoneyAgent(Agent):
        def **init**(self, unique_id, model):
            super().**init**(unique_id, model)
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
    class MoneyModel(ModelDF):
        def **init**(self, N):
            super().**init**()
            self.agents += MoneyAgentSet(N, self)

        def step(self):
            self.agents.do("step")
    ```

=== "mesa"
    ```python
    class MoneyModel(Model):
        def **init**(self, N):
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
2. **Leverage DataFrame Operations 🛠️**: Familiarize yourself with pandas or Polars operations for efficient agent manipulation.
3. **Vectorize Logic 🚅**: Convert loops and conditionals to vectorized operations where possible.
4. **Use AgentSets 📦**: Group similar agents into AgentSets instead of creating many individual agent classes.

### Handling Race Conditions 🏁

When simultaneous activation is not possible, you need to handle race conditions carefully. There are two main approaches:

1. **Custom UDF with Numba 🔧**: Use a custom User Defined Function (UDF) with Numba for efficient sequential processing.

   - [Polars UDF Guide](https://docs.pola.rs/user-guide/expressions/user-defined-functions/)
   - [pandas Numba Engine](https://pandas.pydata.org/pandas-docs/stable/user_guide/window.html#numba-engine)

2. **Looping Mechanism 🔁**: Implement a looping mechanism on vectorized operations.

For a more detailed implementation of handling race conditions, please refer to the `examples/sugarscape-ig` in the mesa-frames repository. This example demonstrates how to implement the Sugarscape model with instantaneous growback, which requires careful handling of sequential agent actions.
