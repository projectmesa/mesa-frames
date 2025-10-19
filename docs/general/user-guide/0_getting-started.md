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

### From Imperative Code to Behavioral Rules 💭

When scientists describe an ABM-like process they typically write a **system of state-transition functions**:

$$
x_i(t+1) = f_i\big(x_i(t),\; \mathcal{N}(i,t),\; E(t)\big)
$$

Here, $x_i(t)$ is the agent’s state, $\mathcal{N}(i,t)$ its neighborhood or local environment, and $E(t)$ a global environment; $f_i$ is the behavioral law.

In classic `mesa`, agent behavior is implemented through explicit loops: each agent individually gathers information from its neighbors, computes its next state, and often stores this in a buffer to ensure synchronous updates. The behavioral law $f_i$ is distributed across multiple steps: neighbor iteration, temporary buffers, and scheduling logic, resulting in procedural, step-by-step control flow.

In `mesa-frames`, these stages are unified into a single vectorized transformation. Agent interactions, state transitions, and updates are expressed as DataFrame operations (such as joins, group-bys, and column expressions) allowing all agents to process perceptions and commit actions simultaneously. This approach centralizes the behavioral law $f_i$ into concise, declarative rules, improving clarity and performance.

#### Example: Network contagion (Linear Threshold)

Behavioral rule: a node activates if the number of active neighbors ≥ its threshold.

=== "mesa-frames"

    Single vectorized transformation. A join brings in source activity, a group-by aggregates exposures per destination, and a column expression applies the activation equation and commits in one pass, no explicit loops or staging structure needed.

    ```python
    class Nodes(AgentSet):
        # self.df columns: agent_id, active (bool), theta (int)
        # self.model.space.edges: DataFrame[src, dst]
        def step(self):
            E = self.model.space.edges  # [src, dst]
            # Exposure: active neighbors per dst (vectorized join + groupby)
            exposures = (
                E.join(
                    self.df.select(pl.col("agent_id").alias("src"),
                                   pl.col("active").alias("src_active")),
                    on="src", how="left"
                )
                .with_columns(pl.col("src_active").fill_null(False))
                .group_by("dst")
                .agg(pl.col("src_active").sum().alias("k_active"))
            )
            # Behavioral equation applied to all agents, committed in-place
            self.df = (
                self.df
                .join(exposures, left_on="agent_id", right_on="dst", how="left")
                .with_columns(pl.col("k_active").fill_null(0))
                .with_columns(
                    (pl.col("active") | (pl.col("k_active") >= pl.col("theta")))
                    .alias("active")
                )
                .drop(["k_active", "dst"])
            )
    ```

=== "mesa"

    Two-phase imperative procedure. Each agent loops over its neighbors to count active ones (exposure), stores a provisional next state to avoid premature mutation, then a separate pass commits all buffered states for synchronicity.

    ```python
    class Node(mesa.Agent):
        def step(self):
            # (1) Gather exposure: count active neighbors right now
            k_active = sum(
                1 for j in self.model.G.neighbors(self.unique_id)
                if self.model.id2agent[j].active
            )
            # (2) Compute next state (don't mutate yet to stay synchronous)
            self.next_active = self.active or (k_active >= self.theta)

    # Second pass (outside the agent method) performs the commit:
    for a in model.agents:
        a.active = a.next_active
    ```

!!! tip "Transition tips — quick summary"
    1. Think in sets: operate on AgentSets/DataFrames, not per-agent objects.
    2. Write transitions as Polars column expressions; avoid Python loops.
    3. Use joins + group-bys to compute interactions/exposure across relations.
    4. Commit state synchronously in one vectorized pass.
    5. Group similar agents into one AgentSet with typed columns.
    6. Use UDFs or staged/iterative patterns only for true race/conflict cases.

### Handling Race Conditions 🏁

When simultaneous activation is not possible, you need to handle race conditions carefully. There are two main approaches:

1. **Custom UDF with Numba 🔧**: Use a custom User Defined Function (UDF) with Numba for efficient sequential processing.

   - [Polars UDF Guide](https://docs.pola.rs/user-guide/expressions/user-defined-functions/)

2. **Looping Mechanism 🔁**: Implement a looping mechanism on vectorized operations.

For a more detailed implementation of handling race conditions, see the [Advanced Tutorial](../tutorials/3_advanced_tutorial.ipynb). It walks through the Sugarscape model with instantaneous growback and shows practical patterns for staged vectorization and conflict resolution.
