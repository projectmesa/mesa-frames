# Classes 📚

## AgentSetDF 👥

To create your own AgentSetDF class, you need to subclass the AgentSetPolars class and make sure to call `super().__init__(model)`.

Typically, the next step would be to populate the class with your agents. To do that, you need to add a DataFrame to the AgentSetDF. You can do `self += agents` or `self.add(agents)`, where `agents` is a DataFrame or something that could be passed to a DataFrame constructor, like a dictionary or lists of lists. You need to make sure your DataFrame doesn't have a 'unique_id' column because IDs are generated automatically, otherwise you will get an error raised. In the DataFrame, you should also put any attribute of the agent you are using.

How can you choose which agents should be in the same AgentSet? The idea is that you should minimize the missing values in the DataFrame (so they should have similar/same attributes) and mostly everybody should do the same actions.

Example:

```python
class MoneyAgent(AgentSetPolars):
    def __init__(self, n: int, model: ModelDF):
        super().__init__(model)
        self.initial_wealth = pl.ones(n)
        self += pl.DataFrame({
            "wealth": self.initial_wealth
        })

    def step(self):
        self["wealth"] = self["wealth"] + self.random.integers(n)
```

You can access the underlying DataFrame where agents are stored with `self.df`. This allows you to use DataFrame methods like `self.df.sample` or `self.df.group_by("wealth")` and more.

## ModelDF 🏗️

To add your AgentSetDF to your ModelDF, you should also add it to the agents with `+=` or `add`.

NOTE: ModelDF.agents are stored in a class which is entirely similar to AgentSetDF called AgentsDF. The API of the two are the same. If you try accessing AgentsDF.df, you will get a dictionary of `[AgentSetDF, DataFrame]`.

Example:

```python
class EcosystemModel(ModelDF):
    def __init__(self, n_prey, n_predators):
        super().__init__()
        self.agents += Preys(n_prey, self)
        self.agents += Predators(n_predators, self)

    def step(self):
        self.agents.do("move")
        self.agents.do("hunt")
        self.prey.do("reproduce")
```

## Space: GridDF 🌐

mesa-frames provides efficient implementations of spatial environments:

- Spatial operations (like moving agents) are vectorized for performance

Example:

```python
class GridWorld(ModelDF):
    def __init__(self, width, height):
        super().__init__()
        self.space = GridPolars(self, (width, height))
        self.agents += AgentSet(100, self)
        self.space.place_to_empty(self.agents)
```

A continuous GeoSpace, NetworkSpace, and a collection to have multiple spaces in the models are in the works! 🚧

## DataCollector 🗂️  

`DataCollector` records model- and agent-level data during simulation.  
You configure what to collect, how to store it, and when to trigger collection.  

Example:  

```python
class ExampleModel(ModelDF):
    def __init__(self):
        super().__init__()
        self.agents = ExampleAgentSet1(self)
        self.datacollector = DataCollector(
            model=self,
            model_reporters={"total_wealth": lambda m: m.agents["wealth"].sum()},
            agent_reporters={"wealth": "wealth", "age": "age"},
            storage="csv",
            storage_uri="./data",
            trigger=lambda m: m.schedule.steps % 2 == 0
        )

    def step(self):
        self.agents.step()
        self.datacollector.conditional_collect()
        self.datacollector.flush()
```
