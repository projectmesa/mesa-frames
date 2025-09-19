# Classes 📚

## AgentSet 👥

To create your own AgentSet class, you need to subclass the AgentSet class and make sure to call `super().__init__(model)`.

Typically, the next step would be to populate the class with your agents. To do that, you need to add a DataFrame to the AgentSet. You can do `self += agents` or `self.add(agents)`, where `agents` is a DataFrame or something that could be passed to a DataFrame constructor, like a dictionary or lists of lists. You need to make sure your DataFrame doesn't have a 'unique_id' column because IDs are generated automatically, otherwise you will get an error raised. In the DataFrame, you should also put any attribute of the agent you are using.

How can you choose which agents should be in the same AgentSet? The idea is that you should minimize the missing values in the DataFrame (so they should have similar/same attributes) and mostly everybody should do the same actions.

Example:

```python
class MoneyAgents(AgentSet):
    def __init__(self, n: int, model: Model):
        super().__init__(model)
        self.initial_wealth = pl.ones(n)
        self += pl.DataFrame({
            "wealth": self.initial_wealth
        })

    def step(self):
        self["wealth"] = self["wealth"] + self.random.integers(n)
```

You can access the underlying DataFrame where agents are stored with `self.df`. This allows you to use DataFrame methods like `self.df.sample` or `self.df.group_by("wealth")` and more.

## Model 🏗️

<<<<<<< HEAD
To add your AgentSet to your Model, use the registry `self.sets` with `+=` or `add`.

Note: All agent sets live inside `AgentSetRegistry` (available as `model.sets`). Access sets through the registry, and access DataFrames from the set itself. For example: `self.sets["Preys"].df`
=======

To add your AgentSet to your Model, you should also add it to the sets with `+=` or `add`.

NOTE: Model.sets are stored in a class which is entirely similar to AgentSet called AgentSetRegistry. The API of the two are the same. If you try accessing AgentSetRegistry.df, you will get a dictionary of `[AgentSet, DataFrame]`.
>>>>>>> 51c54cd666d876a5debb1b7dd71556ee9c458956

Example:

```python
class EcosystemModel(Model):
    def __init__(self, n_prey, n_predators):
        super().__init__()
        self.sets += Preys(n_prey, self)
        self.sets += Predators(n_predators, self)

    def step(self):
        self.sets.do("move")
        self.sets.do("hunt")
<<<<<<< HEAD
        # Access specific sets via the registry
        self.sets["Preys"].do("reproduce")
=======
        self.prey.do("reproduce")
>>>>>>> 51c54cd666d876a5debb1b7dd71556ee9c458956
```

## Space: Grid 🌐

mesa-frames provides efficient implementations of spatial environments:

- Spatial operations (like moving agents) are vectorized for performance

Example:

```python
class GridWorld(Model):
    def __init__(self, width, height):
        super().__init__()
        self.space = Grid(self, (width, height))
        self.sets += AgentSet(100, self)
        self.space.place_to_empty(self.sets)
```

A continuous GeoSpace, NetworkSpace, and a collection to have multiple spaces in the models are in the works! 🚧

## DataCollector 🗂️

`DataCollector` records model- and agent-level data during simulation.
You configure what to collect, how to store it, and when to trigger collection.

Example:

```python
class ExampleModel(Model):
    def __init__(self):
        super().__init__()
<<<<<<< HEAD
        # Add the set to the registry
        self.sets.add(MoneyAgents(100, self))
        # Configure reporters: use the registry to locate sets; get df from the set
        self.datacollector = DataCollector(
            model=self,
            model_reporters={
                "total_wealth": lambda m: m.sets["MoneyAgents"].df["wealth"].sum(),
            },
=======
        self.sets = MoneyAgent(self)
        self.datacollector = DataCollector(
            model=self,
            model_reporters={"total_wealth": lambda m: lambda m: list(m.sets.df.values())[0]["wealth"].sum()},
>>>>>>> 51c54cd666d876a5debb1b7dd71556ee9c458956
            agent_reporters={"wealth": "wealth"},
            storage="csv",
            storage_uri="./data",
            trigger=lambda m: m.steps % 2 == 0,
        )

    def step(self):
<<<<<<< HEAD
        # Step all sets via the registry
        self.sets.do("step")
=======
        self.sets.step()
>>>>>>> 51c54cd666d876a5debb1b7dd71556ee9c458956
        self.datacollector.conditional_collect()
        self.datacollector.flush()
```
