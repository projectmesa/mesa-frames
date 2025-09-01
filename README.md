# mesa-frames 🚀

mesa-frames is an extension of the [mesa](https://github.com/projectmesa/mesa) framework, designed for complex simulations with thousands of agents. By storing agents in a DataFrame, mesa-frames significantly enhances the performance and scalability of mesa, while maintaining a similar syntax. mesa-frames allows for the use of [vectorized functions](https://stackoverflow.com/a/1422198) which significantly speeds up operations whenever simultaneous activation of agents is possible.

## Why DataFrames? 📊

DataFrames are optimized for simultaneous operations through [SIMD processing](https://en.wikipedia.org/wiki/Single_instruction,_multiple_data). At the moment, mesa-frames supports the use of Polars library.

- [Polars](https://pola.rs/) is a new DataFrame library with a syntax similar to pandas but with several innovations, including a backend implemented in Rust, the Apache Arrow memory format, query optimization, and support for larger-than-memory DataFrames.

The following is a performance graph showing execution time using mesa and mesa-frames for the [Boltzmann Wealth model](https://mesa.readthedocs.io/en/stable/tutorials/intro_tutorial.html).

![Performance Graph with Mesa](https://github.com/projectmesa/mesa-frames/blob/main/examples/boltzmann_wealth/boltzmann_with_mesa.png)

![Performance Graph without Mesa](https://github.com/projectmesa/mesa-frames/blob/main/examples/boltzmann_wealth/boltzmann_no_mesa.png)

([You can check the script used to generate the graph here](https://github.com/projectmesa/mesa-frames/blob/main/examples/boltzmann_wealth/performance_plot.py), but if you want to additionally compare vs Mesa, you have to uncomment `mesa_implementation` and its label)

## Installation

### Install from PyPI

```bash
pip install mesa-frames
```

### Install from Source

To install the most updated version of mesa-frames, you can clone the repository and install the package in editable mode.

#### Cloning the Repository

To get started with mesa-frames, first clone the repository from GitHub:

```bash
git clone https://github.com/projectmesa/mesa-frames.git
cd mesa_frames
```

#### Installing in a Conda Environment

If you want to install it into a new environment:

```bash
conda create -n myenv
```

If you want to install it into an existing environment:

```bash
conda activate myenv
```

Then, to install mesa-frames itself:

```bash
pip install -e .
```

#### Installing in a Python Virtual Environment

If you want to install it into a new environment:

```bash
python3 -m venv myenv
source myenv/bin/activate  # On Windows, use `myenv\Scripts\activate`
```

If you want to install it into an existing environment:

```bash
source myenv/bin/activate  # On Windows, use `myenv\Scripts\activate`
```

Then, to install mesa-frames itself:

```bash
pip install -e .
```

## Usage

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/projectmesa/mesa-frames/blob/main/docs/general/user-guide/2_introductory-tutorial.ipynb)

**Note:** mesa-frames is currently in its early stages of development. As such, the usage patterns and API are subject to change. Breaking changes may be introduced. Reports of feedback and issues are encouraged.

[You can find the API documentation here](https://projectmesa.github.io/mesa-frames/api).

### Creation of an Agent

The agent implementation differs from base mesa. Agents are only defined at the AgentSet level. You can import `AgentSet`. As in mesa, you subclass and make sure to call `super().__init__(model)`. You can use the `add` method or the `+=` operator to add agents to the AgentSet. Most methods mirror the functionality of `mesa.AgentSet`. Additionally, `mesa-frames.AgentSet` implements many dunder methods such as `AgentSet[mask, attr]` to get and set items intuitively. All operations are by default inplace, but if you'd like to use functional programming, mesa-frames implements a fast copy method which aims to reduce memory usage, relying on reference-only and native copy methods.

```python
from mesa-frames import AgentSet

class MoneyAgentDF(AgentSet):
    def __init__(self, n: int, model: Model):
        super().__init__(model)
        # Adding the agents to the agent set
        self += pl.DataFrame(
            {"wealth": pl.ones(n, eager=True)}
        )

    def step(self) -> None:
        # The give_money method is called
        self.do("give_money")

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

### Creation of the Model

Creation of the model is fairly similar to the process in mesa. You subclass `Model` and call `super().__init__()`. The `model.sets` attribute has the same interface as `mesa-frames.AgentSet`. You can use `+=` or `self.sets.add` with a `mesa-frames.AgentSet` (or a list of `AgentSet`) to add agents to the model.

```python
from mesa-frames import Model

class MoneyModelDF(Model):
    def __init__(self, N: int, agents_cls):
        super().__init__()
        self.n_agents = N
        self.sets += MoneyAgentDF(N, self)

    def step(self):
        # Executes the step method for every agentset in self.sets
        self.sets.do("step")

    def run_model(self, n):
        for _ in range(n):
            self.step()
```

## What's Next? 🔮

- Refine the API to make it more understandable for someone who is already familiar with the mesa package. The goal is to provide a seamless experience for users transitioning to or incorporating mesa-frames.
- Adding support for default mesa functions to ensure that the standard mesa functionality is preserved.
- Adding GPU functionality (cuDF and Dask-cuDF).
- Creating a decorator that will automatically vectorize an existing mesa model. This feature will allow users to easily tap into the performance enhancements that mesa-frames offers without significant code alterations.
- Creating a unique class for AgentSet, independent of the backend implementation.

## License

Copyright 2024 Adam Amer, Project Mesa team and contributors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

For the full license text, see the [LICENSE](https://github.com/projectmesa/mesa-frames/blob/main/LICENSE) file in the GitHub repository.
