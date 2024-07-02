# mesa-frames

mesa-frames is an extension of the [mesa](https://github.com/projectmesa/mesa) framework, designed for complex simulations with thousands of agents. By storing agents in a DataFrame, mesa-frames significantly enhances the performance and scalability of mesa, while maintaining a similar syntax. mesa-frames allows for the use of [vectorized functions](https://vegibit.com/what-is-a-vectorized-operation-in-pandas/) whenever simultaneous activation of agents is possible.

## Why DataFrames?

DataFrames are optimized for simultaneous operations through [SIMD processing](https://en.wikipedia.org/wiki/Single_instruction,_multiple_data). At the moment, mesa-frames supports the use of two main libraries: Pandas and Polars.

- [Pandas](https://pandas.pydata.org/) is a popular data-manipulation Python library, developed using C and Cython. Pandas is known for its ease of use, allowing for declarative programming and high performance.
- [Polars](https://pola.rs/) is a new DataFrame library with a syntax similar to Pandas but with several innovations, including a backend implemented in Rust, the Apache Arrow memory format, query optimization, and support for larger-than-memory DataFrames.

The following is a performance graph showing execution time using mesa and mesa-frames for the [Boltzmann Wealth model](https://mesa.readthedocs.io/en/stable/tutorials/intro_tutorial.html).

![Performance Graph](https://github.com/adamamer20/mesa_frames/blob/main/docs/images/readme_plot.png)

(The script used to generate the graph can be found [here](https://github.com/adamamer20/mesa_frames/blob/main/docs/scripts/readme_plot.py))

## Installation

### Cloning the Repository

To get started with mesa-frames, first clone the repository from GitHub:

```bash
git clone https://github.com/adamamer20/mesa_frames.git
cd mesa_frames
```

### Installing in a Conda Environment

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
# For pandas backend
pip install -e .[pandas]
# Alternatively, for Polars backend
pip install -e .[polars]
```

### Installing in a Python Virtual Environment

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
# For pandas backend
pip install -e .[pandas]
# Alternatively, for Polars backend
pip install -e .[polars]
```


## Usage

**Note:** mesa-frames is currently in its early stages of development. As such, the usage patterns and API are subject to change. Breaking changes may be introduced. Reports of feedback and issues are encouraged.

You can find the API documentation [here](https://adamamer20.github.io/mesa-frames/api).

### Creation of an Agent

The agent implementation differs from base mesa. Agents are only defined at the AgentSet level. You can import either `AgentSetPandas` or `AgentSetPolars`. As in mesa, you subclass and make sure to call `super().__init__(model)`. You can use the `add` method or the `+=` operator to add agents to the AgentSet. Most methods mirror the functionality of `mesa.AgentSet`. Additionally, `mesa-frames.AgentSet` implements many dunder methods such as `AgentSet[mask, attr]` to get and set items intuitively. All operations are by default inplace, but if you'd like to use functional programming, mesa-frames implements a fast copy method which aims to reduce memory usage, relying on reference-only and native copy methods.

```python
from mesa-frames import AgentSetPolars

class MoneyAgentPolars(AgentSetPolars):
    def __init__(self, n: int, model: ModelDF):
        super().__init__(model)
        # Adding the agents to the agent set
        self += pl.DataFrame(
            {"unique_id": pl.arange(n, eager=True), "wealth": pl.ones(n, eager=True)}
        )

    def step(self) -> None:
        # The give_money method is called
        self.do("give_money")

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

### Creation of the Model

Creation of the model is fairly similar to the process in mesa. You subclass `ModelDF` and call `super().__init__()`. The `model.agents` attribute has the same interface as `mesa-frames.AgentSet`. You can use `+=` or `self.agents.add` with a `mesa-frames.AgentSet` (or a list of `AgentSet`) to add agents to the model.

```python
from mesa-frames import ModelDF

class MoneyModelDF(ModelDF):
    def __init__(self, N: int, agents_cls):
        super().__init__()
        self.n_agents = N
        self.agents += MoneyAgentPolars(N, self)

    def step(self):
        # Executes the step method for every agentset in self.agents
        self.agents.do("step")

    def run_model(self, n):
        for _ in range(n):
            self.step()
```

## What's Next?

- Refine the API to make it more understandable for someone who is already familiar with the mesa package. The goal is to provide a seamless experience for users transitioning to or incorporating mesa-frames.
- Adding support for default mesa functions to ensure that the standard mesa functionality is preserved.
- Adding GPU functionality (cuDF and Rapids).
- Creating a decorator that will automatically vectorize an existing mesa model. This feature will allow users to easily tap into the performance enhancements that mesa-frames offers without significant code alterations.
- Creating a unique class for AgentSet, independent of the backend implementation.

## License

mesa-frames is made available under the MIT License. This license allows you to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the software, subject to the following conditions:

- The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
- The software is provided "as is", without warranty of any kind.

For the full license text, see the [LICENSE](https://github.com/adamamer20/mesa_frames/blob/main/LICENSE) file in the GitHub repository.
