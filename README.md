<p align="center">
  <img src="https://raw.githubusercontent.com/projectmesa/mesa/main/docs/images/mesa_logo.png" alt="Mesa logo" width="96">
</p>

<h1 align="center">mesa-frames</h1>

|         |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| ------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| CI/CD   | [![CI Checks](https://github.com/projectmesa/mesa-frames/actions/workflows/build.yml/badge.svg?branch=main)](https://github.com/projectmesa/mesa-frames/actions/workflows/build.yml) [![codecov](https://codecov.io/gh/projectmesa/mesa-frames/branch/main/graph/badge.svg)](https://app.codecov.io/gh/projectmesa/mesa-frames)                                                                                                                     |
| Package | [![PyPI - Version](https://img.shields.io/pypi/v/mesa-frames.svg?logo=pypi&label=PyPI&logoColor=gold)](https://pypi.org/project/mesa-frames/) [![PyPI - Downloads](https://img.shields.io/pypi/dm/mesa-frames.svg?color=blue&label=Downloads&logo=pypi&logoColor=gold)](https://pypi.org/project/mesa-frames/) [![PyPI - Python Version](https://img.shields.io/pypi/pyversions/mesa-frames.svg?logo=python&label=Python&logoColor=gold)](https://pypi.org/project/mesa-frames/) |
| Meta    | [![linting - Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://docs.astral.sh/ruff/) [![formatter - Ruff](https://img.shields.io/badge/formatter-Ruff-0f172a?logo=ruff&logoColor=white)](https://docs.astral.sh/ruff/formatter/) [![Hatch project](https://img.shields.io/badge/%F0%9F%A5%9A-Hatch-4051b5.svg)](https://github.com/pypa/hatch) [![Managed with uv](https://img.shields.io/badge/managed%20with-uv-5a4fcf?logo=uv&logoColor=white)](https://github.com/astral-sh/uv) |
| Chat    | [![chat](https://img.shields.io/matrix/project-mesa:matrix.org?label=chat&logo=Matrix)](https://matrix.to/#/#project-mesa:matrix.org)                                                                                                                                                                                                                                                                                                                                                      |

---

## Scale Mesa beyond its limits

Classic [Mesa](https://github.com/projectmesa/mesa) stores each agent as a Python object, which quickly becomes a bottleneck at scale.  
**mesa-frames** reimagines agent storage using **Polars DataFrames**, so agents live in a columnar store rather than the Python heap.  

You keep the Mesa-style `Model` / `AgentSet` structure, but updates are vectorized and memory-efficient.

### Why it matters
- ⚡ **10× faster** bulk updates on 10k+ agents (see benchmarks)  
- 📊 **Columnar execution** via [Polars](https://docs.pola.rs/): [SIMD](https://en.wikipedia.org/wiki/Single_instruction,_multiple_data) ops, multi-core support  
- 🔄 **Declarative logic**: agent rules as transformations, not Python loops  
- 🚀 **Roadmap**: Lazy queries and GPU support for even faster models

---

## Who is it for?

- Researchers needing to scale to **tens or hundreds of thousands of agents**  
- Users whose agent logic can be written as **vectorized, set-based operations**  

❌ **Not a good fit if:** your model depends on strict per-agent sequencing, complex non-vectorizable methods, or fine-grained identity tracking.


### Install from Source (development)

Clone the repository and install dependencies with [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/projectmesa/mesa-frames.git
cd mesa-frames
uv sync --all-extras
```

`uv sync` creates a local `.venv/` with mesa-frames and its development extras. Run tooling through uv to keep the virtual environment isolated:

```bash
uv run pytest -q --cov=mesa_frames --cov-report=term-missing
uv run ruff check . --fix
uv run pre-commit run -a
```

## Usage

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/projectmesa/mesa-frames/blob/main/docs/general/user-guide/2_introductory-tutorial.ipynb)

**Note:** mesa-frames is currently in its early stages of development. As such, the usage patterns and API are subject to change. Breaking changes may be introduced. Reports of feedback and issues are encouraged.

[You can find the API documentation here](https://projectmesa.github.io/mesa-frames/api).

### Creation of an Agent

The agent implementation differs from base mesa. Agents are only defined at the AgentSet level. You can import `AgentSet`. As in mesa, you subclass and make sure to call `super().__init__(model)`. You can use the `add` method or the `+=` operator to add agents to the AgentSet. Most methods mirror the functionality of `mesa.AgentSet`. Additionally, `mesa-frames.AgentSet` implements many dunder methods such as `AgentSet[mask, attr]` to get and set items intuitively. All operations are by default inplace, but if you'd like to use functional programming, mesa-frames implements a fast copy method which aims to reduce memory usage, relying on reference-only and native copy methods.

```python
from mesa-frames import AgentSet

class MoneyAgents(AgentSet):
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
        self.sets += MoneyAgents(N, self)

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
