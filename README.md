# mesa-frames
 `mesa-frames` is an extension of the [`mesa`](https://github.com/projectmesa/mesa) designed for complex simulations with thousands of agents. By storing agents in a Pandas dataframe, `mesa-frames` significantly enhances the performance and scalability of `mesa`, while mantaining a similar syntax.  `mesa-frames` allows for the use of [vectorized functions](https://vegibit.com/what-is-a-vectorized-operation-in-pandas/) whenever simultaneous activation of agents is possible.

## Why Pandas?
[Pandas](https://pandas.pydata.org/) is a popular data-manipulation Python library, developed using C and Cython. Pandas it's known both for its ease of use, allowing for declarative programming and performance. The following is a performance graph showing execution time using `mesa` and `mesa-frames` for the [Boltzmann Wealth model](https://mesa.readthedocs.io/en/stable/tutorials/intro_tutorial.html). 
![](https://github.com/adamamer20/mesa_frames/blob/main/docs/images/readme_plot.png).
(The script used to generate the graph can be found [here](https://github.com/adamamer20/mesa_frames/blob/main/docs/scripts/readme_plot.py))

## Installation

### Cloning the Repository
To get started with `mesa-frames`, first clone the repository from GitHub:
```bash
git clone https://github.com/adamamer20/mesa_frames.git
cd mesa_frame
```
- ### Installing in a conda environment
If you want to install it into a new environment:
```bash
conda create -n myenv -f requirements.txt
pip install -e .
```
If you want to install it into an existing environment:
```bash
conda activate myenv
conda install -f requirements.txt
pip install -e .
```

- ### Installing in a Python virtual environment
If you want to install into a new environment:
```bash
python3 -m venv myenv
source myenv/bin/activate  # On Windows, use `myenv\Scripts\activate`
pip install -r requirements.txt
pip install -e .
```

If you want to install into an existing environment:
```bash
source myenv/bin/activate  # On Windows, use `myenv\Scripts\activate`
pip install -r requirements.txt
pip install -e .
```

## Usage
NOTE: `mesa-frames` is currently in its early stages of development. As such, the usage patterns and API are subject to change. Breaking changes may be introduced. Reports of feedback and issues are encouraged.

### Creation of the model

Creation of the model is fairly similar to the process in `mesa`. 
A new method [`create_agents`](https://github.com/adamamer20/mesa_frames/blob/main/mesa_frames/model.py#L131) is introduced which substitutes the addition of agents to the scheduler.
The [`run_model`](https://github.com/adamamer20/mesa_frames/blob/main/mesa_frames/model.py#L71) method also accepts the number of steps which the model has run.

```python
from mesa_frames.model import ModelDF
from mesa_frames.agent import AgentDF

class MyModel(ModelDF):
    def __init__(self, N):
        super().__init__()
        self.num_agents = N
        self.create_agents(N, {MyAgent: 1})
```

### Creation of an agent
The agent implementation differs from base `mesa`. Agents are only defined at the class level.
The `dtypes` dictionary attribute defines the name and data types of the new columns that will be added to the MyModel.agents dataframe during the execution `create_agents`.
All methods of an `AgentDF` should be class methods and every operation within class methods should operate on the `cls.model.agents` DataFrame, filtering by the class's mask initialized during the execution of `create_agents`. The step method is defined for the entire class.

```python
from mesa_frames.agent import AgentDF

class MyAgent(AgentDF):
    dtypes: dict[str, str] = {"wealth": "int64"}

    @classmethod
    def __init__(cls):
        super().__init__()
        #The next line sets the wealth attribute of all agents that are type MyAgent to 1
        cls.model.agents.loc[cls.mask, "wealth"] = 1

    @classmethod
    def step(cls):
        wealthy_agents = cls.model.agents.loc[cls.mask, "wealth"] > 0
        if wealthy_agents.any():
            #The next line finds the mask of a random sample of agents which ù
            is of the same length as wealthy agents
            other_agents = cls.model.agents.index.isin(
                cls.model.agents.sample(n=wealthy_agents.sum()).index
            )
            cls.model.agents.loc[wealthy_agents, "wealth"] -= 1
            cls.model.agents.loc[other_agents, "wealth"] += 1

```
## What's Next?
- Refine the API to make it more understandable for someone who is already familiar with the `mesa` package. The goal is to provide a seamless experience for users transitioning to or incorporating `mesa-frames`.
- Adding support for default `mesa` functions, ensure that the standard mesa functionality is preserved.
- Adding support for or shifting to `polars` instead of pandas. This change aims to boost performance and decrease memory consumption, which is crucial for models with milions of agents.
- Creating a decorator that will automatically vectorize an existing mesa model. This feature will allow users to easily tap into the performance enhancements that mesa-frames offers without significant code alterations.

## License

`mesa-frames` is made available under the MIT License. This license allows you to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the software, subject to the following conditions:

- The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
- The software is provided "as is", without warranty of any kind.

For the full license text, see the [LICENSE](https://github.com/adamamer20/mesa_frames/blob/main/LICENSE) file in the GitHub repository.