# Welcome to mesa-frames 🚀

mesa-frames is an extension of the [mesa](https://github.com/projectmesa/mesa) framework, designed for complex simulations with thousands of agents. By storing agents in a DataFrame, mesa-frames significantly enhances the performance and scalability of mesa, while maintaining a similar syntax.

## Why DataFrames? 📊

DataFrames are optimized for simultaneous operations through [SIMD processing](https://en.wikipedia.org/wiki/Single_instruction,_multiple_data). Currently, mesa-frames supports two main libraries:

- [pandas](https://pandas.pydata.org/): A popular data-manipulation Python library, known for its ease of use and high performance.
- [Polars](https://pola.rs/): A new DataFrame library with a Rust backend, offering innovations like Apache Arrow memory format and support for larger-than-memory DataFrames.

## Performance Boost 🏎️

Check out our performance graphs comparing mesa and mesa-frames for the [Boltzmann Wealth model](https://mesa.readthedocs.io/en/stable/tutorials/intro_tutorial.html):

![Performance Graph](images/readme_plot_0.png)

![Performance Graph without Mesa](images/readme_plot_1.png)

## Quick Start 🚀

### Installation

```bash
git clone https://github.com/adamamer20/mesa_frames.git
cd mesa_frames
pip install -e .[pandas]  # For pandas backend
# or
pip install -e .[polars]  # For Polars backend
```

### Basic Usage

Here's a quick example of how to create an agent set using mesa-frames:

```python
from mesa-frames import AgentSetPolars, ModelDF

class MoneyAgentPolars(AgentSetPolars):
    def __init__(self, n: int, model: ModelDF):
        super().__init__(model)
        self += pl.DataFrame(
            {"unique_id": pl.arange(n, eager=True), "wealth": pl.ones(n, eager=True)}
        )

    def step(self) -> None:
        self.do("give_money")

    def give_money(self):
        # ... (implementation details)

class MoneyModelDF(ModelDF):
    def __init__(self, N: int):
        super().__init__()
        self.agents += MoneyAgentPolars(N, self)

    def step(self):
        self.agents.do("step")

    def run_model(self, n):
        for _ in range(n):
            self.step()
```

## What's Next? 🔮

- API refinement for seamless transition from mesa
- Support for default mesa functions
- GPU functionality (cuDF and Rapids)
- Automatic vectorization of existing mesa models
- Backend-independent AgentSet class

## Get Involved! 🤝

mesa-frames is in its early stages, and we welcome your feedback and contributions! Check out our [GitHub repository](https://github.com/adamamer20/mesa_frames) to get started.

## License

mesa-frames is available under the MIT License. See the [LICENSE](https://github.com/adamamer20/mesa_frames/blob/main/LICENSE) file for full details.
