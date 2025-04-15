# Roadmap 🗺️

This document outlines the development roadmap for the mesa-frames project. It provides insights into our current priorities, upcoming features, and long-term vision.

## 0.1.0 Stable Release Goals 🎯

### 1. Deprecating pandas and Transitioning to polars

One of our major priorities is to move from pandas to polars as the primary dataframe backend. This transition is motivated by performance considerations. We should use the lazily evaluated version of polars.

**Related issues:** [#89: Investigate using Ibis for the common interface library to any DF backend](https://github.com/projectmesa/mesa-frames/issues/89), [#10: GPU integration: Dask, cuda (cudf) and RAPIDS (Polars)](https://github.com/projectmesa/mesa-frames/issues/10)

#### Progress and Next Steps

- We are exploring [Ibis](https://ibis-project.org/) or [narwhals](https://github.com/narwhals-dev/narwhals) as a common interface library that could support multiple backends (Polars, DuckDB, Spark etc.), but since most of the development is currently in polars, we will currently continue using Polars.
- The pandas backend is becoming increasingly problematic to maintain and will eventually be deprecated
- Benchmarking is underway to quantify performance differences between different backends
- We're investigating GPU acceleration options, including the potential integration with RAPIDS ecosystem

### 2. Handling Concurrency Management

A critical aspect of agent-based models is efficiently managing concurrent agent movements, especially when multiple agents attempt to move to the same location simultaneously. We aim to implement abstractions that handle these concurrency conditions automatically.

**Related issues:** [#108: Adding abstraction of optimal agent movement](https://github.com/projectmesa/mesa-frames/issues/108), [#48: Emulate RandomActivation with DataFrame.rolling](https://github.com/projectmesa/mesa-frames/issues/48)

#### Sugarscape Example of Concurrency Issues

Testing with many potential collisions revealed a specific issue:

**Problem scenario:**

- Consider two agents targeting the same cell:
  - A mid-priority agent (higher in the agent order)
  - A low-priority agent (lower in the agent order)
- The mid-priority agent has low preference for the cell
- The low-priority agent has high preference for the cell
- Without accounting for priority:
  - The mid-priority agent's best moves kept getting "stolen" by higher priority agents
  - This forced it to resort to lower preference target cells
  - However, these lower preference cells were often already taken by lower priority agents in previous iterations

**Solution approach:**

- Implement a "priority" count to ensure that each action is "legal"
- This prevents race conditions but requires recomputing the priority at each iteration
- Current implementation may be slower than Numba due to this overhead
- After the Ibis refactoring, we can investigate if lazy evaluation can help mitigate this performance issue

The Sugarscape example demonstrates the need for this abstraction, as multiple agents often attempt to move to the same cell simultaneously. By generalizing this functionality, we can eliminate the need for users to implement complex conflict resolution logic repeatedly.

#### Progress and Next Steps

- Create utility functions in `DiscreteSpaceDF` and `AgentContainer` to move agents optimally based on specified attributes
- Provide built-in resolution strategies for common concurrency scenarios
- Ensure the implementation works efficiently with the vectorized approach of mesa-frames

### Additional 0.1.0 Goals

- Complete core API stabilization
- Completely mirror mesa's functionality
- Improve documentation and examples
- Address outstanding bugs and performance issues

## Beyond 0.1.0

Future roadmap items will be added as the project evolves and new priorities emerge.

We welcome community feedback on our roadmap! Please open an issue if you have suggestions or would like to contribute to any of these initiatives.
