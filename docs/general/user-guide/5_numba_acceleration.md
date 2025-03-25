# Numba Acceleration in mesa-frames

## Introduction

This guide explains how to use Numba to accelerate agent-based models in mesa-frames. [Numba](https://numba.pydata.org/) is a just-in-time (JIT) compiler for Python that can significantly improve performance of numerical Python code by compiling it to optimized machine code at runtime.

Mesa-frames already offers substantial performance improvements over standard mesa by using DataFrame-based storage (especially with Polars), but for computationally intensive simulations, Numba can provide additional acceleration.

## When to Use Numba

Consider using Numba acceleration in the following scenarios:

1. **Large agent populations**: When your simulation involves thousands or millions of agents
2. **Computationally intensive agent methods**: When agents perform complex calculations or numerical operations
3. **Spatial operations**: For optimizing neighbor search and spatial movement calculations
4. **Performance bottlenecks**: When profiling reveals specific methods as performance bottlenecks

## Numba Integration Options

Mesa-frames supports several Numba integration approaches:

1. **CPU acceleration**: Standard Numba acceleration on a single CPU core
2. **Parallel CPU acceleration**: Utilizing multiple CPU cores for parallel processing
3. **GPU acceleration**: Leveraging NVIDIA GPUs through CUDA (requires a compatible GPU and CUDA installation)

## Basic Implementation Pattern

The recommended pattern for implementing Numba acceleration in mesa-frames follows these steps:

1. Identify the performance-critical method in your agent class
2. Extract the numerical computation into a separate function
3. Decorate this function with Numba's `@jit`, `@vectorize`, or `@guvectorize` decorators
4. Call this accelerated function from your agent class method

## Example: Basic Numba Acceleration

Here's a simple example of using Numba to accelerate an agent method:

```python
import numpy as np
from numba import jit
from mesa_frames import AgentSetPolars, ModelDF

class MyAgentSet(AgentSetPolars):
    def __init__(self, model: ModelDF, n_agents: int):
        super().__init__(model)
        # Initialize agents
        self += pl.DataFrame({
            "unique_id": pl.arange(n_agents, eager=True),
            "value": pl.ones(n_agents, eager=True)
        })
    
    def complex_calculation(self):
        # Extract data to numpy arrays for Numba processing
        values = self.agents["value"].to_numpy()
        
        # Call the Numba-accelerated function
        results = self._calculate_with_numba(values)
        
        # Update the agent values
        self["value"] = results
    
    @staticmethod
    @jit(nopython=True)
    def _calculate_with_numba(values):
        # This function will be compiled by Numba
        result = np.empty_like(values)
        for i in range(len(values)):
            # Complex calculation that benefits from Numba
            result[i] = values[i] ** 2 + np.sin(values[i])
        return result
```

## Advanced Implementation: Vectorized Operations

For even better performance, you can use Numba's vectorization capabilities:

```python
import numpy as np
from numba import vectorize, float64
from mesa_frames import AgentSetPolars, ModelDF

class MyVectorizedAgentSet(AgentSetPolars):
    def __init__(self, model: ModelDF, n_agents: int):
        super().__init__(model)
        # Initialize agents
        self += pl.DataFrame({
            "unique_id": pl.arange(n_agents, eager=True),
            "value": pl.ones(n_agents, eager=True)
        })
    
    def complex_calculation(self):
        # Extract data to numpy arrays
        values = self.agents["value"].to_numpy()
        
        # Call the vectorized function
        results = self._vectorized_calculation(values)
        
        # Update the agent values
        self["value"] = results
    
    @staticmethod
    @vectorize([float64(float64)], nopython=True)
    def _vectorized_calculation(x):
        # This function will be applied to each element
        return x ** 2 + np.sin(x)
```

## GPU Acceleration with CUDA

If you have a compatible NVIDIA GPU, you can use Numba's CUDA capabilities for massive parallelization:

```python
import numpy as np
from numba import cuda
from mesa_frames import AgentSetPolars, ModelDF

class MyCudaAgentSet(AgentSetPolars):
    def __init__(self, model: ModelDF, n_agents: int):
        super().__init__(model)
        # Initialize agents
        self += pl.DataFrame({
            "unique_id": pl.arange(n_agents, eager=True),
            "value": pl.ones(n_agents, eager=True)
        })
    
    def complex_calculation(self):
        # Extract data to numpy arrays
        values = self.agents["value"].to_numpy()
        
        # Prepare output array
        results = np.empty_like(values)
        
        # Call the CUDA kernel
        threads_per_block = 256
        blocks_per_grid = (len(values) + threads_per_block - 1) // threads_per_block
        self._cuda_calculation[blocks_per_grid, threads_per_block](values, results)
        
        # Update the agent values
        self["value"] = results
    
    @staticmethod
    @cuda.jit
    def _cuda_calculation(values, results):
        # Calculate thread index
        i = cuda.grid(1)
        
        # Check array bounds
        if i < values.size:
            # Complex calculation
            results[i] = values[i] ** 2 + math.sin(values[i])
```

## General Usage Pattern with guvectorize

The Sugarscape example in mesa-frames demonstrates a more advanced pattern using `guvectorize`:

```python
class AgentSetWithNumba(AgentSetPolars):
    numba_target = "cpu"  # Can be "cpu", "parallel", or "cuda"
    
    def _get_accelerated_function(self):
        @guvectorize(
            [
                (
                    int32[:],  # input array 1
                    int32[:],  # input array 2
                    # ... more input arrays
                    int32[:],  # output array
                )
            ],
            "(n), (m), ... -> (p)",  # Signature defining array shapes
            nopython=True,
            target=self.numba_target,
        )
        def vectorized_function(
            input1, input2, ..., output
        ):
            # Function implementation
            # This will be compiled for the specified target
            # (CPU, parallel, or CUDA)
            
            # Perform calculations and populate output array
            
        return vectorized_function
```

## Real-World Example: Sugarscape Implementation

The mesa-frames repository includes a complete example of Numba acceleration in the Sugarscape model. 
The implementation includes three variants:

1. **AntPolarsNumbaCPU**: Single-core CPU acceleration
2. **AntPolarsNumbaParallel**: Multi-core CPU acceleration
3. **AntPolarsNumbaGPU**: GPU acceleration using CUDA

You can find this implementation in the `examples/sugarscape_ig/ss_polars/agents.py` file.

## Performance Considerations

When using Numba with mesa-frames, keep the following in mind:

1. **Compilation overhead**: The first call to a Numba function includes compilation time
2. **Data transfer overhead**: Moving data between DataFrame and NumPy arrays has a cost
3. **Function complexity**: Numba benefits most for computationally intensive functions
4. **Best practices**: Follow [Numba's best practices](https://numba.pydata.org/numba-doc/latest/user/performance-tips.html) for maximum performance

## Installation

To use Numba with mesa-frames, install it as an optional dependency:

```bash
pip install mesa-frames[numba]
```

Or if you're installing from source:

```bash
pip install -e ".[numba]"
```

## Conclusion

Numba acceleration provides a powerful way to optimize performance-critical parts of your mesa-frames models. By selectively applying Numba to computationally intensive methods, you can achieve significant performance improvements while maintaining the overall structure and readability of your model code. 