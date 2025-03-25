"""
A simple example demonstrating Numba acceleration in mesa-frames.

This example compares different implementations of the same model:
1. A standard Polars-based implementation
2. A basic Numba-accelerated implementation
3. A vectorized Numba implementation 
4. A parallel Numba implementation

The model simulates agents on a 2D grid with a simple diffusion process.
"""

import numpy as np
import polars as pl
from numba import jit, vectorize, prange, float64, int64
from mesa_frames import AgentSetPolars, GridPolars, ModelDF


class DiffusionAgentStandard(AgentSetPolars):
    """Standard implementation using Polars without Numba."""
    
    def __init__(self, model, n_agents):
        super().__init__(model)
        # Initialize agents with random values
        self += pl.DataFrame({
            "unique_id": pl.arange(n_agents, eager=True),
            "value": model.random.random(n_agents),
        })
    
    def step(self):
        """Standard implementation using pure Polars operations."""
        # Get neighborhood
        neighborhood = self.space.get_neighbors(agents=self, include_center=True)
        
        # Group by center agent to get all neighbors for each agent
        grouped = neighborhood.group_by("unique_id_center")
        
        # For each agent, calculate new value based on neighbor average
        for group in grouped:
            center_id = group["unique_id_center"][0]
            neighbor_ids = group["unique_id"]
            neighbor_values = self[neighbor_ids, "value"]
            new_value = neighbor_values.mean()
            self[center_id, "value"] = new_value


class DiffusionAgentNumba(AgentSetPolars):
    """Implementation using basic Numba acceleration."""
    
    def __init__(self, model, n_agents):
        super().__init__(model)
        # Initialize agents with random values
        self += pl.DataFrame({
            "unique_id": pl.arange(n_agents, eager=True),
            "value": model.random.random(n_agents),
        })
    
    def step(self):
        """Numba-accelerated implementation."""
        # Get neighborhood
        neighborhood = self.space.get_neighbors(agents=self, include_center=True)
        
        # Extract arrays for Numba processing
        center_ids = neighborhood["unique_id_center"].to_numpy()
        neighbor_ids = neighborhood["unique_id"].to_numpy()
        values = self.agents["value"].to_numpy()
        
        # Use Numba to calculate new values
        new_values = self._calculate_new_values(center_ids, neighbor_ids, values)
        
        # Update agent values
        for i, agent_id in enumerate(self.agents["unique_id"]):
            if i < len(new_values):
                self[agent_id, "value"] = new_values[i]
    
    @staticmethod
    @jit(nopython=True)
    def _calculate_new_values(center_ids, neighbor_ids, values):
        """Numba-accelerated calculation of new values."""
        # Get unique center IDs
        unique_centers = np.unique(center_ids)
        new_values = np.zeros_like(unique_centers, dtype=np.float64)
        
        # For each center, calculate average of neighbors
        for i, center in enumerate(unique_centers):
            # Find indices where center_ids matches this center
            indices = np.where(center_ids == center)[0]
            
            # Get neighbor IDs for this center
            neighbors = neighbor_ids[indices]
            
            # Calculate mean of neighbor values
            neighbor_values = np.array([values[n] for n in neighbors])
            new_values[i] = np.mean(neighbor_values)
            
        return new_values


class DiffusionAgentVectorized(AgentSetPolars):
    """Implementation using vectorized Numba operations."""
    
    def __init__(self, model, n_agents):
        super().__init__(model)
        # Initialize agents with random values
        self += pl.DataFrame({
            "unique_id": pl.arange(n_agents, eager=True),
            "value": model.random.random(n_agents),
        })
    
    def step(self):
        """Implementation using vectorized operations where possible."""
        # Get neighborhood
        neighborhood = self.space.get_neighbors(agents=self, include_center=True)
        
        # Extract data for processing
        unique_ids = self.agents["unique_id"].to_numpy()
        values = self.agents["value"].to_numpy()
        
        # Create a lookup dictionary for values
        value_dict = {uid: val for uid, val in zip(unique_ids, values)}
        
        # Process the neighborhoods
        new_values = np.zeros_like(values)
        
        # Group by center ID and process each group
        for center_id, group in neighborhood.group_by("unique_id_center"):
            neighbor_ids = group["unique_id"].to_numpy()
            neighbor_values = np.array([value_dict[nid] for nid in neighbor_ids])
            
            # Use vectorized functions for calculations (mean in this case)
            idx = np.where(unique_ids == center_id)[0][0]
            new_values[idx] = np.mean(neighbor_values)  # Use NumPy's mean directly
        
        # Update all values at once
        self["value"] = new_values
    
    # The vectorize decorator doesn't work with arrays as input types in this context
    # We'll use a different approach with jit instead
    @staticmethod
    @jit(nopython=True)
    def _calculate_mean(values):
        """Numba-accelerated calculation of mean."""
        return np.mean(values)


class DiffusionAgentParallel(AgentSetPolars):
    """Implementation using parallel Numba operations."""
    
    def __init__(self, model, n_agents):
        super().__init__(model)
        # Initialize agents with random values
        self += pl.DataFrame({
            "unique_id": pl.arange(n_agents, eager=True),
            "value": model.random.random(n_agents),
        })
    
    def step(self):
        """Implementation using parallel processing."""
        # Get neighborhood
        neighborhood = self.space.get_neighbors(agents=self, include_center=True)
        
        # Process in parallel using Numba
        unique_ids = self.agents["unique_id"].to_numpy()
        values = self.agents["value"].to_numpy()
        
        # Create arrays for center IDs and their neighbors
        centers = []
        neighbors_list = []
        
        # Group neighborhoods by center ID
        for center_id, group in neighborhood.group_by("unique_id_center"):
            centers.append(center_id)
            neighbors_list.append(group["unique_id"].to_numpy())
        
        # Convert to arrays for Numba
        centers = np.array(centers)
        max_neighbors = max(len(n) for n in neighbors_list)
        
        # Create 2D array of neighbor IDs with padding
        neighbors_array = np.zeros((len(centers), max_neighbors), dtype=np.int64)
        for i, neighbors in enumerate(neighbors_list):
            neighbors_array[i, :len(neighbors)] = neighbors
        
        # Calculate new values in parallel
        new_values = self._parallel_calculate(centers, neighbors_array, unique_ids, values, max_neighbors)
        
        # Update agent values
        for center_id, new_value in zip(centers, new_values):
            self[center_id, "value"] = new_value
    
    @staticmethod
    @jit(nopython=True, parallel=True)
    def _parallel_calculate(centers, neighbors_array, unique_ids, values, max_neighbors):
        """Parallel calculation of new values using Numba."""
        result = np.zeros_like(centers, dtype=np.float64)
        
        # Process each center in parallel
        for i in prange(len(centers)):
            center = centers[i]
            neighbors = neighbors_array[i]
            
            # Calculate mean of neighbor values
            sum_val = 0.0
            count = 0
            
            for j in range(max_neighbors):
                neighbor = neighbors[j]
                if neighbor == 0 and j > 0:  # Padding value
                    break
                    
                # Find this neighbor's value
                for k in range(len(unique_ids)):
                    if unique_ids[k] == neighbor:
                        sum_val += values[k]
                        count += 1
                        break
            
            result[i] = sum_val / count if count > 0 else 0
            
        return result


class DiffusionModel(ModelDF):
    """Model demonstrating different implementation approaches."""
    
    def __init__(self, width, height, n_agents, agent_class):
        super().__init__()
        self.grid = GridPolars(self, (width, height))
        self.agents += agent_class(self, n_agents)
        self.grid.place_to_empty(self.agents)
        
    def step(self):
        self.agents.do("step")
        
    def run(self, steps):
        for _ in range(steps):
            self.step()


def run_comparison(width, height, n_agents, steps):
    """Run and compare different implementations."""
    import time
    
    results = {}
    
    for name, agent_class in [
        ("Standard", DiffusionAgentStandard),
        ("Basic Numba", DiffusionAgentNumba),
        ("Vectorized", DiffusionAgentVectorized),
        ("Parallel", DiffusionAgentParallel)
    ]:
        # Initialize model
        model = DiffusionModel(width, height, n_agents, agent_class)
        
        # Run with timing
        start = time.time()
        model.run(steps)
        end = time.time()
        
        results[name] = end - start
        print(f"{name}: {results[name]:.4f} seconds")
    
    # Return the results
    return results


if __name__ == "__main__":
    print("Running comparison of Numba acceleration approaches")
    results = run_comparison(
        width=50,
        height=50,
        n_agents=1000,
        steps=10
    )
    
    # Plot results if matplotlib is available
    try:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 6))
        plt.bar(results.keys(), results.values())
        plt.ylabel("Execution time (seconds)")
        plt.title("Comparison of Numba Acceleration Approaches")
        plt.savefig("numba_comparison.png")
        plt.close()
        
        print("Results saved to numba_comparison.png")
    except ImportError:
        print("Matplotlib not available for plotting") 