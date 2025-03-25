"""
Unit tests for Numba acceleration in mesa-frames.

This module tests the Numba acceleration functionality in mesa-frames, focusing
on the acceleration functions themselves rather than full model integration.
"""

import numpy as np
import pytest

from examples.numba_example.model import (
    DiffusionAgentNumba,
    DiffusionAgentVectorized,
    DiffusionAgentParallel
)


class Test_NumbaAcceleration:
    """Test suite for Numba acceleration functionality."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up common test parameters."""
        self.seed = 42

    def test_basic_numba_function(self):
        """Test the basic Numba function directly."""
        # Test the basic Numba function
        center_ids = np.array([0, 0, 0, 1, 1, 2])
        neighbor_ids = np.array([0, 1, 2, 0, 1, 2])
        values = np.array([0.5, 0.3, 0.8])
        
        new_values = DiffusionAgentNumba._calculate_new_values(
            center_ids, neighbor_ids, values
        )
        
        # We should get 3 new values (one for each unique center ID)
        assert len(new_values) == 3
        
        # For center ID 0, the neighbors are [0, 1, 2], so mean is (0.5 + 0.3 + 0.8)/3 = 0.53333
        assert abs(new_values[0] - 0.53333) < 0.0001
        
        # For center ID 1, the neighbors are [0, 1], so mean is (0.5 + 0.3)/2 = 0.4
        assert abs(new_values[1] - 0.4) < 0.0001
        
        # For center ID 2, the neighbor is [2], so mean is 0.8
        assert abs(new_values[2] - 0.8) < 0.0001

    def test_vectorized_function(self):
        """Test the vectorized Numba function."""
        # Call the vectorized function directly
        values = np.array([0.5, 0.3, 0.8])
        
        # The function is now a standard Numba jit function
        result = DiffusionAgentVectorized._calculate_mean(values)
        
        # The mean should be (0.5 + 0.3 + 0.8)/3 = 0.53333
        assert abs(result - 0.53333) < 0.0001

    def test_parallel_function(self):
        """Test the parallel Numba function."""
        # Set up test data
        centers = np.array([0, 1, 2])
        neighbors_array = np.array([
            [0, 1, 2],  # neighbors for center 0
            [0, 1, 0],  # neighbors for center 1
            [2, 0, 0]   # neighbors for center 2
        ])
        unique_ids = np.array([0, 1, 2])
        values = np.array([0.5, 0.3, 0.8])
        max_neighbors = 3
        
        # Call the parallel function
        results = DiffusionAgentParallel._parallel_calculate(
            centers, neighbors_array, unique_ids, values, max_neighbors
        )
        
        # Check the results
        # For center 0, the neighbors are [0, 1, 2], so mean is (0.5 + 0.3 + 0.8)/3 = 0.53333
        assert abs(results[0] - 0.53333) < 0.0001
        
        # For center 1, the neighbors are [0, 1], so mean is (0.5 + 0.3)/2 = 0.4
        assert abs(results[1] - 0.4) < 0.0001
        
        # For center 2, the neighbor is [2], so mean is 0.8/1 = 0.8
        assert abs(results[2] - 0.8) < 0.0001 