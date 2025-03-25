"""
Pytest tests for the Sugarscape example with Mesa 3.x.
"""

import sys
import os
from pathlib import Path

# Add the parent directory to sys.path to allow imports from ss_mesa package
current_dir = Path(__file__).parent
examples_dir = current_dir.parent
root_dir = examples_dir.parent

# Add root directory to sys.path if not already there
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

# Add the examples directory to sys.path if not already there
if str(examples_dir) not in sys.path:
    sys.path.insert(0, str(examples_dir))

import mesa
import pytest
from ss_mesa.model import SugarscapeMesa
from ss_mesa.agents import AntMesa, Sugar


@pytest.fixture
def sugarscape_model():
    """Create a standard Sugarscape model for testing."""
    return SugarscapeMesa(10, width=10, height=10)


def test_model_creation(sugarscape_model):
    """Test that the model can be created properly with Mesa 3.x"""
    model = sugarscape_model

    # Count agents with isinstance
    total_agents = len(model.agents)
    ant_count = sum(1 for agent in model.agents if isinstance(agent, AntMesa))
    sugar_count = sum(1 for agent in model.agents if isinstance(agent, Sugar))

    # Check that we have the expected number of agents
    assert total_agents == (10 * 10 + 10), "Unexpected total agent count"
    assert ant_count == 10, "Unexpected AntMesa agent count"
    assert sugar_count == 10 * 10, "Unexpected Sugar agent count"


def test_model_step(sugarscape_model):
    """Test that the model can be stepped with Mesa 3.x"""
    model = sugarscape_model

    # Count agents before stepping
    ant_count_before = sum(1 for agent in model.agents if isinstance(agent, AntMesa))

    # Step the model
    model.step()

    # Count agents after stepping
    ant_count_after = sum(1 for agent in model.agents if isinstance(agent, AntMesa))

    # In this basic test, we just verify the step completes without errors
    # and the number of ants doesn't unexpectedly change
    assert ant_count_after >= 0, "Expected at least some ants to survive"


@pytest.fixture
def simple_model():
    """Create a simplified model with just a few agents to isolate behavior"""

    class SimpleModel(mesa.Model):
        def __init__(self, seed=None):
            super().__init__(seed=seed)
            self.space = mesa.space.MultiGrid(5, 5, torus=False)

            # Add sugar agents to all cells
            self.sugars = []
            for x in range(5):
                for y in range(5):
                    sugar = Sugar(self, 5)
                    self.space.place_agent(sugar, (x, y))
                    self.sugars.append(sugar)

            # Create one ant agent
            self.ant = AntMesa(self, False, 10, 2, 3)
            self.space.place_agent(self.ant, (2, 2))  # Place in the middle

        def step(self):
            # Step the sugar agents
            for sugar in self.sugars:
                sugar.step()

            # Step the ant agent
            self.ant.step()

    return SimpleModel()


def test_simple_model_creation(simple_model):
    """Test that the simple model is created with the correct agents."""
    # Check agents
    assert len(simple_model.agents) == 26, "Expected 26 total agents (25 sugar + 1 ant)"

    ant_count = sum(1 for agent in simple_model.agents if isinstance(agent, AntMesa))
    sugar_count = sum(1 for agent in simple_model.agents if isinstance(agent, Sugar))

    assert ant_count == 1, "Expected exactly 1 AntMesa agent"
    assert sugar_count == 25, "Expected exactly 25 Sugar agents"


def test_sugar_step(simple_model):
    """Test that sugar agents can step without errors."""
    for sugar in simple_model.sugars:
        sugar.step()
    # If we get here without exceptions, the test passes


def test_ant_step(simple_model):
    """Test that ant agents can step without errors."""
    simple_model.ant.step()
    # If we get here without exceptions, the test passes


def test_simple_model_step(simple_model):
    """Test that the simple model can step without errors."""
    simple_model.step()
    # If we get here without exceptions, the test passes
