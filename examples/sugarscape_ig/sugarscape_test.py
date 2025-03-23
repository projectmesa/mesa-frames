import mesa
import sys
from ss_mesa.model import SugarscapeMesa
from ss_mesa.agents import AntMesa, Sugar


def test_model_creation():
    """Test that the model can be created properly with Mesa 3.x"""
    print("\n--- Testing Model Creation ---")
    
    # Print Mesa version
    print(f"Using Mesa version: {mesa.__version__}")
    
    # Create the model
    print("Creating Sugarscape model...")
    model = SugarscapeMesa(10, width=10, height=10)
    
    # Count agents with isinstance
    total_agents = len(model.agents)
    ant_count = sum(1 for agent in model.agents if isinstance(agent, AntMesa))
    sugar_count = sum(1 for agent in model.agents if isinstance(agent, Sugar))
    
    print(f"Total agents: {total_agents}")
    print(f"AntMesa agents: {ant_count}")
    print(f"Sugar agents: {sugar_count}")
    
    # Check that we have the expected number of agents
    assert total_agents == (10*10 + 10), "Unexpected total agent count"
    assert ant_count == 10, "Unexpected AntMesa agent count"
    assert sugar_count == 10*10, "Unexpected Sugar agent count"
    
    # Show some of the agents
    print("\nSample of agents in model:")
    for i, agent in enumerate(model.agents):
        if i < 5:
            print(f"Agent {i} - Type: {type(agent).__name__}, ID: {agent.unique_id}")
        else:
            print("...")
            break
    
    print("Model creation test passed!")
    return model


def test_model_step(model):
    """Test that the model can be stepped with Mesa 3.x"""
    print("\n--- Testing Model Step ---")
    
    # Count agents before stepping
    ant_count_before = sum(1 for agent in model.agents if isinstance(agent, AntMesa))
    print(f"AntMesa agents before step: {ant_count_before}")
    
    # Step the model
    print("Running one step...")
    try:
        model.step()
        print("Step completed successfully!")
        
        # Count agents after stepping
        ant_count_after = sum(1 for agent in model.agents if isinstance(agent, AntMesa))
        print(f"AntMesa agents after step: {ant_count_after}")
        
    except Exception as e:
        print(f"Error running step: {e}")
        assert False, f"Model step failed: {e}"
    
    print("Model step test passed!")


def test_simple_model():
    """Test a simplified model with just a few agents to isolate behavior"""
    print("\n--- Testing Simple Model ---")
    
    # Create a simpler model
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
    
    # Create and test
    print("Creating simple model...")
    simple_model = SimpleModel()
    print("Created model")
    
    # Check agents
    print(f"Agents in model: {len(simple_model.agents)}")
    ant_count = sum(1 for agent in simple_model.agents if isinstance(agent, AntMesa))
    sugar_count = sum(1 for agent in simple_model.agents if isinstance(agent, Sugar))
    print(f"AntMesa agents: {ant_count}")
    print(f"Sugar agents: {sugar_count}")
    
    # Try to step sugar agents directly
    print("\nStepping sugar agents directly...")
    try:
        for sugar in simple_model.sugars:
            sugar.step()
        print("Sugar steps successful!")
    except Exception as e:
        print(f"Error stepping sugar: {e}")
        assert False, f"Sugar step failed: {e}"
    
    # Try to step ant agent directly
    print("\nStepping ant agent directly...")
    try:
        simple_model.ant.step()
        print("Ant step successful!")
    except Exception as e:
        print(f"Error stepping ant: {e}")
        assert False, f"Ant step failed: {e}"
    
    # Try to step the model
    print("\nStepping simple model...")
    try:
        simple_model.step()
        print("Model step successful!")
    except Exception as e:
        print(f"Error stepping model: {e}")
        assert False, f"Simple model step failed: {e}"
    
    print("Simple model test passed!")


def run_tests():
    """Run all tests in sequence"""
    try:
        # Test 1: Model Creation
        model = test_model_creation()
        
        # Test 2: Model Step
        test_model_step(model)
        
        # Test 3: Simple Model
        test_simple_model()
        
        print("\n--- All Tests Passed Successfully! ---")
        print(f"The Sugarscape model is working correctly with Mesa {mesa.__version__}")
        return True
    except AssertionError as e:
        print(f"\nTest failed: {e}")
        return False


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1) 