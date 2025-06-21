from mesa_frames.concrete.datacollector import DataCollector
from mesa_frames import ModelDF,AgentSetPolars
import pytest
import polars as pl


def custom_trigger(model):
    return True


class ExampleAgentSetPolars(AgentSetPolars):
    def __init__(self, model: ModelDF):
        super().__init__(model)
        self.starting_wealth = pl.Series("wealth", [1, 2, 3, 4])
        self.dc = DataCollector(
            model=model,
            trigger=custom_trigger,
            model_reporters={
                "total_agents": lambda model: len(model._agents._agentsets[0])
            },
            agent_reporters={
                "wealth": lambda model: model._agents._agentsets[0]["wealth"]
            }
        )

    def add_wealth(self, amount: int) -> None:
        self["wealth"] += amount

    def step(self) -> None:
        self.add_wealth(1)

@pytest.fixture
def fix1_AgentSetPolars() -> ExampleAgentSetPolars:
    model = ModelDF()
    agents = ExampleAgentSetPolars(model)
    agents["wealth"] = agents.starting_wealth
    agents["age"] = [10, 20, 30, 40]
    model.agents.add(agents)
    return agents
@pytest.fixture
def fix3_AgentSetPolars() -> ExampleAgentSetPolars:
    model = ModelDF()
    agents = ExampleAgentSetPolars(model)
    agents["wealth"] = agents.starting_wealth + 7
    agents["age"] = [12, 13, 14, 116]
    return agents

class Test_DataCollector:
    def test_model(self,fix1_AgentSetPolars):
        agents = fix1_AgentSetPolars
        assert agents["wealth"] == [1,2,3,4]
        assert agents["wealth"] == [1,5,3,4]
        