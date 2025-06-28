from mesa_frames.concrete.datacollector import DataCollector
from mesa_frames import ModelDF, AgentSetPolars, AgentsDF
import pytest
import polars as pl

def custom_trigger(model):
    return True


class ExampleAgentSetPolars(AgentSetPolars):
    def __init__(self, model: ModelDF):
        super().__init__(model)
        self["wealth"] = pl.Series("wealth", [1, 2, 3, 4])
        self["age"] = pl.Series("age", [10, 20, 30, 40])

    def add_wealth(self, amount: int) -> None:
        self["wealth"] += amount

    def step(self) -> None:
        self.add_wealth(1)


class ExampleModel(ModelDF):
    def __init__(self, agents: AgentsDF):
        super().__init__()
        self.agents = agents
        self.dc = DataCollector(
            model=self,
            trigger=custom_trigger,
            model_reporters={
                "total_agents": lambda model: sum(
                    len(agentset) for agentset in model.agents._agentsets
                )
            },
            # agent_reporters=  {
            #     "wealth":"wealth"
            # }
        )

    def step(self):
        self.agents.do("step")

    def run_model(self, n):
        for _ in range(n):
            self.step()


@pytest.fixture
def fix1_AgentSetPolars() -> ExampleAgentSetPolars:
    return ExampleAgentSetPolars(ModelDF())


@pytest.fixture
def fix_AgentsDF(fix1_AgentSetPolars: ExampleAgentSetPolars) -> AgentsDF:
    model = ModelDF()
    agents = AgentsDF(model)
    agents.add([fix1_AgentSetPolars])
    return agents


@pytest.fixture
def fix1_model(fix_AgentsDF: AgentsDF) -> ExampleModel:
    return ExampleModel(fix_AgentsDF)


class Test_DataCollector:
    def test__init__(self,fix1_model):
        model = fix1_model
        with pytest.raises(beartype.roar.BeartypeCallHintParamViolation, match="not instance of .*Callable"):
            model.test_dc = DataCollector(
                model=model,
                trigger=custom_trigger,
                model_reporters={
                    "total_agents": "sum"
                }
            )
        with pytest.raises(beartype.roar.BeartypeCallHintParamViolation, match="not instance of .*Callable"):
            model.test_dc = DataCollector(
                model=model,
                trigger=custom_trigger,
                model_reporters={
                    "total_agents": "sum"
                }
            )
        with pytest.raises(beartype.roar.BeartypeCallHintParamViolation, match="not instance of .*Callable"):
            model.test_dc = DataCollector(
                model=model,
                trigger=custom_trigger,
                model_reporters={
                    "total_agents": "sum"
                }
            )
    def test_collect(self, fix1_model):
        model = fix1_model

        agent_data_dict = {}
        agent_data_dict["wealth"] = model._agents._agentsets[0]["wealth"]
        
        agent_lazy_frame = pl.LazyFrame(agent_data_dict)

        model.dc.collect()
        collected_data = model.dc.data

        # test collected_model_data
        print(collected_data)
        assert collected_data["model"]["step"].to_list() == [0]
        assert collected_data["model"]["total_agents"].to_list() == [4]
        with pytest.raises(pl.exceptions.ColumnNotFoundError, match="max_wealth"):
            collected_data["model"]["max_wealth"]

        # fails agent level due to agentsdf

        # test collected_agent_data

        # assert collected_data["agent"]["step"].to_list() == [0,0,0,0]
        # assert collected_data["agent"]["wealth"].to_list() == [1,2,3,4]
        # with pytest.raises(
        #     pl.exceptions.ColumnNotFoundError, match='max_wealth'
        # ):
        #      collected_data["agent"]["max_wealth"]

    # def test_collect_step(self, fix1_model):
    #     # base check
    #     model = fix1_model
    #     model.run_model(5)

    #     model.dc.collect()
    #     collected_data = model.dc.data

    #     # test collected_model_data
    #     print(collected_data)
    #     assert collected_data["model"]["step"].to_list() == [5]
    #     assert collected_data["model"]["total_agents"].to_list() == [4]
    #     with pytest.raises(pl.exceptions.ColumnNotFoundError, match="max_wealth"):
    #         collected_data["model"]["max_wealth"]

    #     # fails agent level due to agentsdf
