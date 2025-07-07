from mesa_frames.concrete.datacollector import DataCollector
from mesa_frames import ModelDF, AgentSetPolars, AgentsDF
import pytest
import polars as pl
import beartype
import tempfile
import os


def custom_trigger(model):
    return model._steps % 2 == 0


class ExampleAgentSet1(AgentSetPolars):
    def __init__(self, model: ModelDF):
        super().__init__(model)
        self["wealth"] = pl.Series("wealth", [1, 2, 3, 4])
        self["age"] = pl.Series("age", [10, 20, 30, 40])

    def add_wealth(self, amount: int) -> None:
        self["wealth"] += amount

    def step(self) -> None:
        self.add_wealth(1)

class ExampleAgentSet2(AgentSetPolars):
    def __init__(self, model: ModelDF):
        super().__init__(model)
        self["wealth"] = pl.Series("wealth", [10, 20, 30, 40])
        self["age"] = pl.Series("age", [11, 22, 33, 44])

    def add_wealth(self, amount: int) -> None:
        self["wealth"] += amount

    def step(self) -> None:
        self.add_wealth(2)

class ExampleAgentSet3(AgentSetPolars):
    def __init__(self, model: ModelDF):
        super().__init__(model)
        self["age"] = pl.Series("age", [1, 2, 3, 4])
        self["wealth"] = pl.Series("wealth", [1, 2, 3, 4])

    def age_agents(self, amount: int) -> None:
        self["age"] += amount

    def step(self) -> None:
        self.age_agents(1)



class ExampleModel(ModelDF):
    def __init__(self, agents: AgentsDF):
        super().__init__()
        self.agents = agents

    def step(self):
        self.agents.do("step")

    def run_model(self, n):
        for _ in range(n):
            self.step()

    def run_model(self, n):
        for _ in range(n):
            self.step()

    def run_model_with_collect(self, n):
        for _ in range(n):
            self.step()
            self.dc.collect()

    def run_model_with_conditional_collect(self, n):
        for _ in range(n):
            self.step()
            self.dc.conditional_collect()


@pytest.fixture
def fix1_AgentSetPolars() -> ExampleAgentSet1:
    return ExampleAgentSet1(ModelDF())

@pytest.fixture
def fix2_AgentSetPolars() -> ExampleAgentSet2:
    return ExampleAgentSet2(ModelDF())

@pytest.fixture
def fix3_AgentSetPolars() -> ExampleAgentSet3:
    return ExampleAgentSet3(ModelDF())


@pytest.fixture
def fix_AgentsDF(fix1_AgentSetPolars: ExampleAgentSet1,fix2_AgentSetPolars : ExampleAgentSet2, fix3_AgentSetPolars: ExampleAgentSet3) -> AgentsDF:
    model = ModelDF()
    agents = AgentsDF(model)
    agents.add([fix1_AgentSetPolars,fix2_AgentSetPolars,fix3_AgentSetPolars])
    return agents


@pytest.fixture
def fix1_model(fix_AgentsDF: AgentsDF) -> ExampleModel:
    return ExampleModel(fix_AgentsDF)


class TestDataCollector:
    def test__init__(self, fix1_model):
        model = fix1_model
        with pytest.raises(
            beartype.roar.BeartypeCallHintParamViolation,
            match="not instance of .*Callable",
        ):
            model.test_dc = DataCollector(
                model=model, model_reporters={"total_agents": "sum"}
            )
        with pytest.raises(
            ValueError,
            match="Please define a storage_uri to if to be stored not in memory",
        ):
            model.test_dc = DataCollector(model=model, storage="S3-csv")

    def test_collect(self, fix1_model):
        model = fix1_model
    
        model.dc = DataCollector(
            model=model,
            model_reporters={
                "total_agents": lambda model: sum(
                    len(agentset) for agentset in model.agents._agentsets
                )
            },
            agent_reporters={
                "wealth": lambda agents: agents._agentsets[0]["wealth"],
                "age":"age"
            },
        )


        agent_data_dict = {}
        agent_data_dict["wealth"] = model.agents._agentsets[0]["wealth"]

        model.dc.collect()
        collected_data = model.dc.data

        # test collected_model_data
        assert collected_data["model"]["step"].to_list() == [0]
        assert collected_data["model"]["total_agents"].to_list() == [12]
        with pytest.raises(pl.exceptions.ColumnNotFoundError, match="max_wealth"):
            collected_data["model"]["max_wealth"]

        assert collected_data["agent"]["step"].to_list() == [0, 0, 0, 0]
        assert collected_data["agent"]["wealth"].to_list() == [1, 2, 3, 4]
        with pytest.raises(pl.exceptions.ColumnNotFoundError, match="max_wealth"):
            collected_data["agent"]["max_wealth"]

    def test_collect_step(self, fix1_model):
        model = fix1_model
        model.dc = DataCollector(
            model=model,
            model_reporters={
                "total_agents": lambda model: sum(
                    len(agentset) for agentset in model.agents._agentsets
                )
            },
            agent_reporters={
                "wealth": lambda agents: agents._agentsets[0]["wealth"]
            },
        )
        model.run_model(5)

        model.dc.collect()
        collected_data = model.dc.data

        assert collected_data["model"]["step"].to_list() == [5]
        assert collected_data["model"]["total_agents"].to_list() == [12]

        assert collected_data["agent"]["step"].to_list() == [5, 5, 5, 5]
        assert collected_data["agent"]["wealth"].to_list() == [6, 7, 8, 9]

    def test_conditional_collect(self, fix1_model):
        model = fix1_model
        model.dc = DataCollector(
            model=model,
            trigger=custom_trigger,
            model_reporters={
                "total_agents": lambda model: sum(
                    len(agentset) for agentset in model.agents._agentsets
                )
            },
            agent_reporters={
                "wealth": lambda agents: agents._agentsets[0]["wealth"]
            },
        )

        model.run_model_with_conditional_collect(5)
        collected_data = model.dc.data

        assert collected_data["model"]["step"].to_list() == [2, 4]
        assert collected_data["model"]["total_agents"].to_list() == [12, 12]

        assert collected_data["agent"]["step"].to_list() == [2, 2, 2, 2, 4, 4, 4, 4]
        assert collected_data["agent"]["wealth"].to_list() == [3, 4, 5, 6, 5, 6, 7, 8]

    def test_flush_local_csv(self, fix1_model):
        with tempfile.TemporaryDirectory() as tmpdir:
            model = fix1_model
            model.dc = DataCollector(
                model=model,
                trigger=custom_trigger,
                model_reporters={
                    "total_agents": lambda model: sum(
                        len(agentset) for agentset in model.agents._agentsets
                    )
                },
                agent_reporters={
                    "wealth": lambda agents: agents._agentsets[0]["wealth"]
                },
                storage="csv",
                storage_uri=tmpdir,
            )

            model.run_model_with_conditional_collect(4)
            model.dc.flush()

            collected_data = model.dc.data
            assert collected_data["model"].shape == (0, 0)
            assert collected_data["agent"].shape == (0, 0)

            created_files = os.listdir(tmpdir)
            assert len(created_files) == 4, (
                f"Expected 4 files, found {len(created_files)}: {created_files}"
            )

            model_df = pl.read_csv(
                os.path.join(tmpdir, "model_step2.csv"),
                schema_overrides={"seed": pl.Utf8},
            )
            assert model_df["step"].to_list() == [2]
            assert model_df["total_agents"].to_list() == [12]

            model_df = pl.read_csv(
                os.path.join(tmpdir, "model_step4.csv"),
                schema_overrides={"seed": pl.Utf8},
            )
            assert model_df["step"].to_list() == [4]
            assert model_df["total_agents"].to_list() == [12]

            agent_df = pl.read_csv(
                os.path.join(tmpdir, "agent_step2.csv"),
                schema_overrides={"seed": pl.Utf8},
            )
            assert agent_df["step"].to_list() == [2, 2, 2, 2]
            assert agent_df["wealth"].to_list() == [3, 4, 5, 6]

            agent_df = pl.read_csv(
                os.path.join(tmpdir, "agent_step4.csv"),
                schema_overrides={"seed": pl.Utf8},
            )
            assert agent_df["step"].to_list() == [4, 4, 4, 4]
            assert agent_df["wealth"].to_list() == [5, 6, 7, 8]

    def test_flush_local_parquet(self, fix1_model):
        with tempfile.TemporaryDirectory() as tmpdir:
            model = fix1_model
            model.dc = DataCollector(
                model=model,
                trigger=custom_trigger,
                model_reporters={
                    "total_agents": lambda model: sum(
                        len(agentset) for agentset in model.agents._agentsets
                    )
                },
                agent_reporters={
                    "wealth": lambda agents: agents._agentsets[0]["wealth"]
                },
                storage="parquet",
                storage_uri=tmpdir,
            )

            model.dc.collect()
            model.dc.flush()

            created_files = os.listdir(tmpdir)
            assert len(created_files) == 2, (
                f"Expected 2 files, found {len(created_files)}: {created_files}"
            )

            model_df = pl.read_parquet(os.path.join(tmpdir, "model_step0.parquet"))
            assert model_df["step"].to_list() == [0]
            assert model_df["total_agents"].to_list() == [12]

            agent_df = pl.read_parquet(os.path.join(tmpdir, "agent_step0.parquet"))
            assert agent_df["step"].to_list() == [0, 0, 0, 0]
            assert agent_df["wealth"].to_list() == [1, 2, 3, 4]
