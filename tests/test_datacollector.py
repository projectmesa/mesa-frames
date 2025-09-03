from mesa_frames.concrete.datacollector import DataCollector
from mesa_frames import Model, AgentSet, AgentSetRegistry
import pytest
import polars as pl
import beartype
import tempfile
import os
import time


def custom_trigger(model):
    return model._steps % 2 == 0


class ExampleAgentSet1(AgentSet):
    def __init__(self, model: Model):
        super().__init__(model)
        self["wealth"] = pl.Series("wealth", [1, 2, 3, 4])
        self["age"] = pl.Series("age", [10, 20, 30, 40])

    def add_wealth(self, amount: int) -> None:
        self["wealth"] += amount

    def step(self) -> None:
        self.add_wealth(1)


class ExampleAgentSet2(AgentSet):
    def __init__(self, model: Model):
        super().__init__(model)
        self["wealth"] = pl.Series("wealth", [10, 20, 30, 40])
        self["age"] = pl.Series("age", [11, 22, 33, 44])

    def add_wealth(self, amount: int) -> None:
        self["wealth"] += amount

    def step(self) -> None:
        self.add_wealth(2)


class ExampleAgentSet3(AgentSet):
    def __init__(self, model: Model):
        super().__init__(model)
        self["age"] = pl.Series("age", [1, 2, 3, 4])
        self["wealth"] = pl.Series("wealth", [1, 2, 3, 4])

    def age_agents(self, amount: int) -> None:
        self["age"] += amount

    def step(self) -> None:
        self.age_agents(1)


class ExampleModel(Model):
    def __init__(self, sets: AgentSetRegistry):
        super().__init__()
        self.sets = sets

    def step(self):
        self.sets.do("step")

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


class ExampleModelWithMultipleCollects(Model):
    def __init__(self, agents: AgentSetRegistry):
        super().__init__()
        self.sets = agents

    def step(self):
        self.dc.conditional_collect()
        self.sets.do("step")
        self.dc.conditional_collect()

    def run_model_with_conditional_collect_multiple_batch(self, n):
        for _ in range(n):
            self.step()


@pytest.fixture(scope="session")
def postgres_uri():
    return os.getenv("POSTGRES_URI", "postgresql://user:password@localhost:5432/testdb")


@pytest.fixture
def fix1_AgentSet() -> ExampleAgentSet1:
    return ExampleAgentSet1(Model())


@pytest.fixture
def fix2_AgentSet() -> ExampleAgentSet2:
    return ExampleAgentSet2(Model())


@pytest.fixture
def fix3_AgentSet() -> ExampleAgentSet3:
    return ExampleAgentSet3(Model())


@pytest.fixture
def fix_AgentSetRegistry(
    fix1_AgentSet: ExampleAgentSet1,
    fix2_AgentSet: ExampleAgentSet2,
    fix3_AgentSet: ExampleAgentSet3,
) -> AgentSetRegistry:
    model = Model()
    agents = AgentSetRegistry(model)
    agents.add([fix1_AgentSet, fix2_AgentSet, fix3_AgentSet])
    return agents


@pytest.fixture
def fix1_model(fix_AgentSetRegistry: AgentSetRegistry) -> ExampleModel:
    return ExampleModel(fix_AgentSetRegistry)


@pytest.fixture
def fix2_model(fix_AgentSetRegistry: AgentSetRegistry) -> ExampleModel:
    return ExampleModelWithMultipleCollects(fix_AgentSetRegistry)


class TestDataCollector:
    def test__init__(self, fix1_model, postgres_uri):
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

        with pytest.raises(
            ValueError,
            match="Please define a storage_uri to if to be stored not in memory",
        ):
            model.test_dc = DataCollector(model=model, storage="postgresql")

    def test_collect(self, fix1_model):
        model = fix1_model

        model.dc = DataCollector(
            model=model,
            model_reporters={
                "total_agents": lambda model: sum(
                    len(agentset) for agentset in model.sets._agentsets
                )
            },
            agent_reporters={
                "wealth": lambda model: model.sets._agentsets[0]["wealth"],
                "age": "age",
            },
        )

        model.dc.collect()
        collected_data = model.dc.data

        # test collected_model_data
        assert collected_data["model"].shape == (1, 4)
        assert set(collected_data["model"].columns) == {
            "step",
            "seed",
            "batch",
            "total_agents",
        }
        assert collected_data["model"]["step"].to_list() == [0]
        assert collected_data["model"]["total_agents"].to_list() == [12]
        with pytest.raises(pl.exceptions.ColumnNotFoundError, match="max_wealth"):
            collected_data["model"]["max_wealth"]

        assert collected_data["agent"].shape == (4, 7)
        assert set(collected_data["agent"].columns) == {
            "wealth",
            "age_ExampleAgentSet1",
            "age_ExampleAgentSet2",
            "age_ExampleAgentSet3",
            "step",
            "seed",
            "batch",
        }
        assert collected_data["agent"]["wealth"].to_list() == [1, 2, 3, 4]
        assert collected_data["agent"]["age_ExampleAgentSet1"].to_list() == [
            10,
            20,
            30,
            40,
        ]
        assert collected_data["agent"]["age_ExampleAgentSet2"].to_list() == [
            11,
            22,
            33,
            44,
        ]
        assert collected_data["agent"]["age_ExampleAgentSet3"].to_list() == [1, 2, 3, 4]
        assert collected_data["agent"]["step"].to_list() == [0, 0, 0, 0]
        with pytest.raises(pl.exceptions.ColumnNotFoundError, match="max_wealth"):
            collected_data["agent"]["max_wealth"]

    def test_collect_step(self, fix1_model):
        model = fix1_model
        model.dc = DataCollector(
            model=model,
            model_reporters={
                "total_agents": lambda model: sum(
                    len(agentset) for agentset in model.sets._agentsets
                )
            },
            agent_reporters={
                "wealth": lambda model: model.sets._agentsets[0]["wealth"],
                "age": "age",
            },
        )
        model.run_model(5)

        model.dc.collect()
        collected_data = model.dc.data

        assert collected_data["model"].shape == (1, 4)
        assert set(collected_data["model"].columns) == {
            "step",
            "seed",
            "batch",
            "total_agents",
        }
        assert collected_data["model"]["step"].to_list() == [5]
        assert collected_data["model"]["total_agents"].to_list() == [12]

        assert collected_data["agent"].shape == (4, 7)
        assert set(collected_data["agent"].columns) == {
            "wealth",
            "age_ExampleAgentSet1",
            "age_ExampleAgentSet2",
            "age_ExampleAgentSet3",
            "step",
            "seed",
            "batch",
        }
        assert collected_data["agent"]["wealth"].to_list() == [6, 7, 8, 9]
        assert collected_data["agent"]["age_ExampleAgentSet1"].to_list() == [
            10,
            20,
            30,
            40,
        ]
        assert collected_data["agent"]["age_ExampleAgentSet2"].to_list() == [
            11,
            22,
            33,
            44,
        ]
        assert collected_data["agent"]["age_ExampleAgentSet3"].to_list() == [6, 7, 8, 9]
        assert collected_data["agent"]["step"].to_list() == [5, 5, 5, 5]

    def test_conditional_collect(self, fix1_model):
        model = fix1_model
        model.dc = DataCollector(
            model=model,
            trigger=custom_trigger,
            model_reporters={
                "total_agents": lambda model: sum(
                    len(agentset) for agentset in model.sets._agentsets
                )
            },
            agent_reporters={
                "wealth": lambda model: model.sets._agentsets[0]["wealth"],
                "age": "age",
            },
        )

        model.run_model_with_conditional_collect(5)
        collected_data = model.dc.data

        assert collected_data["model"].shape == (2, 4)
        assert set(collected_data["model"].columns) == {
            "step",
            "seed",
            "batch",
            "total_agents",
        }
        assert collected_data["model"]["step"].to_list() == [2, 4]
        assert collected_data["model"]["total_agents"].to_list() == [12, 12]

        assert collected_data["agent"].shape == (8, 7)
        assert set(collected_data["agent"].columns) == {
            "wealth",
            "age_ExampleAgentSet1",
            "age_ExampleAgentSet2",
            "age_ExampleAgentSet3",
            "step",
            "seed",
            "batch",
        }
        assert set(collected_data["agent"].columns) == {
            "wealth",
            "age_ExampleAgentSet1",
            "age_ExampleAgentSet2",
            "age_ExampleAgentSet3",
            "step",
            "seed",
            "batch",
        }
        assert collected_data["agent"]["wealth"].to_list() == [3, 4, 5, 6, 5, 6, 7, 8]
        assert collected_data["agent"]["age_ExampleAgentSet1"].to_list() == [
            10,
            20,
            30,
            40,
            10,
            20,
            30,
            40,
        ]
        assert collected_data["agent"]["age_ExampleAgentSet2"].to_list() == [
            11,
            22,
            33,
            44,
            11,
            22,
            33,
            44,
        ]
        assert collected_data["agent"]["age_ExampleAgentSet3"].to_list() == [
            3,
            4,
            5,
            6,
            5,
            6,
            7,
            8,
        ]
        assert collected_data["agent"]["step"].to_list() == [2, 2, 2, 2, 4, 4, 4, 4]

    def test_flush_local_csv(self, fix1_model):
        with tempfile.TemporaryDirectory() as tmpdir:
            model = fix1_model
            model.dc = DataCollector(
                model=model,
                trigger=custom_trigger,
                model_reporters={
                    "total_agents": lambda model: sum(
                        len(agentset) for agentset in model.sets._agentsets
                    )
                },
                agent_reporters={
                    "wealth": lambda model: model.sets._agentsets[0]["wealth"],
                    "age": "age",
                },
                storage="csv",
                storage_uri=tmpdir,
            )

            model.run_model_with_conditional_collect(4)
            model.dc.flush()
            for _ in range(20):  # wait up to ~2 seconds
                created_files = os.listdir(tmpdir)
                if len(created_files) >= 4:
                    break
                time.sleep(0.1)

            # check deletion after flush
            collected_data = model.dc.data
            assert collected_data["model"].shape == (0, 0)
            assert collected_data["agent"].shape == (0, 0)

            created_files = os.listdir(tmpdir)
            assert len(created_files) == 4, (
                f"Expected 4 files, found {len(created_files)}: {created_files}"
            )

            model_df = pl.read_csv(
                os.path.join(tmpdir, "model_step2_batch0.csv"),
                schema_overrides={"seed": pl.Utf8},
            )
            assert set(model_df.columns) == {"step", "seed", "batch", "total_agents"}
            assert model_df["step"].to_list() == [2]
            assert model_df["total_agents"].to_list() == [12]

            agent_df = pl.read_csv(
                os.path.join(tmpdir, "agent_step2_batch0.csv"),
                schema_overrides={"seed": pl.Utf8},
            )
            assert set(agent_df.columns) == {
                "wealth",
                "age_ExampleAgentSet1",
                "age_ExampleAgentSet2",
                "age_ExampleAgentSet3",
                "step",
                "seed",
                "batch",
            }
            assert agent_df["step"].to_list() == [2, 2, 2, 2]
            assert agent_df["wealth"].to_list() == [3, 4, 5, 6]
            assert agent_df["age_ExampleAgentSet1"].to_list() == [10, 20, 30, 40]
            assert agent_df["age_ExampleAgentSet2"].to_list() == [11, 22, 33, 44]
            assert agent_df["age_ExampleAgentSet3"].to_list() == [
                3,
                4,
                5,
                6,
            ]

            agent_df = pl.read_csv(
                os.path.join(tmpdir, "agent_step4_batch0.csv"),
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
                        len(agentset) for agentset in model.sets._agentsets
                    )
                },
                agent_reporters={
                    "wealth": lambda model: model.sets._agentsets[0]["wealth"],
                },
                storage="parquet",
                storage_uri=tmpdir,
            )

            model.dc.collect()
            model.dc.flush()

            for _ in range(20):  # wait up to ~2 seconds
                created_files = os.listdir(tmpdir)
                if len(created_files) >= 4:
                    break
                time.sleep(0.1)

            created_files = os.listdir(tmpdir)
            assert len(created_files) == 2, (
                f"Expected 2 files, found {len(created_files)}: {created_files}"
            )

            model_df = pl.read_parquet(
                os.path.join(tmpdir, "model_step0_batch0.parquet")
            )
            assert model_df["step"].to_list() == [0]
            assert model_df["total_agents"].to_list() == [12]

            agent_df = pl.read_parquet(
                os.path.join(tmpdir, "agent_step0_batch0.parquet")
            )
            assert agent_df["step"].to_list() == [0, 0, 0, 0]
            assert agent_df["wealth"].to_list() == [1, 2, 3, 4]

    @pytest.mark.skipif(
        os.getenv("SKIP_PG_TESTS") == "true",
        reason="PostgreSQL tests are skipped on Windows runners",
    )
    def test_postgress(self, fix1_model, postgres_uri):
        model = fix1_model

        # Connect directly and validate data
        import psycopg2

        conn = psycopg2.connect(postgres_uri)
        cur = conn.cursor()

        cur.execute("""
            CREATE TABLE public.model_data (
                step INTEGER,
                seed VARCHAR,
                batch INTEGER,
                total_agents INTEGER
            )
        """)

        cur.execute("""
            CREATE TABLE public.agent_data (
                step INTEGER,
                seed VARCHAR,
                batch INTEGER,
                age_ExampleAgentSet1 INTEGER,
                age_ExampleAgentSet2 INTEGER,
                age_ExampleAgentSet3 INTEGER,
                wealth INTEGER
            )
        """)
        conn.commit()

        model.dc = DataCollector(
            model=model,
            trigger=custom_trigger,
            model_reporters={
                "total_agents": lambda model: sum(
                    len(agentset) for agentset in model.sets._agentsets
                )
            },
            agent_reporters={
                "wealth": lambda model: model.sets._agentsets[0]["wealth"],
                "age": "age",
            },
            storage="postgresql",
            schema="public",
            storage_uri=postgres_uri,
        )

        model.run_model_with_conditional_collect(4)
        model.dc.flush()

        # Connect directly and validate data
        for _ in range(20):
            cur.execute("SELECT COUNT(*) FROM model_data")
            (count,) = cur.fetchone()
            if count >= 2:  # expecting 2 rows
                break
            time.sleep(0.1)

        # Check model data
        cur.execute("SELECT step, total_agents FROM model_data ORDER BY step")
        model_rows = cur.fetchall()
        assert model_rows == [(2, 12), (4, 12)]

        cur.execute(
            "SELECT step, batch, wealth,age_ExampleAgentSet1, age_ExampleAgentSet2, age_ExampleAgentSet3 FROM agent_data WHERE step=2 ORDER BY wealth"
        )
        agent_rows = cur.fetchall()
        assert agent_rows == [
            (2, 0, 3, 10, 11, 3),
            (2, 0, 4, 20, 22, 4),
            (2, 0, 5, 30, 33, 5),
            (2, 0, 6, 40, 44, 6),
        ]

        cur.close()
        conn.close()

    def test_batch_memory(self, fix2_model):
        model = fix2_model
        model.dc = DataCollector(
            model=model,
            trigger=custom_trigger,
            model_reporters={
                "total_agents": lambda model: sum(
                    len(agentset) for agentset in model.sets._agentsets
                )
            },
            agent_reporters={
                "wealth": lambda model: model.sets._agentsets[0]["wealth"],
                "age": "age",
            },
        )

        model.run_model_with_conditional_collect_multiple_batch(5)
        collected_data = model.dc.data
        assert collected_data["model"].shape == (4, 4)
        assert set(collected_data["model"].columns) == {
            "step",
            "seed",
            "batch",
            "total_agents",
        }
        assert collected_data["model"]["step"].to_list() == [2, 2, 4, 4]
        assert collected_data["model"]["batch"].to_list() == [0, 1, 0, 1]
        assert collected_data["model"]["total_agents"].to_list() == [12, 12, 12, 12]

        assert collected_data["agent"].shape == (16, 7)
        assert set(collected_data["agent"].columns) == {
            "wealth",
            "age_ExampleAgentSet1",
            "age_ExampleAgentSet2",
            "age_ExampleAgentSet3",
            "step",
            "seed",
            "batch",
        }

        assert set(collected_data["agent"].columns) == {
            "wealth",
            "age_ExampleAgentSet1",
            "age_ExampleAgentSet2",
            "age_ExampleAgentSet3",
            "step",
            "seed",
            "batch",
        }

        assert collected_data["agent"]["step"].to_list() == [
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            4,
            4,
            4,
            4,
            4,
            4,
            4,
            4,
        ]
        assert collected_data["agent"]["wealth"].to_list() == [
            2,
            3,
            4,
            5,
            3,
            4,
            5,
            6,
            4,
            5,
            6,
            7,
            5,
            6,
            7,
            8,
        ]
        assert collected_data["agent"]["age_ExampleAgentSet1"].to_list() == [
            10,
            20,
            30,
            40,
            10,
            20,
            30,
            40,
            10,
            20,
            30,
            40,
            10,
            20,
            30,
            40,
        ]
        assert collected_data["agent"]["age_ExampleAgentSet2"].to_list() == [
            11,
            22,
            33,
            44,
            11,
            22,
            33,
            44,
            11,
            22,
            33,
            44,
            11,
            22,
            33,
            44,
        ]
        assert collected_data["agent"]["age_ExampleAgentSet3"].to_list() == [
            2,
            3,
            4,
            5,
            3,
            4,
            5,
            6,
            4,
            5,
            6,
            7,
            5,
            6,
            7,
            8,
        ]

        with pytest.raises(pl.exceptions.ColumnNotFoundError, match="max_wealth"):
            collected_data["agent"]["max_wealth"]

    def test_batch_save(self, fix2_model):
        with tempfile.TemporaryDirectory() as tmpdir:
            model = fix2_model
            model.dc = DataCollector(
                model=model,
                trigger=custom_trigger,
                model_reporters={
                    "total_agents": lambda model: sum(
                        len(agentset) for agentset in model.sets._agentsets
                    )
                },
                agent_reporters={
                    "wealth": lambda model: model.sets._agentsets[0]["wealth"],
                    "age": "age",
                },
                storage="csv",
                storage_uri=tmpdir,
            )

            model.run_model_with_conditional_collect_multiple_batch(5)
            model.dc.flush()
            for _ in range(20):  # wait up to ~2 seconds
                created_files = os.listdir(tmpdir)
                if len(created_files) >= 4:
                    break
                time.sleep(0.1)

            # check deletion after flush
            collected_data = model.dc.data
            assert collected_data["model"].shape == (0, 0)
            assert collected_data["agent"].shape == (0, 0)

            created_files = os.listdir(tmpdir)
            print(created_files)
            assert len(created_files) == 8, (
                f"Expected 4 files, found {len(created_files)}: {created_files}"
            )

            # test model batch reset
            model_df_step2_batch0 = pl.read_csv(
                os.path.join(tmpdir, "model_step2_batch0.csv"),
                schema_overrides={"seed": pl.Utf8},
            )
            assert set(model_df_step2_batch0.columns) == {
                "step",
                "seed",
                "batch",
                "total_agents",
            }
            assert model_df_step2_batch0["step"].to_list() == [2]
            assert model_df_step2_batch0["total_agents"].to_list() == [12]

            model_df_step2_batch0 = pl.read_csv(
                os.path.join(tmpdir, "model_step2_batch1.csv"),
                schema_overrides={"seed": pl.Utf8},
            )
            assert set(model_df_step2_batch0.columns) == {
                "step",
                "seed",
                "batch",
                "total_agents",
            }
            assert model_df_step2_batch0["step"].to_list() == [2]
            assert model_df_step2_batch0["total_agents"].to_list() == [12]

            model_df_step4_batch0 = pl.read_csv(
                os.path.join(tmpdir, "model_step4_batch0.csv"),
                schema_overrides={"seed": pl.Utf8},
            )
            assert set(model_df_step4_batch0.columns) == {
                "step",
                "seed",
                "batch",
                "total_agents",
            }
            assert model_df_step4_batch0["step"].to_list() == [4]
            assert model_df_step4_batch0["total_agents"].to_list() == [12]

            # test agent batch reset
            agent_df_step2_batch0 = pl.read_csv(
                os.path.join(tmpdir, "agent_step2_batch0.csv"),
                schema_overrides={"seed": pl.Utf8},
            )
            assert set(agent_df_step2_batch0.columns) == {
                "wealth",
                "age_ExampleAgentSet1",
                "age_ExampleAgentSet2",
                "age_ExampleAgentSet3",
                "step",
                "seed",
                "batch",
            }
            assert agent_df_step2_batch0["step"].to_list() == [2, 2, 2, 2]
            assert agent_df_step2_batch0["wealth"].to_list() == [2, 3, 4, 5]
            assert agent_df_step2_batch0["age_ExampleAgentSet1"].to_list() == [
                10,
                20,
                30,
                40,
            ]
            assert agent_df_step2_batch0["age_ExampleAgentSet2"].to_list() == [
                11,
                22,
                33,
                44,
            ]
            assert agent_df_step2_batch0["age_ExampleAgentSet3"].to_list() == [
                2,
                3,
                4,
                5,
            ]

            agent_df_step2_batch1 = pl.read_csv(
                os.path.join(tmpdir, "agent_step2_batch1.csv"),
                schema_overrides={"seed": pl.Utf8},
            )
            assert set(agent_df_step2_batch1.columns) == {
                "wealth",
                "age_ExampleAgentSet1",
                "age_ExampleAgentSet2",
                "age_ExampleAgentSet3",
                "step",
                "seed",
                "batch",
            }
            assert agent_df_step2_batch1["step"].to_list() == [2, 2, 2, 2]
            assert agent_df_step2_batch1["wealth"].to_list() == [3, 4, 5, 6]
            assert agent_df_step2_batch1["age_ExampleAgentSet1"].to_list() == [
                10,
                20,
                30,
                40,
            ]
            assert agent_df_step2_batch1["age_ExampleAgentSet2"].to_list() == [
                11,
                22,
                33,
                44,
            ]
            assert agent_df_step2_batch1["age_ExampleAgentSet3"].to_list() == [
                3,
                4,
                5,
                6,
            ]

            agent_df_step4_batch0 = pl.read_csv(
                os.path.join(tmpdir, "agent_step4_batch0.csv"),
                schema_overrides={"seed": pl.Utf8},
            )
            assert set(agent_df_step4_batch0.columns) == {
                "wealth",
                "age_ExampleAgentSet1",
                "age_ExampleAgentSet2",
                "age_ExampleAgentSet3",
                "step",
                "seed",
                "batch",
            }
            assert agent_df_step4_batch0["step"].to_list() == [4, 4, 4, 4]
            assert agent_df_step4_batch0["wealth"].to_list() == [4, 5, 6, 7]
            assert agent_df_step4_batch0["age_ExampleAgentSet1"].to_list() == [
                10,
                20,
                30,
                40,
            ]
            assert agent_df_step4_batch0["age_ExampleAgentSet2"].to_list() == [
                11,
                22,
                33,
                44,
            ]
            assert agent_df_step4_batch0["age_ExampleAgentSet3"].to_list() == [
                4,
                5,
                6,
                7,
            ]
