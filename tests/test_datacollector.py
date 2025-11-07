from mesa_frames.concrete.datacollector import DataCollector
from mesa_frames import Model, AgentSet, AgentSetRegistry
import pytest
import polars as pl
import beartype
import tempfile
import os
import time


def custom_trigger(model):
    return model.steps % 2 == 0


class ExampleAgentSet1(AgentSet):
    def __init__(self, model: Model):
        super().__init__(model)
        self._df = pl.DataFrame(
            {
                "unique_id": [101, 102, 103, 104],
                "wealth": [1, 2, 3, 4],
                "age": [10, 20, 30, 40],
            }
        )

    def add_wealth(self, amount: int) -> None:
        self.set("wealth", self["wealth"] + amount)

    def step(self) -> None:
        self.add_wealth(1)


class ExampleAgentSet2(AgentSet):
    def __init__(self, model: Model):
        super().__init__(model)
        self._df = pl.DataFrame(
            {
                "unique_id": [201, 202, 203, 204],
                "wealth": [10, 20, 30, 40],
                "age": [11, 22, 33, 44],
            }
        )

    def add_wealth(self, amount: int) -> None:
        self.set("wealth", self["wealth"] + amount)

    def step(self) -> None:
        self.add_wealth(2)


class ExampleAgentSet3(AgentSet):
    def __init__(self, model: Model):
        super().__init__(model)
        self._df = pl.DataFrame(
            {
                "unique_id": [301, 302, 303, 304],
                "age": [1, 2, 3, 4],
                "wealth": [1, 2, 3, 4],
            }
        )

    def age_agents(self, amount: int) -> None:
        self.set("age", self["age"] + amount)

    def step(self) -> None:
        self.age_agents(1)


class ExampleModel(Model):
    def __init__(self, sets: AgentSetRegistry):
        super().__init__()
        self.sets = sets
        self._steps = 0

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
        self._seed = 0

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
    return ExampleAgentSet1(Model(seed=1))


@pytest.fixture
def fix2_AgentSet() -> ExampleAgentSet2:
    return ExampleAgentSet2(Model(seed=1))


@pytest.fixture
def fix3_AgentSet() -> ExampleAgentSet3:
    return ExampleAgentSet3(Model(seed=1))


@pytest.fixture
def fix_AgentSetRegistry(
    fix1_AgentSet: ExampleAgentSet1,
    fix2_AgentSet: ExampleAgentSet2,
    fix3_AgentSet: ExampleAgentSet3,
) -> AgentSetRegistry:
    model = Model()
    agents = AgentSetRegistry(model)
    agents._agentsets = [fix1_AgentSet, fix2_AgentSet, fix3_AgentSet]
    # Manually update model link for agent sets
    fix1_AgentSet._model = model
    fix2_AgentSet._model = model
    fix3_AgentSet._model = model
    return agents


@pytest.fixture
def fix1_model(fix_AgentSetRegistry: AgentSetRegistry) -> ExampleModel:
    model = ExampleModel(fix_AgentSetRegistry)
    fix_AgentSetRegistry._model = model
    for s in fix_AgentSetRegistry._agentsets:
        s._model = model
    return model


@pytest.fixture
def fix2_model(fix_AgentSetRegistry: AgentSetRegistry) -> ExampleModel:
    model = ExampleModelWithMultipleCollects(fix_AgentSetRegistry)
    fix_AgentSetRegistry._model = model
    for s in fix_AgentSetRegistry._agentsets:
        s._model = model
    return model


class TestDataCollector:
    def test__init__(self, fix1_model, postgres_uri):
        model = fix1_model
        # with pytest.raises(
        #     beartype.roar.BeartypeCallHintParamViolation,
        #     match="not instance of .*Callable",
        # ):
        #     model.test_dc = DataCollector(
        #         model=model, model_reporters={"total_agents": "sum"}
        #     )

        model.test_dc = DataCollector(
            model=model, agent_reporters={"wealth": lambda m: 1}
        )
        assert model.test_dc is not None

        with pytest.raises(
            beartype.roar.BeartypeCallHintParamViolation,
            match="not instance of str",
        ):
            model.test_dc = DataCollector(model=model, agent_reporters={123: "wealth"})

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
                "wealth": "wealth",
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

        agent_df = collected_data["agent"]

        ## 3 agent sets * 4 agents/set = 12 rows
        assert agent_df.shape == (12, 7)
        assert set(agent_df.columns) == {
            "unique_id",
            "agent_type",
            "wealth",
            "age",
            "step",
            "seed",
            "batch",
        }

        expected_wealth = [1, 2, 3, 4] + [10, 20, 30, 40] + [1, 2, 3, 4]
        expected_age = [10, 20, 30, 40] + [11, 22, 33, 44] + [1, 2, 3, 4]

        assert sorted(agent_df["wealth"].to_list()) == sorted(expected_wealth)
        assert sorted(agent_df["age"].to_list()) == sorted(expected_age)

        type_counts = agent_df["agent_type"].value_counts(sort=True)
        assert (
            type_counts.filter(pl.col("agent_type") == "ExampleAgentSet1")["count"][0]
            == 4
        )
        assert (
            type_counts.filter(pl.col("agent_type") == "ExampleAgentSet2")["count"][0]
            == 4
        )
        assert (
            type_counts.filter(pl.col("agent_type") == "ExampleAgentSet3")["count"][0]
            == 4
        )
        assert agent_df["step"].to_list() == [0] * 12
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
                "wealth": "wealth",
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

        agent_df = collected_data["agent"]
        assert agent_df.shape == (12, 7)
        assert set(agent_df.columns) == {
            "unique_id",
            "agent_type",
            "wealth",
            "age",
            "step",
            "seed",
            "batch",
        }

        expected_wealth = [6, 7, 8, 9] + [20, 30, 40, 50] + [1, 2, 3, 4]
        expected_age = [10, 20, 30, 40] + [11, 22, 33, 44] + [6, 7, 8, 9]

        assert sorted(agent_df["wealth"].to_list()) == sorted(expected_wealth)
        assert sorted(agent_df["age"].to_list()) == sorted(expected_age)
        assert agent_df["step"].to_list() == [5] * 12

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
                "wealth": "wealth",
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

        agent_df = collected_data["agent"]

        # 12 agents * 2 steps = 24 rows
        assert agent_df.shape == (24, 7)
        assert set(agent_df.columns) == {
            "unique_id",
            "agent_type",
            "wealth",
            "age",
            "step",
            "seed",
            "batch",
        }

        df_step_2 = agent_df.filter(pl.col("step") == 2)
        expected_wealth_s2 = [3, 4, 5, 6] + [14, 24, 34, 44] + [1, 2, 3, 4]
        expected_age_s2 = [10, 20, 30, 40] + [11, 22, 33, 44] + [3, 4, 5, 6]

        assert df_step_2.shape == (12, 7)
        assert sorted(df_step_2["wealth"].to_list()) == sorted(expected_wealth_s2)
        assert sorted(df_step_2["age"].to_list()) == sorted(expected_age_s2)

        df_step_4 = agent_df.filter(pl.col("step") == 4)
        expected_wealth_s4 = [5, 6, 7, 8] + [18, 28, 38, 48] + [1, 2, 3, 4]
        expected_age_s4 = [10, 20, 30, 40] + [11, 22, 33, 44] + [5, 6, 7, 8]

        assert df_step_4.shape == (12, 7)
        assert sorted(df_step_4["wealth"].to_list()) == sorted(expected_wealth_s4)
        assert sorted(df_step_4["age"].to_list()) == sorted(expected_age_s4)

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
                    "wealth": "wealth",
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
            assert agent_df.shape == (12, 7)
            assert set(agent_df.columns) == {
                "unique_id",
                "agent_type",
                "wealth",
                "age",
                "step",
                "seed",
                "batch",
            }

            expected_wealth_s2 = [3, 4, 5, 6] + [14, 24, 34, 44] + [1, 2, 3, 4]
            expected_age_s2 = [10, 20, 30, 40] + [11, 22, 33, 44] + [3, 4, 5, 6]

            assert agent_df["step"].to_list() == [2] * 12
            assert sorted(agent_df["wealth"].to_list()) == sorted(expected_wealth_s2)
            assert sorted(agent_df["age"].to_list()) == sorted(expected_age_s2)

            agent_df_s4 = pl.read_csv(
                os.path.join(tmpdir, "agent_step4_batch0.csv"),
                schema_overrides={"seed": pl.Utf8},
            )
            expected_wealth_s4 = [5, 6, 7, 8] + [18, 28, 38, 48] + [1, 2, 3, 4]
            assert agent_df_s4["step"].to_list() == [4] * 12
            assert sorted(agent_df_s4["wealth"].to_list()) == sorted(expected_wealth_s4)

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
                    "wealth": "wealth",
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
            # 12 rows. 6 cols: unique_id, agent_type, wealth, step, seed, batch
            assert agent_df.shape == (12, 6)
            assert set(agent_df.columns) == {
                "unique_id",
                "agent_type",
                "wealth",
                "step",
                "seed",
                "batch",
            }

            expected_wealth = [1, 2, 3, 4] + [10, 20, 30, 40] + [1, 2, 3, 4]
            assert agent_df["step"].to_list() == [0] * 12
            assert sorted(agent_df["wealth"].to_list()) == sorted(expected_wealth)

    @pytest.mark.skipif(
        os.getenv("SKIP_PG_TESTS") == "true",
        reason="PostgreSQL tests are skipped on Windows runners",
    )
    def test_postgress(self, fix1_model, postgres_uri):
        model = fix1_model

        # Connect directly and validate data
        import psycopg2

        try:
            conn = psycopg2.connect(postgres_uri)
        except psycopg2.OperationalError as e:
            pytest.skip(f"Could not connect to PostgreSQL: {e}")

        cur = conn.cursor()

        ## Cleaning up tables first
        cur.execute("DROP TABLE IF EXISTS public.model_data, public.agent_data;")
        conn.commit()

        cur.execute("""
            CREATE TABLE public.model_data (
                step INTEGER,
                seed VARCHAR,
                batch INTEGER,
                total_agents INTEGER
            )
        """)

        ## MODIFIED: CREATE TABLE for long format
        cur.execute("""
            CREATE TABLE public.agent_data (
                step INTEGER,
                seed VARCHAR,
                batch INTEGER,
                unique_id BIGINT,
                agent_type VARCHAR,
                wealth INTEGER,
                age INTEGER
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
            ##  MODIFIED : long format agent reporters
            agent_reporters={
                "wealth": "wealth",
                "age": "age",
            },
            storage="postgresql",
            schema="public",
            storage_uri=postgres_uri,
        )

        model.run_model_with_conditional_collect(4)  ## Runs 1,2,3,4. Collects at 2, 4.
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

        # MODIFIED: Check agent data
        cur.execute(
            "SELECT wealth, age FROM agent_data WHERE step=2 ORDER BY wealth, age"
        )
        agent_rows = cur.fetchall()

        expected_rows_s2 = [
            (1, 3),
            (2, 4),
            (3, 5),
            (3, 10),
            (4, 6),
            (4, 20),
            (5, 30),
            (6, 40),
            (14, 11),
            (24, 22),
            (34, 33),
            (44, 44),
        ]

        assert sorted(agent_rows) == sorted(expected_rows_s2)

        cur.execute("DROP TABLE public.model_data, public.agent_data;")
        conn.commit()
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
            agent_reporters={"wealth": "wealth", "age": "age"},
        )

        model.run_model_with_conditional_collect_multiple_batch(5)
        collected_data = model.dc.data

        assert collected_data["model"].shape == (4, 4)
        assert collected_data["model"]["step"].to_list() == [2, 2, 4, 4]
        assert collected_data["model"]["batch"].to_list() == [0, 1, 0, 1]

        agent_df = collected_data["agent"]
        assert agent_df.shape == (48, 7)
        assert set(agent_df.columns) == {
            "unique_id",
            "agent_type",
            "wealth",
            "age",
            "step",
            "seed",
            "batch",
        }

        df_s2_b0 = agent_df.filter((pl.col("step") == 2) & (pl.col("batch") == 0))
        expected_wealth_s2b0 = [2, 3, 4, 5] + [12, 22, 32, 42] + [1, 2, 3, 4]
        assert sorted(df_s2_b0["wealth"].to_list()) == sorted(expected_wealth_s2b0)

        df_s2_b1 = agent_df.filter((pl.col("step") == 2) & (pl.col("batch") == 1))
        expected_wealth_s2b1 = [3, 4, 5, 6] + [14, 24, 34, 44] + [1, 2, 3, 4]
        assert sorted(df_s2_b1["wealth"].to_list()) == sorted(expected_wealth_s2b1)

        df_s4_b0 = agent_df.filter((pl.col("step") == 4) & (pl.col("batch") == 0))
        expected_wealth_s4b0 = [4, 5, 6, 7] + [16, 26, 36, 46] + [1, 2, 3, 4]
        assert sorted(df_s4_b0["wealth"].to_list()) == sorted(expected_wealth_s4b0)

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
                    "wealth": "wealth",
                    "age": "age",
                },
                storage="csv",
                storage_uri=tmpdir,
            )

            model.run_model_with_conditional_collect_multiple_batch(5)
            model.dc.flush()

            for _ in range(20):  # wait up to ~2 seconds
                created_files = os.listdir(tmpdir)
                if len(created_files) >= 8:  # 4 collects * 2 files/collect = 8 files
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
            assert model_df_step2_batch0["step"].to_list() == [2]
            assert model_df_step2_batch0["total_agents"].to_list() == [12]

            model_df_step2_batch1 = pl.read_csv(
                os.path.join(tmpdir, "model_step2_batch1.csv"),
                schema_overrides={"seed": pl.Utf8},
            )
            assert model_df_step2_batch1["step"].to_list() == [2]
            assert model_df_step2_batch1["total_agents"].to_list() == [12]

            model_df_step4_batch0 = pl.read_csv(
                os.path.join(tmpdir, "model_step4_batch0.csv"),
                schema_overrides={"seed": pl.Utf8},
            )
            assert model_df_step4_batch0["step"].to_list() == [4]
            assert model_df_step4_batch0["total_agents"].to_list() == [12]

            # test agent batch reset
            agent_df_step2_batch0 = pl.read_csv(
                os.path.join(tmpdir, "agent_step2_batch0.csv"),
                schema_overrides={"seed": pl.Utf8, "unique_id": pl.UInt64},
            )

            expected_wealth_s2b0 = [2, 3, 4, 5] + [12, 22, 32, 42] + [1, 2, 3, 4]
            assert sorted(agent_df_step2_batch0["wealth"].to_list()) == sorted(
                expected_wealth_s2b0
            )

            agent_df_step2_batch1 = pl.read_csv(
                os.path.join(tmpdir, "agent_step2_batch1.csv"),
                schema_overrides={"seed": pl.Utf8, "unique_id": pl.UInt64},
            )
            expected_wealth_s2b1 = [3, 4, 5, 6] + [14, 24, 34, 44] + [1, 2, 3, 4]
            assert sorted(agent_df_step2_batch1["wealth"].to_list()) == sorted(
                expected_wealth_s2b1
            )

    def test_collect_no_agentsets_list(self, fix1_model, caplog):
        """Tests that the collector logs an error and exits gracefully if _agentsets is missing."""
        model = fix1_model
        del model.sets._agentsets

        dc = DataCollector(model=model, agent_reporters={"wealth": "wealth"})
        dc.collect()

        assert "could not find '_agentsets'" in caplog.text
        assert dc.data["agent"].shape == (0, 0)

    def test_collect_agent_set_no_df(self, fix1_model, caplog):
        """Tests that the collector logs a warning and skips a set if it has no .df attribute."""

        class NoDfSet:
            def __init__(self):
                self.__class__ = type("NoDfSet", (object,), {})

        fix1_model.sets._agentsets.append(NoDfSet())

        fix1_model.dc = DataCollector(
            fix1_model, agent_reporters={"wealth": "wealth", "age": "age"}
        )
        fix1_model.dc.collect()

        assert "has no 'df' attribute" in caplog.text
        assert fix1_model.dc.data["agent"].shape == (12, 7)

    def test_collect_df_no_unique_id(self, fix1_model, caplog):
        """Tests that the collector logs a warning and skips a set if its df has no unique_id."""
        bad_set = fix1_model.sets._agentsets[0]
        bad_set._df = bad_set._df.drop("unique_id")

        fix1_model.dc = DataCollector(
            fix1_model, agent_reporters={"wealth": "wealth", "age": "age"}
        )
        fix1_model.dc.collect()

        assert "has no 'unique_id' column" in caplog.text
        assert fix1_model.dc.data["agent"].shape == (8, 7)

    def test_collect_no_matching_reporters(self, fix1_model):
        """Tests that the collector returns an empty frame if no reporters match any columns."""
        fix1_model.dc = DataCollector(
            fix1_model, agent_reporters={"baz": "foo", "qux": "bar"}
        )
        fix1_model.dc.collect()

        assert fix1_model.dc.data["agent"].shape == (0, 0)
