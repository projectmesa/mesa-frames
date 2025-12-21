from __future__ import annotations

# %% [markdown]
"""# Data Collector Tutorial

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/projectmesa/mesa-frames/blob/main/docs/general/user-guide/4_datacollector.ipynb)

This notebook walks you through using the concrete `DataCollector` in `mesa-frames` to collect model- and agent-level data and write it to different storage backends: **memory, CSV, Parquet, S3, and PostgreSQL**.

It also shows how to use **conditional triggers** and how the **schema validation** behaves for PostgreSQL."""

# %% [markdown]
"""## Installation (Colab or fresh env)

Uncomment and run the next cell if you're in Colab or a clean environment."""

# %%
# #!pip install git+https://github.com/projectmesa/mesa-frames mesa

# %% [markdown]
"""## Minimal Example Model

We create a tiny model using the `Model` and an `AgentSet`-style agent container. This is just to demonstrate collection APIs."""

# %%
from mesa_frames import Model, AgentSet, DataCollector
import polars as pl


class MoneyAgents(AgentSet):
    def __init__(self, n: int, model: Model):
        super().__init__(model)
        # one column, one unit of wealth each
        self += pl.DataFrame({"wealth": pl.ones(n, eager=True)})

    def step(self) -> None:
        self.select(self.wealth > 0)
        receivers = self.df.sample(n=len(self.active_agents), with_replacement=True)
        self["active", "wealth"] -= 1
        income = receivers.group_by("unique_id").len()
        self[income["unique_id"], "wealth"] += income["len"]


class MoneyModel(Model):
    def __init__(self, n: int):
        super().__init__()
        self.sets.add(MoneyAgents(n, self))
        self.dc = DataCollector(
            model=self,
            model_reporters={
                "total_wealth": lambda m: m.sets["MoneyAgents"].df["wealth"].sum(),
                "n_agents": lambda m: len(m.sets["MoneyAgents"]),
            },
            agent_reporters={
                "wealth": "wealth",  # pull existing column
            },
            storage="memory",  # we'll switch this per example
            storage_uri=None,
            trigger=lambda m: m.steps % 2
            == 0,  # collect every 2 steps via conditional_collect
            reset_memory=True,
        )

    def step(self):
        self.sets.do("step")

    def run(self, steps: int, conditional: bool = True):
        for _ in range(steps):
            self.step()
            self.dc.conditional_collect()  # or .collect if you want to collect every step regardless of trigger


model = MoneyModel(1000)
model.run(10)
model.dc.data  # peek in-memory dataframes

# %% [markdown]
"""## Saving the data for later use

`DataCollector` supports multiple storage backends.
Files are saved with **step number** and **batch number** (e.g., `model_step10_batch2.csv`) so multiple collects at the same step don’t overwrite.

- **CSV:** `storage="csv"` → writes `model_step{n}_batch{k}.csv`, easy to open anywhere.
- **Parquet:** `storage="parquet"` → compressed, efficient for large datasets.
- **S3:** `storage="S3-csv"`/`storage="S3-parquet"` → saves CSV/Parquet directly to Amazon S3.
- **PostgreSQL:** `storage="postgresql"` → inserts results into `model_data` and `agent_data` tables for querying."""

# %% [markdown]
"""## Writing to Local CSV

Switch the storage to `csv` and provide a folder path. Files are written as `model_step{n}.csv` and `agent_step{n}.csv`."""

# %%
import os

os.makedirs("./data_csv", exist_ok=True)
model_csv = MoneyModel(1000)
model_csv.dc = DataCollector(
    model=model_csv,
    model_reporters={
        "total_wealth": lambda m: m.sets["MoneyAgents"].df["wealth"].sum(),
        "n_agents": lambda m: len(m.sets["MoneyAgents"]),
    },
    agent_reporters={
        "wealth": "wealth",
    },
    storage="csv",  # saving as csv
    storage_uri="./data_csv",
    trigger=lambda m: m._steps % 2 == 0,
    reset_memory=True,
)
model_csv.run(10)
model_csv.dc.flush()
os.listdir("./data_csv")

# %% [markdown]
"""## Writing to Local Parquet

Use `parquet` for columnar output."""

# %%
os.makedirs("./data_parquet", exist_ok=True)
model_parq = MoneyModel(1000)
model_parq.dc = DataCollector(
    model=model_parq,
    model_reporters={
        "total_wealth": lambda m: m.sets["MoneyAgents"].df["wealth"].sum(),
        "n_agents": lambda m: len(m.sets["MoneyAgents"]),
    },
    agent_reporters={
        "wealth": "wealth",
    },
    storage="parquet",  # save as parquet
    storage_uri="data_parquet",
    trigger=lambda m: m._steps % 2 == 0,
    reset_memory=True,
)
model_parq.run(10)
model_parq.dc.flush()
os.listdir("./data_parquet")

# %% [markdown]
"""## Writing to Amazon S3 (CSV or Parquet)

Set AWS credentials via environment variables or your usual config. Then choose `S3-csv` or `S3-parquet` and pass an S3 URI (e.g., `s3://my-bucket/experiments/run-1`).

> **Note:** This cell requires network access & credentials when actually run."""

# %%
model_s3 = MoneyModel(1000)
model_s3.dc = DataCollector(
    model=model_s3,
    model_reporters={
        "total_wealth": lambda m: m.sets["MoneyAgents"].df["wealth"].sum(),
        "n_agents": lambda m: len(m.sets["MoneyAgents"]),
    },
    agent_reporters={
        "wealth": "wealth",
    },
    storage="S3-csv",  # save as csv in S3
    storage_uri="s3://my-bucket/experiments/run-1",  # change it to required path
    trigger=lambda m: m._steps % 2 == 0,
    reset_memory=True,
)
model_s3.run(10)
model_s3.dc.flush()

# %% [markdown]
"""## Writing to PostgreSQL

PostgreSQL requires that the target tables exist and that the expected reporter columns are present. The collector will validate tables/columns up front and raise descriptive errors if something is missing.

Below is a minimal schema example. Adjust columns to your configured reporters."""

# %%
DDL_MODEL = r"""
CREATE SCHEMA IF NOT EXISTS public;
CREATE TABLE IF NOT EXISTS public.model_data (
  step INTEGER,
  seed VARCHAR,
  total_wealth BIGINT,
  n_agents INTEGER
);
"""
DDL_AGENT = r"""
CREATE TABLE IF NOT EXISTS public.agent_data (
  step INTEGER,
  seed VARCHAR,
  unique_id BIGINT,
  wealth BIGINT
);
"""
print(DDL_MODEL)
print(DDL_AGENT)

# %% [markdown]
"""After creating the tables (outside this notebook or via a DB connection cell), configure and flush:"""

# %%
POSTGRES_URI = "postgresql://user:pass@localhost:5432/mydb"
m_pg = MoneyModel(300)
m_pg.dc._storage = "postgresql"
m_pg.dc._storage_uri = POSTGRES_URI
m_pg.run(6)
m_pg.dc.flush()

# %% [markdown]
"""## Triggers & Conditional Collection

The collector accepts a `trigger: Callable[[Model], bool]`. When using `conditional_collect()`, the collector checks the trigger and collects only if it returns `True`.

You can always call `collect()` to gather data unconditionally."""

# %%
m = MoneyModel(100)
m.dc.trigger = lambda model: model._steps % 3 == 0  # every 3rd step
m.run(10, conditional=True)
m.dc.data["model"].head()

# %% [markdown]
"""## Troubleshooting

- **ValueError: Please define a storage_uri** — for non-memory backends you must set `_storage_uri`.
- **Missing columns in table** — check the PostgreSQL error text; create/alter the table to include the columns for your configured `model_reporters` and `agent_reporters`, plus required `step` and `seed`.
- **Permissions/credentials errors** (S3/PostgreSQL) — ensure correct IAM/credentials or database permissions."""

# %% [markdown]
"""---
*Generated on 2025-08-30.*"""
