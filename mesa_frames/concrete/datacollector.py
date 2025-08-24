"""
Concrete class for data collection in mesa-frames.

This module defines a `DataCollector` implementation that gathers and optionally persists
model-level and agent-level data during simulations. It supports multiple storage backends,
including in-memory, CSV, Parquet, S3, and PostgreSQL, using Polars for efficient lazy
data processing.

Classes:
    DataCollector:
        A concrete class defining logic for all data collector implementations.
        It supports flexible reporting of model and agent attributes, conditional
        data collection using a trigger function, and pluggable backends for storage.

Supported Storage Backends:
    - memory         : In-memory collection (default)
    - csv            : Local CSV file output
    - parquet        : Local Parquet file output
    - S3-csv         : CSV files stored on Amazon S3
    - S3-parquet     : Parquet files stored on Amazon S3
    - postgresql     : PostgreSQL database with schema support

Triggers:
    - A `trigger` parameter can be provided to control conditional collection.
      This is a callable taking the model as input and returning a boolean.
      If true, data is collected during `conditional_collect()`.

Usage:
    The `DataCollector` class is designed to be used within a `ModelDF` instance
    to collect model-level and/or agent-level data.

    Example:
    --------
    from mesa_frames.concrete.model import ModelDF
    from mesa_frames.concrete.datacollector import DataCollector

    class ExampleModel(ModelDF):
        def __init__(self, agents: AgentsDF):
            super().__init__()
            self.agents = agents
            self.dc = DataCollector(
                model=self,
                # other required arguments
            )

        def step(self):
            # Option 1: collect immediately
            self.dc.collect()

            # Option 2: collect based on condition
            self.dc.conditional_collect()

            # Write the collected data to the destination
            self.dc.flush()
"""

import polars as pl
import boto3
from urllib.parse import urlparse
import tempfile
import psycopg2
from mesa_frames.abstract.datacollector import AbstractDataCollector
from typing import Any, Literal
from collections.abc import Callable
from mesa_frames import ModelDF
from psycopg2.extensions import connection


class DataCollector(AbstractDataCollector):
    def __init__(
        self,
        model: ModelDF,
        model_reporters: dict[str, Callable] | None = None,
        agent_reporters: dict[str, str | Callable] | None = None,
        trigger: Callable[[Any], bool] | None = None,
        reset_memory: bool = True,
        storage: Literal[
            "memory", "csv", "parquet", "S3-csv", "S3-parquet", "postgresql"
        ] = "memory",
        storage_uri: str | None = None,
        schema: str = "public",
    ):
        """
        Initialize the DataCollector with configuration options.

        Parameters
        ----------
        model : ModelDF
            The model object from which data is collected.
        model_reporters : dict[str, Callable] | None
            Functions to collect data at the model level.
        agent_reporters : dict[str, str | Callable] | None
            Attributes or functions to collect data at the agent level.
        trigger : Callable[[Any], bool] | None
            A function(model) -> bool that determines whether to collect data.
        reset_memory : bool
            Whether to reset in-memory data after flushing. Default is True.
        storage : Literal["memory", "csv", "parquet", "S3-csv", "S3-parquet", "postgresql"        ]
            Storage backend URI (e.g. 'memory:', 'csv:', 'postgresql:').
        storage_uri: str | None
            URI or path corresponding to the selected storage backend.
        schema: str
            Schema name used for PostgreSQL storage.

        """
        super().__init__(
            model=model,
            model_reporters=model_reporters,
            agent_reporters=agent_reporters,
            trigger=trigger,
            reset_memory=reset_memory,
            storage=storage,  # literal won't work
        )
        self._writers = {
            "csv": self._write_csv_local,
            "parquet": self._write_parquet_local,
            "S3-csv": self._write_csv_s3,
            "S3-parquet": self._write_parquet_s3,
            "postgresql": self._write_postgres,
        }
        self._storage_uri = storage_uri
        self._schema = schema

        self._validate_inputs()

    def _collect(self):
        """
        Collect data from the model and agents for the current step.

        This method checks for the presence of model and agent reporters
        and calls the appropriate collection routines for each.
        """
        if self._model_reporters:
            self._collect_model_reporters()

        if self._agent_reporters:
            self._collect_agent_reporters()

    def _collect_model_reporters(self):
        """
        Collect model-level data using the model_reporters.

        Creates a LazyFrame containing the step, seed, and values
        returned by each model reporter. Appends the LazyFrame to internal storage.
        """
        model_data_dict = {}
        model_data_dict["step"] = self._model._steps
        model_data_dict["seed"] = str(self.seed)
        for column_name, reporter in self._model_reporters.items():
            model_data_dict[column_name] = reporter(self._model)
        model_lazy_frame = pl.LazyFrame([model_data_dict])
        self._frames.append(("model", str(self._model._steps), model_lazy_frame))

    def _collect_agent_reporters(self):
        """
        Collect agent-level data using the agent_reporters.

        Constructs a LazyFrame with one column per reporter and
        includes `step` and `seed` metadata. Appends it to internal storage.
        """
        agent_data_dict = {}
        for col_name, reporter in self._agent_reporters.items():
            if isinstance(reporter, str):
                for k, v in self._model.agents[reporter].items():
                    agent_data_dict[col_name + "_" + str(k.__class__.__name__)] = v
            else:
                agent_data_dict[col_name] = reporter(self._model)
        agent_lazy_frame = pl.LazyFrame(agent_data_dict)
        agent_lazy_frame = agent_lazy_frame.with_columns(
            [
                pl.lit(self._model._steps).alias("step"),
                pl.lit(str(self.seed)).alias("seed"),
            ]
        )
        self._frames.append(("agent", str(self._model._steps), agent_lazy_frame))

    @property
    def data(self) -> dict[str, pl.DataFrame]:
        """
        Retrieve the collected data as eagerly evaluated Polars DataFrames.

        Returns
        -------
        dict[str, pl.DataFrame]
            A dictionary with keys "model" and "agent" mapping to concatenated DataFrames of collected data.
        """
        model_frames = [
            lf.collect() for kind, step, lf in self._frames if kind == "model"
        ]
        agent_frames = [
            lf.collect() for kind, step, lf in self._frames if kind == "agent"
        ]
        return {
            "model": pl.concat(model_frames) if model_frames else pl.DataFrame(),
            "agent": pl.concat(agent_frames) if agent_frames else pl.DataFrame(),
        }

    def _flush(self, frames_to_flush: list):
        """
        Flush the collected data to the configured external storage backend.

        Uses the appropriate writer function based on the specified storage option.
        """
        self._writers[self._storage](
            uri=self._storage_uri, frames_to_flush=frames_to_flush
        )

    def _write_csv_local(self, uri: str, frames_to_flush: list):
        """
        Write collected data to local CSV files.

        Parameters
        ----------
        uri : str
            Local directory path to write files into.
        frames_to_flush : list
            the collected data in the current thread.
        """
        for kind, step, df in frames_to_flush:
            df.collect().write_csv(f"{uri}/{kind}_step{step}.csv")

    def _write_parquet_local(self, uri: str, frames_to_flush: list):
        """
        Write collected data to local Parquet files.

        Parameters
        ----------
        uri: str
            Local directory path to write files into.
        frames_to_flush : list
            the collected data in the current thread.
        """
        for kind, step, df in frames_to_flush:
            df.collect().write_parquet(f"{uri}/{kind}_step{step}.parquet")

    def _write_csv_s3(self, uri: str, frames_to_flush: list):
        """
        Write collected data to AWS S3 in CSV format.

        Parameters
        ----------
        uri: str
            S3 URI (e.g., s3://bucket/path) to upload files to.
        frames_to_flush : list
            the collected data in the current thread.
        """
        self._write_s3(uri=uri, frames_to_flush=frames_to_flush, format_="csv")

    def _write_parquet_s3(self, uri: str, frames_to_flush: list):
        """
        Write collected data to AWS S3 in Parquet format.

        Parameters
        ----------
        uri: str
            S3 URI (e.g., s3://bucket/path) to upload files to.
        frames_to_flush : list
            the collected data in the current thread.
        """
        self._write_s3(uri=uri, frames_to_flush=frames_to_flush, format_="parquet")

    def _write_s3(self, uri: str, frames_to_flush: list, format_: str):
        """
        Upload collected data to S3 in a specified format.

        Parameters
        ----------
        uri: str
            S3 URI to upload to.
        frames_to_flush : list
            the collected data in the current thread.
        format_: str
            Format of the output files ("csv" or "parquet").
        """
        s3 = boto3.client("s3")
        parsed = urlparse(uri)
        bucket = parsed.netloc
        prefix = parsed.path.lstrip("/")
        for kind, step, lf in frames_to_flush:
            df = lf.collect()
            with tempfile.NamedTemporaryFile(suffix=f".{format_}") as tmp:
                if format_ == "csv":
                    df.write_csv(tmp.name)
                elif format_ == "parquet":
                    df.write_parquet(tmp.name)
                key = f"{prefix}/{kind}_step{step}.{format_}"
                s3.upload_file(tmp.name, bucket, key)

    def _write_postgres(self, uri: str, frames_to_flush: list):
        """
        Write collected data to a PostgreSQL database.

        Each frame is inserted into the appropriate table (`model_data` or `agent_data`)
        using batched insert queries.

        Parameters
        ----------
        uri: str
            PostgreSQL connection URI in the form postgresql://testuser:testpass@localhost:5432/testdb
        frames_to_flush : list
            the collected data in the current thread.
        """
        conn = self._get_db_connection(uri=uri)
        cur = conn.cursor()
        for kind, step, lf in frames_to_flush:
            df = lf.collect()
            table = f"{kind}_data"
            cols = df.columns
            values = [tuple(row) for row in df.rows()]
            placeholders = ", ".join(["%s"] * len(cols))
            columns = ", ".join(cols)
            cur.executemany(
                f"INSERT INTO {self._schema}.{table} ({columns}) VALUES ({placeholders})",
                values,
            )
        conn.commit()
        cur.close()
        conn.close()

    def _get_db_connection(self, uri: str) -> connection:
        """
        Uri should be like: postgresql://user:pass@host:port/dbname.

        Parameters
        ----------
        uri: str
            PostgreSQL connection URI in the form postgresql://testuser:testpass@localhost:5432/testdb

        Returns
        -------
        connection
            psycopg2 connection
        """
        parsed = urlparse(uri)
        conn = psycopg2.connect(
            dbname=parsed.path[1:],  # remove leading slash
            user=parsed.username,
            password=parsed.password,
            host=parsed.hostname,
            port=parsed.port,
        )
        return conn

    def _validate_inputs(self):
        """
        Validate configuration and required schema for non-memory storage backends.

        - Ensures a `storage_uri` is provided if needed.
        - For PostgreSQL, validates that required tables and columns exist.
        """
        if self._storage != "memory" and self._storage_uri == None:
            raise ValueError(
                "Please define a storage_uri to if to be stored not in memory"
            )

        if self._storage == "postgresql":
            conn = self._get_db_connection(self._storage_uri)
            try:
                self._validate_postgress_table_exists(conn)
                self._validate_postgress_columns_exists(conn)
            finally:
                conn.close()

    def _validate_postgress_table_exists(self, conn: connection):
        """
        Validate that the required PostgreSQL tables exist for storing model and agent data.

        Parameters
        ----------
        conn: connection
            Open database connection.
        """
        if self._model_reporters:
            self._validate_reporter_table(conn=conn, table_name="model_data")
        if self._agent_reporters:
            self._validate_reporter_table(conn=conn, table_name="agent_data")

    def _validate_postgress_columns_exists(self, conn: connection):
        """
        Validate that required columns are present in the PostgreSQL tables.

        Parameters
        ----------
        conn: connection
            Open database connection.
        """
        if self._model_reporters:
            self._validate_reporter_table_columns(
                conn=conn, table_name="model_data", reporter=self._model_reporters
            )
        if self._agent_reporters:
            self._validate_reporter_table_columns(
                conn=conn, table_name="agent_data", reporter=self._agent_reporters
            )

    def _validate_reporter_table(self, conn: connection, table_name: str):
        """
        Check if a given table exists in the PostgreSQL schema.

        Parameters
        ----------
        conn : connection
            Open database connection.
        table_name : str
            Name of the table to check.

        Raises
        ------
        ValueError
            If the table does not exist in the schema.
        """
        query = f"""
            SELECT EXISTS (
            SELECT FROM information_schema.tables
            WHERE table_schema = '{self._schema}' AND table_name = '{table_name}'
            );"""
        result = self._execute_query_with_result(conn, query)
        if result == [(False,)]:
            raise ValueError(
                f"{self._schema}.{table_name} does not exist. To store collected data in DB please create a table with required columns"
            )

    def _validate_reporter_table_columns(
        self, conn: connection, table_name: str, reporter: dict[str, Callable | str]
    ):
        """
        Check if the expected columns are present in a given PostgreSQL table.

        Parameters
        ----------
        conn : connection
            Open database connection.
        table_name :str
            Name of the table to validate.
        reporter : dict[str, Callable | str]
            Dictionary of reporters whose keys are expected as columns.

        Raises
        ------
        ValueError
            If any expected columns are missing from the table.
        """
        expected_columns = set()
        for col_name, required_column in reporter.items():
            if isinstance(required_column, str):
                for k, v in self._model.agents[required_column].items():
                    expected_columns.add(
                        (col_name + "_" + str(k.__class__.__name__)).lower()
                    )
            else:
                expected_columns.add(col_name.lower())

        query = f"""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = '{self._schema}' AND table_name = '{table_name}';
        """

        result = self._execute_query_with_result(conn, query)
        if not result:
            raise ValueError(
                f"Could not retrieve columns for table {self._schema}.{table_name}"
            )

        existing_columns = {row[0] for row in result}
        missing_columns = expected_columns - existing_columns
        required_columns = {
            "step": "Integer",
            "seed": "Varchar",
        }

        missing_required = {
            col: col_type
            for col, col_type in required_columns.items()
            if col not in existing_columns
        }

        if missing_columns or missing_required:
            error_parts = []

            if missing_columns:
                error_parts.append(f"Missing columns: {sorted(missing_columns)}")

            if missing_required:
                required_list = [
                    f"`{col}` column of type ({col_type})"
                    for col, col_type in missing_required.items()
                ]
                error_parts.append(
                    "Missing specific columns: " + ", ".join(required_list)
                )

            raise ValueError(
                f"Missing columns in table {self._schema}.{table_name}: "
                + "; ".join(error_parts)
            )

    def _execute_query_with_result(self, conn: connection, query: str) -> list[tuple]:
        """
        Execute a SQL query and return the fetched results.

        Parameters
        ----------
        conn : connection
            Open database connection.
        query : str
            SQL query string.

        Returns
        -------
        list[tuple]
            Query result rows.
        """
        with conn.cursor() as cur:
            cur.execute(query)
            return cur.fetchall()
