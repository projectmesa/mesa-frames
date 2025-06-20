"""
Concrete class for data collection in mesa-frames

This modulde defines the core functionalities for data collection in mesa-frames
It provides a 
and.
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


class DataCollector(AbstractDataCollector):
    def __init__(
        self,
        model: ModelDF,
        model_reporters: dict[str, Callable] | None = None,
        agent_reporters: dict[str, str | Callable] | None = None,
        trigger: Callable[[Any], bool] | None = None,
        reset_memory: bool = True,
        storage: Literal["memory", "csv", "postgresql"] = "memory",
        storage_uri: str = None,
        schema: str = 'public'
    ):
        super().__init__(
            model=model,
            model_reporters=model_reporters,
            agent_reporters=agent_reporters,
            trigger=trigger,
            reset_memory=reset_memory,
            storage=storage,  # literal won't work
        )
        self._writers = {
            "csv": self.write_csv_local,
            "parquet": self.write_parquet_local,
            "S3-csv": self.write_csv_s3,
            "S3-parquet": self.write_parquet_s3,
            "postgres": self.write_postgres,
        }
        self._storage_uri = storage_uri
        self._schema = schema

        self._validate_inputs()
    
    def _collect(self):

        if self._model_reporters:
            self._collect_model_reporters()

        if self._agent_reporters:
            self._collect_agent_reporters()
    
    def _collect_model_reporters(self):
        model_data_dict = {}
        model_data_dict["step"] = self._model._steps
        model_data_dict["seed"] = str(self.seed)
        for column_name, reporter in self._model_reporters.items():
            model_data_dict[column_name] = reporter(self._model)
        model_lazy_frame = pl.LazyFrame([model_data_dict])
        self._frames.append(("model", str(self._model._steps), model_lazy_frame))

    def _collect_agent_reporters(self):
        agent_data_dict = {}
        for col_name, reporter in self._agent_reporters.items():
            agent_data_dict[col_name] = reporter(self._model)
        agent_lazy_frame = pl.LazyFrame(agent_data_dict)
        agent_lazy_frame = agent_lazy_frame.with_columns(
            [
                pl.lit(self._model._steps).alias("step"),
                pl.lit(str(self.seed)).alias("seed"),
            ]
        )
        self._frames.append(("agent", str(self._model._steps), agent_lazy_frame))

    def _flush(self):
        self._writers[self.schema](self._storage_uri)

    def write_csv_local(self, uri):
        for kind, step, df in self._frames:
            df.collect().write_csv(f"{uri}/{kind}_step{step}.csv")

    def write_parquet_local(self, uri):
        for kind, step, df in self._frames:
            df.collect().write_parquet(f"{uri}/{kind}_step{step}.parquet")

    def write_csv_s3(self, uri):
        self._write_s3(uri, format_="csv")

    def _write_s3(self, uri, format_):
        s3 = boto3.client("s3")
        parsed = urlparse(uri)
        bucket = parsed.netloc
        prefix = parsed.path.lstrip("/")
        for kind, step, lf in self._frames:
            df = lf.collect()
            with tempfile.NamedTemporaryFile(suffix=f".{format_}") as tmp:
                if format_ == "csv":
                    df.write_csv(tmp.name)
                else:
                    df.write_parquet(tmp.name)
                key = f"{prefix}/{kind}_step{step}.{format_}"
                s3.upload_file(tmp.name, bucket, key)

    def write_postgres(self, uri):
        conn = self._get_db_connection(uri=uri)
        cur = conn.cursor()
        for kind, step, lf in self._frames:
            df = lf.collect()
            table = f"{kind}_data"
            cols = df.columns
            values = [tuple(row) for row in df.rows()]
            placeholders = ", ".join(["%s"] * len(cols))
            columns = ", ".join(cols)
            cur.executemany(
                f"INSERT INTO {table} ({columns}) VALUES ({placeholders})", values
            )
        conn.commit()
        cur.close()
        conn.close()

    @property
    def data(self):
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
    
    def _get_db_connection(self,uri):
        parsed = urlparse(f"//{uri}")
        conn = psycopg2.connect(
            dbname=parsed.path[1:],
            user=parsed.username,
            password=parsed.password,
            host=parsed.hostname,
            port=parsed.port,
        )
        return conn
    def _validate_inputs(self):
        if self.storage != "memory" and self._storage_uri ==None:
            raise ValueError("Please define a storage_uri to if to be stored not in memory")

        if self.storage == "postgresql":
            
            conn = self._get_db_connection(self._storage_uri)
            self._validate_postgress_table_exists(conn)
            self._validate_postgress_columns_exists(conn)

    def _validate_postgress_table_exists(self,conn):
        if self._model_reporters:
            self._validate_reporter_table(conn = conn,table_name = "model_data")
            self._validate_reporter_table_columns(conn = conn,table_name = "model_data")
        
    def _validate_reporter_table(self,conn,table_name):
        query = f"""
            SELECT EXISTS (
            SELECT FROM information_schema.tables 
            WHERE table_schema = '{self._schema}' AND table_name = '{table_name}'
            );"""
        if not self._execute_query_with_result(conn,query):
            raise ValueError(f"Table model_data does not exist in the schema : {self._schema}")
        
    def _validate_reporter_table_columns(self,conn,table_name):
        expected_columns = set(self._model_reporters.keys())
        query = f"""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = '{self._schema}' AND table_name = '{table_name}';
        """
        
        result = self._execute_query_with_result(conn, query)
        if not result:
            raise ValueError(f"Could not retrieve columns for table {self._schema}.{table_name}")
        
        existing_columns = set(row[0] for row in result)
        missing_columns = expected_columns - existing_columns
        
        if missing_columns:
            raise ValueError(f"Missing columns in table {self._schema}.{table_name}: {missing_columns}")


    def _execute_query_with_result(self, conn, query):
        with conn.cursor() as cur:
            cur.execute(query)
            return cur.fetchall()