"""
DATA COLLECTOR.

and.
"""

from mesa_frames.abstract.datacollector import AbstractDataCollector
import polars as pl
import boto3
from urllib.parse import urlparse
import tempfile
import psycopg2

class DataCollector(AbstractDataCollector):
    def __init__(
        self,
        model,
        model_reporters=None,
        agent_reporters=None,
        trigger=None,
        reset_memory=True,
        storage="memory:",
    ):
        super().__init__(
            model=model,
            model_reporters=model_reporters,
            agent_reporters=agent_reporters,
            trigger=trigger,
            reset_memory=reset_memory,
            storage=storage,
        )


    def _collect(self):
        model_data_dict ={}
        model_data_dict['step'] = self._model.steps
        model_data_dict['seed'] = self.seed
        for column_name,reporter in self._model_reporters.items():
            model_data_dict[column_name] = reporter(self._model)
        model_lazy_frame = pl.LazyFrame([model_data_dict])
        self._frames.append(("model",self._model.steps, model_lazy_frame))

    def _flush(self):
        kind_to_frames = {"model": [], "agent": []}
        for kind,step,lf in self._frames:
            kind_to_frames[kind].append((step,lf.collect()))

        if self._storage_uri.startswith("csv:"):
            base = self._storage_uri[4:]
            for kind, dfs in kind_to_frames.items():
                for i, df in dfs:
                    df.write_csv(f"{base}/{kind}_step{i}.csv")

        elif self._storage_uri.startswith("parquet:"):
            base = self._storage_uri[8:]
            for kind, dfs in kind_to_frames.items():
                for i, df in dfs:
                    df.write_parquet(f"{base}/{kind}_step{i}.parquet")

        elif self._storage_uri.startswith("csvs3:") or self._storage_uri.startswith("parquets3:"):
            format_ = "csv" if self._storage_uri.startswith("csvs3:") else "parquet"
            s3_prefix = self._storage_uri.split(":", 1)[1]
            s3 = boto3.client("s3")
            parsed = urlparse(s3_prefix)
            bucket = parsed.netloc
            prefix = parsed.path.lstrip("/")

            for kind, dfs in kind_to_frames.items():
                for i, df in dfs:
                    with tempfile.NamedTemporaryFile(suffix=f".{format_}") as tmp:
                        if format_ == "csv":
                            df.write_csv(tmp.name)
                        else:
                            df.write_parquet(tmp.name)
                        key = f"{prefix}/{kind}_step{i}.{format_}"
                        s3.upload_file(tmp.name, bucket, key)

        elif self._storage_uri.startswith("postgres:"):
            parsed = urlparse(self._storage_uri)
            conn = psycopg2.connect(
                dbname=parsed.path[1:],
                user=parsed.username,
                password=parsed.password,
                host=parsed.hostname,
                port=parsed.port
            )
            cur = conn.cursor()
            # create table if it doesnt exist
            # check every column is same
            for kind, dfs in kind_to_frames.items():
                table = f"{kind}_data"
                for step,df in dfs:
                    cols = df.columns
                    values = [tuple(row) for row in df.rows()]
                    placeholders = ", ".join(["%s"] * len(cols))
                    columns = ", ".join(cols)
                    cur.executemany(
                        f"INSERT INTO {table} ({columns}) VALUES ({placeholders})",
                        values
                    )
            conn.commit()
            cur.close()
            conn.close()