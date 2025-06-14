"""
DATA COLLECTOR.

and.
"""

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
            storage=storage,  # literal won't work
        )
        self._writers = {
            "csv": self.write_csv_local,
            "parquet": self.write_parquet_local,
            "S3-csv": self.write_csv_s3,
            # "S3-parquet": self.write_parquet_s3,
            "postgres": self.write_postgres,
        }

    def _collect(self):
        model_data_dict = {}
        model_data_dict["step"] = self._model._steps
        model_data_dict["seed"] = str(self.seed)
        for column_name, reporter in self._model_reporters.items():
            model_data_dict[column_name] = reporter(self._model)
        model_lazy_frame = pl.LazyFrame([model_data_dict])
        self._frames.append(("model", str(self._model._steps), model_lazy_frame))
        if self._agent_reporters:
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
        schema = self._storage_uri.split(":", 1)[0]
        uri = self._storage_uri.split(":", 1)[1]

        if schema not in self._writers:
            raise ValueError("Unknown writer")

        self._writers[schema](uri)

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
        parsed = urlparse(f"//{uri}")
        conn = psycopg2.connect(
            dbname=parsed.path[1:],
            user=parsed.username,
            password=parsed.password,
            host=parsed.hostname,
            port=parsed.port,
        )
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
