import pandas as pd
import numpy as np

class Subject:
    def __init__(self, name: str, df: pd.DataFrame, fs: float):
        """
        df:'time' column, 'Integrated EMG' (mV) and contains 'EMG'  (mV) of subject
        fs: sampling frequency (Hz)
        """
        self.name = name
        self.df = df.copy()
        self.fs = fs

        self.stats: dict = {}
        self.features: dict = {}
        self.chunks: list[pd.DataFrame] = []
        self.sqi: dict = {}

    @classmethod

    def from_csv(cls, name: str, path: str, fs: float, target_len: int | None=None):
        df = pd.read_csv(path, delimiter=";", encoding="utf-8")
        df = df.replace(",", ".", regex=True)

        for col in df.columns:
            if col.lower() != "time":
                df[col] = pd.to_numeric(df[col], errors="coerce")
        if target_len is not None:
            current_len=len(df)
            if current_len < target_len:
                pad_size = target_len - current_len
                pad_dict = {}
                for col in df.columns:
                    if col.lower() != "time":
                        continue
                    pad_dict[col] = np.zeros(pad_size)
                pad_df=pd.DataFrame(pad_dict)
                df = pd.concat([df, pad_df], ignore_index=True)
            elif current_len > target_len:
                df = df.iloc[:target_len].reset_index(drop=True)

        if "time" not in df.columns:
            n = len(df)
            df["time"] = np.arange(n)/fs
        cols = list(df.columns)

        if "time" in cols:
            cols.insert(0, cols.pop(cols.index("time")))
            df = df[cols]

        return cls(name=name, df=df, fs=fs)

    #def load_csv(self):
    #    return self.df

    def get_column(self, column: str) -> pd.Series:
        return self.df[column]

    def set_column(self, column: str, value):
        self.df[column] = value

    def summary(self) -> pd.DataFrame:
        self.stats["describe"] = self.df.describe()
        return self.stats["describe"]

    def add_feature(self, name: str, values):
        self.features[name] = values

    def get_feature(self, name: str):
        return self.features.get(name)

    def add_sqi(self, name: str, sqi_values):
        self.sqi[name] = sqi_values

    def add_chunk(self, chunk_df: pd.DataFrame):
        self.chunks.append(chunk_df)
