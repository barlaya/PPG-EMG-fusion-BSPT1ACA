import pandas as pd
import numpy as np

#for downsampling
from math import gcd
from scipy.signal import resample_poly, windows

class Subject:
    def __init__(self, name: str, df: pd.DataFrame, fs: int):
        """
        df:'time' column, 'Integrated EMG' (mV) and contains 'EMG'  (mV) of subject
        fs: sampling frequency (Hz)
        """
        self.name = name
        self.df = df.copy()
        self.fs = fs

        self.stats: dict = {}
        self.features: dict = {}
        self.active_chunks: list[pd.DataFrame] = []
        self.passive_chunks: list[pd.DataFrame] = []
        self.active_windows: list[pd.DataFrame] = []
        self.passive_windows: list[pd.DataFrame] = []
        self.sqi: dict = {}

    @classmethod

    def from_csv(cls, name: str, path: str, fs: int, target_len: int | None=None):
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

    def resample_data(self, new_frequency):
        g = gcd(self.fs, new_frequency)
        up = new_frequency // g
        down = self.fs // g

        resampled_signals = {}

        for col in [col for col in self.df.columns if col != "time"]:
            x = self.df[col].to_numpy()
            y = resample_poly(x, up=up, down=down)
            resampled_signals[col] = y

        n_new = len(next(iter(resampled_signals.values())))

        t0 = self.df["time"].iloc[0]
        resampled_signals["time"] = t0 + np.arange(n_new) / new_frequency

        self.df = pd.DataFrame(resampled_signals)
        self.fs = new_frequency

        return self

    def get_column(self, column: str) -> pd.Series:
        return self.df[column]

    def set_column(self, column: str, value):
        self.df[column] = value
        return self

    def summary(self) -> pd.DataFrame:
        self.stats["describe"] = self.df.describe()
        return self.stats["describe"]

    def add_feature(self, name: str, values):
        self.features[name] = values

    def get_feature(self, name: str):
        return self.features.get(name)

    def add_sqi(self, name: str, sqi_values):
        self.sqi[name] = sqi_values

    def chunk_subject_by_events(self, events: list[dict]):
        for event in events:
            event_type = event["phase"]

            event_start = event["start_s"]
            event_end = event["end_s"]

            query = (self.df["time"] >= event_start) & (self.df["time"] <= event_end)

            chunk = self.df[query]

            if event_type == "rest":
                self.passive_chunks.append(chunk)
            else:
                self.active_chunks.append(chunk)

        return self

    def apply_gaussian_windowing(self, columns, sigma=0.4):
        for idx in range(len(self.active_windows)):
            window_length = len(self.active_windows[idx])
            gauss_win = windows.gaussian(window_length, std=sigma * window_length).reshape(-1, 1)
            self.active_windows[idx][columns] = self.active_windows[idx][columns].to_numpy() * gauss_win

        for idx in range(len(self.passive_windows)):
            window_length = len(self.passive_windows[idx])
            gauss_win = windows.gaussian(window_length, std=sigma * window_length).reshape(-1, 1)

            self.passive_windows[idx][columns] = self.passive_windows[idx][columns].to_numpy() * gauss_win

        return self