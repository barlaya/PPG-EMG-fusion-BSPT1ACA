import pandas as pd
import numpy as np


class Subject:
    def __init__(self, name: str, df: pd.DataFrame, fs: float):
        """
        df:'time' column, contains 'EMG' of subject
        fs: sampling frequency (Hz)
        """
        self.name = name
        self.df = df.copy()
        self.fs = fs

        self.stats: dict = {}
        self.features: dict = {}
        self.chunks: list[pd.DataFrame] = []
        self.sqi: dict = {}

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


class ManagerClass:
    def __init__(self):
        """
        dict of all Subjects, to perform same methods on all
        """
        self.patient_objects: dict[str, Subject] = {}

    def add_subject(self, patient: Subject):
        self.patient_objects[patient.name] = patient

    def get_subject(self, name: str) -> Subject | None:
        return self.patient_objects.get(name)

    def all_names(self):
        return list(self.patient_objects.keys())

    def apply(self, func):
        for patient in self.patient_objects.values():
            func(patient)
