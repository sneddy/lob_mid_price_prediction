from typing import List

import numpy as np
import pandas as pd


class TimeFolds:
    def __init__(
        self,
        n_folds=5,
        minifold_size=1000,
        neutral_ratio=0.2,
        test_ratio=0.2,
        test_neutral_ratio=0.1,
    ):
        """
        Example with default parameters:
        n_folds=5, minifold_size=1000, neutral_ratio=0.2

        0001-1000 - train1
        1000-2000 - train2
        2000-3000 - train3
        3000-4000 - train4
        -----------------------
        4000-4200 - empty zone
        4200-4800 - validation
        4800-5000 - empty zone
        -----------------------
        """
        self.n_folds = n_folds
        self.minifold_size = minifold_size
        self.neutral_ratio = neutral_ratio
        self.test_ratio = test_ratio
        self.test_neutral_ratio = test_neutral_ratio

        self.skip_bot_thr = int(self.neutral_ratio * self.minifold_size)
        self.skip_top_thr = self.minifold_size - self.skip_bot_thr

        self.df = None
        self.target = None
        self.train_size = 0
        self.test_size = 0
        self.n = 0
        self.folds = None
        self.minifolds = None

    def __len__(self):
        if self.df is None:
            return 0
        return self.df.shape[0]

    @property
    def columns(self):
        if self.df is None:
            return
        return self.df.columns

    def fit(self, df: pd.DataFrame, target: pd.Series):
        self.df = df
        self.target = target

        self.n = self.target.shape[0]

        self.test_size = int(self.n * self.test_ratio)
        self.train_size = self.n - self.test_size

        self.minifolds = np.arange(self.n) // self.minifold_size
        self.folds = np.arange(self.train_size) // self.minifold_size % self.n_folds

    @property
    def whole_train(self):
        return self.df.loc[: self.train_size, :]

    def get_train_data(self, fold_id: int, include_minifolds=False):
        selected_idx = np.where(self.folds != fold_id)[0]
        train_data = self.df.iloc[selected_idx, :].copy()
        if include_minifolds:
            train_data["minifold"] = self.minifolds[selected_idx]
        return train_data

    def get_train_target(self, fold_id: int):
        selected_idx = np.where(self.folds != fold_id)[0]
        return self.target.iloc[selected_idx].copy()

    def get_validation_idxs(self, fold_id) -> np.ndarray:
        selected_idx = np.where(self.folds == fold_id)[0]
        minifold_idx = selected_idx % self.minifold_size
        selected_pos_in_idx = np.where((self.skip_bot_thr < minifold_idx) & (minifold_idx < self.skip_top_thr))[0]
        return selected_idx[selected_pos_in_idx]

    def get_valid_data(self, fold_id: int, include_minifolds=False):
        selected_idx = self.get_validation_idxs(fold_id)
        valid_data = self.df.iloc[selected_idx, :].copy()
        if include_minifolds:
            valid_data["minifold"] = self.minifolds[selected_idx]
        return valid_data

    def get_valid_target(self, fold_id: int):
        selected_idx = self.get_validation_idxs(fold_id)
        return self.target.iloc[selected_idx].copy()

    @property
    def test_size_without_neutral(self):
        return self.test_size - int(self.test_size * self.test_neutral_ratio)

    def get_test_data(self, include_minifolds=False):
        test_data = self.df.iloc[-self.test_size_without_neutral :, :].copy()

        if include_minifolds:
            test_data["minifold"] = self.minifolds[-self.test_size :]
        return test_data

    def get_test_target(self):
        return self.target[-self.test_size_without_neutral :]

    def get_oof_features(self, usecols: List[str]):
        pass
