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
        train_test_gap=0,
        pseudo_target_shift=None,
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
        self.train_test_gap = train_test_gap
        self.pseudo_target_shift = pseudo_target_shift

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

        self.n: int = self.target.shape[0]
        self.test_size: int = int(self.n * self.test_ratio)
        self.train_size: int = self.n - self.train_test_gap - self.test_size

        self.minifolds: np.ndarray = np.arange(self.train_size) // self.minifold_size
        self.folds: np.ndarray = self.minifolds % self.n_folds

    @property
    def whole_train_data(self):
        """data including train and validation in the same order"""
        return self.df.loc[: self.train_size, :]

    def get_train_data(self, fold_id: int, include_minifolds=False):
        selected_idx = self.get_train_idxs(fold_id)
        train_data = self.df.iloc[selected_idx, :].copy()
        if include_minifolds:
            train_data["minifold"] = self.minifolds[selected_idx]
        return train_data

    @property
    def whole_pseudo_target(self):
        """
        Expect mid_price that not so clear but works
        Also amount of rows filtered out by neutral_ratio should be larget then pseudo_target_shift.
        """
        assert (
            self.skip_top_thr >= self.pseudo_target_shift
        ), f"Pseudo target shouldn't be None. Please increase {self.neutral_ratio}"
        assert "mid_price" in self.df.columns, ValueError("Expected mid_price in features to calculate pseudo target")
        return -self.df["mid_price"].diff(-self.pseudo_target_shift)

    def get_train_target(self, fold_id: int):
        selected_idx = self.get_train_idxs(fold_id)
        if self.pseudo_target_shift is not None:
            return self.whole_pseudo_target.iloc[selected_idx].copy()
        return self.target.iloc[selected_idx].copy()

    def get_train_idxs(self, fold_id) -> np.ndarray:
        return np.where(self.folds != fold_id)[0]

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

    def get_test_data(self):
        test_data = self.df.iloc[-self.test_size :, :].copy()
        return test_data

    def get_test_target(self):
        return self.target[-self.test_size :]

    def get_oof_features(self, usecols: List[str]):
        pass
