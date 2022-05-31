from __future__ import annotations

import numpy as np
import pandas as pd

# import sklearn.preprocessing
from sklearn.decomposition import PCA

from xtx.modeling.time_folds import TimeFolds


class DatasetPreprocessor:
    def __init__(
        self,
        time_folds: TimeFolds(),
        pca_components: int | None = None,
    ):
        self.time_folds = time_folds
        self.pca_components = pca_components

        self.scaler = self.time_folds.scaler
        # self.scaler = sklearn.preprocessing.RobustScaler()
        # scaled_whole_data = self.scaler.fit_transform(time_folds.whole_train_data.dropna())

        # if self.pca_components is not None:
        #     self.pca = PCA(n_components=self.pca_components)
        #     self.pca.fit(scaled_whole_data)

    def prepare_fold(self, fold_id) -> FoldPreprocessor:
        # pca = self.pca if self.pca_components is not None else None
        return FoldPreprocessor(self.time_folds, fold_id, self.scaler, pca=None)

    def transform(self, data: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Preprocessing of unseen data
        Args:
            data (pd.DataFrame): features to preprocess
        Returns:
           Tuple[np.ndarray, np.ndarray]: processed data and not nan indexes
        """
        not_nan_idxs = np.where(data.notnull().all(1))[0]
        processed = self.scaler.transform(data.fillna(0))
        # if self.pca_components is not None:
        #     processed = self.pca.transform(processed)
        return processed, not_nan_idxs


class FoldPreprocessor:
    """
    Store:
        - train/valid/test data after postprocessing without NaN values
        - train/valid target corresponding to postprocessed data
        - test target with the same size as in input
        - train/valid idxs - indexes of the not Nan values inside coresponding dataset.
            Always indexed from 0 (not global index)

    """

    def __init__(
        self,
        time_folds: TimeFolds,
        fold_id: int,
        scaler,
        pca: PCA | None = None,
    ):
        self.time_folds = time_folds
        self.fold_id = fold_id
        self.scaler = scaler
        self.pca = pca

        self.train_data, self.train_target, self.train_idxs = self._init_train()
        self.valid_data, self.valid_target, self.valid_idxs = self._init_valid()
        if self.time_folds.test_ratio > 0:
            self.test_data, self.test_target = self._init_test()

    def _init_train(self):
        train_data = self.time_folds.get_train_data(self.fold_id)
        train_target = self.time_folds.get_train_target(self.fold_id)
        train_idxs = np.where(train_data.notnull().all(1))[0]
        # train_idxs = [idx for idx in train_idxs if idx > 87]

        train_data.dropna(inplace=True)
        train_data = self.scaler.transform(train_data)
        if self.pca is not None:
            train_data = self.pca.transform(train_data)
        train_target = train_target.values[train_idxs]

        return train_data, train_target, train_idxs

    def _init_valid(self):
        valid_data = self.time_folds.get_valid_data(self.fold_id)
        valid_target = self.time_folds.get_valid_target(self.fold_id)
        valid_idxs = np.where(valid_data.notnull().all(1))[0]
        # valid_idxs = [idx for idx in valid_idxs if idx > 87]

        valid_data.dropna(inplace=True)
        valid_target = valid_target.values[valid_idxs]

        valid_data = self.scaler.transform(valid_data)
        if self.pca is not None:
            valid_data = self.pca.transform(valid_data)

        return valid_data, valid_target, valid_idxs

    def _init_test(self):
        test_data = self.time_folds.get_test_data()
        test_target = self.time_folds.get_test_target()
        test_data.fillna(0, inplace=True)
        test_data = self.scaler.transform(test_data)
        if self.pca is not None:
            test_data = self.pca.transform(test_data)

        return test_data, test_target
