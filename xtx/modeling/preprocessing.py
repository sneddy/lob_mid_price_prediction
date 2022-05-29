from typing import Optional

import numpy as np
import sklearn.preprocessing
from sklearn.decomposition import PCA

from xtx.modeling.time_folds import TimeFolds


class FoldPreprocessor:
    """
    Store:
        - train/valid/test data after postprocessing without NaN values
        - train/valid target corresponding to postprocessed data
        - test target with the same size as in input
        - train/valid idxs - indexes of the not Nan values inside coresponding dataset.
            Always indexed from 0 (not global index)

    """

    def __init__(self, time_folds: TimeFolds, fold_id: int, pca_components: Optional[int] = None):
        self.time_folds = time_folds
        self.fold_id = fold_id
        self.pca_components = pca_components

        # self.scaler = sklearn.preprocessing.RobustScaler()
        self.scaler = sklearn.preprocessing.StandardScaler()
        if self.pca_components is not None:
            self.pca = PCA(n_components=self.pca_components)

        self.train_data, self.train_target, self.train_idxs = self._init_train()
        self.valid_data, self.valid_target, self.valid_idxs = self._init_valid()
        self.test_data, self.test_target = self._init_test()

        # self.train_data = self.pca.fit_transform(self.train_data)
        # self.valid_data = self.pca.transform(self.valid_data)
        # self.test_data = self.pca.transform(self.test_data)

    def _init_train(self):
        train_data = self.time_folds.get_train_data(self.fold_id)
        train_target = self.time_folds.get_train_target(self.fold_id)
        train_idxs = np.where(train_data.notnull().all(1))

        train_data.dropna(inplace=True)
        self.scaler.fit(train_data)
        train_data = self.scaler.transform(train_data)
        if self.pca_components is not None:
            train_data = self.pca.fit_transform(train_data)
        train_target = train_target.values[train_idxs]

        return train_data, train_target, train_idxs

    def _init_valid(self):
        valid_data = self.time_folds.get_valid_data(self.fold_id)
        valid_target = self.time_folds.get_valid_target(self.fold_id)
        valid_idxs = np.where(valid_data.notnull().all(1))

        valid_data.dropna(inplace=True)
        valid_target = valid_target.values[valid_idxs]

        valid_data = self.scaler.transform(valid_data)
        if self.pca_components is not None:
            valid_data = self.pca.transform(valid_data)

        return valid_data, valid_target, valid_idxs

    def _init_test(self):
        test_data = self.time_folds.get_test_data()
        test_target = self.time_folds.get_test_target()
        test_data.fillna(0, inplace=True)
        test_data = self.scaler.transform(test_data)
        if self.pca_components is not None:
            test_data = self.pca.transform(test_data)

        return test_data, test_target
