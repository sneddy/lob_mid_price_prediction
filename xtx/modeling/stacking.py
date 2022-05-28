import itertools
from typing import Dict, List, Optional

import numpy as np

from xtx.modeling.evaluation import get_mse_and_corr_scores
from xtx.modeling.runners import CrossValClassificationRunner, CrossValRunner


class RunnersStacking:
    def __init__(
        self,
        reg_runners: Optional[Dict[str, CrossValRunner]] = None,
        clf_runners: Optional[Dict[str, CrossValClassificationRunner]] = None,
        additional_features: Optional[np.ndarray] = None,
    ):
        self.reg_runners = {} if reg_runners is None else reg_runners
        self.clf_runners = {} if clf_runners is None else clf_runners
        self.additional_features = additional_features

        self.runner_columns = None
        self.test_target = None
        self.check_consistency()

    def _check_runner_consistensy(self, runner):
        if self.runner_columns is None:
            self.runner_columns = runner.columns
        else:
            assert (self.runner_columns == runner.columns).all(), ValueError("Inconsistent columns")
        if self.test_target is None:
            self.test_target = runner.test_target
        else:
            assert (self.test_target == runner.test_target).all(), ValueError("Inconsistent test target")

    def check_consistency(self):
        for name, runner in itertools.chain(self.reg_runners.items(), self.clf_runners.items()):
            self._check_runner_consistensy(runner)

    @property
    def oof_ensembling_features(self) -> np.ndarray:
        """Return OutOfFold features only from scores"""
        oof_scores_list = []
        for name, runner in itertools.chain(self.reg_runners.items(), self.clf_runners.items()):
            oof_scores_list.append(runner.oof_features)
        return np.vstack(oof_scores_list).T

    @property
    def oof_stacking_features(self) -> np.ndarray:
        """Return OutOfFold features from regression scores and classification probas"""
        oof_scores_list = []
        if self.reg_runners is not None:
            oof_scores_list.extend([runner.oof_features for name, runner in self.reg_runners.items()])
        if self.clf_runners is not None:
            oof_scores_list.extend([runner.oof_class_probas[:, :-1] for name, runner in self.clf_runners.items()])
        return np.column_stack(oof_scores_list)

    @property
    def test_ensembling_features(self) -> np.ndarray:
        test_scores_list = []
        if self.reg_runners is not None:
            test_scores_list.extend([runner.averaged_test for name, runner in self.reg_runners.items()])
        if self.clf_runners is not None:
            test_scores_list.extend([runner.averaged_test for name, runner in self.clf_runners.items()])
        return np.vstack(test_scores_list).T

    @property
    def test_stacking_features(self) -> np.ndarray:
        """Return OutOfFold features from regression scores and classification probas"""
        test_features_list = []
        if self.reg_runners is not None:
            test_features_list.extend([runner.averaged_test for name, runner in self.reg_runners.items()])
        if self.clf_runners is not None:
            # removed one columns to prevent linear depended columns
            test_features_list.extend([runner.test_class_probas[:, :-1] for name, runner in self.clf_runners.items()])
        return np.column_stack(test_features_list)

    @property
    def oof_target(self) -> np.ndarray:
        if self.reg_runners is not None:
            name = list(self.reg_runners.keys())[0]
            return self.reg_runners[name].oof_target
        if self.clf_runners is not None:
            name = list(self.clf_runners.keys())[0]
            return self.clf_runners[name].oof_target

    def make_ridge_stacking(self):
        from sklearn.linear_model import Ridge

        model = Ridge(alpha=0.1)
        model.fit(self.oof_stacking_features, self.oof_target)
        stacking_predicted = model.predict(self.test_stacking_features)
        get_mse_and_corr_scores(self.test_target, stacking_predicted, verbose=True, prefix="Stacking test")

    def make_oof_ensemble(self, weights: List[float] = None) -> np.ndarray:
        if weights is None:
            ensemble_prediction = self.oof_ensembling_features.mean(1)
        else:
            ensemble_prediction = np.dot(self.oof_ensembling_features, weights) / np.sum(weights)
        get_mse_and_corr_scores(self.oof_target, ensemble_prediction, verbose=True, prefix="OOF Ensemble validation")

    def make_test_ensemble(self, weights: List[float] = None) -> np.ndarray:
        if weights is None:
            ensemble_prediction = self.test_ensembling_features.mean(1)
        else:
            ensemble_prediction = np.dot(self.test_ensembling_features, weights) / np.sum(weights)
        get_mse_and_corr_scores(self.test_target, ensemble_prediction, verbose=True, prefix="OOF Ensemble test")
