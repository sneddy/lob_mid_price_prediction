import importlib
import itertools
from functools import cached_property
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from xtx.modeling.evaluation import get_mse_and_corr_scores
from xtx.modeling.runners import CrossValClassificationRunner, CrossValRunner


class RunnersStacking:
    def __init__(
        self,
        model_module: str,
        model_cls: str,
        model_params: Optional[Dict[str, Any]] = None,
        reg_runners: Optional[Dict[str, CrossValRunner]] = None,
        clf_runners: Optional[Dict[str, CrossValClassificationRunner]] = None,
        additional_features: Optional[np.ndarray] = None,
        test_eval: bool = True,
    ):
        self.test_eval = test_eval
        self.model_module = model_module
        self.model_cls = model_cls
        self.model_params = {} if model_params is None else model_params

        self.reg_runners = {} if reg_runners is None else reg_runners
        self.clf_runners = {} if clf_runners is None else clf_runners
        self.additional_features = additional_features

        self.runner_columns = None
        self.test_target = None
        self.check_consistency()
        self.stacking_model = self.init_model()

    @property
    def runner_names(self):
        return sorted(list(self.reg_runners) + list(self.clf_runners))

    @cached_property
    def name2runner(self) -> Dict[str, CrossValRunner]:
        mapping = {}
        for name, runner in itertools.chain(self.reg_runners.items(), self.clf_runners.items()):
            mapping[name] = runner
        return mapping

    def init_model(self):
        """Init model object"""
        module = importlib.import_module(self.model_module)
        model_cls = getattr(module, self.model_cls)
        return model_cls(**self.model_params)

    def _check_runner_consistensy(self, runner):
        assert len(self.reg_runners) + len(self.clf_runners) == len(self.runner_names), ValueError(
            "Model names conflict"
        )
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

    def fit_stacking(self):
        self.stacking_model.fit(self.oof_stacking_features, self.oof_target)
        if self.test_eval:
            stacking_predicted = self.stacking_model.predict(self.test_stacking_features)
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

    def predict_by_ensemble(self, unseen_features: pd.DataFrame, weights: List[float] = None) -> np.ndarray:
        predictions = []
        for name, runner in itertools.chain(self.reg_runners.items(), self.clf_runners.items()):
            predictions.append(runner.predict(unseen_features))
        predictions = np.array(predictions).T
        if weights is None:
            ensemble_prediction = predictions.mean(1)
        else:
            ensemble_prediction = np.dot(predictions, weights) / np.sum(weights)
        return ensemble_prediction

    def predict_by_stacking(self, unseen_features: pd.DataFrame):
        stacking_features = []
        for name, runner in itertools.chain(self.reg_runners.items(), self.clf_runners.items()):
            if name in self.reg_runners:
                predictions = runner.predict(unseen_features).reshape(-1, 1)
                print(name, predictions.shape)
                stacking_features.append(predictions)
            else:
                # removed one columns to prevent linear depended columns
                predictions = runner.predict_proba(unseen_features)[:, :-1]
                print(name, predictions.shape)
                stacking_features.append(predictions)
        stacking_features = np.hstack(stacking_features)

        return self.stacking_model.predict(stacking_features)
