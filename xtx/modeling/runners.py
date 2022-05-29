from __future__ import annotations

import importlib
import os
import pickle
import warnings
from typing import Any, Dict, Optional

import lightgbm as lgb
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from xtx.modeling.evaluation import CrossValReport
from xtx.modeling.preprocessing import DatasetPreprocessor
from xtx.modeling.time_folds import TimeFolds


class CrossValRunner:
    """
    Run cross-validation of sklearn linear model
    Store out-of-folds features, averaged test, trained models and reports
    """

    def __init__(
        self,
        time_folds: TimeFolds,
        model_module: str,
        model_cls: str,
        model_params: Optional[Dict[str, Any]] = None,
        test_eval: bool = True,
    ):
        # assert time_folds.neutral_ratio == 0, "Cant calculate oof on split with neutral"
        self.time_folds = time_folds
        self.model_module = model_module
        self.model_cls = model_cls
        self.model_params = {} if model_params is None else model_params
        self.test_eval = test_eval and self.time_folds.test_ratio > 0

        self.oof = np.full(self.time_folds.train_size, np.nan)
        self.scores = []
        self.trained_models = []
        self.preprocessor = DatasetPreprocessor(time_folds)
        self.report = CrossValReport(time_folds.n_folds, test_eval=self.test_eval)

    def init_model(self):
        """Init model object"""
        module = importlib.import_module(self.model_module)
        model_cls = getattr(module, self.model_cls)
        return model_cls(**self.model_params)

    @property
    def columns(self):
        return self.time_folds.columns

    def _fit(self, model, fold_processor):
        if model.__class__ == lgb.LGBMRegressor:
            callbacks = [lgb.log_evaluation(100)]
            # callbacks.append(lgb.early_stopping(50))
            warnings.filterwarnings("ignore", category=UserWarning)
            model.fit(
                fold_processor.train_data,
                fold_processor.train_target,
                eval_set=[
                    (fold_processor.train_data, fold_processor.train_target),
                    (fold_processor.valid_data, fold_processor.valid_target),
                ],
                callbacks=callbacks,
            )
        else:
            model.fit(fold_processor.train_data, fold_processor.train_target)
        self.trained_models.append(model)

    def _val_predict(self, model, fold_processor):
        val_predicted = model.predict(fold_processor.valid_data)
        fold_id = fold_processor.fold_id
        valid_idxs = self.time_folds.get_validation_idxs(fold_id)[fold_processor.valid_idxs]
        self.oof[valid_idxs] = val_predicted
        self.report.update_val(fold_processor.valid_target, val_predicted)

    def _test_predict(self, model, fold_processor):
        test_predicted = model.predict(fold_processor.test_data)
        self.report.update_test(fold_processor.test_target, test_predicted)

    def fit(self, verbose: bool = False):
        progress_bar = tqdm(range(self.time_folds.n_folds), desc=self.model_cls)
        for fold_id in progress_bar:
            model = self.init_model()
            fold_processor = self.preprocessor.prepare_fold(fold_id)
            self._fit(model, fold_processor)
            self._val_predict(model, fold_processor)
            if self.test_eval:
                self._test_predict(model, fold_processor)

            if self.test_eval:
                bar_description = f"{self.model_cls}: \
                    val mse: {self.report.val_mse_mean:.3f}, \
                    val corr: {self.report.val_corr_mean:.3f} \
                    test mse: {self.report.test_mse_mean:.3f}, \
                    test corr: {self.report.test_corr_mean:.3f}"
            else:
                bar_description = f"{self.model_cls}:  \
                    val mse: {self.report.val_mse_mean:.3f}, \
                    val corr: {self.report.val_corr_mean:.3f}"
            progress_bar.set_description(bar_description)
        if verbose:
            print(self.report)

    @property
    def averaged_test(self):
        return self.report.averaged_test_predicted

    @property
    def test_target(self):
        return self.report.test_target

    @property
    def oof_target(self):
        nan_idxs = np.isnan(self.oof)
        return self.time_folds.target[: self.time_folds.train_size].values[~nan_idxs]

    @property
    def oof_features(self):
        nan_idxs = np.isnan(self.oof)
        return self.oof[~nan_idxs]

    def predict(self, unseen_features: pd.DataFrame):
        """Make ensemble prediction by some unseen features"""
        processed_features, non_nan_idxs = self.preprocessor.transform(unseen_features)
        averaged_predictions = np.zeros(unseen_features.shape[0])

        for fold_id in range(self.time_folds.n_folds):
            model = self.trained_models[fold_id]
            predicted = model.predict(processed_features[non_nan_idxs, :])
            averaged_predictions[non_nan_idxs] = (
                averaged_predictions[non_nan_idxs] + predicted
            ) / self.time_folds.n_folds

        return averaged_predictions

    def save(self, runner_dir: str):
        """save models, oof, test"""
        print(f"Saving runner to {runner_dir}")
        os.makedirs(runner_dir, exist_ok=True)
        pickle.dump(self, open(os.path.join(runner_dir, "runner.pkl"), "wb"))
        self.report.save(os.path.join(runner_dir, "report.txt"))

        np.save(os.path.join(runner_dir, "oof_predictions"), self.oof_features)
        np.save(os.path.join(runner_dir, "oof_target"), self.oof_target)
        np.save(os.path.join(runner_dir, "test_predictions"), self.averaged_test)
        np.save(os.path.join(runner_dir, "test_target"), self.test_target)

    @classmethod
    def load(cls, runner_dir: str) -> CrossValRunner:
        print(f"Loading runner from {runner_dir}")
        runner_path = os.path.join(runner_dir, "runner.pkl")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            return np.load(runner_path, allow_pickle=True)

    @classmethod
    def cache_exists(cls, runner_dir) -> bool:
        runner_path = os.path.join(runner_dir, "runner.pkl")
        return os.path.exists(runner_path)


class CrossValClassificationRunner(CrossValRunner):
    def __init__(
        self,
        n_classes,
        time_folds: TimeFolds,
        model_module: str,
        model_cls: str,
        model_params: dict[str, Any] | None = None,
    ):
        super().__init__(time_folds, model_module, model_cls, model_params)
        self.n_classes = n_classes
        self.oof_probas = np.full((self.time_folds.train_size, self.n_classes), np.nan)
        self.test_class_probas = np.zeros((self.time_folds.test_size, self.n_classes))

    def _fit(self, model, fold_processor):
        model.fit(fold_processor.train_data, np.sign(fold_processor.train_target))
        self.trained_models.append(model)

    @classmethod
    def _probas2prediction(self, probas: np.ndarray) -> np.ndarray:
        return (probas[:, 2] - probas[:, 0]) * (1 - probas[:, 1])

    def _val_predict(self, model, fold_processor):
        val_probas = model.predict_proba(fold_processor.valid_data)
        val_predicted = self._probas2prediction(val_probas)

        fold_id = fold_processor.fold_id
        valid_idxs = self.time_folds.get_validation_idxs(fold_id)[fold_processor.valid_idxs]
        self.oof[valid_idxs] = val_predicted
        self.oof_probas[valid_idxs] = val_probas
        self.report.update_val(fold_processor.valid_target, val_predicted)

    def _test_predict(self, model, fold_processor):
        test_probas = model.predict_proba(fold_processor.test_data)
        self.test_class_probas += test_probas / self.time_folds.n_folds
        test_predicted = self._probas2prediction(test_probas)
        self.report.update_test(fold_processor.test_target, test_predicted)

    @property
    def oof_class_probas(self):
        nan_idxs = np.isnan(self.oof)
        return self.oof_probas[~nan_idxs]


class Stacking:
    def __init__(self):
        pass
