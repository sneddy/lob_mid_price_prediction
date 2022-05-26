import importlib
from typing import Any, Dict, List, Optional, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from tqdm.auto import tqdm

from xtx.preprocessing import FoldPreprocessor
from xtx.time_folds import TimeFolds


class CrossValReport:
    """Calculate and store relevant metrics and statistics"""

    def __init__(self, n_folds: int):
        self.n_folds = n_folds
        self.test_target: np.ndarray = None
        self.averaged_test_predicted: np.ndarray = None
        self.val_mse_scores: List[float] = []
        self.val_corr_scores: List[float] = []
        self.test_mse_scores: List[float] = []
        self.test_corr_scores: List[float] = []

    def update_val(self, val_predicted: np.ndarray, val_target: np.ndarray):
        corr_score = np.corrcoef(val_predicted, val_target)[0, 1]
        mse_score = mean_squared_error(val_predicted, val_target)
        self.val_corr_scores.append(corr_score)
        self.val_mse_scores.append(mse_score)

    def update_test(self, test_predicted: np.ndarray, test_target: np.ndarray):
        if self.test_target is None:
            self.test_target = test_target
        else:
            assert (self.test_target == test_target).all(), ValueError(
                "Got different test target by folds"
            )
        if self.averaged_test_predicted is None:
            self.averaged_test_predicted = test_predicted / self.n_folds
        else:
            self.averaged_test_predicted += test_predicted / self.n_folds

        corr_score = np.corrcoef(test_predicted, test_target)[0, 1]
        mse_score = mean_squared_error(test_predicted, test_target)
        self.test_corr_scores.append(corr_score)
        self.test_mse_scores.append(mse_score)

    @property
    def val_mse_mean(self) -> float:
        return np.mean(self.val_mse_scores)

    @property
    def val_corr_mean(self) -> float:
        return np.mean(self.val_corr_scores)

    @property
    def test_mse_mean(self) -> float:
        return np.mean(self.test_mse_scores)

    @property
    def test_corr_mean(self) -> float:
        return np.mean(self.test_corr_scores)

    @property
    def val_mse_conf(self) -> Tuple[float, float]:
        sigma = np.std(self.val_mse_scores)
        return self.val_mse_mean - 2 * sigma, self.val_mse_mean + 2 * sigma

    @property
    def val_corr_conf(self) -> Tuple[float, float]:
        sigma = np.std(self.val_corr_scores)
        return self.val_corr_mean - 2 * sigma, self.val_corr_mean + 2 * sigma

    @property
    def test_mse_conf(self) -> Tuple[float, float]:
        sigma = np.std(self.test_mse_scores)
        return self.test_mse_mean - 2 * sigma, self.test_mse_mean + 2 * sigma

    @property
    def test_corr_conf(self) -> Tuple[float, float]:
        sigma = np.std(self.test_corr_scores)
        return self.test_corr_mean - 2 * sigma, self.test_corr_mean + 2 * sigma

    @property
    def test_averaged_mse(self) -> float:
        return mean_squared_error(self.averaged_test_predicted, self.test_target)

    @property
    def test_averaged_corr(self) -> float:
        return np.corrcoef(self.averaged_test_predicted, self.test_target)[0, 1]

    @property
    def folds_report(self) -> pd.DataFrame:
        df_dict = {
            ("val", "mse"): self.val_mse_scores,
            ("val", "corr"): self.val_corr_scores,
            ("test", "mse"): self.test_mse_scores,
            ("test", "corr"): self.test_corr_scores,
        }
        df = pd.DataFrame.from_dict(df_dict).T.reset_index()
        df.columns = ["dataset", "metric_name"] + [
            f"fold_{idx}" for idx in range(self.n_folds)
        ]
        return df.round(3)

    def __repr__(self) -> str:
        aggregated_repr = f"""
        \t\tVal  corr averaged: {self.val_corr_mean:.3f}
        \t\tVal   MSE averaged: {self.val_mse_mean:.3f}
        \t\tTest corr averaged: {self.test_corr_mean:.3f}
        \t\tTest  MSE averaged: {self.test_mse_mean:.3f}
        ------------------------------------------------------------------
        \t\tAveraged test  MSE: {self.test_averaged_mse:.3f}
        \t\tAveraged test corr: {self.test_averaged_corr:.3f}
        """
        return self.folds_report.to_markdown() + aggregated_repr


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
    ):
        # assert time_folds.neutral_ratio == 0, "Cant calculate oof on split with neutral"
        self.time_folds = time_folds
        self.model_module = model_module
        self.model_cls = model_cls
        self.model_params = {} if model_params is None else model_params

        self.oof = np.full(self.time_folds.train_size, np.nan)
        self.averaged_test = None
        self.scores = []
        self.trained_models = []
        self.report = CrossValReport(time_folds.n_folds)

    def init_model(self):
        """Init model object"""
        module = importlib.import_module(self.model_module)
        model_cls = getattr(module, self.model_cls)
        return model_cls(**self.model_params)

    def fit(self, verbose: bool = False):
        progress_bar = tqdm(range(self.time_folds.n_folds))
        for fold_id in progress_bar:
            model = self.init_model()
            fold_processor = FoldPreprocessor(self.time_folds, fold_id)
            if model.__class__ == lgb.LGBMRegressor:
                model.fit(
                    fold_processor.train_data,
                    fold_processor.train_target,
                    eval_set=[(fold_processor.valid_data, fold_processor.valid_target)],
                    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)],
                )
            else:
                model.fit(fold_processor.train_data, fold_processor.train_target)

            val_predicted = model.predict(fold_processor.valid_data)

            valid_idxs = self.time_folds.get_validation_idxs(fold_id)[
                fold_processor.valid_idxs
            ]
            self.oof[valid_idxs] = val_predicted
            self.report.update_val(val_predicted, fold_processor.valid_target)
            test_predicted = model.predict(fold_processor.test_data)
            self.report.update_test(test_predicted, fold_processor.test_target)
            self.trained_models.append(model)
            progress_bar.set_description(
                f"Val_mse: {self.report.val_mse_mean:.3f}, \
                 val_corr: {self.report.val_corr_mean:.3f}"
            )
        self.averaged_test = self.report.averaged_test_predicted
        if verbose:
            print(self.report)

    def predict(self, unseen_features: pd.DataFrame):
        raise NotImplementedError()


def ridge_eval(time_folds, fold_id, ridge_alpha=100, verbose=True):
    fold_processor = FoldPreprocessor(time_folds, fold_id)

    model = Ridge(alpha=ridge_alpha)

    model.fit(fold_processor.train_data, fold_processor.train_target)
    predicted = model.predict(fold_processor.valid_data)
    corr_score = np.corrcoef(predicted, fold_processor.valid_target)[0, 1]
    mse_score = mean_squared_error(predicted, fold_processor.valid_target)

    test_predicted = model.predict(fold_processor.test_data)
    test_corr_score = np.corrcoef(test_predicted, fold_processor.test_target)[0, 1]
    test_mse_score = mean_squared_error(test_predicted, fold_processor.test_target)
    if verbose:
        print(f"Val correlation: {corr_score:.4f}")
        print(f"Val MSE: {mse_score:.4f}")
        print(f"Test correlation: {test_corr_score:.4f}")
        print(f"Test MSE: {test_mse_score:.4f}")
    return mse_score, test_mse_score
