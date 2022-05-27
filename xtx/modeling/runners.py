import importlib
from typing import Any, Dict, Optional

import lightgbm as lgb
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from xtx.modeling.evaluation import CrossValReport
from xtx.modeling.preprocessing import FoldPreprocessor
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

            valid_idxs = self.time_folds.get_validation_idxs(fold_id)[fold_processor.valid_idxs]
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

    def save(self):
        """save models, oof, test"""
        pass
