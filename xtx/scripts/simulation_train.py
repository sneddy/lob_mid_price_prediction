import os
from typing import Optional

import numpy as np
import pandas as pd

import xtx.utils.dev_utils as dev_utils
from xtx.factory import build_runners, load_features
from xtx.modeling.stacking import RunnersStacking
from xtx.modeling.time_folds import TimeFolds

pd.set_option("display.max_columns", 100)

logger = dev_utils.init_logger("logging/train.log")


def train(data_path: str, features_path: Optional[str] = None):
    merged_features, target = load_features(data_path, features_path, use_cache=True)

    time_folds = TimeFolds(
        n_folds=5,
        minifold_size=60000,
        neutral_ratio=0.05,
        test_ratio=0,
        train_test_gap=10000,
    )

    time_folds.fit(merged_features, target)

    model_zoo = dev_utils.load_yaml("configs/models_zoo.yaml")["train_zoo"]
    stacking_model_zoo = dev_utils.load_yaml("configs/models_zoo.yaml")["stacking_zoo"]

    use_regression_models = [
        "default_ridge",
        "default_dart",
        "default_lasso",
        # "default_lgbm",
        # "default_sgd",
        # "wild_lgbm",
        # "bayesian_ridge",
    ]
    use_classification_models = [
        # "default_logreg"
    ]
    clf_model_configs = {model_name: model_zoo[model_name] for model_name in use_classification_models}
    reg_model_configs = {model_name: model_zoo[model_name] for model_name in use_regression_models}

    clf_runners = build_runners(time_folds, clf_model_configs, runners_dir="runners/5_folds_sim", regression=False)
    reg_runners = build_runners(time_folds, reg_model_configs, runners_dir="runners/5_folds_sim", regression=True)

    runners_stacking = RunnersStacking(
        reg_runners=reg_runners,
        clf_runners=clf_runners,
        test_eval=False,
        **stacking_model_zoo["stacking_ridge"],
    )
    runners_stacking.make_oof_ensemble()
    # runners_stacking.make_test_ensemble()
    runners_stacking.fit_stacking()
    return runners_stacking


def test(data_path: str, runners_stacking: RunnersStacking, predictions_dir: str, features_path: Optional[str] = None):
    merged_features, _ = load_features(data_path, features_path, use_cache=True)
    predicted_by_ensemble = runners_stacking.predict_by_ensemble(merged_features)
    predicted_by_stacking = runners_stacking.predict_by_stacking(merged_features)

    os.makedirs(predictions_dir, exist_ok=True)
    np.save(os.path.join(predictions_dir, "ensemble"), predicted_by_ensemble)
    np.save(os.path.join(predictions_dir, "stacking"), predicted_by_stacking)
    return predicted_by_ensemble


if __name__ == "__main__":
    DATA_PATH = "data/simulation/train.csv"
    features_path = "data/simulation_features.pkl"
    runners_stacking = train(DATA_PATH, features_path)

    prediction_path = "predictions/debug"
    test_data_path = "data/simulation/test.csv"
    test_features_path = "data/simulation_test_features.pkl"
    test(test_data_path, runners_stacking, prediction_path, test_features_path)
