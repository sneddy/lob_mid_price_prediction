import os
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from contexttimer import Timer

import xtx.utils.dev_utils as dev_utils
from xtx.features.feature_extractor import FeatureExtractor
from xtx.modeling.runners import CrossValClassificationRunner, CrossValRunner
from xtx.modeling.stacking import RunnersStacking
from xtx.modeling.time_folds import TimeFolds

pd.set_option("display.max_columns", 100)

logger = dev_utils.init_logger("logging/train.log")


def load_features(data_path: str, features_path: str, fake_target: int = None, use_cache=False):
    feature_extractor = FeatureExtractor(data_path)
    data = feature_extractor.data

    if use_cache and features_path is not None and os.path.exists(features_path):
        logger.info(f"Loading cached features from {features_path}")
        os.makedirs(os.path.dirname(features_path), exist_ok=True)
        merged_features = pd.read_pickle(features_path)
        # return merged_features, data.y
    else:
        with Timer() as base_features_time:
            base_features = feature_extractor.get_base_features()
        logger.info(f"Base features extraction time: {base_features_time.elapsed:.1f} sec")
        logger.info(f"Extracted base features: \n{base_features.columns.tolist()}")

        with Timer() as topk_features_time:
            topk_features = feature_extractor.get_topk_features()
        logger.info(f"Topk features extraction time: {topk_features_time.elapsed:.1f} sec")
        logger.info(f"Extracted topk features: \n{topk_features.columns.tolist()}")

        with Timer() as time_features_time:
            time_features = feature_extractor.get_time_base_features(base_features)
        logger.info(f"Topk features extraction time: {time_features_time.elapsed:.1f} sec")
        logger.info(f"Extracted topk features: \n{time_features.columns.tolist()}")

        merged_features = pd.concat((base_features, topk_features, time_features), axis=1)

        logger.info(f"cache features into {features_path}")
        merged_features.to_pickle(features_path)

    target = data.y if "y" in data.columns else None
    if fake_target is not None:
        logger.info(f"Making fake target: mid_price_diff_{fake_target}")
        target = feature_extractor.get_fake_target(fake_target).iloc[fake_target:]
        merged_features = merged_features.iloc[fake_target:, :]
    return merged_features, target


def build_runners(
    time_folds,
    model_congigs: Dict[str, Dict[str, Any]],
    runners_dir: str,
    use_cache: bool = True,
    regression: bool = True,
) -> Dict[str, CrossValRunner]:
    runners = {}

    for name, model_config in model_congigs.items():
        cached_runner_dir = os.path.join(runners_dir, name)
        if use_cache and CrossValRunner.cache_exists(cached_runner_dir):
            runners[name] = CrossValRunner.load(cached_runner_dir)
            logger.info(runners[name].report)
        else:
            if regression:
                current_runner = CrossValRunner(time_folds, **model_config)
            else:
                current_runner = CrossValClassificationRunner(n_classes=3, time_folds=time_folds, **model_config)
            current_runner.fit(verbose=True)
            runners[name] = current_runner
            current_runner.save(cached_runner_dir)
    return runners


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
        # reg_runners=reg_runners, clf_runners=clf_runners, **stacking_model_zoo["stacking_elastic"]
        reg_runners=reg_runners,
        clf_runners=clf_runners,
        test_eval=False,
        **stacking_model_zoo["stacking_ridge"],
    )
    runners_stacking.make_oof_ensemble()
    # runners_stacking.make_test_ensemble()
    runners_stacking.fit_stacking()
    return runners_stacking


def test(data_path: str, runners_stacking: RunnersStacking, prediction_path: str, features_path: Optional[str] = None):
    merged_features, _ = load_features(data_path, features_path, use_cache=True)
    predicted = runners_stacking.predict_by_ensemble(merged_features)
    np.save(prediction_path, predicted)
    return predicted


if __name__ == "__main__":
    DATA_PATH = "data/simulation/train.csv"
    features_path = "data/simulation_features.pkl"
    runners_stacking = train(DATA_PATH, features_path)

    prediction_path = "predictions/debug_ensemble"
    test_data_path = "data/simulation/test.csv"
    test_features_path = "data/simulation_test_features.pkl"
    test(test_data_path, runners_stacking, prediction_path, test_features_path)
