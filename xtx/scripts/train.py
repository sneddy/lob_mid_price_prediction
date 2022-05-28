import os
from typing import Any, Dict, Optional

import pandas as pd

import xtx.utils.dev_utils as dev_utils
from xtx.features.feature_extractor import FeatureExtractor
from xtx.modeling.runners import CrossValClassificationRunner, CrossValRunner
from xtx.modeling.stacking import RunnersStacking
from xtx.modeling.time_folds import TimeFolds

pd.set_option("display.max_columns", 100)

logger = dev_utils.init_logger("logging/train.log")


def load_features(data_path: str, features_path: str, use_cache=False):
    feature_extractor = FeatureExtractor(data_path)
    data = feature_extractor.data

    if use_cache and features_path is not None and os.path.exists(features_path):
        logger.info(f"Loading cached features from {features_path}")
        os.makedirs(os.path.dirname(features_path), exist_ok=True)
        merged_features = pd.read_pickle(features_path)
        return merged_features, data.y

    base_features = feature_extractor.get_base_features()
    logger.info(f"Extracted base features for train: \n{base_features.columns.tolist()}")

    flatten_usecols = [
        "bid_flatten_mean_5",
        "wap_flatten_5",
        # "ask_flatten_iqr_15",
        "bid_flatten_mean_50",
        "ask_flatten_skew_50",
        "bid_flatten_kurtosis_50",
        "ask_flatten_mean_50",
        "ask_flatten_iqr_50",
        # "ask_flatten_mean_100",
        # "ask_flatten_iqr_100",
    ]
    flatten_useranks = sorted({int(col.split("_")[-1]) for col in flatten_usecols})
    flatten_features = feature_extractor.load_flatten_features(
        features_directory="artefacts", useranks=flatten_useranks, usecols=flatten_usecols
    )
    logger.info(f"Extracted flatten features for train: \n{flatten_features.columns.tolist()}")

    time_features = feature_extractor.get_time_base_features(base_features)
    merged_features = pd.concat((base_features, flatten_features, time_features), axis=1)

    if use_cache:
        logger.info(f"cache features into {features_path}")
        merged_features.to_pickle(features_path)
    return merged_features, data.y


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


def main(data_path: str, features_path: Optional[str] = None):
    merged_features, target = load_features(data_path, features_path, use_cache=True)
    time_folds = TimeFolds(
        n_folds=5,
        minifold_size=60000,
        neutral_ratio=0.05,
        test_ratio=0.25,
        test_neutral_ratio=0.1,
    )
    time_folds.fit(merged_features, target)

    model_zoo = dev_utils.load_yaml("configs/models_zoo.yaml")["zoo"]
    use_regression_models = [
        "default_ridge",
        # "default_lasso",
        # "default_lgbm_v4",
        # "wild_lgbm",
        # "bayesian_ridge",
        # "default_fm",
    ]
    use_classification_models = [
        # "default_logreg"
    ]
    clf_model_configs = {model_name: model_zoo[model_name] for model_name in use_classification_models}
    reg_model_configs = {model_name: model_zoo[model_name] for model_name in use_regression_models}

    clf_runners = build_runners(time_folds, clf_model_configs, runners_dir="runners/5_folds_new_v2", regression=False)
    reg_runners = build_runners(time_folds, reg_model_configs, runners_dir="runners/5_folds_new_v2", regression=True)

    runners_stacking = RunnersStacking(reg_runners, clf_runners)
    runners_stacking.make_oof_ensemble()
    runners_stacking.make_test_ensemble()
    runners_stacking.make_ridge_stacking()


if __name__ == "__main__":
    print("run")
    DATA_PATH = "data/data.pkl"
    features_path = "data/features.pkl"
    main(DATA_PATH, features_path)
