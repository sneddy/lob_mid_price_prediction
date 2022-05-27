import pandas as pd

from xtx.features.feature_extractor import FeatureExtractor
from xtx.modeling.runners import CrossValRunner
from xtx.modeling.time_folds import TimeFolds
from xtx.utils.dev_utils import init_logger

pd.set_option("display.max_columns", 100)

logger = init_logger("logging/train.log")

MODEL_CONFIGS = {
    "default_ridge": {
        "model_module": "sklearn.linear_model",
        "model_cls": "Ridge",
        "model_params": {"alpha": 100},
    },
    "default_lasso": {
        "model_module": "sklearn.linear_model",
        "model_cls": "Lasso",
        "model_params": {"alpha": 0.01},
    },
    "default_lgbm": {
        "model_module": "lightgbm",
        "model_cls": "LGBMRegressor",
        "model_params": {
            "n_jobs": -1,
            "num_leaves": 13,
            "learning_rate": 0.01,
            "n_estimators": 500,
            "reg_lambda": 1,
            "colsample_bytree": 0.7,
            "subsample": 0.05,
        },
    },
}


def load_features(data_path: str):
    feature_extractor = FeatureExtractor(data_path)
    data = feature_extractor.data
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
    return merged_features, data.y


def main(data_path: str):
    merged_features, target = load_features(data_path)
    time_folds = TimeFolds(
        n_folds=5,
        minifold_size=60000,
        neutral_ratio=0.05,
        test_ratio=0.25,
        test_neutral_ratio=0.1,
    )
    time_folds.fit(merged_features, target)

    ridge_runner = CrossValRunner(time_folds, **MODEL_CONFIGS["default_ridge"])
    ridge_runner.fit(verbose=True)


if __name__ == "__main__":
    DATA_PATH = "data/data.pkl"
    main(DATA_PATH)
