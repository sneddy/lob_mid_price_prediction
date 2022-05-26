import pandas as pd
from sklearn.linear_model import Ridge

from xtx.evaluation import CrossValRunner
from xtx.feature_extractor import FeatureExtractor
from xtx.time_folds import TimeFolds
from xtx.utils import init_logger

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


def main(data_path: str):
    feature_extractor = FeatureExtractor(data_path)
    data = feature_extractor.data
    base_features = feature_extractor.get_base_features()
    logger.info(f"Extracted base features for train: \n{base_features.columns}")

    time_folds = TimeFolds(
        n_folds=5,
        minifold_size=60000,
        neutral_ratio=0.15,
        test_ratio=0.25,
        test_neutral_ratio=0.1,
    )
    time_folds.fit(base_features, data.y)

    cross_val_runner = CrossValRunner(time_folds, **MODEL_CONFIGS["default_ridge"])
    cross_val_runner.fit(verbose=True)


if __name__ == "__main__":
    DATA_PATH = "data/data.pkl"
    main(DATA_PATH)
