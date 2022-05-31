import argparse
from typing import Any, Dict

import pandas as pd

import xtx.utils.dev_utils as dev_utils
import xtx.utils.modeling_utils as modeling_utils
from xtx.factory import load_features

pd.set_option("display.max_columns", 100)

logger = dev_utils.init_logger("logging/train.log")


def argparser():
    parser = argparse.ArgumentParser(description="Xtx pipeline")
    parser.add_argument("train_cfg", type=str, help="train config path")
    return parser.parse_args()


if __name__ == "__main__":
    args = argparser()
    experiment_config_path = args.train_cfg.strip("/")

    experiment: Dict[str, Any] = dev_utils.load_yaml(experiment_config_path)
    modeling_utils.set_all_seeds(experiment["random_seed"])
    logger.info(f"Experiment parameters:\n{experiment}")
    assert (
        experiment.get("cached_features", None) is not None
    ), f"Feature extraction should have cached_features in config {experiment_config_path}"
    load_features(
        data_path=experiment["train_data_path"],
        features_path=experiment["cached_features"],
        use_cache=False,
        pseudo_target=None,
        from_pool=True,
    )
    assert (
        experiment.get("cached_test_features", None) is not None
    ), f"Feature extraction should have cached_test_features in config {experiment_config_path}"

    load_features(
        data_path=experiment["test_data_path"],
        features_path=experiment["cached_test_features"],
        use_cache=False,
        pseudo_target=None,
        from_pool=True,
    )
