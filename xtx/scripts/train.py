import argparse
import os
from typing import Any, Dict

import numpy as np
import pandas as pd

import xtx.utils.dev_utils as dev_utils
import xtx.utils.modeling_utils as modeling_utils
from xtx.factory import build_runners, load_features
from xtx.modeling.stacking import RunnersStacking
from xtx.modeling.time_folds import TimeFolds

pd.set_option("display.max_columns", 100)

logger = dev_utils.init_logger("logging/train.log")


def argparser():
    parser = argparse.ArgumentParser(description="Xtx pipeline")
    parser.add_argument("train_cfg", type=str, help="train config path")
    return parser.parse_args()


def train(experiment: Dict[str, Any]):
    clf_runners = {}
    reg_runners = {}
    pseudo_target_list = experiment.get("pseudo_target", [0])

    merged_features, target = load_features(
        data_path=experiment["train_data_path"],
        features_path=experiment["cached_features"],
        use_cache=True,
        pseudo_target=None,
        from_pool=experiment["from_pool"],
        usecols=experiment.get("usecols", None),
    )

    for pseudo_target in pseudo_target_list:
        logger.info(f"Building runners for pseudo_target: {pseudo_target}")

        time_folds = TimeFolds(pseudo_target_shift=pseudo_target, **experiment["TimeFolds"])
        test_eval = False  # time_folds.test_ratio > 0
        time_folds.fit(merged_features, target)

        model_zoo = dev_utils.load_yaml(experiment["model_zoo"])
        use_regression_models = experiment.get("use_regression_models", [])
        # use_classification_models = experiment.get("use_classification_models", [])
        # clf_model_configs = {
        #     model_name: model_zoo["train_zoo"][model_name] for model_name in use_classification_models
        # }
        reg_model_configs = {model_name: model_zoo["train_zoo"][model_name] for model_name in use_regression_models}

        runners_dir = experiment["runners_dir"]
        current_clf_runners = {}
        # current_clf_runners = build_runners(
        #     time_folds, clf_model_configs, runners_dir=runners_dir, regression=False, pseudo_target=pseudo_target
        # )
        current_reg_runners = build_runners(
            time_folds, reg_model_configs, runners_dir=runners_dir, regression=True, pseudo_target=pseudo_target
        )
        clf_runners.update(current_clf_runners)
        reg_runners.update(current_reg_runners)

    runners_stacking = RunnersStacking(
        reg_runners=reg_runners,
        clf_runners=clf_runners,
        test_eval=test_eval,
        **model_zoo["stacking_zoo"][experiment["stacking_model"]],
    )
    runners_stacking.make_oof_ensemble()
    if test_eval > 0:
        runners_stacking.make_test_ensemble()
    runners_stacking.fit_stacking()
    return runners_stacking


def test(experiment: Dict[str, Any], runners_stacking: RunnersStacking):

    merged_features, target = load_features(
        data_path=experiment["test_data_path"],
        features_path=experiment["cached_test_features"],
        use_cache=True,
        pseudo_target=None,
        from_pool=experiment["from_pool"],
        usecols=experiment.get("usecols", None),
    )

    predicted_by_ensemble = runners_stacking.predict_by_ensemble(merged_features)
    predicted_by_stacking = runners_stacking.predict_by_stacking(merged_features)

    predictions_dir = experiment["predictions_dir"]
    os.makedirs(predictions_dir, exist_ok=True)
    ensemble_predictions_fpath = os.path.join(predictions_dir, "ensemble")
    logger.info(f"Saving ensemble prediction to {ensemble_predictions_fpath}.npy")
    np.save(ensemble_predictions_fpath, predicted_by_ensemble)

    stacking_predictions_fpath = os.path.join(predictions_dir, "stacking")
    logger.info(f"Saving stacking prediction to {stacking_predictions_fpath}.npy")
    np.save(stacking_predictions_fpath, predicted_by_stacking)
    return predicted_by_ensemble


if __name__ == "__main__":
    args = argparser()
    experiment_config_path = args.train_cfg.strip("/")

    experiment: Dict[str, Any] = dev_utils.load_yaml(experiment_config_path)
    modeling_utils.set_all_seeds(experiment["random_seed"])
    logger.info(f"Experiment parameters:\n{experiment}")

    runners_stacking = train(experiment)
    if experiment.get("test_data_path", None) is not None:
        test(experiment, runners_stacking)
